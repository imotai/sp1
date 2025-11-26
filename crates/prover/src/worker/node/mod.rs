mod init;

use std::sync::Arc;

pub use init::*;
use mti::prelude::{MagicTypeIdExt, V7};
use sp1_core_executor::{
    CompressedMemory, ExecutionReport, GasEstimatingVM, MinimalExecutor, Program, SP1Context,
    SP1CoreOpts,
};
use sp1_core_machine::io::SP1Stdin;
use sp1_primitives::io::SP1PublicValues;
use sp1_prover_types::{
    network_base_types::ProofMode, Artifact, ArtifactClient, ArtifactType, InMemoryArtifactClient,
    TaskStatus, TaskType,
};
pub use sp1_verifier::{ProofFromNetwork, SP1Proof};
use tracing::{instrument, Instrument};

use crate::{
    verify::SP1Verifier,
    worker::{LocalWorkerClient, ProofId, RawTaskRequest, RequesterId, TaskContext, WorkerClient},
    SP1CoreProofData, SP1VerifyingKey,
};

pub(crate) struct SP1NodeInner {
    artifact_client: InMemoryArtifactClient,
    worker_client: LocalWorkerClient,
    verifier: SP1Verifier,
    opts: SP1CoreOpts,
}

pub struct SP1LocalNode {
    inner: Arc<SP1NodeInner>,
}

impl Clone for SP1LocalNode {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

impl SP1LocalNode {
    fn blocking_execute_program(
        &self,
        program: Arc<Program>,
        stdin: SP1Stdin,
        context: SP1Context<'static>,
    ) -> anyhow::Result<(SP1PublicValues, [u8; 32], ExecutionReport)> {
        // Phase 1: Use MinimalExecutor for fast execution and public values stream
        const MINIMAL_TRACE_CHUNK_THRESHOLD: u64 =
            2147483648 / std::mem::size_of::<sp1_jit::MemValue>() as u64;
        let max_number_trace_entries = std::env::var("MINIMAL_TRACE_CHUNK_THRESHOLD").map_or_else(
            |_| MINIMAL_TRACE_CHUNK_THRESHOLD,
            |s| s.parse::<u64>().unwrap_or(MINIMAL_TRACE_CHUNK_THRESHOLD),
        );

        let mut minimal_executor =
            MinimalExecutor::new(program.clone(), false, Some(max_number_trace_entries));

        // Feed stdin buffers to the executor
        for buf in stdin.buffer {
            minimal_executor.with_input(&buf);
        }

        // Execute the program to completion, collecting all trace chunks
        let mut chunks = Vec::new();
        while let Some(chunk) = minimal_executor.execute_chunk() {
            chunks.push(chunk);
        }

        tracing::info!("chunks: {:?}", chunks.len());

        // Extract the public values stream from minimal executor
        let public_value_stream = minimal_executor.into_public_values_stream();
        let public_values = SP1PublicValues::from(&public_value_stream);

        tracing::info!("public_value_stream: {:?}", public_value_stream);

        let mut accumulated_report = ExecutionReport::default();
        let filler: [u8; 32] = [0; 32];

        let mut touched_addresses = CompressedMemory::new();

        for chunk in chunks {
            let mut gas_estimating_vm = GasEstimatingVM::new(
                &chunk,
                program.clone(),
                context.proof_nonce,
                &mut touched_addresses,
                self.inner.opts.clone(),
            );
            let report = gas_estimating_vm.execute().unwrap();
            accumulated_report += report;
        }

        Ok((public_values, filler, accumulated_report))
    }

    pub async fn setup(&self, elf: &[u8]) -> anyhow::Result<SP1VerifyingKey> {
        let elf_artifact = self.inner.artifact_client.create_artifact()?;
        self.inner.artifact_client.upload_program(&elf_artifact, elf.to_vec()).await?;

        // Create a setup task and wait for the vk
        let vk_artifact = self.inner.artifact_client.create_artifact()?;
        let context = TaskContext {
            proof_id: ProofId::new("core_proof"),
            parent_id: None,
            parent_context: None,
            requester_id: RequesterId::new("local node"),
        };
        let setup_request = RawTaskRequest {
            inputs: vec![elf_artifact.clone()],
            outputs: vec![vk_artifact.clone()],
            context: context.clone(),
        };
        tracing::trace!("submitting setup task");
        let setup_id =
            self.inner.worker_client.submit_task(TaskType::SetupVkey, setup_request).await?;
        // Wait for the setup task to finish
        let subscriber =
            self.inner.worker_client.subscriber(context.proof_id.clone()).await?.per_task();
        let status =
            subscriber.wait_task(setup_id).instrument(tracing::debug_span!("setup task")).await?;
        if status != TaskStatus::Succeeded {
            return Err(anyhow::anyhow!("setup task failed"));
        }
        tracing::trace!("setup task succeeded");
        // Download the vk
        let vk = self.inner.artifact_client.download::<SP1VerifyingKey>(&vk_artifact).await?;

        // Clean up the artifacts
        self.inner.artifact_client.try_delete(&elf_artifact, ArtifactType::Program).await?;
        self.inner
            .artifact_client
            .try_delete(&vk_artifact, ArtifactType::UnspecifiedArtifactType)
            .await?;

        Ok(vk)
    }

    #[instrument(name = "execute_program", skip_all)]
    pub async fn execute(
        &self,
        elf: &[u8],
        stdin: SP1Stdin,
        context: SP1Context<'static>,
    ) -> anyhow::Result<(SP1PublicValues, [u8; 32], ExecutionReport)> {
        let node = self.clone();
        let program = Program::from(elf)
            .map_err(|e| anyhow::anyhow!("failed to dissassemble program: {}", e))?;
        let program = Arc::new(program);
        tokio::task::spawn_blocking(move || node.blocking_execute_program(program, stdin, context))
            .await?
    }

    pub async fn prove(
        &self,
        elf: &[u8],
        stdin: SP1Stdin,
        context: SP1Context<'static>,
    ) -> anyhow::Result<ProofFromNetwork> {
        self.prove_with_mode(elf, stdin, context, ProofMode::Compressed).await
    }

    pub async fn prove_with_mode(
        &self,
        elf: &[u8],
        stdin: SP1Stdin,
        sp1_context: SP1Context<'static>,
        mode: ProofMode,
    ) -> anyhow::Result<ProofFromNetwork> {
        // Create a request for the controller task.
        let pid = std::process::id();
        let context = TaskContext {
            proof_id: ProofId::new("proof".create_type_id::<V7>().to_string()),
            parent_id: None,
            parent_context: None,
            requester_id: RequesterId::new(format!("local-node-{pid}")),
        };

        let elf_artifact = self.inner.artifact_client.create_artifact()?;
        self.inner.artifact_client.upload_program(&elf_artifact, elf.to_vec()).await?;

        let proof_nonce_artifact = self.inner.artifact_client.create_artifact()?;
        self.inner
            .artifact_client
            .upload::<[u32; 4]>(&proof_nonce_artifact, sp1_context.proof_nonce)
            .await?;

        let stdin_artifact = self.inner.artifact_client.create_artifact()?;
        self.inner
            .artifact_client
            .upload_with_type(&stdin_artifact, ArtifactType::Stdin, stdin)
            .await?;

        let mode_artifact = Artifact((mode as i32).to_string());

        // Create an artifact for the output
        let output_artifact = self.inner.artifact_client.create_artifact()?;

        let request = RawTaskRequest {
            inputs: vec![
                elf_artifact.clone(),
                stdin_artifact.clone(),
                mode_artifact.clone(),
                proof_nonce_artifact.clone(),
            ],
            outputs: vec![output_artifact.clone()],
            context: context.clone(),
        };

        let task_id = self.inner.worker_client.submit_task(TaskType::Controller, request).await?;
        let subscriber =
            self.inner.worker_client.subscriber(context.proof_id.clone()).await?.per_task();
        let status = subscriber.wait_task(task_id).await?;
        if status != TaskStatus::Succeeded {
            return Err(anyhow::anyhow!("controller task failed"));
        }

        // Clean up the input artifacts
        self.inner.artifact_client.try_delete(&elf_artifact, ArtifactType::Program).await?;
        self.inner.artifact_client.try_delete(&stdin_artifact, ArtifactType::Stdin).await?;

        // Download the output proof and return it.
        let proof =
            self.inner.artifact_client.download::<ProofFromNetwork>(&output_artifact).await?;
        // Clean up the output artifact
        self.inner
            .artifact_client
            .try_delete(&output_artifact, ArtifactType::UnspecifiedArtifactType)
            .await?;

        self.inner
            .artifact_client
            .try_delete(&proof_nonce_artifact, ArtifactType::UnspecifiedArtifactType)
            .await?;

        Ok(proof)
    }

    /// Get a reference to the verifier.
    #[must_use]
    #[inline]
    pub fn verifier(&self) -> &SP1Verifier {
        &self.inner.verifier
    }

    pub fn verify(&self, vk: &SP1VerifyingKey, proof: &SP1Proof) -> anyhow::Result<()> {
        // Verify the underlying proof.
        match proof {
            SP1Proof::Core(proof) => {
                let core_proof = SP1CoreProofData(proof.clone());
                self.verifier().verify(&core_proof, vk)?;
            }
            SP1Proof::Compressed(proof) => {
                self.verifier().verify_compressed(proof, vk)?;
            }
            SP1Proof::Plonk(_) => {
                todo!()
            }
            SP1Proof::Groth16(_) => {
                todo!()
            }
        }

        Ok(())
    }
}

#[cfg(all(test, feature = "experimental"))]
mod tests {
    use serial_test::serial;
    use sp1_core_machine::utils::setup_logger;

    use crate::worker::cpu_worker_builder;

    use super::*;

    #[tokio::test]
    #[serial]
    async fn test_e2e_node() -> anyhow::Result<()> {
        setup_logger();

        let elf = test_artifacts::FIBONACCI_ELF;
        let stdin = SP1Stdin::default();
        let mode = ProofMode::Compressed;

        let client = SP1LocalNodeBuilder::from_worker_client_builder(cpu_worker_builder())
            .build()
            .await
            .unwrap();

        let proof_nonce = [0x6284, 0xC0DE, 0x4242, 0xCAFE];

        let time = tokio::time::Instant::now();
        let context = SP1Context { proof_nonce, ..Default::default() };

        let (_, _, report) = client.execute(&elf, stdin.clone(), context.clone()).await.unwrap();

        let execute_time = time.elapsed();
        let cycles = report.total_instruction_count() as usize;
        tracing::info!(
            "execute time: {:?}, cycles: {}, gas: {:?}",
            execute_time,
            cycles,
            report.gas()
        );

        let time = tokio::time::Instant::now();
        let vk = client.setup(&elf).await.unwrap();
        let setup_time = time.elapsed();
        tracing::info!("setup time: {:?}", setup_time);

        let time = tokio::time::Instant::now();

        tracing::info!("proving with mode: {mode:?}");
        let proof = client.prove_with_mode(&elf, stdin, context, mode).await.unwrap();
        let proof_time = time.elapsed();
        tracing::info!("proof time: {:?}", proof_time);

        // Verify the proof
        client.verify(&vk, &proof.proof).unwrap();

        Ok(())
    }
}
