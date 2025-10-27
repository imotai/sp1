mod init;

use std::sync::Arc;

pub use init::*;
use sp1_core_executor::{
    CompressedMemory, ExecutionReport, GasEstimatingVM, MinimalExecutor, Program, SP1Context,
};
use sp1_core_machine::io::SP1Stdin;
use sp1_primitives::io::SP1PublicValues;
use sp1_prover_types::{
    network_base_types::ProofMode, Artifact, ArtifactClient, ArtifactType, InMemoryArtifactClient,
    TaskStatus, TaskType,
};
use tracing::{instrument, Instrument};

use crate::{
    verify::SP1Verifier,
    worker::{
        cluster_opts, LocalWorkerClient, ProofId, RawTaskRequest, RequesterId, SP1WorkerProof,
        WorkerClient,
    },
    SP1CoreProof, SP1VerifyingKey,
};

pub(crate) struct SP1NodeInner {
    artifact_client: InMemoryArtifactClient,
    worker_client: LocalWorkerClient,
    verifier: SP1Verifier,
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
        _context: SP1Context<'static>,
    ) -> anyhow::Result<(SP1PublicValues, [u8; 32], ExecutionReport)> {
        // Phase 1: Use MinimalExecutor for fast execution and public values stream
        const MAX_NUMBER_TRACE_ENTRIES: u64 =
            2147483648 / std::mem::size_of::<sp1_jit::MemValue>() as u64;

        let mut minimal_executor =
            MinimalExecutor::new(program.clone(), false, Some(MAX_NUMBER_TRACE_ENTRIES));

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

        let opts = cluster_opts();
        for chunk in chunks {
            let mut gas_estimating_vm =
                GasEstimatingVM::new(&chunk, program.clone(), &mut touched_addresses, opts.clone());
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
        let proof_id = ProofId::new("core_proof");
        let requester_id = RequesterId::new("local node");
        let setup_request = RawTaskRequest {
            inputs: vec![elf_artifact.clone()],
            outputs: vec![vk_artifact.clone()],
            proof_id: proof_id.clone(),
            parent_id: None,
            parent_context: None,
            requester_id: requester_id.clone(),
        };
        tracing::trace!("submitting setup task");
        let setup_id =
            self.inner.worker_client.submit_task(TaskType::SetupVkey, setup_request).await?;
        // Wait for the setup task to finish
        let subscriber = self.inner.worker_client.subscriber(proof_id.clone()).await?.per_task();
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
        _context: SP1Context<'static>,
    ) -> anyhow::Result<(SP1PublicValues, [u8; 32], ExecutionReport)> {
        let node = self.clone();
        let program = Program::from(elf)
            .map_err(|e| anyhow::anyhow!("failed to dissassemble program: {}", e))?;
        let program = Arc::new(program);
        tokio::task::spawn_blocking(move || node.blocking_execute_program(program, stdin, _context))
            .await?
    }

    pub async fn prove_core(
        &self,
        elf: &[u8],
        stdin: SP1Stdin,
        _context: SP1Context<'static>,
    ) -> anyhow::Result<SP1CoreProof> {
        let proof = self.prove_with_mode(elf, stdin, _context, ProofMode::Core).await?;
        match proof {
            SP1WorkerProof::Core(proof) => Ok(proof),
            _ => Err(anyhow::anyhow!("proof is not a core proof")),
        }
    }

    pub async fn prove(
        &self,
        elf: &[u8],
        stdin: SP1Stdin,
        _context: SP1Context<'static>,
    ) -> anyhow::Result<SP1WorkerProof> {
        self.prove_with_mode(elf, stdin, _context, ProofMode::Compressed).await
    }

    pub async fn prove_with_mode(
        &self,
        elf: &[u8],
        stdin: SP1Stdin,
        _context: SP1Context<'static>,
        mode: ProofMode,
    ) -> anyhow::Result<SP1WorkerProof> {
        // Create a request for the controller task.

        let proof_id = ProofId::new("core_proof");
        let requester_id = RequesterId::new("local node");

        let elf_artifact = self.inner.artifact_client.create_artifact()?;
        self.inner.artifact_client.upload_program(&elf_artifact, elf.to_vec()).await?;

        let stdin_artifact = self.inner.artifact_client.create_artifact()?;
        self.inner
            .artifact_client
            .upload_with_type(&stdin_artifact, ArtifactType::Stdin, stdin)
            .await?;

        let mode_artifact = Artifact((mode as i32).to_string());

        // Create an artifact for the output
        let output_artifact = self.inner.artifact_client.create_artifact()?;

        let request = RawTaskRequest {
            inputs: vec![elf_artifact.clone(), stdin_artifact.clone(), mode_artifact.clone()],
            outputs: vec![output_artifact.clone()],
            proof_id: proof_id.clone(),
            parent_id: None,
            parent_context: None,
            requester_id,
        };

        let task_id = self.inner.worker_client.submit_task(TaskType::Controller, request).await?;
        let subscriber = self.inner.worker_client.subscriber(proof_id.clone()).await?.per_task();
        let status = subscriber.wait_task(task_id).await?;
        if status != TaskStatus::Succeeded {
            return Err(anyhow::anyhow!("controller task failed"));
        }

        // Clean up the input artifacts
        self.inner.artifact_client.try_delete(&elf_artifact, ArtifactType::Program).await?;
        self.inner.artifact_client.try_delete(&stdin_artifact, ArtifactType::Stdin).await?;

        // Download the output proof and return it.
        let proof =
            SP1WorkerProof::download(mode, &output_artifact, &self.inner.artifact_client).await?;
        // Clean up the output artifact
        self.inner
            .artifact_client
            .try_delete(&output_artifact, ArtifactType::UnspecifiedArtifactType)
            .await?;

        Ok(proof)
    }

    /// Get a reference to the verifier.
    #[must_use]
    #[inline]
    pub fn verifier(&self) -> &SP1Verifier {
        &self.inner.verifier
    }

    pub fn verify(&self, vk: &SP1VerifyingKey, proof: &SP1WorkerProof) -> anyhow::Result<()> {
        match proof {
            SP1WorkerProof::Core(proof) => self
                .verifier()
                .verify(&proof.proof, vk)
                .map_err(|e| anyhow::anyhow!("failed to verify core proof: {:?}", e)),
            SP1WorkerProof::Compressed(proof) => self
                .verifier()
                .verify_compressed(proof, vk)
                .map_err(|e| anyhow::anyhow!("failed to verify compressed proof:{:?}", e)),
        }
    }
}
