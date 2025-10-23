mod init;

use std::borrow::Borrow;

pub use init::*;
use sp1_core_executor::SP1Context;
use sp1_core_machine::io::SP1Stdin;
use sp1_hypercube::air::PublicValues;
use sp1_prover_types::{
    network_base_types::ProofMode, Artifact, ArtifactClient, ArtifactType, InMemoryArtifactClient,
    TaskStatus, TaskType,
};
use tracing::Instrument;

use crate::{
    verify::SP1Verifier,
    worker::{LocalWorkerClient, ProofId, RawTaskRequest, RequesterId, WorkerClient},
    SP1CoreProof, SP1VerifyingKey,
};

pub struct SP1LocalNode {
    artifact_client: InMemoryArtifactClient,
    worker_client: LocalWorkerClient,
    verifier: SP1Verifier,
}

impl SP1LocalNode {
    pub async fn setup(&self, elf: &[u8]) -> anyhow::Result<SP1VerifyingKey> {
        let elf_artifact = self.artifact_client.create_artifact()?;
        self.artifact_client.upload_program(&elf_artifact, elf.to_vec()).await?;

        // Create a setup task and wait for the vk
        let vk_artifact = self.artifact_client.create_artifact()?;
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
        let setup_id = self.worker_client.submit_task(TaskType::SetupVkey, setup_request).await?;
        // Wait for the setup task to finish
        let subscriber = self.worker_client.subscriber(proof_id.clone()).await?.per_task();
        let status =
            subscriber.wait_task(setup_id).instrument(tracing::debug_span!("setup task")).await?;
        if status != TaskStatus::Succeeded {
            return Err(anyhow::anyhow!("setup task failed"));
        }
        tracing::trace!("setup task succeeded");
        // Download the vk
        let vk = self.artifact_client.download::<SP1VerifyingKey>(&vk_artifact).await?;

        // Clean up the artifacts
        self.artifact_client.try_delete(&elf_artifact, ArtifactType::Program).await?;
        self.artifact_client
            .try_delete(&vk_artifact, ArtifactType::UnspecifiedArtifactType)
            .await?;

        Ok(vk)
    }

    pub async fn prove_core(
        &self,
        elf: &[u8],
        stdin: SP1Stdin,
        _context: SP1Context<'static>,
    ) -> anyhow::Result<SP1CoreProof> {
        // Create a request for the controller task.

        let proof_id = ProofId::new("core_proof");
        let requester_id = RequesterId::new("local node");

        let elf_artifact = self.artifact_client.create_artifact()?;
        self.artifact_client.upload_program(&elf_artifact, elf.to_vec()).await?;

        let stdin_artifact = self.artifact_client.create_artifact()?;
        self.artifact_client.upload_with_type(&stdin_artifact, ArtifactType::Stdin, stdin).await?;

        let mode_artifact = Artifact((ProofMode::Core as i32).to_string());

        // Create an artifact for the output
        let output_artifact = self.artifact_client.create_artifact()?;

        let request = RawTaskRequest {
            inputs: vec![elf_artifact.clone(), stdin_artifact.clone(), mode_artifact.clone()],
            outputs: vec![output_artifact.clone()],
            proof_id: proof_id.clone(),
            parent_id: None,
            parent_context: None,
            requester_id,
        };

        let task_id = self.worker_client.submit_task(TaskType::Controller, request).await?;
        let subscriber = self.worker_client.subscriber(proof_id.clone()).await?.per_task();
        let status = subscriber.wait_task(task_id).await?;
        if status != TaskStatus::Succeeded {
            return Err(anyhow::anyhow!("controller task failed"));
        }
        // Download the proof
        let proof = self.artifact_client.download::<SP1CoreProof>(&output_artifact).await?;

        // Clean up the artifacts
        self.artifact_client.try_delete(&elf_artifact, ArtifactType::Program).await?;
        self.artifact_client.try_delete(&stdin_artifact, ArtifactType::Stdin).await?;
        self.artifact_client
            .try_delete(&mode_artifact, ArtifactType::UnspecifiedArtifactType)
            .await?;
        self.artifact_client
            .try_delete(&output_artifact, ArtifactType::UnspecifiedArtifactType)
            .await?;

        // Sort the shard proofs by its range.
        let proof = tokio::task::spawn_blocking(move || {
            let mut proof = proof.clone();
            proof.proof.0.sort_by_key(|shard_proof| {
                let public_values: &PublicValues<[_; 4], [_; 3], [_; 4], _> =
                    shard_proof.public_values.as_slice().borrow();
                public_values.range()
            });
            proof
        })
        .await?;

        Ok(proof)
    }

    /// Get a reference to the verifier.
    #[must_use]
    #[inline]
    pub fn verifier(&self) -> &SP1Verifier {
        &self.verifier
    }
}
