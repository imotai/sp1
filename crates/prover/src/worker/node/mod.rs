mod config;
mod init;

pub use config::*;
pub use init::*;
use sp1_core_executor::{SP1Context, SP1CoreOpts};
use sp1_core_machine::io::SP1Stdin;
use sp1_prover_types::{
    ArtifactClient, ArtifactType, InMemoryArtifactClient, TaskStatus, TaskType,
};

use crate::{
    worker::{LocalWorkerClient, ProofId, RawTaskRequest, RequesterId, WorkerClient},
    SP1CoreProof,
};

pub struct SP1LocalNode {
    artifact_client: InMemoryArtifactClient,
    worker_client: LocalWorkerClient,
}

impl SP1LocalNode {
    pub async fn prove_core(
        &self,
        elf: &[u8],
        stdin: SP1Stdin,
        opts: SP1CoreOpts,
        _context: SP1Context<'static>,
    ) -> anyhow::Result<SP1CoreProof> {
        // Create a request for the controller task.

        let proof_id = ProofId::new("core_proof");
        let requester_id = RequesterId::new("local node");

        let elf_artifact = self.artifact_client.create_artifact()?;
        self.artifact_client
            .upload_with_type(&elf_artifact, ArtifactType::Program, elf.to_vec())
            .await?;

        let stdin_artifact = self.artifact_client.create_artifact()?;
        self.artifact_client.upload_with_type(&stdin_artifact, ArtifactType::Stdin, stdin).await?;

        let opts_artifact = self.artifact_client.create_artifact()?;
        self.artifact_client
            .upload_with_type(&opts_artifact, ArtifactType::UnspecifiedArtifactType, opts)
            .await?;

        // Create an artifact for the output
        let output_artifact = self.artifact_client.create_artifact()?;

        let request = RawTaskRequest {
            inputs: vec![elf_artifact.clone(), stdin_artifact.clone(), opts_artifact.clone()],
            outputs: vec![output_artifact.clone()],
            proof_id,
            parent_id: None,
            parent_context: None,
            requester_id,
        };

        let task_id = self.worker_client.submit_task(TaskType::Controller, request).await?;
        let subscriber = self.worker_client.subscriber().await.per_task();
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
            .try_delete(&opts_artifact, ArtifactType::UnspecifiedArtifactType)
            .await?;
        self.artifact_client
            .try_delete(&output_artifact, ArtifactType::UnspecifiedArtifactType)
            .await?;

        Ok(proof)
    }
}
