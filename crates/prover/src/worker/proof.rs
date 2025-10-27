use serde::{Deserialize, Serialize};
use sp1_hypercube::SP1RecursionProof;
use sp1_primitives::SP1GlobalContext;
use sp1_prover_types::{network_base_types::ProofMode, Artifact, ArtifactClient};

use crate::{worker::TaskError, InnerSC, SP1CoreProof};

// TODO: Consider unifying with the same type in the sdk crate.
#[derive(Clone, Serialize, Deserialize)]
pub enum SP1WorkerProof {
    Core(SP1CoreProof),
    Compressed(Box<SP1RecursionProof<SP1GlobalContext, InnerSC>>),
}

impl SP1WorkerProof {
    pub fn num_shards(&self) -> Option<usize> {
        match self {
            SP1WorkerProof::Core(proof) => Some(proof.proof.0.len()),
            SP1WorkerProof::Compressed(_) => Some(0),
        }
    }

    pub async fn download(
        mode: ProofMode,
        output: &Artifact,
        artifact_client: &impl ArtifactClient,
    ) -> Result<SP1WorkerProof, TaskError> {
        match mode {
            ProofMode::Core => {
                let proof = artifact_client.download::<SP1CoreProof>(output).await?;
                Ok(SP1WorkerProof::Core(proof))
            }
            ProofMode::Compressed => {
                let proof = artifact_client
                    .download::<SP1RecursionProof<SP1GlobalContext, InnerSC>>(output)
                    .await?;
                Ok(SP1WorkerProof::Compressed(Box::new(proof)))
            }
            _ => unimplemented!("proof mode not supported: {:?}", mode),
        }
    }
}
