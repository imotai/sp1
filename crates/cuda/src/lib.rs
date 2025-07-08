/// The shared API between the client and server.
pub mod api;

/// The client that interacts with the CUDA server.
pub mod client;

/// The proving key type, which is a "remote" reference to a key held by the CUDA server.
pub mod pk;

/// The server startup logic.
mod server;

use std::path::PathBuf;

pub use client::CudaClientError;
pub use pk::CudaProvingKey;
use semver::Version;
use sp1_core_machine::{io::SP1Stdin, recursion::SP1RecursionProof};
use sp1_primitives::Elf;
use sp1_prover::{InnerSC, OuterSC, SP1CoreProof, SP1VerifyingKey};

use crate::client::CudaClient;

const MIN_CUDA_VERSION: Version = Version::new(12, 6, 0);

#[derive(Clone)]
pub struct CudaProver {
    client: CudaClient,
}

impl CudaProver {
    /// Create a new prover, using the 0th CUDA device.
    pub async fn new() -> Result<Self, CudaClientError> {
        Ok(Self { client: CudaClient::connect(0).await? })
    }

    /// Create a new prover, using the given CUDA device.
    pub async fn new_with_id(cuda_id: u32) -> Result<Self, CudaClientError> {
        Ok(Self { client: CudaClient::connect(cuda_id).await? })
    }

    pub async fn setup(&self, elf: Elf) -> Result<CudaProvingKey, CudaClientError> {
        self.client.setup(elf).await
    }

    pub async fn core(
        &self,
        key: &CudaProvingKey,
        stdin: SP1Stdin,
    ) -> Result<SP1CoreProof, CudaClientError> {
        self.client.core(key, stdin).await
    }

    pub async fn compress(
        &self,
        vk: &SP1VerifyingKey,
        proof: SP1CoreProof,
        deferred: Vec<SP1RecursionProof<InnerSC>>,
    ) -> Result<SP1RecursionProof<InnerSC>, CudaClientError> {
        self.client.compress(vk, proof, deferred).await
    }

    pub async fn shrink(
        &self,
        proof: SP1RecursionProof<InnerSC>,
    ) -> Result<SP1RecursionProof<InnerSC>, CudaClientError> {
        self.client.shrink(proof).await
    }

    pub async fn wrap(
        &self,
        proof: SP1RecursionProof<InnerSC>,
    ) -> Result<SP1RecursionProof<OuterSC>, CudaClientError> {
        self.client.wrap(proof).await
    }
}

async fn check_cuda_version() -> Result<(), CudaClientError> {
    #[derive(serde::Deserialize, Debug)]
    struct CudaVersions {
        cuda_cudart: CudaVersion,
    }

    #[derive(serde::Deserialize, Debug)]
    struct CudaVersion {
        version: semver::Version,
    }

    let cuda_path: PathBuf = std::env::var("CUDA_PATH")
        .map_err(|_| CudaClientError::Unexpected("CUDA_PATH env var is not set".to_string()))?
        .into();

    let version_file = cuda_path.join("version.json");
    let version_file = tokio::fs::read_to_string(&version_file)
        .await
        .map_err(|_| CudaClientError::Unexpected("Failed to read version.json".to_string()))?;

    let versions: CudaVersions = serde_json::from_str(&version_file)
        .map_err(|_| CudaClientError::Unexpected("Failed to parse version.json".to_string()))?;

    if versions.cuda_cudart.version < MIN_CUDA_VERSION {
        return Err(CudaClientError::CudaVersionTooOld);
    }

    Ok(())
}
