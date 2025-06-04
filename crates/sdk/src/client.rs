//! # SP1 Prover Client
//!
//! A client for interacting with the prover for the SP1 RISC-V zkVM.

use crate::{cpu::builder::CpuProverBuilder, cuda::builder::CudaProverBuilder, env::EnvProver};

#[cfg(feature = "network")]
use crate::network::builder::NetworkProverBuilder;

/// An entrypoint for interacting with the prover for the SP1 RISC-V zkVM.
///
/// IMPORTANT: `ProverClient` only needs to be initialized ONCE and can be reused for subsequent
/// proving operations, all provers types are cheap to clone and share across threads.
///
/// Note that the initialization may be slow as it loads necessary proving parameters and sets up
/// the environment.
pub struct ProverClient;

impl ProverClient {
    /// Builds an [`EnvProver`], which loads the mode and any settings from the environment.
    ///
    /// # Usage
    /// ```no_run
    /// use sp1_sdk::{Prover, ProverClient, SP1Stdin};
    ///
    /// std::env::set_var("SP1_PROVER", "network");
    /// std::env::set_var("NETWORK_PRIVATE_KEY", "...");
    /// let prover = ProverClient::from_env();
    ///
    /// let elf = &[1, 2, 3];
    /// let stdin = SP1Stdin::new();
    ///
    /// let (pk, vk) = prover.setup(elf);
    /// let proof = prover.prove(&pk, &stdin).compressed().run().unwrap();
    /// ```
    #[must_use]
    pub async fn from_env() -> EnvProver {
        EnvProver::new().await
    }

    /// Creates a new [`ProverClientBuilder`] so that you can configure the prover client.
    #[must_use]
    pub fn builder() -> ProverClientBuilder {
        ProverClientBuilder
    }
}

/// A builder to define which proving client to use.
pub struct ProverClientBuilder;

impl ProverClientBuilder {
    /// Builds a [`CpuProver`] specifically for local CPU proving.
    ///
    /// # Usage
    /// ```no_run
    /// use sp1_sdk::{Prover, ProverClient, SP1Stdin};
    ///
    /// let elf = &[1, 2, 3];
    /// let stdin = SP1Stdin::new();
    ///
    /// let prover = ProverClient::builder().cpu().build();
    /// let (pk, vk) = prover.setup(elf).await;
    /// let proof = prover.prove(pk, stdin).compressed().run().await.unwrap();
    /// ```
    #[must_use]
    pub fn cpu(&self) -> CpuProverBuilder {
        CpuProverBuilder
    }

    /// Builds a [`CudaProver`] specifically for local proving on NVIDIA GPUs.
    ///
    /// # Example
    /// ```no_run
    /// use sp1_sdk::{Prover, ProverClient, SP1Stdin};
    ///
    /// let elf = &[1, 2, 3];
    /// let stdin = SP1Stdin::new();
    ///
    /// let prover = ProverClient::builder().cuda().build();
    /// let (pk, vk) = prover.setup(elf);
    /// let proof = prover.prove(&pk, &stdin).compressed().run().unwrap();
    /// ```
    #[must_use]
    pub fn cuda(&self) -> CudaProverBuilder {
        CudaProverBuilder::default()
    }

    /// Builds a [`NetworkProver`] specifically for proving on the network.
    ///
    /// # Example
    /// ```no_run
    /// use sp1_sdk::{Prover, ProverClient, SP1Stdin};
    ///
    /// let elf = &[1, 2, 3];
    /// let stdin = SP1Stdin::new();
    ///
    /// let prover = ProverClient::builder().network().build().await;
    /// let (pk, vk) = prover.setup(elf).await;
    /// let proof = prover.prove(pk, stdin).compressed().await.unwrap();
    /// ```
    #[cfg(feature = "network")]
    #[must_use]
    pub fn network(&self) -> NetworkProverBuilder {
        NetworkProverBuilder { private_key: None, rpc_url: None, tee_signers: None }
    }
}
