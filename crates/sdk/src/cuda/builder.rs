//! # CPU Prover Builder
//!
//! This module provides a builder for the [`CpuProver`].

use super::CudaProver;
use crate::cpu::CpuProver;
use sp1_cuda::CudaProver as CudaProverImpl;

/// A builder for the [`CudaProver`].
///
/// The builder is used to configure the [`CudaProver`] before it is built.
#[derive(Debug, Default)]
pub struct CudaProverBuilder {
    cuda_device_id: Option<u32>,
}

impl CudaProverBuilder {
    /// Sets the CUDA device id.
    ///
    /// # Details
    /// Run the CUDA prover with the provided device id, all operations will be performed on this
    /// device index.
    ///
    /// # Example
    /// ```rust,no_run
    /// use sp1_sdk::ProverClient;
    ///
    /// let prover = ProverClient::builder().cuda().with_device_id(0).build();
    /// ```
    #[must_use]
    pub fn with_device_id(mut self, id: u32) -> Self {
        self.cuda_device_id = Some(id);
        self
    }

    /// Builds a [`CudaProver`].
    ///
    /// # Details
    /// This method will build a [`CudaProver`] with the given parameters.
    ///
    /// # Example
    /// ```rust,no_run
    /// use sp1_sdk::ProverClient;
    ///
    /// let prover = ProverClient::builder().cuda().build();
    /// ```
    #[must_use]
    pub async fn build(self) -> CudaProver {
        let cpu_prover = CpuProver::new().await;
        let cuda_prover = match self.cuda_device_id {
            Some(id) => CudaProverImpl::new_with_id(id).await,
            None => CudaProverImpl::new().await,
        };

        CudaProver {
            cpu_prover,
            prover: cuda_prover.expect("Failed to create the CUDA prover impl"),
        }
    }
}
