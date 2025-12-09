//! # SP1 CUDA Prover
//!
//! A prover that uses the CUDA to execute and prove programs.

/// The builder for the CUDA prover.
pub mod builder;
/// The CUDA prove request type.
pub mod prove;

use std::sync::Arc;

use crate::blocking::{cpu::CpuProver, prover::BaseProveRequest, Prover};

use prove::CudaProveRequest;
use sp1_core_machine::io::SP1Stdin;
use sp1_cuda::{CudaClientError, CudaProver as CudaProverImpl, CudaProvingKey};
use sp1_primitives::Elf;
use sp1_prover::worker::SP1LocalNode;

/// A prover that uses the CPU for execution and the CUDA for proving.
#[derive(Clone)]
pub struct CudaProver {
    pub(crate) cpu_prover: CpuProver,
    pub(crate) prover: CudaProverImpl,
}

impl Prover for CudaProver {
    type ProvingKey = CudaProvingKey;
    type Error = CudaClientError;
    type ProveRequest<'a> = CudaProveRequest<'a>;

    fn inner(&self) -> Arc<SP1LocalNode> {
        self.cpu_prover.inner()
    }

    fn setup(&self, elf: Elf) -> Result<Self::ProvingKey, Self::Error> {
        crate::blocking::block_on(self.prover.setup(elf))
    }

    fn prove<'a>(&'a self, pk: &'a Self::ProvingKey, stdin: SP1Stdin) -> Self::ProveRequest<'a> {
        CudaProveRequest { base: BaseProveRequest::new(self, pk, stdin) }
    }
}
