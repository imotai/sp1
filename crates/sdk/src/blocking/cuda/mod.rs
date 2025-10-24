//! # SP1 CUDA Prover
//!
//! A prover that uses the CUDA to execute and prove programs.

/// The builder for the CUDA prover.
pub mod builder;
/// The CUDA prove request type.
pub mod prove;

use std::sync::Arc;

use crate::{
    blocking::{cpu::CpuProver, prover::BaseProveRequest, Prover},
    cpu::{prove_groth16, prove_plonk},
    SP1Proof, SP1ProofMode, SP1ProofWithPublicValues,
};

use prove::CudaProveRequest;
use sp1_core_executor::SP1Context;
use sp1_core_machine::io::SP1Stdin;
use sp1_cuda::{CudaClientError, CudaProver as CudaProverImpl, CudaProvingKey};
use sp1_primitives::Elf;
use sp1_prover::{
    components::CpuSP1ProverComponents, local::LocalProver, SP1CoreProofData, SP1ProofWithMetadata,
};

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

    fn inner(&self) -> Arc<LocalProver<CpuSP1ProverComponents>> {
        self.cpu_prover.inner()
    }

    fn setup(&self, elf: Elf) -> Result<Self::ProvingKey, Self::Error> {
        crate::blocking::block_on(self.prover.setup(elf))
    }

    fn prove<'a>(&'a self, pk: &'a Self::ProvingKey, stdin: SP1Stdin) -> Self::ProveRequest<'a> {
        CudaProveRequest { base: BaseProveRequest::new(self, pk, stdin) }
    }
}

impl CudaProver {
    #[allow(clippy::needless_pass_by_value)]
    fn prove_impl(
        &self,
        pk: &CudaProvingKey,
        stdin: SP1Stdin,
        context: SP1Context<'static>,
        mode: SP1ProofMode,
    ) -> Result<SP1ProofWithPublicValues, CudaClientError> {
        crate::blocking::block_on(async move {
            // Collect the deferred proofs
            let deferred_proofs =
                stdin.proofs.iter().map(|(reduce_proof, _)| reduce_proof.clone()).collect();

            // Generate the core proof.
            let proof: SP1ProofWithMetadata<SP1CoreProofData> =
                self.prover.core(pk, stdin, context.proof_nonce).await?;
            if mode == SP1ProofMode::Core {
                return Ok(SP1ProofWithPublicValues::new(
                    SP1Proof::Core(proof.proof.0),
                    proof.public_values,
                    self.version().to_string(),
                ));
            }

            // Generate the compressed proof.
            let public_values = proof.public_values.clone();
            let compressed_proof =
                self.prover.compress(pk.verifying_key(), proof, deferred_proofs).await?;
            if mode == SP1ProofMode::Compressed {
                return Ok(SP1ProofWithPublicValues::new(
                    SP1Proof::Compressed(Box::new(compressed_proof)),
                    public_values,
                    self.version().to_string(),
                ));
            }

            let shrink_proof =
                crate::blocking::block_on(self.prover.clone().shrink(compressed_proof))?;
            let wrap_proof = crate::blocking::block_on(self.prover.clone().wrap(shrink_proof))?;
            match mode {
                SP1ProofMode::Groth16 => {
                    let groth16_proof = crate::blocking::block_on(prove_groth16(
                        &self.cpu_prover.prover,
                        wrap_proof,
                    ));
                    Ok(SP1ProofWithPublicValues::new(
                        SP1Proof::Groth16(groth16_proof),
                        public_values,
                        self.version().to_string(),
                    ))
                }
                SP1ProofMode::Plonk => {
                    let plonk_proof =
                        crate::blocking::block_on(prove_plonk(&self.cpu_prover.prover, wrap_proof));
                    Ok(SP1ProofWithPublicValues::new(
                        SP1Proof::Plonk(plonk_proof),
                        public_values,
                        self.version().to_string(),
                    ))
                }
                _ => unreachable!(),
            }
        })
    }
}
