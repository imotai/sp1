//! # SP1 CPU Prover
//!
//! A prover that uses the CPU to execute and prove programs.

pub mod builder;
pub mod prove;

use std::sync::Arc;

use anyhow::Result;
use prove::CpuProveBuilder;
use sp1_core_executor::{ExecutionError, Program, SP1Context};
use sp1_core_machine::io::SP1Stdin;
use sp1_hypercube::{prover::MachineProvingKey, SP1RecursionProof};
use sp1_primitives::{Elf, SP1GlobalContext, SP1OuterGlobalContext};
use sp1_prover::{
    build::{try_build_groth16_bn254_artifacts_dev, try_build_plonk_bn254_artifacts_dev},
    components::SP1ProverComponents,
    error::SP1ProverError,
    local::{LocalProver, LocalProverOpts},
    Groth16Bn254Proof, InnerSC, OuterSC, PlonkBn254Proof, SP1ProverBuilder, SP1VerifyingKey,
};
use sp1_prover::{
    components::CpuSP1ProverComponents,
    // verify::{verify_groth16_bn254_public_inputs, verify_plonk_bn254_public_inputs},
    SP1CoreProofData,
    SP1ProofWithMetadata,
};

use crate::{
    install::try_install_circuit_artifacts,
    prover::{Prover, ProvingKey, SendFutureResult},
    SP1Proof, SP1ProofMode, SP1ProofWithPublicValues,
};

use thiserror::Error;

/// A prover that uses the CPU to execute and prove programs.
#[derive(Clone)]
pub struct CpuProver {
    pub(crate) prover: Arc<LocalProver<CpuSP1ProverComponents>>,
}

/// A proving key for the [`CpuProver`].
///
/// This struct is used to store the proving key for the [`CpuProver`].
#[derive(Clone)]
pub struct CPUProvingKey {
    pub(crate) raw: Arc<
        MachineProvingKey<
            SP1GlobalContext,
            <CpuSP1ProverComponents as SP1ProverComponents>::CoreComponents,
        >,
    >,
    pub(crate) vk: SP1VerifyingKey,
    pub(crate) program: Arc<Program>,
    pub(crate) elf: Elf,
}

/// An error occurred while proving.
#[derive(Debug, Error)]
pub enum CPUProverError {
    /// An error occurred while proving.
    #[error(transparent)]
    Prover(#[from] SP1ProverError),

    /// An error occurred while executing.
    #[error(transparent)]
    Execution(#[from] ExecutionError),

    /// An unexpected error occurred.
    #[error("An unexpected error occurred: {:?}", .0)]
    Unexpected(#[from] anyhow::Error),
}

impl ProvingKey for CPUProvingKey {
    fn verifying_key(&self) -> &SP1VerifyingKey {
        &self.vk
    }

    fn elf(&self) -> &Elf {
        &self.elf
    }
}

impl Prover for CpuProver {
    type ProvingKey = CPUProvingKey;
    type Error = CPUProverError;
    type ProveRequest<'a> = CpuProveBuilder<'a>;

    fn inner(&self) -> Arc<LocalProver<CpuSP1ProverComponents>> {
        self.prover.clone()
    }

    fn setup(&self, elf: Elf) -> impl SendFutureResult<Self::ProvingKey, Self::Error> {
        async move {
            let (raw, program, vk) = self.prover.prover().core().setup(&elf).await;

            // todo!(n): safety comments
            let inner = unsafe { raw.into_inner() };

            Ok(CPUProvingKey { raw: inner, vk, elf, program })
        }
    }

    fn prove<'a>(&'a self, pk: &'a Self::ProvingKey, stdin: SP1Stdin) -> Self::ProveRequest<'a> {
        CpuProveBuilder::new(self, pk, stdin)
    }
}

impl CpuProver {
    /// Creates a new [`CpuProver`], using the default [`LocalProverOpts`].
    #[must_use]
    pub async fn new() -> Self {
        Self::new_with_opts(None).await
    }

    /// Creates a new [`CpuProver`] with optional custom [`SP1CoreOpts`].
    #[must_use]
    pub async fn new_with_opts(core_opts: Option<sp1_core_executor::SP1CoreOpts>) -> Self {
        let prover = SP1ProverBuilder::<CpuSP1ProverComponents>::new().build().await;
        let mut opts = LocalProverOpts::default();

        // Override core_opts if provided
        if let Some(core_opts) = core_opts {
            opts.core_opts = core_opts;
        }

        let prover = Arc::new(LocalProver::new(prover, opts));

        Self { prover }
    }

    /// # ⚠️ WARNING: This prover is experimental and should not be used in production.
    /// It is intended for development and debugging purposes.
    ///
    /// Creates a new [`CpuProver`], using the default [`LocalProverOpts`].
    /// Verification of the proof system's verification key is skipped, meaning that the
    /// recursion proofs are not guaranteed to be about a permitted recursion program.
    #[cfg(feature = "experimental")]
    #[must_use]
    pub async fn new_experimental() -> Self {
        let prover = SP1ProverBuilder::<CpuSP1ProverComponents>::new()
            .without_vk_verification()
            .build()
            .await;
        let opts = LocalProverOpts::default();
        let prover = Arc::new(LocalProver::new(prover, opts));

        Self { prover }
    }

    pub(crate) async fn prove_impl(
        &self,
        pk: &CPUProvingKey,
        stdin: SP1Stdin,
        context: SP1Context<'static>,
        mode: SP1ProofMode,
    ) -> Result<SP1ProofWithPublicValues, CPUProverError> {
        let core_proof = prove_core(&self.prover, pk, &stdin, context).await?;
        let public_values = core_proof.public_values.clone();

        if mode == SP1ProofMode::Core {
            let SP1CoreProofData(proof) = core_proof.proof;
            return Ok(SP1ProofWithPublicValues::new(
                SP1Proof::Core(proof),
                public_values,
                self.version().to_string(),
            ));
        }

        // Generate the compressed proof.
        let compress_proof = compress_proof(&self.prover, pk, &stdin, core_proof).await?;
        if mode == SP1ProofMode::Compressed {
            return Ok(SP1ProofWithPublicValues::new(
                SP1Proof::Compressed(Box::new(compress_proof)),
                public_values,
                self.version().to_string(),
            ));
        }

        let shrink_proof = self.prover.shrink(compress_proof).await?;
        let wrap_proof = self.prover.wrap(shrink_proof).await?;
        match mode {
            SP1ProofMode::Groth16 => {
                let groth16_proof = prove_groth16(&self.prover, wrap_proof).await;
                Ok(SP1ProofWithPublicValues::new(
                    SP1Proof::Groth16(groth16_proof),
                    public_values,
                    self.version().to_string(),
                ))
            }
            SP1ProofMode::Plonk => {
                let plonk_proof = prove_plonk(&self.prover, wrap_proof).await;
                Ok(SP1ProofWithPublicValues::new(
                    SP1Proof::Plonk(plonk_proof),
                    public_values,
                    self.version().to_string(),
                ))
            }
            _ => unreachable!(),
        }
    }
}

async fn prove_core(
    prover: &Arc<LocalProver<CpuSP1ProverComponents>>,
    pk: &CPUProvingKey,
    stdin: &SP1Stdin,
    context: SP1Context<'static>,
) -> Result<SP1ProofWithMetadata<SP1CoreProofData>, SP1ProverError> {
    prover.clone().prove_core(pk.raw.clone(), pk.program.clone(), stdin.clone(), context).await
}

async fn compress_proof(
    prover: &Arc<LocalProver<CpuSP1ProverComponents>>,
    pk: &CPUProvingKey,
    stdin: &SP1Stdin,
    core_proof: SP1ProofWithMetadata<SP1CoreProofData>,
) -> Result<SP1RecursionProof<SP1GlobalContext, InnerSC>, SP1ProverError> {
    let deferred_proofs =
        stdin.proofs.iter().map(|(reduce_proof, _)| reduce_proof.clone()).collect();
    prover.clone().compress(&pk.vk, core_proof, deferred_proofs).await
}

pub(crate) async fn prove_groth16(
    prover: &LocalProver<CpuSP1ProverComponents>,
    wrap_proof: SP1RecursionProof<SP1OuterGlobalContext, OuterSC>,
) -> Groth16Bn254Proof {
    #[cfg(feature = "experimental")]
    let artifacts_dir = try_build_groth16_bn254_artifacts_dev(&wrap_proof.vk, &wrap_proof.proof);

    #[cfg(not(feature = "experimental"))]
    // TODO: Test that this works after v6.0.0 release
    let artifacts_dir = try_install_circuit_artifacts("groth16").await;

    prover.wrap_groth16_bn254(wrap_proof, &artifacts_dir).await
}

pub(crate) async fn prove_plonk(
    prover: &LocalProver<CpuSP1ProverComponents>,
    wrap_proof: SP1RecursionProof<SP1OuterGlobalContext, OuterSC>,
) -> PlonkBn254Proof {
    #[cfg(feature = "experimental")]
    let artifacts_dir = try_build_plonk_bn254_artifacts_dev(&wrap_proof.vk, &wrap_proof.proof);

    #[cfg(not(feature = "experimental"))]
    // TODO: Test that this works after v6.0.0 release
    let artifacts_dir = try_install_circuit_artifacts("plonk").await;

    prover.wrap_plonk_bn254(wrap_proof, &artifacts_dir).await
}
