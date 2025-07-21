use derive_where::derive_where;
use slop_algebra::PrimeField32;
use slop_baby_bear::BabyBear;
use slop_basefold::FriConfig;
use slop_jagged::{BabyBearPoseidon2, JaggedConfig};

use serde::{Deserialize, Serialize};
use slop_air::Air;
use slop_multilinear::MultilinearPcsVerifier;
use thiserror::Error;

use crate::{air::MachineAir, prover::CoreProofShape, Machine, VerifierConstraintFolder};

use super::{MachineConfig, MachineVerifyingKey, ShardProof, ShardVerifier, ShardVerifierError};
/// A complete proof of program execution.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "C: MachineConfig, C::Challenger: Serialize",
    deserialize = "C: MachineConfig, C::Challenger: Deserialize<'de>"
))]
pub struct MachineProof<C: MachineConfig> {
    /// The shard proofs.
    pub shard_proofs: Vec<ShardProof<C>>,
}

impl<C: MachineConfig> From<Vec<ShardProof<C>>> for MachineProof<C> {
    fn from(shard_proofs: Vec<ShardProof<C>>) -> Self {
        Self { shard_proofs }
    }
}

/// An error that occurs during the verification of a machine proof.
#[derive(Debug, Error)]
pub enum MachineVerifierError<EF, PcsError> {
    /// An error that occurs during the verification of a shard proof.
    #[error("invalid shard proof: {0}")]
    InvalidShardProof(#[from] ShardVerifierError<EF, PcsError>),
    /// The public values are invalid
    #[error("invalid public values")]
    InvalidPublicValues(&'static str),
    /// There are too many shards.
    #[error("too many shards")]
    TooManyShards,
    /// Invalid verification key.
    #[error("invalid verification key")]
    InvalidVerificationKey,
    /// Empty proof.
    #[error("empty proof")]
    EmptyProof,
}

/// Derive the error type from the machine config.
pub type MachineVerifierConfigError<C> = MachineVerifierError<
    <C as JaggedConfig>::EF,
    <<C as JaggedConfig>::BatchPcsVerifier as MultilinearPcsVerifier>::VerifierError,
>;

/// A verifier for a machine proof.
#[derive_where(Clone)]
pub struct MachineVerifier<C: MachineConfig, A: MachineAir<C::F>> {
    /// Shard proof verifier.
    shard_verifier: ShardVerifier<C, A>,
}

impl<C: MachineConfig, A: MachineAir<C::F>> MachineVerifier<C, A> {
    /// Create a new machine verifier.
    pub fn new(shard_verifier: ShardVerifier<C, A>) -> Self {
        Self { shard_verifier }
    }

    /// Get a new challenger.
    pub fn challenger(&self) -> C::Challenger {
        self.shard_verifier.challenger()
    }

    /// Get the machine.
    pub fn machine(&self) -> &Machine<C::F, A> {
        &self.shard_verifier.machine
    }

    /// Get the maximum log row count.
    pub fn max_log_row_count(&self) -> usize {
        self.shard_verifier.pcs_verifier.max_log_row_count
    }

    /// Get the log stacking height.
    #[must_use]
    #[inline]
    pub fn log_stacking_height(&self) -> u32 {
        self.shard_verifier.log_stacking_height()
    }

    /// Get the shape of a shard proof.
    pub fn shape_from_proof(&self, proof: &ShardProof<C>) -> CoreProofShape<C::F, A> {
        self.shard_verifier.shape_from_proof(proof)
    }

    /// Get the shard verifier.
    #[must_use]
    #[inline]
    pub fn shard_verifier(&self) -> &ShardVerifier<C, A> {
        &self.shard_verifier
    }
}

impl<C: MachineConfig, A: MachineAir<C::F>> MachineVerifier<C, A>
where
    C::F: PrimeField32,
{
    /// Verify the machine proof.
    pub fn verify(
        &self,
        vk: &MachineVerifyingKey<C>,
        proof: &MachineProof<C>,
    ) -> Result<
        (),
        MachineVerifierError<C::EF, <C::BatchPcsVerifier as MultilinearPcsVerifier>::VerifierError>,
    >
    where
        A: for<'a> Air<VerifierConstraintFolder<'a, C>>,
    {
        let mut challenger = self.challenger();
        // Observe the verifying key.
        vk.observe_into(&mut challenger);

        // Verify the shard proofs.
        for (i, shard_proof) in proof.shard_proofs.iter().enumerate() {
            let mut challenger = challenger.clone();
            let span = tracing::debug_span!("verify shard", i).entered();
            self.verify_shard(vk, shard_proof, &mut challenger)
                .map_err(MachineVerifierError::InvalidShardProof)?;
            span.exit();
        }

        Ok(())
    }

    /// Verify a shard proof.
    pub fn verify_shard(
        &self,
        vk: &MachineVerifyingKey<C>,
        proof: &ShardProof<C>,
        challenger: &mut C::Challenger,
    ) -> Result<
        (),
        ShardVerifierError<C::EF, <C::BatchPcsVerifier as MultilinearPcsVerifier>::VerifierError>,
    >
    where
        A: for<'a> Air<VerifierConstraintFolder<'a, C>>,
    {
        self.shard_verifier.verify_shard(vk, proof, challenger)
    }
}

impl<A: MachineAir<BabyBear>> MachineVerifier<BabyBearPoseidon2, A> {
    /// Get the FRI config.
    #[must_use]
    #[inline]
    pub fn fri_config(&self) -> &FriConfig<BabyBear> {
        &self.shard_verifier.pcs_verifier.stacked_pcs_verifier.pcs_verifier.fri_config
    }
}
