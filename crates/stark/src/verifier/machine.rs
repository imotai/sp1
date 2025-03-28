use serde::{Deserialize, Serialize};
use slop_air::Air;
use slop_challenger::Synchronizable;
use thiserror::Error;

use crate::{air::MachineAir, VerifierConstraintFolder};

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

/// An error that occurs during the verification of a machine proof.
#[derive(Debug, Error)]
pub enum MachineVerifierError<C: MachineConfig> {
    /// An error that occurs during the verification of a shard proof.
    #[error("invalid shard proof: {0}")]
    InvalidShardProof(ShardVerifierError<C>),
    /// The global cumulative sum check fails.
    #[error("non-zero global cumulative sum")]
    NonZeroCumulativeSum,
    /// The public values are invalid
    #[error("invalid public values")]
    InvalidPublicValues(&'static str),
    /// There are too many shards.
    #[error("too many shards")]
    TooManyShards,
}

/// A verifier for a machine proof.
pub struct MachineVerifier<C: MachineConfig, A: MachineAir<C::F>> {
    /// Shard proof verifier.
    shard_verifier: ShardVerifier<C, A>,
}

impl<C: MachineConfig, A: MachineAir<C::F>> MachineVerifier<C, A> {
    /// Create a new machine verifier.
    pub fn new(shard_verifier: ShardVerifier<C, A>) -> Self {
        Self { shard_verifier }
    }

    /// Verify the machine proof.
    pub fn verify(
        &self,
        vk: &MachineVerifyingKey<C>,
        proof: &MachineProof<C>,
        challenger: &mut C::Challenger,
    ) -> Result<(), MachineVerifierError<C>>
    where
        A: for<'a> Air<VerifierConstraintFolder<'a, C>>,
        C::Challenger: Synchronizable,
    {
        // Observe the verifying key.
        vk.observe_into(challenger);

        // Verify the shard proofs.
        for (i, shard_proof) in proof.shard_proofs.iter().enumerate() {
            let mut challenger = challenger.clone();
            let span = tracing::debug_span!("verify shard", i).entered();
            self.shard_verifier
                .verify_shard(vk, shard_proof, &mut challenger)
                .map_err(MachineVerifierError::InvalidShardProof)?;
            span.exit();
        }

        Ok(())
    }
}
