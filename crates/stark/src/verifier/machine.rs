use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::air::MachineAir;

use super::{MachineConfig, MachineVerifyingKey, ShardProof, ShardVerifier, ShardVerifierError};

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "C: MachineConfig", deserialize = "C: MachineConfig"))]
pub struct MachineProof<C: MachineConfig> {
    pub shard_proofs: Vec<ShardProof<C>>,
}

#[derive(Debug, Error)]
pub enum MachineVerifierError<C: MachineConfig> {
    #[error("invalid shard proof: {0}")]
    InvalidShardProof(ShardVerifierError<C>),
}

pub struct MachineVerifier<C: MachineConfig, A: MachineAir<C::F>> {
    shard_verifier: ShardVerifier<C, A>,
}

impl<C: MachineConfig, A: MachineAir<C::F>> MachineVerifier<C, A> {
    pub fn new(shard_verifier: ShardVerifier<C, A>) -> Self {
        Self { shard_verifier }
    }

    pub fn verify(
        &self,
        vk: &MachineVerifyingKey<C>,
        proof: &MachineProof<C>,
        challenger: &mut C::Challenger,
    ) -> Result<(), MachineVerifierError<C>> {
        // Observe the verifying key.
        vk.observe_into(challenger);

        // Verify the shard proofs.
        for shard_proof in proof.shard_proofs.iter() {
            tracing::info!("verifying shard proof");
            let mut challenger = challenger.clone();
            self.shard_verifier
                .verify_shard(vk, shard_proof, &mut challenger)
                .map_err(MachineVerifierError::InvalidShardProof);
            tracing::info!("shard proof verified");
        }

        Ok(())
    }
}
