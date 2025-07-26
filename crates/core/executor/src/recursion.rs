use serde::{Deserialize, Serialize};
use sp1_stark::{MachineConfig, MachineVerifyingKey, ShardProof};
/// An intermediate proof which proves the execution of a Hypercube verifier.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "C: MachineConfig, C::Challenger: Serialize",
    deserialize = "C: MachineConfig, C::Challenger: Deserialize<'de>"
))]
pub struct SP1RecursionProof<C: MachineConfig> {
    /// The verifying key associated with the proof.
    pub vk: MachineVerifyingKey<C>,
    /// The shard proof representing the shard proof.
    pub proof: ShardProof<C>,
}

impl<C: MachineConfig> std::fmt::Debug for SP1RecursionProof<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("SP1ReduceProof");
        // TODO: comment back after debug enabled.
        // debug_struct.field("vk", &self.vk);
        // debug_struct.field("proof", &self.proof);
        debug_struct.finish()
    }
}
