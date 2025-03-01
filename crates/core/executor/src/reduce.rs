use serde::{Deserialize, Serialize};
use sp1_stark::{MachineConfig, MachineVerifyingKey, ShardProof};
/// An intermediate proof which proves the execution.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(serialize = "C: MachineConfig", deserialize = "C: MachineConfig"))]
pub struct SP1ReduceProof<C: MachineConfig> {
    /// The compress verifying key associated with the proof.
    pub vk: MachineVerifyingKey<C>,
    /// The shard proof representing the compressed proof.
    pub proof: ShardProof<C>,
}

impl<C: MachineConfig> std::fmt::Debug for SP1ReduceProof<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("SP1ReduceProof");
        // TODO: comment back after debug enabled.
        // debug_struct.field("vk", &self.vk);
        // debug_struct.field("proof", &self.proof);
        debug_struct.finish()
    }
}
