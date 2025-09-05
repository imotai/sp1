use serde::{Deserialize, Serialize};
use slop_challenger::IopCtx;
use sp1_hypercube::{MachineConfig, MachineVerifyingKey, ShardProof};
/// An intermediate proof which proves the execution of a Hypercube verifier.
#[derive(Serialize, Deserialize, Clone)]
#[serde(bound(
    serialize = "GC: IopCtx, GC::Challenger: Serialize",
    deserialize = "GC: IopCtx, GC::Challenger: Deserialize<'de>"
))]
pub struct SP1RecursionProof<GC: IopCtx, C: MachineConfig<GC>> {
    /// The verifying key associated with the proof.
    pub vk: MachineVerifyingKey<GC, C>,
    /// The shard proof representing the shard proof.
    pub proof: ShardProof<GC, C>,
}

impl<GC: IopCtx, C: MachineConfig<GC>> std::fmt::Debug for SP1RecursionProof<GC, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("SP1ReduceProof");
        // TODO: comment back after debug enabled.
        // debug_struct.field("vk", &self.vk);
        // debug_struct.field("proof", &self.proof);
        debug_struct.finish()
    }
}
