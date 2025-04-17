use serde::{Deserialize, Serialize};
use slop_basefold_prover::PowProver;

use crate::DeviceGrindingChallenger;

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct GrindingPowCudaProver;

impl<C: DeviceGrindingChallenger + Send + Sync> PowProver<C> for GrindingPowCudaProver {
    async fn grind(&self, challenger: &mut C, bits: usize) -> C::Witness {
        challenger.grind_device(bits).await
    }
}
