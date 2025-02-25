use serde::{Deserialize, Serialize};
use slop_challenger::GrindingChallenger;

pub trait PowProver<C: GrindingChallenger>: 'static + Send + Sync {
    fn grind(&self, challenger: &mut C, bits: usize) -> C::Witness {
        challenger.grind(bits)
    }
}

#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct GrindingPowProver;

impl<C: GrindingChallenger> PowProver<C> for GrindingPowProver {}
