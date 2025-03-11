use serde::{Deserialize, Serialize};
use slop_jagged::JaggedPcsProof;
use slop_matrix::{dense::RowMajorMatrixView, stack::VerticalPair};
use slop_sumcheck::PartialSumcheckProof;

use crate::{septic_digest::SepticDigest, LogupGkrProof};

use super::MachineConfig;

/// The maximum number of elements that can be stored in the public values vec.  Both SP1 and
/// recursive proofs need to pad their public values vec to this length.  This is required since the
/// recursion verification program expects the public values vec to be fixed length.
pub const PROOF_MAX_NUM_PVS: usize = 231;

/// A proof for a shard.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "C: MachineConfig", deserialize = "C: MachineConfig"))]
pub struct ShardProof<C: MachineConfig> {
    /// The commitments to main traces.
    pub main_commitment: C::Commitment,
    /// The values of the traces at the final random point.
    pub opened_values: ShardOpenedValues<C::F, C::EF>,
    /// The evaluation proof.
    pub evaluation_proof: JaggedPcsProof<C>,
    /// TH zerocheck IOP proof.
    pub zerocheck_proof: PartialSumcheckProof<C::EF>,
    /// The public values
    pub public_values: Vec<C::F>,
    /// The `LogUp+GKR` IOP proofs.
    pub gkr_proofs: Vec<LogupGkrProof<C::EF>>,
}

/// The values of the chips in the shard at a random point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardOpenedValues<F, EF> {
    /// For each chip with respect to the canonical ordering, the values of the chip at the random
    /// point.
    pub chips: Vec<ChipOpenedValues<F, EF>>,
}

/// The opening values for a given chip at a random point.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize, EF: Serialize"))]
#[serde(bound(deserialize = "F: Deserialize<'de>, EF: Deserialize<'de>"))]
pub struct ChipOpenedValues<F, EF> {
    /// The opening of the preprocessed trace.
    pub preprocessed: AirOpenedValues<EF>,
    /// The opening of the main trace.
    pub main: AirOpenedValues<EF>,
    /// The global cumulative sum.
    pub global_cumulative_sum: SepticDigest<F>,
    /// The local cumulative sum.
    pub local_cumulative_sum: EF,
    /// The log degree of the chip.
    pub log_degree: Option<u32>,
}

/// The opening values for a given table section at a random point.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize"))]
#[serde(bound(deserialize = "T: Deserialize<'de>"))]
pub struct AirOpenedValues<T> {
    /// The opening of the local trace
    pub local: Vec<T>,
    /// The opening of the next trace.
    pub next: Vec<T>,
}

impl<T> AirOpenedValues<T> {
    /// Organize the opening values into a vertical pair.
    #[must_use]
    pub fn view(&self) -> VerticalPair<RowMajorMatrixView<'_, T>, RowMajorMatrixView<'_, T>>
    where
        T: Clone + Send + Sync,
    {
        let a = RowMajorMatrixView::new_row(&self.local);
        let b = RowMajorMatrixView::new_row(&self.next);
        VerticalPair::new(a, b)
    }
}
