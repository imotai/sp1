use serde::{Deserialize, Serialize};
use slop_algebra::AbstractField;
use slop_challenger::VariableLengthChallenger;
use slop_challenger::{CanObserve, IopCtx};
use slop_multilinear::MultilinearPcsVerifier;

use crate::septic_digest::SepticDigest;

#[allow(clippy::disallowed_types)]
use slop_jagged::KoalaBearPoseidon2;

#[allow(clippy::disallowed_types)]
/// The jagged config for SP1 core, compress, and shrink proofs.
pub type SP1CoreJaggedConfig = KoalaBearPoseidon2;

pub use slop_jagged::SP1OuterConfig;

#[allow(clippy::disallowed_types)]
use slop_basefold::Poseidon2KoalaBear16BasefoldConfig;

#[allow(clippy::disallowed_types)]
/// The basefold configuration (field, extension field, challenger, tensor commitment scheme)
/// for SP1.
pub type SP1BasefoldConfig = Poseidon2KoalaBear16BasefoldConfig;

#[allow(clippy::disallowed_types)]
pub use slop_koala_bear::Poseidon2KoalaBearConfig;

#[allow(clippy::disallowed_types)]
/// The Merkle tree configuration for SP1.
pub type SP1MerkleTreeConfig = Poseidon2KoalaBearConfig;

/// A specification of preprocessed polynomial batch dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ChipDimensions<T> {
    /// The height of the preprocessed polynomial.
    pub height: T,
    /// The number of polynomials in the preprocessed batch.
    pub num_polynomials: T,
}

/// A verifying key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MachineVerifyingKey<C: IopCtx, SC: MultilinearPcsVerifier<C>> {
    /// The start pc of the program.
    pub pc_start: [C::F; 3],
    /// The starting global digest of the program, after incorporating the initial memory.
    pub initial_global_cumulative_sum: SepticDigest<C::F>,
    /// The preprocessed commitments.
    pub preprocessed_commit: C::Digest,
    /// Flag indicating if untrusted programs are allowed.
    pub enable_untrusted_programs: C::F,
    /// Phantom data.
    pub marker: std::marker::PhantomData<SC>,
}

impl<C: IopCtx, SC: MultilinearPcsVerifier<C>> PartialEq for MachineVerifyingKey<C, SC> {
    fn eq(&self, other: &Self) -> bool {
        self.pc_start == other.pc_start
            && self.initial_global_cumulative_sum == other.initial_global_cumulative_sum
            && self.preprocessed_commit == other.preprocessed_commit
            && self.enable_untrusted_programs == other.enable_untrusted_programs
    }
}

impl<C: IopCtx, SC: MultilinearPcsVerifier<C>> Eq for MachineVerifyingKey<C, SC> {}

impl<C: IopCtx, SC: MultilinearPcsVerifier<C>> MachineVerifyingKey<C, SC> {
    /// Observes the values of the proving key into the challenger.
    pub fn observe_into(&self, challenger: &mut C::Challenger) {
        challenger.observe(self.preprocessed_commit);
        challenger.observe_constant_length_slice(&self.pc_start);
        challenger.observe_constant_length_slice(&self.initial_global_cumulative_sum.0.x.0);
        challenger.observe_constant_length_slice(&self.initial_global_cumulative_sum.0.y.0);
        challenger.observe(self.enable_untrusted_programs);
        // Observe the padding.
        challenger.observe_constant_length_slice(&[C::F::zero(); 6]);
    }
}
