//! Prover components.

mod cpu;
mod machine;
mod memory_permit;
mod permits;
mod shard;
mod trace;
mod zerocheck;

pub use cpu::*;
pub use machine::*;
pub use memory_permit::*;
pub use permits::*;
pub use shard::*;
use slop_merkle_tree::BnProver;
use slop_stacked::StackedPcsProver;
use sp1_primitives::{SP1ExtensionField, SP1Field, SP1GlobalContext, SP1OuterGlobalContext};
pub use trace::*;
pub use zerocheck::*;

pub use slop_merkle_tree::Poseidon2KoalaBear16Prover as SP1MerkleTreeProver;

#[allow(clippy::disallowed_types)]
/// The CPU prover components for a jagged PCS prover in SP1.
pub type SP1CpuJaggedProverComponents = StackedPcsProver<SP1MerkleTreeProver, SP1GlobalContext>;

/// The Jagged CPU prover components for the wrap step of recursion in SP1.
pub type SP1OuterCpuJaggedProverComponents =
    StackedPcsProver<BnProver<SP1Field, SP1ExtensionField>, SP1OuterGlobalContext>;
