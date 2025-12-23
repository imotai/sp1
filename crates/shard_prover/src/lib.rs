//! The integration of all the prover components.
mod machine;
mod prover;
mod setup;
mod types;

pub use csl_jagged_tracegen::CORE_MAX_TRACE_SIZE;
pub use csl_utils::{Ext, Felt};
pub use machine::*;
pub use prover::*;
pub use types::*;
