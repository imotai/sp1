//! The integration of all the prover components.
mod machine;
mod prover;
mod setup;
mod types;

pub use cslpc_tracegen::CORE_MAX_TRACE_SIZE;
pub use cslpc_utils::{Ext, Felt};
pub use machine::*;
pub use prover::*;
pub use types::*;
