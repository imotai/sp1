use deepsize2::DeepSizeOf;
use serde::{Deserialize, Serialize};

use crate::events::memory::{MemoryLocalEvent, MemoryWriteRecord};

/// `Poseidon2PrecompileEvent` Event.
///
/// This event is emitted when a `Poseidon2PrecompileEvent` operation is performed.
#[derive(Default, Debug, Clone, Serialize, Deserialize, DeepSizeOf)]
pub struct Poseidon2PrecompileEvent {
    /// The clock cycle.
    pub clk: u64,
    /// The pointer to the input/output array.
    pub ptr: u64,
    /// The memory records for the 8 u64 words (read as input, written as output).
    pub memory_records: Vec<MemoryWriteRecord>,
    /// The local memory access events.
    pub local_mem_access: Vec<MemoryLocalEvent>,
}
