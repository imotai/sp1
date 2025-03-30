use crate::adapter::register::j_type::JTypeReader;
use crate::adapter::state::CPUState;
use sp1_derive::AlignedBorrow;
use std::mem::size_of;

pub const NUM_JAL_COLS: usize = size_of::<JalColumns<u8>>();

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct JalColumns<T> {
    /// The current shard, timestamp, program counter of the CPU.
    pub state: CPUState<T>,

    /// The adapter to read program and register information.
    pub adapter: JTypeReader<T>,

    /// Whether or not the current row is a real row.
    pub is_real: T,
}
