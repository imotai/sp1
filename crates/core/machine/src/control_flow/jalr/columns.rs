use crate::adapter::{register::jalr_type::JalrTypeReader, state::CPUState};
use sp1_derive::AlignedBorrow;
use sp1_stark::Word;
use std::mem::size_of;

use crate::operations::{AddOperation, BabyBearWordRangeChecker};

pub const NUM_JALR_COLS: usize = size_of::<JalrColumns<u8>>();

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct JalrColumns<T> {
    /// The current shard, timestamp, program counter of the CPU.
    pub state: CPUState<T>,

    /// The adapter to read program and register information.
    pub adapter: JalrTypeReader<T>,

    /// The range checker for the next program counter.
    pub next_pc_range_checker: BabyBearWordRangeChecker<T>,

    /// The value of the first operand.
    pub op_a_value: Word<T>,

    /// Whether or not the current row is a real row.
    pub is_real: T,

    /// Instance of `AddOperation` to handle addition logic in `JumpChip`.
    pub add_operation: AddOperation<T>,
}
