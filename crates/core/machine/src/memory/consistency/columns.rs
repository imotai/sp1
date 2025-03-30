use sp1_derive::AlignedBorrow;
use sp1_stark::Word;

use crate::operations::U16toU8Operation;

/// Memory Access Timestamp
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryAccessTimestamp<T> {
    /// The previous shard and timestamp that this memory access is being read from.
    pub prev_shard: T,
    pub prev_clk: T,
    /// This will be true if the current shard == prev_access's shard, else false.
    pub compare_clk: T,
    /// The following columns are decomposed limbs for the difference between the current access's
    /// timestamp and the previous access's timestamp.  Note the actual value of the timestamp
    /// is either the accesses' shard or clk depending on the value of compare_clk.
    ///
    /// This column is the least significant 16 bit limb of current access timestamp - prev access
    /// timestamp - 1.
    pub diff_16bit_limb: T,
    /// This column is the most significant 8 bit limb of current access timestamp - prev access
    /// timestamp - 1.
    pub diff_8bit_limb: T,
}

/// New Memory Access Columns
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryAccessCols<T> {
    pub prev_value: Word<T>,
    pub access_timestamp: MemoryAccessTimestamp<T>,
}

/// New Memory Access Columns for u8 limbs
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryAccessColsU8<T> {
    pub memory_access: MemoryAccessCols<T>,
    pub prev_value_u8: U16toU8Operation<T>,
}
