use sp1_derive::AlignedBorrow;
use sp1_stark::Word;

use crate::operations::U16toU8Operation;

/// Memory Access Timestamp
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryAccessTimestamp<T> {
    /// The previous timestamp's high 24 bits that this memory access is being read from.
    pub prev_high: T,
    /// The previous timestamp's low 24 bits that this memory access is being read from.
    pub prev_low: T,
    /// This will be true if the top 24 bits do not match.
    pub compare_low: T,
    /// The following columns are decomposed limbs for the difference between the current access's
    /// timestamp and the previous access's timestamp.  Note the actual value of the timestamp
    /// is either the accesses' high or low 24 bits depending on the value of compare_low.
    ///
    /// This column is the least significant 16 bit limb of the difference.
    pub diff_low_limb: T,
    /// This column is the most significant 8 bit limb of the difference.
    pub diff_high_limb: T,
}

/// Memory Access Columns
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryAccessCols<T> {
    pub prev_value: Word<T>,
    pub access_timestamp: MemoryAccessTimestamp<T>,
}

/// Memory Access Columns for u8 limbs
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryAccessColsU8<T> {
    pub memory_access: MemoryAccessCols<T>,
    pub prev_value_u8: U16toU8Operation<T>,
}

/// Memory Access Timestamp, when the shard and previous shard are known to be equal
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryAccessInShardTimestamp<T> {
    /// The previous timestamp that this memory access is being read from.
    pub prev_low: T,
    /// The difference in timestamp's least significant 16 bit limb.
    pub diff_low_limb: T,
}

/// Memory Access Columns, when the shard and previous shard are known to be equal
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct MemoryAccessInShardCols<T> {
    pub prev_value: Word<T>,
    pub access_timestamp: MemoryAccessInShardTimestamp<T>,
}
