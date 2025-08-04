use deepsize2::DeepSizeOf;
use serde::{Deserialize, Serialize};

/// The number of local memory entries per row of the memory local chip.
pub const NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC: usize = 1;

/// Memory Record.
///
/// This object encapsulates the information needed to prove a memory access operation. This
/// includes the shard, timestamp, and value of the memory address.
#[derive(Debug, Copy, Clone, Default, Serialize, Deserialize, DeepSizeOf)]
#[repr(C)]
pub struct MemoryRecord {
    /// The shard number.
    pub shard: u32,
    /// The timestamp.
    pub timestamp: u64,
    /// The value.
    pub value: u64,
}

/// A shard number. Used to identify shards in the SP1 proving architecture.
///
/// This is a newtype that guarantees the inner value fits within 31 bits.
#[derive(Debug, Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[repr(transparent)]
pub struct Shard(u32);

impl Shard {
    /// Creates a shard number if the given value fits within the bounds.
    #[must_use]
    pub const fn new(value: u32) -> Option<Self> {
        if value <= SHARD_MASK {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Creates a shard number without checking that the value fits within the bounds.
    ///
    /// # Safety
    /// The value must fit within 31 bits. That is, it must be less than to 2^31.
    #[must_use]
    pub const unsafe fn new_unchecked(value: u32) -> Self {
        Self(value)
    }

    /// Returns the wrapped shard number as a `u32`.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

// We implement `Deserialize` manually to perform the bounds check.
impl<'de> Deserialize<'de> for Shard {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // The `OnceLock` is used to format `SHARD_MASK` in the error message.
        use std::sync::OnceLock;
        static MSG: OnceLock<String> = OnceLock::new();

        let value: u32 = Deserialize::deserialize(deserializer)?;
        // Attempt to create the `Shard`. If creation fails, we return an error.
        Shard::new(value).ok_or_else(|| {
            serde::de::Error::invalid_value(
                serde::de::Unexpected::Unsigned(value as u64),
                &MSG.get_or_init(|| format!("an integer between 0 and {SHARD_MASK:#x}, inclusive"))
                    .as_ref(),
            )
        })
    }
}

/// Memory entry.
///
/// Similar to a [`MemoryRecord`], but it contains/validates data for execution purposes.
#[derive(Debug, Copy, Clone, Default, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// The shard number.
    pub lshard: LogicalShard,
    /// The timestamp.
    pub timestamp: u64,
    /// The value.
    pub value: u64,
}

impl MemoryEntry {
    /// Create a memory entry that represents the program-wide initialization of a value.
    #[must_use]
    pub fn init(value: u64) -> Self {
        Self { lshard: LogicalShard::default(), timestamp: 0, value }
    }
}

impl From<MemoryEntry> for MemoryRecord {
    /// Converts to a `MemoryRecord` from a `MemoryEntry`
    /// by extracting the underlying `Shard` from the `LogicalShard`.
    fn from(value: MemoryEntry) -> Self {
        let MemoryEntry { lshard, timestamp, value } = value;
        Self { shard: lshard.shard().get(), timestamp, value }
    }
}

// Constants used to define `LogicalShard` and `Shard`.
// If we change these constants, be sure to to change the docs and revisit all relevant `unsafe`
// blocks.
const SHARD_MASK: u32 = u32::MAX >> 1;
const EXTERNAL_FLAG_SHIFT: u32 = u32::BITS - 1;

/// A logical shard number. Used to compactly represent shards and compare them.
///
/// This type packs a [`Shard`] and a boolean flag into a `u32`. The flag
/// indicates whether the identified shard is an "external" shard, meaning that the shard associated
/// with the operation will be created later. Currently, this only happens for precompile-associated
/// operations, since precompile events are deferred and processed in later shards.
///
/// In this sense, the packed value is a "logical" shard number. That is, we can compare a
/// `LogicalShard` with the main [`LogicalShard`] number to determine if they are different.
/// For now, this property is only used to predict (in any executor mode) whether a new memory local
/// event would be created in trace mode, and the predicted event count accumulates in the
/// field `local_mem` of the executor's `local_counts: LocalCounts`.
#[allow(clippy::unsafe_derive_deserialize)] // None of the methods themselves are unsafe.
#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(transparent)]
pub struct LogicalShard(u32);

impl LogicalShard {
    /// Creates a logical shard number from a shard number and an external flag.
    #[must_use]
    pub const fn new(shard: Shard, external_flag: bool) -> Self {
        let packed_external_flag = (external_flag as u32) << EXTERNAL_FLAG_SHIFT;
        Self(shard.get() | packed_external_flag)
    }

    /// Get the shard number.
    #[must_use]
    pub const fn shard(self) -> Shard {
        // SAFETY: `Shard` is, by definition, required to be less than or equal to `SHARD_MASK`.
        unsafe { Shard::new_unchecked(self.0 & SHARD_MASK) }
    }

    /// Get the external flag, which indicates whether this shard is an "external" shard.
    #[must_use]
    pub fn external_flag(self) -> bool {
        // No mask is required because the flag is the most significant packed field.
        (self.0 >> EXTERNAL_FLAG_SHIFT) != 0
    }
}

impl std::fmt::Debug for LogicalShard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LogicalShard")
            .field("shard", &self.shard().get())
            .field("external_flag", &self.external_flag())
            .finish()
    }
}

/// Memory Access Position.
///
/// This enum represents the position of a memory access in a register. For example, if a memory
/// access is performed in the C register, it will have a position of C.
///
/// Note: The register positions require that they be read and written in the following order:
/// C, B, A.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MemoryAccessPosition {
    /// Memory access position.
    Memory = 0,
    /// C register access position.
    C = 1,
    /// B register access position.
    B = 2,
    /// A register access position.
    A = 3,
}

/// Memory Read Record.
///
/// This object encapsulates the information needed to prove a memory read operation. This
/// includes the value, shard, timestamp, and previous shard and timestamp.
#[allow(clippy::manual_non_exhaustive)]
#[derive(Debug, Copy, Clone, Default, Serialize, Deserialize, DeepSizeOf)]
#[repr(C)]
pub struct MemoryReadRecord {
    /// The value.
    pub value: u64,
    /// The shard number.
    pub shard: u32,
    /// The timestamp.
    pub timestamp: u64,
    /// The previous shard number.
    pub prev_shard: u32,
    /// The previous timestamp.
    pub prev_timestamp: u64,
}

/// Memory Write Record.
///
/// This object encapsulates the information needed to prove a memory write operation. This
/// includes the value, shard, timestamp, previous value, previous shard, and previous timestamp.
#[allow(clippy::manual_non_exhaustive)]
#[derive(Debug, Copy, Clone, Default, Serialize, Deserialize, DeepSizeOf)]
#[repr(C)]
pub struct MemoryWriteRecord {
    /// The value.
    pub value: u64,
    /// The shard number.
    pub shard: u32,
    /// The timestamp.
    pub timestamp: u64,
    /// The previous value.
    pub prev_value: u64,
    /// The previous shard number.
    pub prev_shard: u32,
    /// The previous timestamp.
    pub prev_timestamp: u64,
}

/// Memory Record Enum.
///
/// This enum represents the different types of memory records that can be stored in the memory
/// event such as reads and writes.
#[derive(Debug, Copy, Clone, Serialize, Deserialize, DeepSizeOf)]
pub enum MemoryRecordEnum {
    /// Read.
    Read(MemoryReadRecord),
    /// Write.
    Write(MemoryWriteRecord),
}

impl MemoryRecordEnum {
    /// Retrieve the current memory record.
    #[must_use]
    pub fn current_record(&self) -> MemoryRecord {
        match self {
            MemoryRecordEnum::Read(record) => MemoryRecord {
                shard: record.shard,
                timestamp: record.timestamp,
                value: record.value,
            },
            MemoryRecordEnum::Write(record) => MemoryRecord {
                shard: record.shard,
                timestamp: record.timestamp,
                value: record.value,
            },
        }
    }

    /// Retrieve the previous memory record.
    #[must_use]
    pub fn previous_record(&self) -> MemoryRecord {
        match self {
            MemoryRecordEnum::Read(record) => MemoryRecord {
                shard: record.prev_shard,
                timestamp: record.prev_timestamp,
                value: record.value,
            },
            MemoryRecordEnum::Write(record) => MemoryRecord {
                shard: record.prev_shard,
                timestamp: record.prev_timestamp,
                value: record.prev_value,
            },
        }
    }
}

/// Memory Initialize/Finalize Event.
///
/// This object encapsulates the information needed to prove a memory initialize or finalize
/// operation. This includes the address, value, shard, timestamp, and whether the memory is
/// initialized or finalized.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, DeepSizeOf)]
#[repr(C)]
pub struct MemoryInitializeFinalizeEvent {
    /// The address.
    pub addr: u64,
    /// The value.
    pub value: u64,
    /// The shard number.
    pub shard: u32,
    /// The timestamp.
    pub timestamp: u64,
}

impl MemoryReadRecord {
    /// Creates a new [``MemoryReadRecord``].
    #[must_use]
    #[inline]
    pub const fn new(entry: &MemoryEntry, prev_entry: &MemoryEntry) -> Self {
        let MemoryEntry { lshard, timestamp, value } = *entry;
        let MemoryEntry { lshard: prev_lshard, timestamp: prev_timestamp, value: _ } = *prev_entry;
        let shard = lshard.shard().get();
        let prev_shard = prev_lshard.shard().get();
        debug_assert!(
            shard > prev_shard || ((shard == prev_shard) && (timestamp > prev_timestamp))
        );
        Self { value, shard, timestamp, prev_shard, prev_timestamp }
    }
}

impl MemoryWriteRecord {
    /// Creates a new [``MemoryWriteRecord``].
    #[must_use]
    #[inline]
    pub const fn new(entry: &MemoryEntry, prev_entry: &MemoryEntry) -> Self {
        let MemoryEntry { lshard, timestamp, value } = *entry;
        let MemoryEntry { lshard: prev_lshard, timestamp: prev_timestamp, value: prev_value } =
            *prev_entry;
        let shard = lshard.shard().get();
        let prev_shard = prev_lshard.shard().get();
        debug_assert!(
            shard > prev_shard || ((shard == prev_shard) && (timestamp > prev_timestamp)),
        );
        Self { value, shard, timestamp, prev_value, prev_shard, prev_timestamp }
    }
}

impl MemoryRecordEnum {
    /// Returns the value of the memory record.
    #[must_use]
    pub const fn value(&self) -> u64 {
        match self {
            MemoryRecordEnum::Read(record) => record.value,
            MemoryRecordEnum::Write(record) => record.value,
        }
    }

    /// Returns the previous value of the memory record.
    #[must_use]
    pub const fn prev_value(&self) -> u64 {
        match self {
            MemoryRecordEnum::Read(record) => record.value,
            MemoryRecordEnum::Write(record) => record.prev_value,
        }
    }
}

impl MemoryInitializeFinalizeEvent {
    /// Creates a new [``MemoryInitializeFinalizeEvent``] for an initialization.
    #[must_use]
    pub const fn initialize(addr: u64, value: u64) -> Self {
        Self { addr, value, shard: 0, timestamp: 0 }
    }

    /// Creates a new [``MemoryInitializeFinalizeEvent``] for a finalization.
    #[must_use]
    pub const fn finalize_from_record(addr: u64, record: &MemoryEntry) -> Self {
        Self {
            addr,
            value: record.value,
            shard: record.lshard.shard().get(),
            timestamp: record.timestamp,
        }
    }
}

impl From<MemoryReadRecord> for MemoryRecordEnum {
    fn from(read_record: MemoryReadRecord) -> Self {
        MemoryRecordEnum::Read(read_record)
    }
}

impl From<MemoryWriteRecord> for MemoryRecordEnum {
    fn from(write_record: MemoryWriteRecord) -> Self {
        MemoryRecordEnum::Write(write_record)
    }
}

/// Memory Local Event.
///
/// This object encapsulates the information needed to prove a memory access operation within a
/// shard. This includes the address, initial memory access, and final memory access within a
/// shard.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, DeepSizeOf)]
#[repr(C)]
pub struct MemoryLocalEvent {
    /// The address.
    pub addr: u64,
    /// The initial memory access.
    pub initial_mem_access: MemoryRecord,
    /// The final memory access.
    pub final_mem_access: MemoryRecord,
}
