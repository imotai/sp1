use serde::{Deserialize, Serialize};
use sp1_curves::{edwards::WORDS_FIELD_ELEMENT, COMPRESSED_POINT_BYTES, NUM_BYTES_FIELD_ELEMENT};

use crate::events::{
    memory::{MemoryReadRecord, MemoryWriteRecord},
    MemoryLocalEvent,
};

/// Edwards Decompress Event.
///
/// This event is emitted when an edwards decompression operation is performed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdDecompressEvent {
    /// The shard number.
    pub shard: u32,
    /// The clock cycle.
    pub clk: u32,
    /// The pointer to the point.
    pub ptr: u64,
    /// The sign bit of the point.
    pub sign: bool,
    /// The comprssed y coordinate as a list of bytes.
    pub y_bytes: [u8; COMPRESSED_POINT_BYTES],
    #[serde(with = "serde_arrays")]
    /// The decompressed x coordinate as a list of bytes.
    pub decompressed_x_bytes: [u8; NUM_BYTES_FIELD_ELEMENT],
    /// The memory records for the x coordinate.
    pub x_memory_records: [MemoryWriteRecord; WORDS_FIELD_ELEMENT],
    /// The memory records for the y coordinate.
    pub y_memory_records: [MemoryReadRecord; WORDS_FIELD_ELEMENT],
    /// The local memory access events.
    pub local_mem_access: Vec<MemoryLocalEvent>,
}

impl Default for EdDecompressEvent {
    fn default() -> Self {
        Self {
            shard: 0,
            clk: 0,
            ptr: 0,
            sign: false,
            y_bytes: [0; COMPRESSED_POINT_BYTES],
            decompressed_x_bytes: [0; NUM_BYTES_FIELD_ELEMENT],
            x_memory_records: [MemoryWriteRecord::default(); WORDS_FIELD_ELEMENT],
            y_memory_records: [MemoryReadRecord::default(); WORDS_FIELD_ELEMENT],
            local_mem_access: Vec::new(),
        }
    }
}
