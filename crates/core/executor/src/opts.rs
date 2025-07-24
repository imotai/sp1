use std::{collections::HashSet, env};

use serde::{Deserialize, Serialize};

use crate::RetainedEventsPreset;

const MAX_SHARD_SIZE: usize = 1 << 24;
const MAX_SHARD_BATCH_SIZE: usize = 8;
const MAX_DEFERRED_SPLIT_THRESHOLD: usize = 1 << 15;

/// The trace area threshold for a shard.
pub const ELEMENT_THRESHOLD: u64 = (1 << 29) - (1 << 27);
/// The height threshold for a shard.
pub const HEIGHT_THRESHOLD: u64 = 1 << 22;

/// The threshold that determines when to split the shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardingThreshold {
    /// The maximum number of elements in the trace.
    pub element_threshold: u64,
    /// The maximum number of rows for a single operation.
    pub height_threshold: u64,
}

/// Options for the core prover.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SP1CoreOpts {
    /// The size of a shard in terms of cycles.
    pub shard_size: usize,
    /// The size of a batch of shards in terms of cycles.
    pub shard_batch_size: usize,
    /// Options for splitting deferred events.
    pub split_opts: SplitOpts,
    /// The threshold that determines when to split the shard.
    pub sharding_threshold: ShardingThreshold,
    /// Preset collections of events to retain in a shard instead of deferring.
    pub retained_events_presets: HashSet<RetainedEventsPreset>,
}

impl Default for SP1CoreOpts {
    fn default() -> Self {
        let split_threshold = env::var("SPLIT_THRESHOLD")
            .map(|s| s.parse::<usize>().unwrap_or(MAX_DEFERRED_SPLIT_THRESHOLD))
            .unwrap_or(MAX_DEFERRED_SPLIT_THRESHOLD)
            .max(MAX_DEFERRED_SPLIT_THRESHOLD);

        let shard_size = env::var("SHARD_SIZE")
            .map_or_else(|_| MAX_SHARD_SIZE, |s| s.parse::<usize>().unwrap_or(MAX_SHARD_SIZE));

        let element_threshold = env::var("ELEMENT_THRESHOLD")
            .map_or_else(|_| ELEMENT_THRESHOLD, |s| s.parse::<u64>().unwrap_or(ELEMENT_THRESHOLD));

        let height_threshold = env::var("HEIGHT_THRESHOLD")
            .map_or_else(|_| HEIGHT_THRESHOLD, |s| s.parse::<u64>().unwrap_or(HEIGHT_THRESHOLD));

        let sharding_threshold = ShardingThreshold { element_threshold, height_threshold };

        let mut retained_events_presets = HashSet::new();
        retained_events_presets.insert(RetainedEventsPreset::Bls12381Field);
        retained_events_presets.insert(RetainedEventsPreset::Bn254Field);
        retained_events_presets.insert(RetainedEventsPreset::Sha256);
        retained_events_presets.insert(RetainedEventsPreset::Poseidon2);
        retained_events_presets.insert(RetainedEventsPreset::U256Ops);

        Self {
            shard_size,
            shard_batch_size: env::var("SHARD_BATCH_SIZE").map_or_else(
                |_| MAX_SHARD_BATCH_SIZE,
                |s| s.parse::<usize>().unwrap_or(MAX_SHARD_BATCH_SIZE),
            ),
            split_opts: SplitOpts::new(split_threshold),
            sharding_threshold,
            retained_events_presets,
        }
    }
}

/// Options for splitting deferred events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitOpts {
    /// The threshold for combining the memory init/finalize events in to the current shard in
    /// terms of total trace area in the shard, and the number of memory init/finalize events.
    pub combine_memory_threshold: (u64, usize),
    /// The threshold for default events.
    pub deferred: usize,
    /// The threshold for keccak events.
    pub keccak: usize,
    /// The threshold for sha extend events.
    pub sha_extend: usize,
    /// The threshold for sha compress events.
    pub sha_compress: usize,
    /// The threshold for memory events.
    pub memory: usize,
    /// The threshold for ec add 256bit events.
    pub ec_add_256bit: usize,
    /// The threshold for ec double 256bit events.
    pub ec_double_256bit: usize,
    /// The threshold for ec add 384bit events.
    pub ec_add_384bit: usize,
    /// The threshold for ec double 384bit events.
    pub ec_double_384bit: usize,
    /// The threshold for fp operation events.
    pub fp_operation_256bit: usize,
    /// The threshold for fp2 operation events.
    pub fp2_operation_256bit: usize,
    /// The threshold for fp operation 384bit events.
    pub fp_operation_384bit: usize,
    /// The threshold for fp2 operation 384bit events.
    pub fp2_operation_384bit: usize,
}

impl SplitOpts {
    /// Create a new [`SplitOpts`] with the given threshold.
    ///
    /// The constants here need to be chosen very carefully to prevent OOM. Consult @jtguibas on
    /// how to change them.
    #[must_use]
    pub fn new(deferred_split_threshold: usize) -> Self {
        Self {
            combine_memory_threshold: (1 << 28, 1 << 17),
            deferred: deferred_split_threshold,
            fp_operation_256bit: deferred_split_threshold * 9 / 5,
            ec_add_256bit: deferred_split_threshold * 37 / 25,
            ec_double_256bit: deferred_split_threshold * 21 / 8,
            ec_add_384bit: deferred_split_threshold * 8 / 15,
            ec_double_384bit: deferred_split_threshold,
            keccak: 7 * deferred_split_threshold / 25,
            sha_extend: 64 * deferred_split_threshold / 129,
            sha_compress: 32 * deferred_split_threshold / 80,
            memory: 26 * deferred_split_threshold,
            fp2_operation_256bit: deferred_split_threshold * 8 / 9,
            fp_operation_384bit: deferred_split_threshold * 11 / 10,
            fp2_operation_384bit: deferred_split_threshold * 14 / 25,
        }
    }
}
