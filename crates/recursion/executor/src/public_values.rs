use std::{borrow::BorrowMut, mem::MaybeUninit};

use serde::{Deserialize, Serialize};
use sp1_derive::AlignedBorrow;
use sp1_stark::{air::POSEIDON_NUM_WORDS, septic_digest::SepticDigest, Word};

use crate::DIGEST_SIZE;

pub const RECURSIVE_PROOF_NUM_PV_ELTS: usize = size_of::<RecursionPublicValues<u8>>();

pub const PV_DIGEST_NUM_WORDS: usize = 8;

/// The PublicValues struct is used to store all of a reduce proof's public values.
#[derive(AlignedBorrow, Serialize, Deserialize, Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct RecursionPublicValues<T> {
    /// The hash of all the bytes that the program has written to public values.
    pub committed_value_digest: [Word<T>; PV_DIGEST_NUM_WORDS],

    /// The hash of all deferred proofs that have been witnessed in the VM.
    pub deferred_proofs_digest: [T; POSEIDON_NUM_WORDS],

    /// The start pc of shards being proven.
    pub start_pc: T,

    /// The expected start pc for the next shard.
    pub next_pc: T,

    /// First shard being proven.
    pub start_shard: T,

    /// Next shard that should be proven.
    pub next_shard: T,

    /// First execution shard being proven.
    pub start_execution_shard: T,

    /// Next execution shard that should be proven.
    pub next_execution_shard: T,

    /// Previous MemoryInit address bits.
    pub previous_init_addr_bits: [T; 32],

    /// Last MemoryInit address bits.
    pub last_init_addr_bits: [T; 32],

    /// Previous MemoryFinalize address bits.
    pub previous_finalize_addr_bits: [T; 32],

    /// Last MemoryFinalize address bits.
    pub last_finalize_addr_bits: [T; 32],

    /// Start state of reconstruct_deferred_digest.
    pub start_reconstruct_deferred_digest: [T; POSEIDON_NUM_WORDS],

    /// End state of reconstruct_deferred_digest.
    pub end_reconstruct_deferred_digest: [T; POSEIDON_NUM_WORDS],

    /// The commitment to the sp1 program being proven.
    pub sp1_vk_digest: [T; DIGEST_SIZE],

    /// The root of the vk merkle tree.
    pub vk_root: [T; DIGEST_SIZE],

    /// Current cumulative sum of lookup bus. Note that for recursive proofs for core proofs, this
    /// contains the global cumulative sum.  
    pub global_cumulative_sum: SepticDigest<T>,

    /// Whether the proof completely proves the program execution.
    pub is_complete: T,

    /// Whether the proof represents a collection of shards which contain at least one execution
    /// shard, i.e. a shard that contains the `cpu` chip.
    pub contains_execution_shard: T,

    /// The exit code of the program.
    pub exit_code: T,

    /// The digest of all the previous public values elements.
    pub digest: [T; DIGEST_SIZE],
}

/// Converts the public values to an array of elements.
impl<F: Copy> RecursionPublicValues<F> {
    pub fn as_array(&self) -> [F; RECURSIVE_PROOF_NUM_PV_ELTS] {
        unsafe {
            let mut ret = [MaybeUninit::<F>::zeroed().assume_init(); RECURSIVE_PROOF_NUM_PV_ELTS];
            let pv: &mut RecursionPublicValues<F> = ret.as_mut_slice().borrow_mut();
            *pv = *self;
            ret
        }
    }
}

impl<T: Copy> IntoIterator for RecursionPublicValues<T> {
    type Item = T;
    type IntoIter = std::array::IntoIter<T, RECURSIVE_PROOF_NUM_PV_ELTS>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_array().into_iter()
    }
}
