use core::{fmt::Debug, mem::size_of};
use std::borrow::{Borrow, BorrowMut};

use deepsize2::DeepSizeOf;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use slop_algebra::{AbstractField, PrimeField32};
use sp1_primitives::consts::split_page_idx;

use crate::{septic_curve::SepticCurve, septic_digest::SepticDigest, PROOF_MAX_NUM_PVS};

/// The number of non padded elements in the SP1 proofs public values vec.
pub const SP1_PROOF_NUM_PV_ELTS: usize = size_of::<PublicValues<[u8; 4], [u8; 3], [u8; 4], u8>>();

/// The number of 32 bit words in the SP1 proof's committed value digest.
pub const PV_DIGEST_NUM_WORDS: usize = 8;

/// The number of field elements in the poseidon2 digest.
pub const POSEIDON_NUM_WORDS: usize = 8;

/// The number of 32 bit words in the SP1 proof's proof nonce.
pub const PROOF_NONCE_NUM_WORDS: usize = 4;

/// Stores all of a shard proof's public values.
#[derive(Serialize, Deserialize, Clone, Copy, Default, Debug, DeepSizeOf)]
#[repr(C)]
pub struct PublicValues<W1, W2, W3, T> {
    /// The `committed_value_digest` value before this shard.
    pub prev_committed_value_digest: [W1; PV_DIGEST_NUM_WORDS],

    /// The hash of all the bytes that the guest program has written to public values.
    pub committed_value_digest: [W1; PV_DIGEST_NUM_WORDS],

    /// The `deferred_proof_digest` value before this shard.
    pub prev_deferred_proofs_digest: [T; POSEIDON_NUM_WORDS],

    /// The hash of all deferred proofs that have been witnessed in the VM. It will be rebuilt in
    /// recursive verification as the proofs get verified. The hash itself is a rolling poseidon2
    /// hash of each proof+vkey hash and the previous hash which is initially zero.
    pub deferred_proofs_digest: [T; POSEIDON_NUM_WORDS],

    /// The shard's start program counter.
    pub pc_start: W2,

    /// The expected start program counter for the next shard.
    pub next_pc: W2,

    /// The expected exit code of the program before this shard.
    pub prev_exit_code: T,

    /// The expected exit code code of the program up to this shard.
    /// This value is only valid if halt has been executed.
    pub exit_code: T,

    /// Whether or not the current shard is an execution shard.
    pub is_execution_shard: T,

    /// The largest address that is witnessed for initialization in the previous shard.
    pub previous_init_addr: W2,

    /// The largest address that is witnessed for initialization in the current shard.
    pub last_init_addr: W2,

    /// The largest address that is witnessed for finalization in the previous shard.
    pub previous_finalize_addr: W2,

    /// The largest address that is witnessed for finalization in the current shard.
    pub last_finalize_addr: W2,

    /// The largest page idx that is witnessed for initialization in the previous shard.
    pub previous_init_page_idx: W2,

    /// The largest page idx that is witnessed for initialization in the current shard.
    pub last_init_page_idx: W2,

    /// The largest page idx that is witnessed for finalization in the previous shard.
    pub previous_finalize_page_idx: W2,

    /// The largest page idx that is witnessed for finalization in the current shard.
    pub last_finalize_page_idx: W2,

    /// The initial timestamp of the shard.
    pub initial_timestamp: W3,

    /// The last timestamp of the shard.
    pub last_timestamp: W3,

    /// If the high bits of timestamp is equal in this shard.
    pub is_timestamp_high_eq: T,

    /// The inverse of the difference of the high bits of timestamp.
    pub inv_timestamp_high: T,

    /// If the low bits of timestamp is equal in this shard.
    pub is_timestamp_low_eq: T,

    /// The inverse of the difference of the low bits of timestamp.
    pub inv_timestamp_low: T,

    /// The number of global memory initializations in the shard.
    pub global_init_count: T,

    /// The number of global memory finalizations in the shard.
    pub global_finalize_count: T,

    /// The number of global page prot initializations in the shard.
    pub global_page_prot_init_count: T,

    /// The number of global page prot finalizations in the shard.
    pub global_page_prot_finalize_count: T,

    /// The number of global interactions in the shard.
    pub global_count: T,

    /// The global cumulative sum of the shard.
    pub global_cumulative_sum: SepticDigest<T>,

    /// The `commit_syscall` value of the previous shard.
    pub prev_commit_syscall: T,

    /// Whether `COMMIT` syscall has been called up to this shard.
    pub commit_syscall: T,

    /// The `commit_deferred_syscall` value of the previous shard.
    pub prev_commit_deferred_syscall: T,

    /// Whether `COMMIT_DEFERRED` syscall has been called up to this shard.
    pub commit_deferred_syscall: T,

    /// The inverse to show that `initial_timestamp != 1` in the shards that aren't the first one.
    pub initial_timestamp_inv: T,

    /// The inverse to show that `last_timestamp != 1` in all shards.
    pub last_timestamp_inv: T,

    /// Whether or not this shard is the first shard of the proof.
    pub is_first_shard: T,

    /// Whether untrusted program support is enabled.  This specifically will enable fetching
    /// instructions from memory during runtime and checking/setting page permissions.
    pub is_untrusted_programs_enabled: T,

    /// The nonce used for this proof.
    pub proof_nonce: [T; PROOF_NONCE_NUM_WORDS],

    /// This field is here to ensure that the size of the public values struct is a multiple of 8.
    pub empty: [T; 4],
}

impl PublicValues<u32, u64, u64, u32> {
    /// Convert the public values into a vector of field elements.  This function will pad the
    /// vector to the maximum number of public values.
    #[must_use]
    pub fn to_vec<F: AbstractField>(&self) -> Vec<F> {
        let mut ret = vec![F::zero(); PROOF_MAX_NUM_PVS];

        let field_values = PublicValues::<[F; 4], [F; 3], [F; 4], F>::from(*self);
        let ret_ref_mut: &mut PublicValues<[F; 4], [F; 3], [F; 4], F> =
            ret.as_mut_slice().borrow_mut();
        *ret_ref_mut = field_values;
        ret
    }

    /// Resets the public values to zero.
    #[must_use]
    pub fn reset(&self) -> Self {
        let mut copy = *self;
        copy.pc_start = 0;
        copy.next_pc = 0;
        copy.previous_init_addr = 0;
        copy.last_init_addr = 0;
        copy.previous_finalize_addr = 0;
        copy.last_finalize_addr = 0;
        copy.previous_init_page_idx = 0;
        copy.last_init_page_idx = 0;
        copy.previous_finalize_page_idx = 0;
        copy.last_finalize_page_idx = 0;
        copy
    }
}

impl<F: PrimeField32> PublicValues<[F; 4], [F; 3], [F; 4], F> {
    /// Returns the commit digest as a vector of little-endian bytes.
    pub fn commit_digest_bytes(&self) -> Vec<u8> {
        self.committed_value_digest
            .iter()
            .flat_map(|w| w.iter().map(|f| f.as_canonical_u32() as u8))
            .collect_vec()
    }
}

impl<T: Clone> Borrow<PublicValues<[T; 4], [T; 3], [T; 4], T>> for [T] {
    fn borrow(&self) -> &PublicValues<[T; 4], [T; 3], [T; 4], T> {
        let size = std::mem::size_of::<PublicValues<[u8; 4], [u8; 3], [u8; 4], u8>>();
        debug_assert!(self.len() >= size);
        let slice = &self[0..size];
        let (prefix, shorts, _suffix) =
            unsafe { slice.align_to::<PublicValues<[T; 4], [T; 3], [T; 4], T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<T: Clone> BorrowMut<PublicValues<[T; 4], [T; 3], [T; 4], T>> for [T] {
    fn borrow_mut(&mut self) -> &mut PublicValues<[T; 4], [T; 3], [T; 4], T> {
        let size = std::mem::size_of::<PublicValues<[u8; 4], [u8; 3], [u8; 4], u8>>();
        debug_assert!(self.len() >= size);
        let slice = &mut self[0..size];
        let (prefix, shorts, _suffix) =
            unsafe { slice.align_to_mut::<PublicValues<[T; 4], [T; 3], [T; 4], T>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

impl<F: AbstractField> From<PublicValues<u32, u64, u64, u32>>
    for PublicValues<[F; 4], [F; 3], [F; 4], F>
{
    #[allow(clippy::too_many_lines)]
    fn from(value: PublicValues<u32, u64, u64, u32>) -> Self {
        let PublicValues {
            prev_committed_value_digest,
            committed_value_digest,
            prev_deferred_proofs_digest,
            deferred_proofs_digest,
            pc_start,
            next_pc,
            prev_exit_code,
            exit_code,
            is_execution_shard,
            previous_init_addr,
            last_init_addr,
            previous_finalize_addr,
            last_finalize_addr,
            previous_init_page_idx,
            last_init_page_idx,
            previous_finalize_page_idx,
            last_finalize_page_idx,
            initial_timestamp,
            last_timestamp,
            is_timestamp_high_eq,
            inv_timestamp_high,
            is_timestamp_low_eq,
            inv_timestamp_low,
            global_init_count,
            global_finalize_count,
            global_page_prot_init_count,
            global_page_prot_finalize_count,
            global_count,
            global_cumulative_sum,
            prev_commit_syscall,
            commit_syscall,
            prev_commit_deferred_syscall,
            commit_deferred_syscall,
            is_untrusted_programs_enabled,
            proof_nonce,
            initial_timestamp_inv,
            last_timestamp_inv,
            is_first_shard,
            ..
        } = value;

        let prev_committed_value_digest: [_; PV_DIGEST_NUM_WORDS] = core::array::from_fn(|i| {
            [
                F::from_canonical_u32(prev_committed_value_digest[i] & 0xFF),
                F::from_canonical_u32((prev_committed_value_digest[i] >> 8) & 0xFF),
                F::from_canonical_u32((prev_committed_value_digest[i] >> 16) & 0xFF),
                F::from_canonical_u32((prev_committed_value_digest[i] >> 24) & 0xFF),
            ]
        });

        let committed_value_digest: [_; PV_DIGEST_NUM_WORDS] = core::array::from_fn(|i| {
            [
                F::from_canonical_u32(committed_value_digest[i] & 0xFF),
                F::from_canonical_u32((committed_value_digest[i] >> 8) & 0xFF),
                F::from_canonical_u32((committed_value_digest[i] >> 16) & 0xFF),
                F::from_canonical_u32((committed_value_digest[i] >> 24) & 0xFF),
            ]
        });

        let prev_deferred_proofs_digest: [_; POSEIDON_NUM_WORDS] =
            core::array::from_fn(|i| F::from_canonical_u32(prev_deferred_proofs_digest[i]));

        let deferred_proofs_digest: [_; POSEIDON_NUM_WORDS] =
            core::array::from_fn(|i| F::from_canonical_u32(deferred_proofs_digest[i]));

        let pc_start = [
            F::from_canonical_u16((pc_start & 0xFFFF) as u16),
            F::from_canonical_u16(((pc_start >> 16) & 0xFFFF) as u16),
            F::from_canonical_u16(((pc_start >> 32) & 0xFFFF) as u16),
        ];
        let next_pc = [
            F::from_canonical_u16((next_pc & 0xFFFF) as u16),
            F::from_canonical_u16(((next_pc >> 16) & 0xFFFF) as u16),
            F::from_canonical_u16(((next_pc >> 32) & 0xFFFF) as u16),
        ];
        let exit_code = F::from_canonical_u32(exit_code);
        let prev_exit_code = F::from_canonical_u32(prev_exit_code);
        let is_execution_shard = F::from_canonical_u32(is_execution_shard);
        let previous_init_addr = [
            F::from_canonical_u16((previous_init_addr & 0xFFFF) as u16),
            F::from_canonical_u16(((previous_init_addr >> 16) & 0xFFFF) as u16),
            F::from_canonical_u16(((previous_init_addr >> 32) & 0xFFFF) as u16),
        ];
        let last_init_addr = [
            F::from_canonical_u16((last_init_addr & 0xFFFF) as u16),
            F::from_canonical_u16(((last_init_addr >> 16) & 0xFFFF) as u16),
            F::from_canonical_u16(((last_init_addr >> 32) & 0xFFFF) as u16),
        ];
        let previous_finalize_addr = [
            F::from_canonical_u16((previous_finalize_addr & 0xFFFF) as u16),
            F::from_canonical_u16(((previous_finalize_addr >> 16) & 0xFFFF) as u16),
            F::from_canonical_u16(((previous_finalize_addr >> 32) & 0xFFFF) as u16),
        ];
        let last_finalize_addr = [
            F::from_canonical_u16((last_finalize_addr & 0xFFFF) as u16),
            F::from_canonical_u16(((last_finalize_addr >> 16) & 0xFFFF) as u16),
            F::from_canonical_u16(((last_finalize_addr >> 32) & 0xFFFF) as u16),
        ];
        let previous_init_page_idx: [F; 3] = core::array::from_fn(|i| {
            F::from_canonical_u16(split_page_idx(previous_init_page_idx)[i])
        });
        let last_init_page_idx: [F; 3] =
            core::array::from_fn(|i| F::from_canonical_u16(split_page_idx(last_init_page_idx)[i]));
        let previous_finalize_page_idx: [F; 3] = core::array::from_fn(|i| {
            F::from_canonical_u16(split_page_idx(previous_finalize_page_idx)[i])
        });
        let last_finalize_page_idx: [F; 3] = core::array::from_fn(|i| {
            F::from_canonical_u16(split_page_idx(last_finalize_page_idx)[i])
        });
        let initial_timestamp = [
            F::from_canonical_u16((initial_timestamp >> 32) as u16),
            F::from_canonical_u8(((initial_timestamp >> 24) & 0xFF) as u8),
            F::from_canonical_u8(((initial_timestamp >> 16) & 0xFF) as u8),
            F::from_canonical_u16((initial_timestamp & 0xFFFF) as u16),
        ];
        let last_timestamp = [
            F::from_canonical_u16((last_timestamp >> 32) as u16),
            F::from_canonical_u8(((last_timestamp >> 24) & 0xFF) as u8),
            F::from_canonical_u8(((last_timestamp >> 16) & 0xFF) as u8),
            F::from_canonical_u16((last_timestamp & 0xFFFF) as u16),
        ];

        let is_timestamp_high_eq = F::from_canonical_u32(is_timestamp_high_eq);
        let inv_timestamp_high = F::from_canonical_u32(inv_timestamp_high);
        let is_timestamp_low_eq = F::from_canonical_u32(is_timestamp_low_eq);
        let inv_timestamp_low = F::from_canonical_u32(inv_timestamp_low);

        let global_init_count = F::from_canonical_u32(global_init_count);
        let global_finalize_count = F::from_canonical_u32(global_finalize_count);
        let global_page_prot_init_count = F::from_canonical_u32(global_page_prot_init_count);
        let global_page_prot_finalize_count =
            F::from_canonical_u32(global_page_prot_finalize_count);
        let global_count = F::from_canonical_u32(global_count);
        let global_cumulative_sum =
            SepticDigest(SepticCurve::convert(global_cumulative_sum.0, F::from_canonical_u32));

        let prev_commit_syscall = F::from_canonical_u32(prev_commit_syscall);
        let commit_syscall = F::from_canonical_u32(commit_syscall);
        let prev_commit_deferred_syscall = F::from_canonical_u32(prev_commit_deferred_syscall);
        let commit_deferred_syscall = F::from_canonical_u32(commit_deferred_syscall);

        let initial_timestamp_inv = F::from_canonical_u32(initial_timestamp_inv);
        let last_timestamp_inv = F::from_canonical_u32(last_timestamp_inv);
        let is_first_shard = F::from_canonical_u32(is_first_shard);
        let is_untrusted_programs_enabled = F::from_canonical_u32(is_untrusted_programs_enabled);

        let proof_nonce: [_; PROOF_NONCE_NUM_WORDS] =
            core::array::from_fn(|i| F::from_canonical_u32(proof_nonce[i]));

        Self {
            prev_committed_value_digest,
            committed_value_digest,
            prev_deferred_proofs_digest,
            deferred_proofs_digest,
            pc_start,
            next_pc,
            prev_exit_code,
            exit_code,
            is_execution_shard,
            previous_init_addr,
            last_init_addr,
            previous_finalize_addr,
            last_finalize_addr,
            previous_init_page_idx,
            last_init_page_idx,
            previous_finalize_page_idx,
            last_finalize_page_idx,
            initial_timestamp,
            last_timestamp,
            is_timestamp_high_eq,
            inv_timestamp_high,
            is_timestamp_low_eq,
            inv_timestamp_low,
            global_init_count,
            global_finalize_count,
            global_page_prot_init_count,
            global_page_prot_finalize_count,
            global_count,
            global_cumulative_sum,
            prev_commit_syscall,
            commit_syscall,
            prev_commit_deferred_syscall,
            commit_deferred_syscall,
            is_untrusted_programs_enabled,
            initial_timestamp_inv,
            last_timestamp_inv,
            is_first_shard,
            proof_nonce,
            empty: core::array::from_fn(|_| F::zero()),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::air::public_values;

    /// Check that the [`PI_DIGEST_NUM_WORDS`] number match the zkVM crate's.
    #[test]
    fn test_public_values_digest_num_words_consistency_zkvm() {
        assert_eq!(public_values::PV_DIGEST_NUM_WORDS, sp1_zkvm::PV_DIGEST_NUM_WORDS);
    }
}
