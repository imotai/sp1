use std::{
    array,
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
    mem::MaybeUninit,
};

use itertools::Itertools;

use slop_air::Air;

use slop_algebra::AbstractField;
use slop_jagged::JaggedConfig;

use serde::{Deserialize, Serialize};
use sp1_recursion_compiler::ir::{Builder, Felt, IrIter};

use sp1_primitives::SP1Field;
use sp1_recursion_executor::{RecursionPublicValues, RECURSIVE_PROOF_NUM_PV_ELTS};

use sp1_hypercube::{
    air::{MachineAir, POSEIDON_NUM_WORDS, PV_DIGEST_NUM_WORDS},
    MachineConfig, MachineVerifyingKey, ShardProof, DIGEST_SIZE,
};

use crate::{
    basefold::{RecursiveBasefoldConfigImpl, RecursiveBasefoldProof, RecursiveBasefoldVerifier},
    challenger::CanObserveVariable,
    jagged::RecursiveJaggedConfig,
    machine::{
        assert_complete, assert_recursion_public_values_valid, recursion_public_values_digest,
        root_public_values_digest,
    },
    shard::{MachineVerifyingKeyVariable, RecursiveShardVerifier, ShardProofVariable},
    zerocheck::RecursiveVerifierConstraintFolder,
    CircuitConfig, SP1FieldConfigVariable, EF,
};

use sp1_recursion_compiler::circuit::CircuitV2Builder;

use super::InnerVal;
/// A program to verify a batch of recursive proofs and aggregate their public values.
#[derive(Debug, Clone, Copy)]
pub struct SP1CompressVerifier<C, SC, A, JC> {
    _phantom: PhantomData<(C, SC, A, JC)>,
}

pub enum PublicValuesOutputDigest {
    Reduce,
    Root,
}

/// Witness layout for the compress stage verifier.
#[allow(clippy::type_complexity)]
pub struct SP1ShapedWitnessVariable<
    C: CircuitConfig<F = SP1Field, EF = EF>,
    SC: SP1FieldConfigVariable<C> + Send + Sync,
    JC: RecursiveJaggedConfig<
        BatchPcsVerifier = RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
    >,
> {
    /// The shard proofs to verify.
    pub vks_and_proofs: Vec<(MachineVerifyingKeyVariable<C, SC>, ShardProofVariable<C, SC, JC>)>,
    pub is_complete: Felt<C::F>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "ShardProof<SC>: Serialize"))]
#[serde(bound(deserialize = "ShardProof<SC>: Deserialize<'de>"))]
/// An input layout for the shard proofs that have been normalized to a standard shape.
pub struct SP1ShapedWitnessValues<SC: MachineConfig> {
    pub vks_and_proofs: Vec<(MachineVerifyingKey<SC>, ShardProof<SC>)>,
    pub is_complete: bool,
}

impl<C, SC, A, JC> SP1CompressVerifier<C, SC, A, JC>
where
    SC: SP1FieldConfigVariable<C> + Send + Sync,
    C: CircuitConfig<F = SP1Field, EF = <SC as JaggedConfig>::EF>,
    A: MachineAir<InnerVal> + for<'a> Air<RecursiveVerifierConstraintFolder<'a, C>>,
    JC: RecursiveJaggedConfig<
        F = SP1Field,
        EF = C::EF,
        Circuit = C,
        Commitment = SC::DigestVariable,
        Challenger = SC::FriChallengerVariable,
        BatchPcsProof = RecursiveBasefoldProof<RecursiveBasefoldConfigImpl<C, SC>>,
        BatchPcsVerifier = RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
    >,
{
    /// Verify a batch of recursive proofs and aggregate their public values.
    ///
    /// The compression verifier can aggregate proofs of different kinds:
    /// - Core proofs: proofs which are recursive proof of a batch of SP1 shard proofs. The
    ///   implementation in this function assumes a fixed recursive verifier specified by
    ///   `recursive_vk`.
    /// - Deferred proofs: proofs which are recursive proof of a batch of deferred proofs. The
    ///   implementation in this function assumes a fixed deferred verification program specified by
    ///   `deferred_vk`.
    /// - Compress proofs: these are proofs which refer to a prove of this program. The key for it
    ///   is part of public values will be propagated across all levels of recursion and will be
    ///   checked against itself as in [sp1_prover::Prover] or as in [super::SP1RootVerifier].
    pub fn verify(
        builder: &mut Builder<C>,
        machine: &RecursiveShardVerifier<A, SC, C, JC>,
        input: SP1ShapedWitnessVariable<C, SC, JC>,
        vk_root: [Felt<C::F>; DIGEST_SIZE],
        kind: PublicValuesOutputDigest,
    ) {
        // Read input.
        let SP1ShapedWitnessVariable { vks_and_proofs, is_complete } = input;

        // Initialize the values for the aggregated public output.
        let mut reduce_public_values_stream: Vec<Felt<_>> = (0..RECURSIVE_PROOF_NUM_PV_ELTS)
            .map(|_| unsafe { MaybeUninit::zeroed().assume_init() })
            .collect();
        let compress_public_values: &mut RecursionPublicValues<_> =
            reduce_public_values_stream.as_mut_slice().borrow_mut();

        // Make sure there is at least one proof.
        assert!(!vks_and_proofs.is_empty());

        // Initialize the consistency check variables.
        let mut sp1_vk_digest: [Felt<_>; DIGEST_SIZE] =
            array::from_fn(|_| unsafe { MaybeUninit::zeroed().assume_init() });
        let mut pc: [Felt<_>; 3] =
            array::from_fn(|_| unsafe { MaybeUninit::zeroed().assume_init() });
        let mut shard: Felt<_> = unsafe { MaybeUninit::zeroed().assume_init() };
        let mut current_exit_code: Felt<_> = unsafe { MaybeUninit::zeroed().assume_init() };
        let mut current_timestamp: [Felt<_>; 4] = array::from_fn(|_| builder.uninit());

        let mut execution_shard: Felt<_> = unsafe { MaybeUninit::zeroed().assume_init() };
        let mut committed_value_digest: [[Felt<_>; 4]; PV_DIGEST_NUM_WORDS] =
            array::from_fn(|_| array::from_fn(|_| unsafe { MaybeUninit::zeroed().assume_init() }));
        let mut deferred_proofs_digest: [Felt<_>; POSEIDON_NUM_WORDS] =
            array::from_fn(|_| unsafe { MaybeUninit::zeroed().assume_init() });
        let mut reconstruct_deferred_digest: [Felt<_>; POSEIDON_NUM_WORDS] =
            core::array::from_fn(|_| unsafe { MaybeUninit::zeroed().assume_init() });
        let mut global_cumulative_sums = Vec::new();
        let mut init_addr: [Felt<_>; 3] =
            array::from_fn(|_| unsafe { MaybeUninit::zeroed().assume_init() });
        let mut finalize_addr: [Felt<_>; 3] =
            array::from_fn(|_| unsafe { MaybeUninit::zeroed().assume_init() });
        let mut commit_syscall: Felt<_> = unsafe { MaybeUninit::zeroed().assume_init() };
        let mut commit_deferred_syscall: Felt<_> = unsafe { MaybeUninit::zeroed().assume_init() };

        // Verify the shard proofs.
        // Verification of proofs can be done in parallel but the aggregation/consistency checks
        // must be done sequentially.
        vks_and_proofs.iter().ir_par_map_collect::<Vec<_>, _, _>(
            builder,
            |builder, (vk, shard_proof)| {
                // Prepare a challenger.
                let mut challenger = SC::challenger_variable(builder);

                // Observe the vk and start pc.
                challenger.observe(builder, vk.preprocessed_commit);
                challenger.observe_slice(builder, vk.pc_start);
                challenger.observe_slice(builder, vk.initial_global_cumulative_sum.0.x.0);
                challenger.observe_slice(builder, vk.initial_global_cumulative_sum.0.y.0);
                // Observe the padding.
                let zero: Felt<_> = builder.eval(C::F::zero());
                for _ in 0..7 {
                    challenger.observe(builder, zero);
                }
                // Verify the shard proof.
                machine.verify_shard(builder, vk, shard_proof, &mut challenger);
            },
        );

        // Check consistency and aggregate public values.
        for (i, (_, shard_proof)) in vks_and_proofs.into_iter().enumerate() {
            // Get the current public values.
            let current_public_values: &RecursionPublicValues<Felt<C::F>> =
                shard_proof.public_values.as_slice().borrow();
            // Assert that the public values are valid.
            assert_recursion_public_values_valid::<C, SC>(builder, current_public_values);
            // Assert that the vk root is the same as the witnessed one.
            for (expected, actual) in vk_root.iter().zip(current_public_values.vk_root.iter()) {
                builder.assert_felt_eq(*expected, *actual);
            }

            // Set the exit code, it is already constrained to be zero in the previous proof.
            if i == 0 {
                // Initialize global and accumulated values.

                // Assign the committed values and deferred proof digests.
                compress_public_values.prev_committed_value_digest =
                    current_public_values.prev_committed_value_digest;
                committed_value_digest = current_public_values.prev_committed_value_digest;

                compress_public_values.prev_deferred_proofs_digest =
                    current_public_values.prev_deferred_proofs_digest;
                deferred_proofs_digest = current_public_values.prev_deferred_proofs_digest;

                // Initiallize start pc.
                compress_public_values.pc_start = current_public_values.pc_start;
                pc = current_public_values.pc_start;

                // Initialize start shard.
                compress_public_values.start_shard = current_public_values.start_shard;
                shard = current_public_values.start_shard;

                // Initialize start execution shard.
                compress_public_values.start_execution_shard =
                    current_public_values.start_execution_shard;
                execution_shard = current_public_values.start_execution_shard;

                // Initialize timestamp.
                compress_public_values.initial_timestamp = current_public_values.initial_timestamp;
                current_timestamp = current_public_values.initial_timestamp;

                // Initialize the MemoryInitialize address.
                compress_public_values.previous_init_addr =
                    current_public_values.previous_init_addr;
                init_addr = current_public_values.previous_init_addr;

                // Initialize the MemoryFinalize address.
                compress_public_values.previous_finalize_addr =
                    current_public_values.previous_finalize_addr;
                finalize_addr = current_public_values.previous_finalize_addr;

                // Initialize the start of deferred digests.
                compress_public_values.start_reconstruct_deferred_digest =
                    current_public_values.start_reconstruct_deferred_digest;
                reconstruct_deferred_digest =
                    current_public_values.start_reconstruct_deferred_digest;

                // Initialize exit code.
                compress_public_values.prev_exit_code = current_public_values.prev_exit_code;
                current_exit_code = current_public_values.prev_exit_code;

                // Initialize `commit_syscall`.
                compress_public_values.prev_commit_syscall =
                    current_public_values.prev_commit_syscall;
                commit_syscall = current_public_values.prev_commit_syscall;

                // Initialize `commit_deferred_syscall`.
                compress_public_values.prev_commit_deferred_syscall =
                    current_public_values.prev_commit_deferred_syscall;
                commit_deferred_syscall = current_public_values.prev_commit_deferred_syscall;

                // Initialize the sp1_vk digest
                compress_public_values.sp1_vk_digest = current_public_values.sp1_vk_digest;
                sp1_vk_digest = current_public_values.sp1_vk_digest;
            }

            // Assert that the current values match the accumulated values and update them.

            // Assert that the `prev_committed_value_digest` is equal to current one, then update.
            for (word, current_word) in committed_value_digest
                .iter()
                .zip_eq(current_public_values.prev_committed_value_digest.iter())
            {
                for (limb, current_limb) in word.iter().zip_eq(current_word.iter()) {
                    builder.assert_felt_eq(*limb, *current_limb);
                }
            }
            committed_value_digest = current_public_values.committed_value_digest;

            // Assert that the `prev_deferred_proofs_digest` is equal to current one, then update.
            for (limb, current_limb) in deferred_proofs_digest
                .iter()
                .zip_eq(current_public_values.prev_deferred_proofs_digest.iter())
            {
                builder.assert_felt_eq(*limb, *current_limb);
            }
            deferred_proofs_digest = current_public_values.deferred_proofs_digest;

            // Assert that the start pc is equal to the current pc, then update.
            for (limb, current_limb) in pc.iter().zip(current_public_values.pc_start.iter()) {
                builder.assert_felt_eq(*limb, *current_limb);
            }
            pc = current_public_values.next_pc;

            // Verify that the shard is equal to the current shard, then update.
            builder.assert_felt_eq(shard, current_public_values.start_shard);
            shard = current_public_values.next_shard;

            // Verify that the execution shard is equal to the current one, then update.
            builder.assert_felt_eq(execution_shard, current_public_values.start_execution_shard);
            execution_shard = current_public_values.next_execution_shard;

            // Verify that the timestamp is equal to the current one, then update.
            for (limb, current_limb) in
                current_timestamp.iter().zip(current_public_values.initial_timestamp.iter())
            {
                builder.assert_felt_eq(*limb, *current_limb);
            }
            current_timestamp = current_public_values.last_timestamp;

            // Verify that the init address is equal to the current one, then update.
            for (limb, current_limb) in
                init_addr.iter().zip(current_public_values.previous_init_addr.iter())
            {
                builder.assert_felt_eq(*limb, *current_limb);
            }
            init_addr = current_public_values.last_init_addr;

            // Verify that the init address is equal to the current one, then update.
            for (limb, current_limb) in
                finalize_addr.iter().zip(current_public_values.previous_finalize_addr.iter())
            {
                builder.assert_felt_eq(*limb, *current_limb);
            }
            finalize_addr = current_public_values.last_finalize_addr;

            // Assert that the start deferred digest is equal to the current one, then update.
            for (digest, current_digest) in reconstruct_deferred_digest
                .iter()
                .zip_eq(current_public_values.start_reconstruct_deferred_digest.iter())
            {
                builder.assert_felt_eq(*digest, *current_digest);
            }
            reconstruct_deferred_digest = current_public_values.end_reconstruct_deferred_digest;

            // Assert that the `prev_exit_code` is equal to the current one, then update.
            builder.assert_felt_eq(current_exit_code, current_public_values.prev_exit_code);
            current_exit_code = current_public_values.exit_code;

            // Assert that the `prev_commit_syscall` is equal to the current one, then update.
            builder.assert_felt_eq(commit_syscall, current_public_values.prev_commit_syscall);
            commit_syscall = current_public_values.commit_syscall;

            // Assert that `prev_commit_deferred_syscall` is equal to the current one, then update.
            builder.assert_felt_eq(
                commit_deferred_syscall,
                current_public_values.prev_commit_deferred_syscall,
            );
            commit_deferred_syscall = current_public_values.commit_deferred_syscall;

            // Assert that the sp1_vk digest is always the same.
            for (digest, current) in sp1_vk_digest.iter().zip(current_public_values.sp1_vk_digest) {
                builder.assert_felt_eq(*digest, current);
            }

            // Add the global cumulative sums to the vector.
            global_cumulative_sums.push(current_public_values.global_cumulative_sum);
        }

        // Sum all the global cumulative sum of the proofs.
        let global_cumulative_sum = builder.sum_digest_v2(global_cumulative_sums);

        // Update the global values from the last accumulated values.
        // Set the `committed_value_digest`.
        compress_public_values.committed_value_digest = committed_value_digest;
        // Set the `deferred_proofs_digest`.
        compress_public_values.deferred_proofs_digest = deferred_proofs_digest;
        // Set next_pc to be the last pc.
        compress_public_values.next_pc = pc;
        // Set next shard to be the last shard
        compress_public_values.next_shard = shard;
        // Set next execution shard to be the last execution shard
        compress_public_values.next_execution_shard = execution_shard;
        // Set the timestamp to be the last timestamp.
        compress_public_values.last_timestamp = current_timestamp;
        // Set the MemoryInitialize address to be the last MemoryInitialize address.
        compress_public_values.last_init_addr = init_addr;
        // Set the MemoryFinalize address to be the last MemoryFinalize address.
        compress_public_values.last_finalize_addr = finalize_addr;
        // Set the start reconstruct deferred digest to be the last reconstruct deferred digest.
        compress_public_values.end_reconstruct_deferred_digest = reconstruct_deferred_digest;
        // Set sp1_vk digest to the one from the proof values.
        compress_public_values.sp1_vk_digest = sp1_vk_digest;
        // Reflect the vk root.
        compress_public_values.vk_root = vk_root;
        // Assign the cumulative sum.
        compress_public_values.global_cumulative_sum = global_cumulative_sum;
        // Assign the `is_complete` flag.
        compress_public_values.is_complete = is_complete;
        // Set the exit code.
        compress_public_values.exit_code = current_exit_code;
        // Set the `commit_syscall` flag.
        compress_public_values.commit_syscall = commit_syscall;
        // Set the `commit_deferred_syscall` flag.
        compress_public_values.commit_deferred_syscall = commit_deferred_syscall;
        // Set the digest according to the previous values.
        compress_public_values.digest = match kind {
            PublicValuesOutputDigest::Reduce => {
                recursion_public_values_digest::<C, SC>(builder, compress_public_values)
            }
            PublicValuesOutputDigest::Root => {
                root_public_values_digest::<C, SC>(builder, compress_public_values)
            }
        };

        // If the proof is complete, make completeness assertions.
        assert_complete(builder, compress_public_values, is_complete);

        SC::commit_recursion_public_values(builder, *compress_public_values);
    }
}
