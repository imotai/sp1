use std::{
    array,
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
    mem::MaybeUninit,
};

use itertools::{izip, Itertools};

use p3_air::Air;
use slop_baby_bear::BabyBear;

use slop_algebra::AbstractField;
use slop_jagged::JaggedConfig;

use serde::{Deserialize, Serialize};
use sp1_recursion_compiler::ir::{Builder, Felt, SymbolicFelt};

use sp1_recursion_executor::{RecursionPublicValues, RECURSIVE_PROOF_NUM_PV_ELTS};

use sp1_stark::{
    air::{MachineAir, POSEIDON_NUM_WORDS, PV_DIGEST_NUM_WORDS},
    shape::OrderedShape,
    MachineConfig, MachineVerifyingKey, ShardProof, Word, DIGEST_SIZE,
};

use crate::{
    basefold::{RecursiveBasefoldConfigImpl, RecursiveBasefoldProof, RecursiveBasefoldVerifier},
    challenger::CanObserveVariable,
    jagged::RecursiveJaggedConfig,
    machine::{
        assert_complete, assert_recursion_public_values_valid, recursion_public_values_digest,
        root_public_values_digest,
    },
    shard::{MachineVerifyingKeyVariable, ShardProofVariable, StarkVerifier},
    zerocheck::RecursiveVerifierConstraintFolder,
    BabyBearFriConfigVariable, CircuitConfig, EF,
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
pub struct SP1CompressWitnessVariable<
    C: CircuitConfig<F = BabyBear, EF = EF>,
    SC: BabyBearFriConfigVariable<C> + Send + Sync,
    JC: RecursiveJaggedConfig<
        BatchPcsVerifier = RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
    >,
> {
    /// The shard proofs to verify.
    pub vks_and_proofs: Vec<(MachineVerifyingKeyVariable<C, SC>, ShardProofVariable<C, SC, JC>)>,
    pub is_complete: Felt<C::F>,
}

/// An input layout for the reduce verifier.
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "ShardProof<SC>: Serialize"))]
#[serde(bound(deserialize = "ShardProof<SC>: Deserialize<'de>"))]
pub struct SP1CompressWitnessValues<SC: MachineConfig> {
    pub vks_and_proofs: Vec<(MachineVerifyingKey<SC>, ShardProof<SC>)>,
    pub is_complete: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SP1CompressShape {
    proof_shapes: Vec<OrderedShape>,
}

impl<C, SC, A, JC> SP1CompressVerifier<C, SC, A, JC>
where
    SC: BabyBearFriConfigVariable<C> + Send + Sync,
    C: CircuitConfig<F = BabyBear, EF = <SC as JaggedConfig>::EF>,
    // <SC::ValMmcs as Mmcs<BabyBear>>::ProverData<RowMajorMatrix<BabyBear>>: Clone,
    A: MachineAir<InnerVal> + for<'a> Air<RecursiveVerifierConstraintFolder<'a, C>>,
    JC: RecursiveJaggedConfig<
        F = BabyBear,
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
        machine: &StarkVerifier<A, SC, C, JC>,
        input: SP1CompressWitnessVariable<C, SC, JC>,
        //vk_root: [Felt<C::F>; DIGEST_SIZE],
        kind: PublicValuesOutputDigest,
        challenger: &mut SC::FriChallengerVariable,
    ) {
        // Read input.
        let SP1CompressWitnessVariable { vks_and_proofs, is_complete } = input;

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
        let mut pc: Felt<_> = unsafe { MaybeUninit::zeroed().assume_init() };
        let mut shard: Felt<_> = unsafe { MaybeUninit::zeroed().assume_init() };

        let mut exit_code: Felt<_> = builder.uninit();

        let mut execution_shard: Felt<_> = unsafe { MaybeUninit::zeroed().assume_init() };
        let mut committed_value_digest: [Word<Felt<_>>; PV_DIGEST_NUM_WORDS] =
            array::from_fn(|_| {
                Word(array::from_fn(|_| unsafe { MaybeUninit::zeroed().assume_init() }))
            });
        let mut deferred_proofs_digest: [Felt<_>; POSEIDON_NUM_WORDS] =
            array::from_fn(|_| unsafe { MaybeUninit::zeroed().assume_init() });
        let mut reconstruct_deferred_digest: [Felt<_>; POSEIDON_NUM_WORDS] =
            core::array::from_fn(|_| unsafe { MaybeUninit::zeroed().assume_init() });
        let mut global_cumulative_sums = Vec::new();
        let mut init_addr_bits: [Felt<_>; 32] =
            core::array::from_fn(|_| unsafe { MaybeUninit::zeroed().assume_init() });
        let mut finalize_addr_bits: [Felt<_>; 32] =
            core::array::from_fn(|_| unsafe { MaybeUninit::zeroed().assume_init() });

        // Initialize a flag to denote if the any of the recursive proofs represents a shard range
        // where at least once of the shards is an execution shard (i.e. contains cpu).
        let mut contains_execution_shard: Felt<_> = builder.eval(C::F::zero());

        // Verify proofs, check consistency, and aggregate public values.
        for (i, (vk, shard_proof)) in vks_and_proofs.into_iter().enumerate() {
            // Verify the shard proof.

            // Prepare a challenger.
            // let mut challenger = machine.pcs_verifier.challenger_variable(builder);

            // Observe the vk and start pc.
            if let Some(vk) = vk.preprocessed_commit.as_ref() {
                challenger.observe(builder, *vk)
            }
            challenger.observe(builder, vk.pc_start);
            challenger.observe_slice(builder, vk.initial_global_cumulative_sum.0.x.0);
            challenger.observe_slice(builder, vk.initial_global_cumulative_sum.0.y.0);
            // Observe the padding.
            let zero: Felt<_> = builder.eval(C::F::zero());
            challenger.observe(builder, zero);

            // Observe the public values.
            challenger.observe_slice(
                builder,
                shard_proof.public_values[0..machine.machine.num_pv_elts()].iter().copied(),
            );

            machine.verify_shard(builder, &vk, &shard_proof, challenger);

            // Get the current public values.
            let current_public_values: &RecursionPublicValues<Felt<C::F>> =
                shard_proof.public_values.as_slice().borrow();
            // Assert that the public values are valid.
            assert_recursion_public_values_valid::<C, SC>(builder, current_public_values);
            // Assert that the vk root is the same as the witnessed one.
            // for (expected, actual) in vk_root.iter().zip(current_public_values.vk_root.iter()) {
            //     builder.assert_felt_eq(*expected, *actual);
            // }

            // Set the exit code, it is already constrained to be zero in the previous proof.
            exit_code = current_public_values.exit_code;

            if i == 0 {
                // Initialize global and accumulated values.

                // Initialize the start of deferred digests.
                for (digest, current_digest, global_digest) in izip!(
                    reconstruct_deferred_digest.iter_mut(),
                    current_public_values.start_reconstruct_deferred_digest.iter(),
                    compress_public_values.start_reconstruct_deferred_digest.iter_mut()
                ) {
                    *digest = *current_digest;
                    *global_digest = *current_digest;
                }

                // Initialize the sp1_vk digest
                for (digest, first_digest) in
                    sp1_vk_digest.iter_mut().zip(current_public_values.sp1_vk_digest)
                {
                    *digest = first_digest;
                }

                // Initiallize start pc.
                compress_public_values.start_pc = current_public_values.start_pc;
                pc = current_public_values.start_pc;

                // Initialize start shard.
                compress_public_values.start_shard = current_public_values.start_shard;
                shard = current_public_values.start_shard;

                // Initialize start execution shard.
                compress_public_values.start_execution_shard =
                    current_public_values.start_execution_shard;
                execution_shard = current_public_values.start_execution_shard;

                // Initialize the MemoryInitialize address bits.
                for (bit, (first_bit, current_bit)) in init_addr_bits.iter_mut().zip(
                    compress_public_values
                        .previous_init_addr_bits
                        .iter_mut()
                        .zip(current_public_values.previous_init_addr_bits.iter()),
                ) {
                    *bit = *current_bit;
                    *first_bit = *current_bit;
                }

                // Initialize the MemoryFinalize address bits.
                for (bit, (first_bit, current_bit)) in finalize_addr_bits.iter_mut().zip(
                    compress_public_values
                        .previous_finalize_addr_bits
                        .iter_mut()
                        .zip(current_public_values.previous_finalize_addr_bits.iter()),
                ) {
                    *bit = *current_bit;
                    *first_bit = *current_bit;
                }

                // Assign the committed values and deferred proof digests.
                for (word, current_word) in committed_value_digest
                    .iter_mut()
                    .zip_eq(current_public_values.committed_value_digest.iter())
                {
                    for (byte, current_byte) in word.0.iter_mut().zip_eq(current_word.0.iter()) {
                        *byte = *current_byte;
                    }
                }

                for (digest, current_digest) in deferred_proofs_digest
                    .iter_mut()
                    .zip_eq(current_public_values.deferred_proofs_digest.iter())
                {
                    *digest = *current_digest;
                }
            }

            // Assert that the current values match the accumulated values.

            // Assert that the start deferred digest is equal to the current deferred digest.
            for (digest, current_digest) in reconstruct_deferred_digest
                .iter()
                .zip_eq(current_public_values.start_reconstruct_deferred_digest.iter())
            {
                builder.assert_felt_eq(*digest, *current_digest);
            }

            // // Consistency checks for all accumulated values.

            // Assert that the sp1_vk digest is always the same.
            for (digest, current) in sp1_vk_digest.iter().zip(current_public_values.sp1_vk_digest) {
                builder.assert_felt_eq(*digest, current);
            }

            // Assert that the start pc is equal to the current pc.
            builder.assert_felt_eq(pc, current_public_values.start_pc);

            // Verify that the shard is equal to the current shard.
            builder.assert_felt_eq(shard, current_public_values.start_shard);

            // Execution shard constraints.
            {
                // Assert that `contains_execution_shard` is boolean.
                builder.assert_felt_eq(
                    current_public_values.contains_execution_shard
                        * (SymbolicFelt::one() - current_public_values.contains_execution_shard),
                    C::F::zero(),
                );
                // A flag to indicate whether the first execution shard has been seen. We have:
                // - `is_first_execution_shard_seen`  = current_contains_execution_shard &&
                //   !execution_shard_seen_before.
                // Since `contains_execution_shard` is the boolean flag used to denote if we have
                // seen an execution shard, we can use it to denote if we have seen an execution
                // shard before.
                let is_first_execution_shard_seen: Felt<_> = builder.eval(
                    current_public_values.contains_execution_shard
                        * (SymbolicFelt::one() - contains_execution_shard),
                );

                // If this is the first execution shard, then we update the start execution shard
                // and the `execution_shard` values.
                compress_public_values.start_execution_shard = builder.eval(
                    current_public_values.start_execution_shard * is_first_execution_shard_seen
                        + compress_public_values.start_execution_shard
                            * (SymbolicFelt::one() - is_first_execution_shard_seen),
                );
                execution_shard = builder.eval(
                    current_public_values.start_execution_shard * is_first_execution_shard_seen
                        + execution_shard * (SymbolicFelt::one() - is_first_execution_shard_seen),
                );

                // If this is an execution shard, make the assertion that the value is consistent.
                builder.assert_felt_eq(
                    current_public_values.contains_execution_shard
                        * (execution_shard - current_public_values.start_execution_shard),
                    C::F::zero(),
                );
            }

            // Assert that the MemoryInitialize address bits are the same.
            for (bit, current_bit) in
                init_addr_bits.iter().zip(current_public_values.previous_init_addr_bits.iter())
            {
                builder.assert_felt_eq(*bit, *current_bit);
            }

            // Assert that the MemoryFinalize address bits are the same.
            for (bit, current_bit) in finalize_addr_bits
                .iter()
                .zip(current_public_values.previous_finalize_addr_bits.iter())
            {
                builder.assert_felt_eq(*bit, *current_bit);
            }

            // Digest constraints.
            {
                // If `committed_value_digest` is not zero, then
                // `public_values.committed_value_digest should be the current.

                // Set a flags to indicate whether `committed_value_digest` is non-zero. The flags
                // are given by the elements of the array, and they will be used as filters to
                // constrain the equality.
                let mut is_non_zero_flags = vec![];
                for word in committed_value_digest {
                    for byte in word {
                        is_non_zero_flags.push(byte);
                    }
                }

                // Using the flags, we can constrain the equality.
                for is_non_zero in is_non_zero_flags {
                    for (word_current, word_public) in committed_value_digest
                        .into_iter()
                        .zip(current_public_values.committed_value_digest)
                    {
                        for (byte_current, byte_public) in word_current.into_iter().zip(word_public)
                        {
                            builder.assert_felt_eq(
                                is_non_zero * (byte_current - byte_public),
                                C::F::zero(),
                            );
                        }
                    }
                }

                // Update the committed value digest.
                for (word, current_word) in committed_value_digest
                    .iter_mut()
                    .zip_eq(current_public_values.committed_value_digest.iter())
                {
                    for (byte, current_byte) in word.0.iter_mut().zip_eq(current_word.0.iter()) {
                        *byte = *current_byte;
                    }
                }

                //  If `deferred_proofs_digest` is not zero, then the current value should be
                // `public_values.deferred_proofs_digest`. We will use a similar approach as above.
                let mut is_non_zero_flags = vec![];
                for element in deferred_proofs_digest {
                    is_non_zero_flags.push(element);
                }

                for is_non_zero in is_non_zero_flags {
                    for (digest_current, digest_public) in deferred_proofs_digest
                        .into_iter()
                        .zip(current_public_values.deferred_proofs_digest)
                    {
                        builder.assert_felt_eq(
                            is_non_zero * (digest_current - digest_public),
                            C::F::zero(),
                        );
                    }
                }

                // Update the deferred proofs digest.
                for (digest, current_digest) in deferred_proofs_digest
                    .iter_mut()
                    .zip_eq(current_public_values.deferred_proofs_digest.iter())
                {
                    *digest = *current_digest;
                }
            }

            // Update the accumulated values.

            // If the current shard has an execution shard, then we update the flag in case it was
            // not already set. That is:
            // - If the current shard has an execution shard and the flag is set to zero, it will be
            //   set to one.
            // - If the current shard has an execution shard and the flag is set to one, it will
            //   remain set to one.
            contains_execution_shard = builder.eval(
                contains_execution_shard
                    + current_public_values.contains_execution_shard
                        * (SymbolicFelt::one() - contains_execution_shard),
            );

            // If this proof contains an execution shard, we update the execution shard value.
            execution_shard = builder.eval(
                current_public_values.next_execution_shard
                    * current_public_values.contains_execution_shard
                    + execution_shard
                        * (SymbolicFelt::one() - current_public_values.contains_execution_shard),
            );

            // Update the reconstruct deferred proof digest.
            for (digest, current_digest) in reconstruct_deferred_digest
                .iter_mut()
                .zip_eq(current_public_values.end_reconstruct_deferred_digest.iter())
            {
                *digest = *current_digest;
            }

            // Update pc to be the next pc.
            pc = current_public_values.next_pc;

            // Update the shard to be the next shard.
            shard = current_public_values.next_shard;

            // Update the MemoryInitialize address bits.
            for (bit, next_bit) in
                init_addr_bits.iter_mut().zip(current_public_values.last_init_addr_bits.iter())
            {
                *bit = *next_bit;
            }

            // Update the MemoryFinalize address bits.
            for (bit, next_bit) in finalize_addr_bits
                .iter_mut()
                .zip(current_public_values.last_finalize_addr_bits.iter())
            {
                *bit = *next_bit;
            }

            // Add the global cumulative sums to the vector.
            global_cumulative_sums.push(current_public_values.global_cumulative_sum);
        }

        // Sum all the global cumulative sum of the proofs.
        let global_cumulative_sum = builder.sum_digest_v2(global_cumulative_sums);

        // Update the global values from the last accumulated values.
        // Set sp1_vk digest to the one from the proof values.
        compress_public_values.sp1_vk_digest = sp1_vk_digest;
        // Set next_pc to be the last pc (which is the same as accumulated pc)
        compress_public_values.next_pc = pc;
        // Set next shard to be the last shard
        compress_public_values.next_shard = shard;
        // Set next execution shard to be the last execution shard
        compress_public_values.next_execution_shard = execution_shard;
        // Set the MemoryInitialize address bits to be the last MemoryInitialize address bits.
        compress_public_values.last_init_addr_bits = init_addr_bits;
        // Set the MemoryFinalize address bits to be the last MemoryFinalize address bits.
        compress_public_values.last_finalize_addr_bits = finalize_addr_bits;
        // Set the start reconstruct deferred digest to be the last reconstruct deferred digest.
        compress_public_values.end_reconstruct_deferred_digest = reconstruct_deferred_digest;
        // Assign the deferred proof digests.
        compress_public_values.deferred_proofs_digest = deferred_proofs_digest;
        // Assign the committed value digests.
        compress_public_values.committed_value_digest = committed_value_digest;
        // Assign the cumulative sum.
        compress_public_values.global_cumulative_sum = global_cumulative_sum;
        // Assign the `is_complete` flag.
        compress_public_values.is_complete = is_complete;
        // Set the contains an execution shard flag.
        compress_public_values.contains_execution_shard = contains_execution_shard;
        // Set the exit code.
        compress_public_values.exit_code = exit_code;
        // Reflect the vk root.
        // compress_public_values.vk_root = vk_root;
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

        // SC::commit_recursion_public_values(builder, *compress_public_values);
    }
}

// impl<SC: BabyBearFriConfig> SP1CompressWitnessValues<SC> {
//     pub fn shape(&self) -> SP1CompressShape {
//         let proof_shapes = self.vks_and_proofs.iter().map(|(_, proof)| proof.shape()).collect();
//         SP1CompressShape { proof_shapes }
//     }
// }

// impl SP1CompressWitnessValues<BabyBearPoseidon2> {
//     pub fn dummy<A: MachineAir<BabyBear>>(
//         machine: &StarkMachine<BabyBearPoseidon2, A>,
//         shape: &SP1CompressShape,
//     ) -> Self {
//         let vks_and_proofs = shape
//             .proof_shapes
//             .iter()
//             .map(|proof_shape| {
//                 let (vk, proof) = dummy_vk_and_shard_proof(machine, proof_shape);
//                 (vk, proof)
//             })
//             .collect();

//         Self { vks_and_proofs, is_complete: false }
//     }
// }

impl From<Vec<OrderedShape>> for SP1CompressShape {
    fn from(proof_shapes: Vec<OrderedShape>) -> Self {
        Self { proof_shapes }
    }
}
