use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
};

use itertools::Itertools;
use slop_air::Air;
use slop_algebra::AbstractField;
use slop_jagged::JaggedConfig;
use sp1_primitives::SP1Field;

use serde::{Deserialize, Serialize};
use sp1_core_machine::riscv::{RiscvAir, MAX_LOG_NUMBER_OF_SHARDS};

use sp1_hypercube::air::PublicValues;

use sp1_hypercube::{MachineConfig, MachineVerifyingKey, ShardProof};

use sp1_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Config, Felt, SymbolicFelt},
};

use sp1_recursion_executor::{RecursionPublicValues, DIGEST_SIZE, RECURSIVE_PROOF_NUM_PV_ELTS};

use crate::{
    basefold::{RecursiveBasefoldConfigImpl, RecursiveBasefoldProof, RecursiveBasefoldVerifier},
    challenger::{CanObserveVariable, DuplexChallengerVariable},
    jagged::RecursiveJaggedConfig,
    machine::{assert_complete, recursion_public_values_digest},
    shard::{MachineVerifyingKeyVariable, RecursiveShardVerifier, ShardProofVariable},
    zerocheck::RecursiveVerifierConstraintFolder,
    CircuitConfig, InnerSC, SP1FieldConfigVariable, SP1FieldFriConfig,
};

pub struct SP1RecursionWitnessVariable<
    C: CircuitConfig<F = SP1Field, EF = crate::EF>,
    SC: SP1FieldConfigVariable<C>,
    JC: RecursiveJaggedConfig<
        BatchPcsVerifier = RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
    >,
> {
    pub vk: MachineVerifyingKeyVariable<C, SC>,
    pub shard_proofs: Vec<ShardProofVariable<C, SC, JC>>,
    pub reconstruct_deferred_digest: [Felt<C::F>; DIGEST_SIZE],
    pub is_complete: Felt<C::F>,
    pub is_first_shard: Felt<C::F>,
    pub vk_root: [Felt<C::F>; DIGEST_SIZE],
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "ShardProof<SC>: Serialize"))]
#[serde(bound(deserialize = "ShardProof<SC>: Deserialize<'de>"))]
/// A struct to contain the inputs to the `normalize` program.
pub struct SP1NormalizeWitnessValues<SC: MachineConfig> {
    pub vk: MachineVerifyingKey<SC>,
    pub shard_proofs: Vec<ShardProof<SC>>,
    pub is_complete: bool,
    pub is_first_shard: bool,
    pub vk_root: [SC::F; DIGEST_SIZE],
    pub reconstruct_deferred_digest: [SC::F; 8],
}

/// A program for recursively verifying a batch of SP1 proofs.
#[derive(Debug, Clone, Copy)]
pub struct SP1RecursiveVerifier<C: Config, SC: SP1FieldFriConfig, JC: RecursiveJaggedConfig> {
    _phantom: PhantomData<(C, SC, JC)>,
}

type InnerVal = <InnerSC as JaggedConfig>::F;
type InnerChallenge = <InnerSC as JaggedConfig>::EF;

impl<C, SC, JC> SP1RecursiveVerifier<C, SC, JC>
where
    SC: SP1FieldConfigVariable<
            C,
            FriChallengerVariable = DuplexChallengerVariable<C>,
            DigestVariable = [Felt<SP1Field>; DIGEST_SIZE],
        > + Send
        + Sync,
    C: CircuitConfig<F = InnerVal, EF = InnerChallenge, Bit = Felt<SP1Field>>,
    JC: RecursiveJaggedConfig<
        BatchPcsVerifier = RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
    >,
    SC: SP1FieldConfigVariable<C> + MachineConfig,
    JC: RecursiveJaggedConfig<
        F = C::F,
        EF = C::EF,
        Circuit = C,
        Commitment = SC::DigestVariable,
        Challenger = SC::FriChallengerVariable,
        BatchPcsProof = RecursiveBasefoldProof<RecursiveBasefoldConfigImpl<C, SC>>,
        BatchPcsVerifier = RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
    >,
{
    /// Verify a batch of SP1 shard proofs and aggregate their public values.
    ///
    /// This program represents a first recursive step in the verification of an SP1 proof
    /// consisting of one or more shards. Each shard proof is verified and its public values are
    /// turned into the recursion public values, which will be aggregated in compress.
    ///
    /// # Constraints
    ///
    /// ## Verifying the core shard proofs.
    /// For each shard, the verifier asserts the correctness of the shard proof which is composed
    /// of verifying the polynomial commitment's proof for openings and verifying the constraints.
    ///
    /// ## Verifing the first shard constraints.
    /// The first shard has some additional constraints for initialization.
    pub fn verify(
        builder: &mut Builder<C>,
        machine: &RecursiveShardVerifier<RiscvAir<<SC as JaggedConfig>::F>, SC, C, JC>,
        input: SP1RecursionWitnessVariable<C, SC, JC>,
    ) where
        RiscvAir<<SC as JaggedConfig>::F>: for<'b> Air<RecursiveVerifierConstraintFolder<'b, C>>,
    {
        // Read input.
        let SP1RecursionWitnessVariable {
            vk,
            shard_proofs,
            is_complete,
            is_first_shard,
            vk_root,
            reconstruct_deferred_digest,
        } = input;

        // Assert that the number of proofs is one.
        assert!(shard_proofs.len() == 1);
        let shard_proof = &shard_proofs[0];

        // Initialize the cumulative sum.
        let mut global_cumulative_sums = Vec::new();

        // Get the public values.
        let public_values: &PublicValues<[Felt<_>; 4], [Felt<_>; 3], [Felt<_>; 4], Felt<_>> =
            shard_proof.public_values.as_slice().borrow();

        // First shard constraints. We verify the validity of the `is_first_shard` boolean
        // flag, and make assertions for that are specific to the first shard using it.
        // We assert that `is_first_shard == (public_values.shard == 1)` in three steps.

        // First, we assert that the `is_first_shard` flag is boolean.
        builder.assert_felt_eq(is_first_shard * (is_first_shard - C::F::one()), C::F::zero());

        // Assert that if `is_first_shard == 1`, then `public_values.shard == 1`.
        builder.assert_felt_eq(is_first_shard * (public_values.shard - C::F::one()), C::F::zero());

        // Assert that if `is_first_shard == 0`, then `public_values.shard != 1`.
        // This asserts that if `public_values.shard == 1`, then `is_first_shard == 1`.
        builder.assert_felt_ne(
            (SymbolicFelt::one() - is_first_shard) * public_values.shard,
            C::F::one(),
        );

        // If it's the first shard, `prev_committed_value_digest` must be zero.
        for word in public_values.prev_committed_value_digest.iter() {
            for limb in word.iter() {
                builder.assert_felt_eq(is_first_shard * *limb, C::F::zero());
            }
        }

        // If it's the first shard, `prev_deferred_proofs_digest` must be zero.
        for limb in public_values.prev_deferred_proofs_digest.iter() {
            builder.assert_felt_eq(is_first_shard * *limb, C::F::zero());
        }

        // If it's the first shard (which is the first execution shard), then the `pc_start`
        // should be vk.pc_start.
        for (pc, vk_pc) in public_values.pc_start.iter().zip_eq(vk.pc_start.iter()) {
            builder.assert_felt_eq(is_first_shard * (*pc - *vk_pc), C::F::zero());
        }

        // If it's the first shard, the `prev_exit_code` must be zero.
        builder.assert_felt_eq(is_first_shard * public_values.prev_exit_code, C::F::zero());

        // If it's the first shard, the execution shard should be 1.
        builder.assert_felt_eq(
            is_first_shard * (public_values.execution_shard - C::F::one()),
            C::F::zero(),
        );
        // If it's the first shard, it must be a CPU shard, so the next execution shard must be 2.
        builder.assert_felt_eq(
            is_first_shard * (public_values.next_execution_shard - C::F::two()),
            C::F::zero(),
        );

        // Assert that `previous_init_addr`, `previous_finalize_addr` are zero for the first shard.
        for limb in public_values.previous_init_addr.iter() {
            builder.assert_felt_eq(is_first_shard * *limb, C::F::zero());
        }
        for limb in public_values.previous_finalize_addr.iter() {
            builder.assert_felt_eq(is_first_shard * *limb, C::F::zero());
        }

        // If it's the first shard, the initial timestamp must be `1`.
        for limb in public_values.initial_timestamp.iter().take(3) {
            builder.assert_felt_eq(is_first_shard * *limb, C::F::zero());
        }
        builder.assert_felt_eq(
            is_first_shard * (public_values.initial_timestamp[3] - C::F::one()),
            C::F::zero(),
        );

        // If it's the first shard, we add the vk's `initial_global_cumulative_sum` to the
        // digest. If it's not the first shard, we add the zero digest to the digest.
        global_cumulative_sums.push(
            builder.select_global_cumulative_sum(is_first_shard, vk.initial_global_cumulative_sum),
        );

        // If it's the first shard, `prev_commit_syscall` must be zero.
        builder.assert_felt_eq(is_first_shard * public_values.prev_commit_syscall, C::F::zero());

        // If it's the first shard, `prev_commit_deferred_syscall` must be zero.
        builder.assert_felt_eq(
            is_first_shard * public_values.prev_commit_deferred_syscall,
            C::F::zero(),
        );

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
        tracing::debug_span!("verify shard")
            .in_scope(|| machine.verify_shard(builder, &vk, shard_proof, &mut challenger));

        // Verify that the shard index has at most `MAX_LOG_NUMBER_OF_SHARDS` bits.
        C::range_check_felt(builder, public_values.shard, MAX_LOG_NUMBER_OF_SHARDS);

        // Verify the shard index is non-zero.
        builder.assert_felt_ne(public_values.shard, C::F::zero());

        // We add the global cumulative sum of the shard.
        global_cumulative_sums.push(public_values.global_cumulative_sum);

        // We sum the digests in `global_cumulative_sums` to get the overall global cumulative sum.
        let global_cumulative_sum = builder.sum_digest_v2(global_cumulative_sums);

        // Write all values to the public values struct and commit to them.
        {
            // Compute the vk digest.
            let vk_digest = vk.hash(builder);

            // Initialize the public values we will commit to.
            let zero: Felt<_> = builder.eval(C::F::zero());
            let mut recursion_public_values_stream = [zero; RECURSIVE_PROOF_NUM_PV_ELTS];
            let recursion_public_values: &mut RecursionPublicValues<_> =
                recursion_public_values_stream.as_mut_slice().borrow_mut();
            recursion_public_values.prev_committed_value_digest =
                public_values.prev_committed_value_digest;
            recursion_public_values.committed_value_digest = public_values.committed_value_digest;
            recursion_public_values.prev_deferred_proofs_digest =
                public_values.prev_deferred_proofs_digest;
            recursion_public_values.deferred_proofs_digest = public_values.deferred_proofs_digest;
            recursion_public_values.pc_start = public_values.pc_start;
            recursion_public_values.next_pc = public_values.next_pc;
            recursion_public_values.start_shard = public_values.shard;
            recursion_public_values.next_shard = builder.eval(public_values.shard + C::F::one());
            recursion_public_values.start_execution_shard = public_values.execution_shard;
            recursion_public_values.next_execution_shard = public_values.next_execution_shard;
            recursion_public_values.initial_timestamp = public_values.initial_timestamp;
            recursion_public_values.last_timestamp = public_values.last_timestamp;
            recursion_public_values.previous_init_addr = public_values.previous_init_addr;
            recursion_public_values.last_init_addr = public_values.last_init_addr;
            recursion_public_values.previous_finalize_addr = public_values.previous_finalize_addr;
            recursion_public_values.last_finalize_addr = public_values.last_finalize_addr;
            recursion_public_values.start_reconstruct_deferred_digest = reconstruct_deferred_digest;
            recursion_public_values.end_reconstruct_deferred_digest = reconstruct_deferred_digest;
            recursion_public_values.sp1_vk_digest = vk_digest;
            recursion_public_values.vk_root = vk_root;
            recursion_public_values.global_cumulative_sum = global_cumulative_sum;
            recursion_public_values.is_complete = is_complete;
            recursion_public_values.prev_exit_code = public_values.prev_exit_code;
            recursion_public_values.exit_code = public_values.exit_code;
            recursion_public_values.prev_commit_syscall = public_values.prev_commit_syscall;
            recursion_public_values.commit_syscall = public_values.commit_syscall;
            recursion_public_values.prev_commit_deferred_syscall =
                public_values.prev_commit_deferred_syscall;
            recursion_public_values.commit_deferred_syscall = public_values.commit_deferred_syscall;

            // Calculate the digest and set it in the public values.
            recursion_public_values.digest =
                recursion_public_values_digest::<C, SC>(builder, recursion_public_values);

            assert_complete(builder, recursion_public_values, is_complete);

            SC::commit_recursion_public_values(builder, *recursion_public_values);
        }
    }
}
