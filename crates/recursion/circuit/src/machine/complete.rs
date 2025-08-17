use itertools::Itertools;
use slop_algebra::AbstractField;
use sp1_primitives::SP1Field;
use sp1_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Config, Felt},
};
use sp1_recursion_executor::RecursionPublicValues;

/// Assertions on recursion public values which represent a complete proof.
///
/// The assertions consist of checking all the expected boundary conditions from a compress proof
/// that represents the end of the recursion tower.
pub(crate) fn assert_complete<C: Config<F = SP1Field>>(
    builder: &mut Builder<C>,
    public_values: &RecursionPublicValues<Felt<C::F>>,
    is_complete: Felt<C::F>,
) {
    let RecursionPublicValues {
        prev_committed_value_digest,
        prev_deferred_proofs_digest,
        deferred_proofs_digest,
        prev_exit_code,
        next_pc,
        initial_timestamp,
        start_reconstruct_deferred_digest,
        end_reconstruct_deferred_digest,
        global_cumulative_sum,
        contains_first_shard,
        previous_init_addr,
        last_init_addr,
        previous_finalize_addr,
        last_finalize_addr,
        prev_commit_syscall,
        commit_syscall,
        prev_commit_deferred_syscall,
        commit_deferred_syscall,
        ..
    } = public_values;

    // Assert that the `is_complete` flag is boolean.
    builder.assert_felt_eq(is_complete * (is_complete - C::F::one()), C::F::zero());

    // Assert the `prev_committed_value_digest` is all zeroes.
    for word in prev_committed_value_digest {
        for limb in word {
            builder.assert_felt_eq(is_complete * *limb, C::F::zero());
        }
    }

    // Assert the `prev_deferred_proofs_digest` is all zeroes.
    for limb in prev_deferred_proofs_digest {
        builder.assert_felt_eq(is_complete * *limb, C::F::zero());
    }

    // Assert that `next_pc` is equal to the `HALT_PC` (so program execution has completed)
    builder.assert_felt_eq(
        is_complete * (next_pc[0] - C::F::from_canonical_u64(sp1_core_executor::HALT_PC)),
        C::F::zero(),
    );
    builder.assert_felt_eq(is_complete * next_pc[1], C::F::zero());
    builder.assert_felt_eq(is_complete * next_pc[2], C::F::zero());

    // Assert that the first shard has been included.
    builder.assert_felt_eq(is_complete * (*contains_first_shard - C::F::one()), C::F::zero());

    // Assert that the initial timestamp is equal to 1.
    for limb in initial_timestamp[0..3].iter() {
        builder.assert_felt_eq(is_complete * *limb, C::F::zero());
    }
    builder.assert_felt_eq(is_complete * (initial_timestamp[3] - C::F::one()), C::F::zero());

    // Assert that the `previous_init_addr` is 0.
    for limb in previous_init_addr.iter() {
        builder.assert_felt_eq(is_complete * *limb, C::F::zero());
    }

    // Assert that the `last_init_addr` is not 0.
    // SAFETY: `last_init_addr` are with valid u16 limbs, as it's checked in each core shard.
    // If `is_complete = 0`, then the right hand side is `p - 1`, which cannot equal sum of three
    // u16 limbs due to the size of `p`. If `is_complete = 1`, then the right hand side is `0`, so
    // this constrains that `last_init_addr` cannot be identical to `0`.
    builder.assert_felt_ne(
        last_init_addr[0] + last_init_addr[1] + last_init_addr[2],
        is_complete - C::F::one(),
    );

    // Assert that the `previous_finalize_addr` is 0.
    for limb in previous_finalize_addr.iter() {
        builder.assert_felt_eq(is_complete * *limb, C::F::zero());
    }

    // Assert that the `last_finalize_addr` is not 0. Same method as `last_init_addr`.
    builder.assert_felt_ne(
        last_finalize_addr[0] + last_finalize_addr[1] + last_finalize_addr[2],
        is_complete - C::F::one(),
    );

    // The start reconstruct deferred digest should be zero.
    for start_digest in start_reconstruct_deferred_digest {
        builder.assert_felt_eq(is_complete * *start_digest, C::F::zero());
    }

    // The end reconstruct deferred digest should be equal to the deferred proofs digest.
    for (end_digest, deferred_digest) in
        end_reconstruct_deferred_digest.iter().zip_eq(deferred_proofs_digest.iter())
    {
        builder.assert_felt_eq(is_complete * (*end_digest - *deferred_digest), C::F::zero());
    }

    // Assert that the starting `prev_exit_code` is equal to 0.
    builder.assert_felt_eq(is_complete * *prev_exit_code, C::F::zero());

    // The starting `prev_commit_syscall` must be zero.
    builder.assert_felt_eq(is_complete * *prev_commit_syscall, C::F::zero());

    // The starting `prev_commit_deferred_syscall` must be zero.
    builder.assert_felt_eq(is_complete * *prev_commit_deferred_syscall, C::F::zero());

    // The final `commit_syscall` must be one.
    builder.assert_felt_eq(is_complete * (*commit_syscall - C::F::one()), C::F::zero());

    // The final `commit_deferred_syscall` must be one.
    builder.assert_felt_eq(is_complete * (*commit_deferred_syscall - C::F::one()), C::F::zero());

    // The global cumulative sum should sum be equal to the zero digest.
    builder.assert_digest_zero_v2(is_complete, *global_cumulative_sum);
}
