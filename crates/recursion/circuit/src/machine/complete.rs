use itertools::Itertools;
use slop_algebra::AbstractField;
use slop_baby_bear::BabyBear;
use sp1_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Config, Felt},
};
use sp1_recursion_executor::RecursionPublicValues;

/// Assertions on recursion public values which represent a complete proof.
///
/// The assertions consist of checking all the expected boundary conditions from a compress proof
/// that represents the end of the recursion tower.
pub(crate) fn assert_complete<C: Config<F = BabyBear>>(
    builder: &mut Builder<C>,
    public_values: &RecursionPublicValues<Felt<C::F>>,
    is_complete: Felt<C::F>,
) {
    let RecursionPublicValues {
        deferred_proofs_digest,
        prev_exit_code,
        next_pc_rel,
        start_shard,
        next_shard,
        initial_timestamp,
        start_execution_shard,
        start_reconstruct_deferred_digest,
        end_reconstruct_deferred_digest,
        global_cumulative_sum,
        ..
    } = public_values;

    // Assert that the `is_complete` flag is boolean.
    builder.assert_felt_eq(is_complete * (is_complete - C::F::one()), C::F::zero());

    // Assert that `next_pc_rel` is equal to one (so program execution has completed)
    builder.assert_felt_eq(is_complete * (*next_pc_rel - C::F::one()), C::F::zero());

    // Assert that start shard is equal to 1.
    builder.assert_felt_eq(is_complete * (*start_shard - C::F::one()), C::F::zero());

    // Assert that the next shard is not equal to one. This guarantees that there is at least one
    // shard that contains CPU.
    builder.assert_felt_ne(is_complete * *next_shard, C::F::one());

    // Assert that the start execution shard is equal to 1.
    builder.assert_felt_eq(is_complete * (*start_execution_shard - C::F::one()), C::F::zero());

    // Assert that the initial timestamp is equal to 1.
    for limb in initial_timestamp.iter().take(3) {
        builder.assert_felt_eq(is_complete * *limb, C::F::zero());
    }

    builder.assert_felt_eq(is_complete * (initial_timestamp[3] - C::F::one()), C::F::zero());

    // Assert that the starting prev_exit_code is equal to 0.
    builder.assert_felt_eq(is_complete * *prev_exit_code, C::F::zero());

    // The start reconstruct deferred digest should be zero.
    for start_digest_word in start_reconstruct_deferred_digest {
        builder.assert_felt_eq(is_complete * *start_digest_word, C::F::zero());
    }

    // The end reconstruct deferred digest should be equal to the deferred proofs digest.
    for (end_digest_word, deferred_digest_word) in
        end_reconstruct_deferred_digest.iter().zip_eq(deferred_proofs_digest.iter())
    {
        builder
            .assert_felt_eq(is_complete * (*end_digest_word - *deferred_digest_word), C::F::zero());
    }

    // The global cumulative sum should sum be equal to the zero digest.
    builder.assert_digest_zero_v2(is_complete, *global_cumulative_sum);
}
