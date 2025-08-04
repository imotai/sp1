use std::marker::PhantomData;

use super::SP1ShapedWitnessVariable;
use crate::{
    basefold::{RecursiveBasefoldConfigImpl, RecursiveBasefoldProof, RecursiveBasefoldVerifier},
    challenger::CanObserveVariable,
    jagged::RecursiveJaggedConfig,
    machine::{assert_complete, assert_root_public_values_valid, RootPublicValues},
    shard::RecursiveShardVerifier,
    zerocheck::RecursiveVerifierConstraintFolder,
    BabyBearFriConfigVariable, CircuitConfig,
};
use slop_air::Air;
use slop_algebra::AbstractField;
use sp1_recursion_compiler::ir::{Builder, Felt};
use sp1_stark::air::MachineAir;
use std::borrow::Borrow;

/// A program to verify a single recursive proof representing a complete proof of program execution.
///
/// The root verifier is simply a `SP1CompressVerifier` with an assertion that the `is_complete`
/// flag is set to true.
#[derive(Debug, Clone, Copy)]
pub struct SP1WrapVerifier<C, SC, A, JC> {
    _phantom: PhantomData<(C, SC, A, JC)>,
}

impl<C, SC, A, JC> SP1WrapVerifier<C, SC, A, JC>
where
    SC: BabyBearFriConfigVariable<C> + Send + Sync,
    C: CircuitConfig<F = SC::F, EF = SC::EF>,
    A: MachineAir<SC::F> + for<'a> Air<RecursiveVerifierConstraintFolder<'a, C>>,
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
    pub fn verify(
        builder: &mut Builder<C>,
        machine: &RecursiveShardVerifier<A, SC, C, JC>,
        input: SP1ShapedWitnessVariable<C, SC, JC>,
    ) {
        // Assert the the proof is not malformed.
        assert!(input.vks_and_proofs.len() == 1);
        // Take the proof from the input.
        let (vk, proof) = &input.vks_and_proofs[0];

        // Assert that the program is complete.
        builder.assert_felt_eq(input.is_complete, C::F::one());
        let public_values: &RootPublicValues<Felt<C::F>> = proof.public_values.as_slice().borrow();
        assert_root_public_values_valid::<C, SC>(builder, public_values);

        let mut challenger = <SC as BabyBearFriConfigVariable<C>>::challenger_variable(builder);
        challenger.observe(builder, vk.preprocessed_commit);
        challenger.observe_slice(builder, vk.pc_start);
        challenger.observe_slice(builder, vk.initial_global_cumulative_sum.0.x.0);
        challenger.observe_slice(builder, vk.initial_global_cumulative_sum.0.y.0);

        // Observe the padding.
        let zero: Felt<_> = builder.eval(C::F::zero());
        for _ in 0..7 {
            challenger.observe(builder, zero);
        }
        machine.verify_shard(builder, vk, proof, &mut challenger);

        assert_complete(builder, &public_values.inner, input.is_complete);

        SC::commit_recursion_public_values(builder, public_values.inner);
    }
}
