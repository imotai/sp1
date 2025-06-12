use std::marker::PhantomData;

use super::PublicValuesOutputDigest;
use crate::{
    basefold::{RecursiveBasefoldConfigImpl, RecursiveBasefoldProof, RecursiveBasefoldVerifier},
    challenger::DuplexChallengerVariable,
    jagged::RecursiveJaggedConfig,
    machine::{SP1CompressWithVKeyVerifier, SP1CompressWithVKeyWitnessVariable},
    shard::RecursiveShardVerifier,
    zerocheck::RecursiveVerifierConstraintFolder,
    BabyBearFriConfigVariable, CircuitConfig,
};
use slop_air::Air;
use slop_algebra::AbstractField;
use slop_baby_bear::BabyBear;
use sp1_recursion_compiler::ir::{Builder, Felt};
use sp1_recursion_executor::DIGEST_SIZE;
use sp1_stark::air::MachineAir;

/// A program to verify a single recursive proof representing a complete proof of program execution.
///
/// The root verifier is simply a `SP1CompressVerifier` with an assertion that the `is_complete`
/// flag is set to true.
#[derive(Debug, Clone, Copy)]
pub struct SP1CompressRootVerifierWithVKey<C, SC, A, JC> {
    _phantom: PhantomData<(C, SC, A, JC)>,
}

impl<C, SC, A, JC> SP1CompressRootVerifierWithVKey<C, SC, A, JC>
where
    SC: BabyBearFriConfigVariable<
            C,
            FriChallengerVariable = DuplexChallengerVariable<C>,
            DigestVariable = [Felt<BabyBear>; DIGEST_SIZE],
        > + Send
        + Sync,
    C: CircuitConfig<F = SC::F, EF = SC::EF, Bit = Felt<BabyBear>>,
    // <SC::ValMmcs as Mmcs<BabyBear>>::ProverData<RowMajorMatrix<BabyBear>>: Clone,
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
        input: SP1CompressWithVKeyWitnessVariable<C, SC, JC>,
        value_assertions: bool,
        kind: PublicValuesOutputDigest,
    ) {
        // Assert that the program is complete.
        builder.assert_felt_eq(input.compress_var.is_complete, C::F::one());
        // Verify the proof, as a compress proof.
        SP1CompressWithVKeyVerifier::verify(builder, machine, input, value_assertions, kind);
    }
}
