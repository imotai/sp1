use std::{collections::BTreeSet, ops::Deref};

use crate::{
    basefold::{RecursiveBasefoldConfigImpl, RecursiveBasefoldVerifier},
    challenger::FieldChallengerVariable,
    jagged::RecursiveJaggedConfig,
    shard::RecursiveShardVerifier,
    sumcheck::verify_sumcheck,
    symbolic::IntoSymbolic,
    BabyBearFriConfigVariable, CircuitConfig,
};
use itertools::Itertools;
use slop_air::{Air, BaseAir};
use slop_algebra::{extension::BinomialExtensionField, AbstractField};
use slop_baby_bear::BabyBear;
use slop_matrix::dense::RowMajorMatrixView;
use slop_multilinear::{full_geq, Mle, Point};
use slop_sumcheck::PartialSumcheckProof;
use sp1_recursion_compiler::{
    ir::{Config, Felt},
    prelude::{Builder, Ext, SymbolicExt},
};
use sp1_stark::{
    air::MachineAir, Chip, ChipOpenedValues, GenericVerifierConstraintFolder, LogUpEvaluations,
    OpeningShapeError, ShardOpenedValues,
};

pub type RecursiveVerifierConstraintFolder<'a, C> = GenericVerifierConstraintFolder<
    'a,
    <C as Config>::F,
    <C as Config>::EF,
    Felt<<C as Config>::F>,
    Ext<<C as Config>::F, <C as Config>::EF>,
    SymbolicExt<<C as Config>::F, <C as Config>::EF>,
>;

#[allow(clippy::type_complexity)]
pub fn eval_constraints<C: CircuitConfig<F = BabyBear>, SC: BabyBearFriConfigVariable<C>, A>(
    builder: &mut Builder<C>,
    chip: &Chip<C::F, A>,
    opening: &ChipOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>,
    alpha: Ext<C::F, C::EF>,
    public_values: &[Felt<C::F>],
) -> Ext<C::F, C::EF>
where
    A: MachineAir<C::F> + for<'a> Air<RecursiveVerifierConstraintFolder<'a, C>>,
{
    let mut folder = RecursiveVerifierConstraintFolder::<C> {
        preprocessed: RowMajorMatrixView::new_row(&opening.preprocessed.local),
        main: RowMajorMatrixView::new_row(&opening.main.local),
        public_values,
        alpha,
        accumulator: SymbolicExt::zero(),
        _marker: std::marker::PhantomData,
    };

    chip.eval(&mut folder);
    builder.eval(folder.accumulator)
}

/// Compute the padded row adjustment for a chip.
pub fn compute_padded_row_adjustment<C: CircuitConfig, A>(
    builder: &mut Builder<C>,
    chip: &Chip<C::F, A>,
    alpha: Ext<C::F, C::EF>,
    public_values: &[Felt<C::F>],
) -> Ext<C::F, C::EF>
where
    A: MachineAir<C::F> + for<'a> Air<RecursiveVerifierConstraintFolder<'a, C>>,
{
    let zero = builder.constant(C::EF::zero());
    let dummy_preprocessed_trace = vec![zero; chip.preprocessed_width()];
    let dummy_main_trace = vec![zero; chip.width()];

    let mut folder = RecursiveVerifierConstraintFolder::<C> {
        preprocessed: RowMajorMatrixView::new_row(&dummy_preprocessed_trace),
        main: RowMajorMatrixView::new_row(&dummy_main_trace),
        alpha,
        accumulator: SymbolicExt::zero(),
        public_values,
        _marker: std::marker::PhantomData,
    };

    chip.eval(&mut folder);
    builder.eval(folder.accumulator)
}

#[allow(clippy::type_complexity)]
pub fn verify_opening_shape<C: CircuitConfig, A>(
    chip: &Chip<C::F, A>,
    opening: &ChipOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>,
) -> Result<(), OpeningShapeError>
where
    A: MachineAir<C::F> + for<'a> Air<RecursiveVerifierConstraintFolder<'a, C>>,
{
    // Verify that the preprocessed width matches the expected value for the chip.
    if opening.preprocessed.local.len() != chip.preprocessed_width() {
        return Err(OpeningShapeError::PreprocessedWidthMismatch(
            chip.preprocessed_width(),
            opening.preprocessed.local.len(),
        ));
    }

    // Verify that the main width matches the expected value for the chip.
    if opening.main.local.len() != chip.width() {
        return Err(OpeningShapeError::MainWidthMismatch(chip.width(), opening.main.local.len()));
    }

    Ok(())
}

impl<C, SC, A, JC> RecursiveShardVerifier<A, SC, C, JC>
where
    C: CircuitConfig<F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>>,
    SC: BabyBearFriConfigVariable<C>,
    A: MachineAir<C::F>,
    JC: RecursiveJaggedConfig<
        BatchPcsVerifier = RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
    >,
{
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub fn verify_zerocheck(
        &self,
        builder: &mut Builder<C>,
        shard_chips: &BTreeSet<Chip<C::F, A>>,
        opened_values: &ShardOpenedValues<Felt<C::F>, Ext<C::F, C::EF>>,
        gkr_evaluations: &LogUpEvaluations<Ext<C::F, C::EF>>,
        zerocheck_proof: &PartialSumcheckProof<Ext<C::F, C::EF>>,
        public_values: &[Felt<C::F>],
        challenger: &mut SC::FriChallengerVariable,
    ) where
        A: for<'a> Air<RecursiveVerifierConstraintFolder<'a, C>>,
    {
        let zero: Ext<C::F, C::EF> = builder.constant(C::EF::zero());
        let one: Ext<C::F, C::EF> = builder.constant(C::EF::one());
        let mut rlc_eval: Ext<C::F, C::EF> = zero;

        let alpha = challenger.sample_ext(builder);
        let gkr_batch_open_challenge: SymbolicExt<C::F, C::EF> =
            challenger.sample_ext(builder).into();
        let lambda = challenger.sample_ext(builder);

        // Get the value of eq(zeta, sumcheck's reduced point).
        let point_symbolic = <Point<Ext<C::F, C::EF>> as IntoSymbolic<C>>::as_symbolic(
            &zerocheck_proof.point_and_eval.0,
        );

        let gkr_evaluations_point = IntoSymbolic::<C>::as_symbolic(&gkr_evaluations.point);

        let zerocheck_eq_value = Mle::full_lagrange_eval(&gkr_evaluations_point, &point_symbolic);

        let zerocheck_eq_vals = vec![zerocheck_eq_value; shard_chips.len()];

        let max_elements = shard_chips
            .iter()
            .map(|chip| chip.width() + chip.preprocessed_width())
            .max()
            .unwrap_or(0);

        let gkr_batch_open_challenge_powers =
            gkr_batch_open_challenge.powers().skip(1).take(max_elements).collect::<Vec<_>>();

        for ((chip, openings), zerocheck_eq_val) in
            shard_chips.iter().zip_eq(opened_values.chips.values()).zip_eq(zerocheck_eq_vals)
        {
            // Verify the shape of the opening arguments matches the expected values.
            verify_opening_shape::<C, A>(chip, openings).unwrap();

            let dimension = zerocheck_proof.point_and_eval.0.dimension();

            assert_eq!(dimension, self.pcs_verifier.max_log_row_count);

            let mut proof_point_extended = point_symbolic.clone();
            proof_point_extended.add_dimension(zero.into());
            let degree_symbolic_ext: Point<SymbolicExt<C::F, C::EF>> =
                openings.degree.iter().map(|x| SymbolicExt::from(*x)).collect::<Point<_>>();
            let point_len = degree_symbolic_ext.dimension();
            degree_symbolic_ext.iter().enumerate().for_each(|(i, x)| {
                builder.assert_ext_eq(*x * (*x - one), zero);
                if i < point_len - 1 {
                    builder.assert_ext_eq(*x * *degree_symbolic_ext.last().unwrap(), zero);
                }
            });
            let geq_val = full_geq(&degree_symbolic_ext, &proof_point_extended);

            let padded_row_adjustment =
                compute_padded_row_adjustment(builder, chip, alpha, public_values);

            let constraint_eval =
                eval_constraints::<C, SC, A>(builder, chip, openings, alpha, public_values)
                    - padded_row_adjustment * geq_val;

            let openings_batch = openings
                .main
                .local
                .iter()
                .chain(openings.preprocessed.local.iter())
                .copied()
                .zip(
                    gkr_batch_open_challenge_powers
                        .iter()
                        .take(openings.main.local.len() + openings.preprocessed.local.len())
                        .copied(),
                )
                .map(|(opening, power)| opening * power)
                .sum::<SymbolicExt<C::F, C::EF>>();

            rlc_eval = builder
                .eval(rlc_eval * lambda + zerocheck_eq_val * (constraint_eval + openings_batch));
        }

        builder.assert_ext_eq(rlc_eval, zerocheck_proof.point_and_eval.1);

        let zerocheck_sum_modifications_from_gkr = gkr_evaluations
            .chip_openings
            .values()
            .map(|chip_evaluation| {
                chip_evaluation
                    .main_trace_evaluations
                    .deref()
                    .iter()
                    .copied()
                    .chain(
                        chip_evaluation
                            .preprocessed_trace_evaluations
                            .as_ref()
                            .iter()
                            .flat_map(|&evals| evals.deref().iter().copied()),
                    )
                    .zip(gkr_batch_open_challenge_powers.iter().copied())
                    .map(|(opening, power)| opening * power)
                    .sum::<SymbolicExt<C::F, C::EF>>()
            })
            .collect::<Vec<_>>();

        let zerocheck_sum_modification: SymbolicExt<C::F, C::EF> =
            zerocheck_sum_modifications_from_gkr
                .iter()
                .fold(zero.into(), |acc, modification| lambda * acc + *modification);

        // Verify that the rlc claim is zero.
        builder.assert_ext_eq(zerocheck_proof.claimed_sum, zerocheck_sum_modification);

        // Verify the zerocheck proof.
        verify_sumcheck::<C, SC>(builder, challenger, zerocheck_proof);

        // Observe the openings
        for opening in opened_values.chips.values() {
            for eval in opening.preprocessed.local.iter() {
                challenger.observe_ext_element(builder, *eval);
            }
            for eval in opening.main.local.iter() {
                challenger.observe_ext_element(builder, *eval);
            }
        }
    }
}

// TODO: Add tests back.
// #[cfg(test)]
// mod tests {
//     use std::{marker::PhantomData, sync::Arc};

//     use slop_algebra::extension::BinomialExtensionField;
//     use slop_baby_bear::DiffusionMatrixBabyBear;
//     use slop_basefold::{BasefoldVerifier, Poseidon2BabyBear16BasefoldConfig};
//     use slop_jagged::BabyBearPoseidon2;
//     use slop_merkle_tree::my_bb_16_perm;
//     use sp1_core_executor::{Program, SP1Context};
//     use sp1_core_machine::{io::SP1Stdin, riscv::RiscvAir, utils::prove_core};
//     use sp1_recursion_compiler::{
//         circuit::{AsmCompiler, AsmConfig},
//         config::InnerConfig,
//     };
//     use sp1_recursion_executor::Runtime;
//     use sp1_stark::{prover::CpuProver, SP1CoreOpts, ShardVerifier};

//     use crate::{
//         basefold::{stacked::RecursiveStackedPcsVerifier, tcs::RecursiveMerkleTreeTcs},
//         challenger::DuplexChallengerVariable,
//         jagged::{
//             RecursiveJaggedConfigImpl, RecursiveJaggedEvalSumcheckConfig,
//             RecursiveJaggedPcsVerifier,
//         },
//         witness::Witnessable,
//     };

//     use super::*;

//     type F = BabyBear;
//     type SC = BabyBearPoseidon2;
//     type JC = RecursiveJaggedConfigImpl<
//         C,
//         SC,
//         RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
//     >;
//     type C = InnerConfig;
//     type EF = BinomialExtensionField<BabyBear, 4>;
//     type A = RiscvAir<BabyBear>;

//     #[tokio::test]
//     async fn test_zerocheck() {
//         let program = Program::from(test_artifacts::FIBONACCI_ELF).unwrap();
//         let log_blowup = 1;
//         let log_stacking_height = 21;
//         let max_log_row_count = 21;
//         let machine = RiscvAir::machine();
//         let verifier = ShardVerifier::from_basefold_parameters(
//             log_blowup,
//             log_stacking_height,
//             max_log_row_count,
//             machine.clone(),
//         );
//         let prover = CpuProver::new(verifier.clone());

//         let (pk, _) = prover.setup(Arc::new(program.clone())).await;

//         let challenger = verifier.pcs_verifier.challenger();

//         let (proof, _) = prove_core(
//             Arc::new(prover),
//             Arc::new(pk),
//             Arc::new(program.clone()),
//             &SP1Stdin::new(),
//             SP1CoreOpts::default(),
//             SP1Context::default(),
//             challenger,
//         )
//         .await
//         .unwrap();

//         let shard_proof = proof.shard_proofs[0].clone();
//         let challenger_state = shard_proof.testing_data.challenger_state.clone();

//         let mut builder = Builder::<C>::default();

//         let mut challenger_variable =
//             DuplexChallengerVariable::from_challenger(&mut builder, &challenger_state);

//         let shard_proof_variable = shard_proof.read(&mut builder);

//         let gkr_points_variable = shard_proof.testing_data.gkr_points.read(&mut builder);
//         let gkr_column_openings_variable = shard_proof
//             .gkr_proofs
//             .iter()
//             .map(|gkr_proof| {
//                 let (main_openings, preprocessed_openings) = &gkr_proof.column_openings;
//                 let main_openings_variable = main_openings.read(&mut builder);
//                 let preprocessed_openings_variable: MleEval<Ext<_, _>> = preprocessed_openings
//                     .as_ref()
//                     .map(MleEval::to_vec)
//                     .unwrap_or_default()
//                     .read(&mut builder)
//                     .into();
//                 (main_openings_variable, preprocessed_openings_variable)
//             })
//             .collect::<Vec<_>>();

//         let verifier = BasefoldVerifier::<Poseidon2BabyBear16BasefoldConfig>::new(log_blowup);
//         let recursive_verifier = RecursiveBasefoldVerifier::<RecursiveBasefoldConfigImpl<C, SC>>
// {             fri_config: verifier.fri_config,
//             tcs: RecursiveMerkleTreeTcs::<C, SC>(PhantomData),
//         };
//         let recursive_verifier =
//             RecursiveStackedPcsVerifier::new(recursive_verifier, log_stacking_height);

//         let recursive_jagged_verifier = RecursiveJaggedPcsVerifier::<
//             SC,
//             C,
//             RecursiveJaggedConfigImpl<
//                 C,
//                 SC,
//                 RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
//             >,
//         > { stacked_pcs_verifier: recursive_verifier, max_log_row_count, jagged_evaluator:
//         > RecursiveJaggedEvalSumcheckConfig::<BabyBearPoseidon2>(PhantomData),
//         };

//         let stark_verifier = StarkVerifier::<A, SC, C, JC> {
//             machine,
//             pcs_verifier: recursive_jagged_verifier,
//             _phantom: std::marker::PhantomData,
//         };

//         stark_verifier.verify_zerocheck(
//             &mut builder,
//             &mut challenger_variable,
//             &shard_proof_variable.opened_values,
//             &shard_proof_variable.zerocheck_proof,
//             &gkr_points_variable,
//             &gkr_column_openings_variable,
//             &shard_proof_variable.public_values,
//         );

//         let mut witness_stream = Vec::new();
//         Witnessable::<AsmConfig<F, EF>>::write(&shard_proof, &mut witness_stream);
//         Witnessable::<AsmConfig<F, EF>>::write(
//             &shard_proof.testing_data.gkr_points,
//             &mut witness_stream,
//         );
//         shard_proof.gkr_proofs.iter().for_each(|gkr_proof| {
//             let (main_openings, preprocessed_openings) = &gkr_proof.column_openings;
//             Witnessable::<AsmConfig<F, EF>>::write(main_openings, &mut witness_stream);
//             let preprocessed_openings_unwrapped: MleEval<_> =
//                 preprocessed_openings.as_ref().map(MleEval::to_vec).unwrap_or_default().into();
//             Witnessable::<AsmConfig<F, EF>>::write(
//                 &preprocessed_openings_unwrapped,
//                 &mut witness_stream,
//             );
//         });

//         let block = builder.into_root_block();
//         let mut compiler = AsmCompiler::<AsmConfig<F, EF>>::default();
//         let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
//         let mut runtime =
//             Runtime::<F, EF, DiffusionMatrixBabyBear>::new(program.clone(), my_bb_16_perm());
//         runtime.witness_stream = witness_stream.into();
//         runtime.run().unwrap();

//         // Test for a bad zerocheck proof.
//         let mut invalid_shard_proof = shard_proof.clone();
//         invalid_shard_proof.zerocheck_proof.univariate_polys[0].coefficients[0] += EF::one();
//         let mut witness_stream = Vec::new();
//         Witnessable::<AsmConfig<F, EF>>::write(&invalid_shard_proof, &mut witness_stream);
//         Witnessable::<AsmConfig<F, EF>>::write(
//             &invalid_shard_proof.testing_data.gkr_points,
//             &mut witness_stream,
//         );
//         invalid_shard_proof.gkr_proofs.iter().for_each(|gkr_proof| {
//             let (main_openings, preprocessed_openings) = &gkr_proof.column_openings;
//             Witnessable::<AsmConfig<F, EF>>::write(main_openings, &mut witness_stream);
//             let preprocessed_openings_unwrapped: MleEval<_> =
//                 preprocessed_openings.as_ref().map(MleEval::to_vec).unwrap_or_default().into();
//             Witnessable::<AsmConfig<F, EF>>::write(
//                 &preprocessed_openings_unwrapped,
//                 &mut witness_stream,
//             );
//         });
//         let mut runtime = Runtime::<F, EF, DiffusionMatrixBabyBear>::new(program,
// my_bb_16_perm());         runtime.witness_stream = witness_stream.into();
//         runtime.run().expect_err("invalid proof should not be verified");
//     }
// }
