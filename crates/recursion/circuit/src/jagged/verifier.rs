use std::marker::PhantomData;

use slop_algebra::{extension::BinomialExtensionField, AbstractField};
use slop_baby_bear::BabyBear;
use slop_jagged::{
    JaggedBasefoldConfig, JaggedLittlePolynomialVerifierParams, JaggedSumcheckEvalProof,
};
use slop_multilinear::{Evaluations, Mle, Point};
use slop_sumcheck::PartialSumcheckProof;
use sp1_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Ext, Felt, SymbolicExt},
};
use sp1_stark::BabyBearPoseidon2;

use crate::{
    basefold::{
        stacked::{RecursiveStackedPcsProof, RecursiveStackedPcsVerifier},
        RecursiveBasefoldConfigImpl, RecursiveBasefoldProof, RecursiveBasefoldVerifier,
        RecursiveMultilinearPcsVerifier,
    },
    challenger::FieldChallengerVariable,
    sumcheck::{evaluate_mle_ext, verify_sumcheck},
    AsRecursive, BabyBearFriConfigVariable, CircuitConfig,
};

use super::jagged_eval::{RecursiveJaggedEvalConfig, RecursiveJaggedEvalSumcheckConfig};

pub trait RecursiveJaggedConfig: Sized {
    type F;
    type EF: AbstractField;
    type Bit;
    type Circuit: CircuitConfig<F = Self::F, EF = Self::EF, Bit = Self::Bit>;
    type Commitment;
    type Challenger: FieldChallengerVariable<Self::Circuit, Self::Bit>;
    type BatchPcsProof;
    type BatchPcsVerifier;
    type JaggedEvaluator: RecursiveJaggedEvalConfig<
        Self::Circuit,
        Self::Challenger,
        JaggedEvalProof = Self::JaggedEvalProof,
    >;
    type JaggedEvalProof;
}

pub struct RecursiveJaggedConfigImpl<C, SC, P> {
    _marker: PhantomData<(C, SC, P)>,
}

impl<
        C: CircuitConfig<F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>>,
        SC: BabyBearFriConfigVariable<C>,
        P: RecursiveMultilinearPcsVerifier<F = C::F, EF = C::EF>,
    > RecursiveJaggedConfig for RecursiveJaggedConfigImpl<C, SC, P>
{
    type F = C::F;
    type EF = C::EF;
    type Bit = C::Bit;
    type Circuit = C;
    type Commitment = SC::DigestVariable;
    type Challenger = SC::FriChallengerVariable;
    type BatchPcsProof = RecursiveBasefoldProof<RecursiveBasefoldConfigImpl<C, SC>>;
    type BatchPcsVerifier = RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>;
    type JaggedEvaluator = RecursiveJaggedEvalSumcheckConfig<SC>;
    type JaggedEvalProof = JaggedSumcheckEvalProof<Ext<C::F, C::EF>>;
}

pub struct JaggedPcsProofVariable<JC: RecursiveJaggedConfig> {
    pub params: JaggedLittlePolynomialVerifierParams<Felt<JC::F>>,
    pub sumcheck_proof: PartialSumcheckProof<Ext<JC::F, JC::EF>>,
    pub jagged_eval_proof: JC::JaggedEvalProof,
    pub stacked_pcs_proof: RecursiveStackedPcsProof<JC::BatchPcsProof, JC::F, JC::EF>,
}

impl<
        C: CircuitConfig<F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>, Bit = Felt<BabyBear>>,
        BC,
        E,
    > AsRecursive<C> for JaggedBasefoldConfig<BC, E>
{
    type Recursive = RecursiveJaggedConfigImpl<
        C,
        BabyBearPoseidon2,
        RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, BabyBearPoseidon2>>,
    >;
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct RecursiveJaggedPcsVerifier<
    SC: BabyBearFriConfigVariable<C>,
    C: CircuitConfig<F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>>,
    JC: RecursiveJaggedConfig<
        BatchPcsVerifier = RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
    >,
> {
    pub stacked_pcs_verifier: RecursiveStackedPcsVerifier<JC::BatchPcsVerifier>,
    pub max_log_row_count: usize,
    pub jagged_evaluator: JC::JaggedEvaluator,
}

impl<
        SC: BabyBearFriConfigVariable<C>,
        C: CircuitConfig<F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>>,
        JC: RecursiveJaggedConfig<
            F = C::F,
            EF = C::EF,
            Circuit = C,
            Commitment = SC::DigestVariable,
            Challenger = SC::FriChallengerVariable,
            BatchPcsProof = RecursiveBasefoldProof<RecursiveBasefoldConfigImpl<C, SC>>,
            BatchPcsVerifier = RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
        >,
    > RecursiveJaggedPcsVerifier<SC, C, JC>
{
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub fn verify_trusted_evaluations(
        &self,
        builder: &mut Builder<JC::Circuit>,
        commitments: &[JC::Commitment],
        point: Point<Ext<JC::F, JC::EF>>,
        evaluation_claims: &[Evaluations<Ext<JC::F, JC::EF>>],
        proof: &JaggedPcsProofVariable<JC>,
        insertion_points: &[usize],
        challenger: &mut JC::Challenger,
    ) -> Vec<Felt<JC::F>> {
        let JaggedPcsProofVariable { stacked_pcs_proof, sumcheck_proof, jagged_eval_proof, params } =
            proof;
        let num_col_variables = (params.col_prefix_sums.len() - 1).next_power_of_two().ilog2();
        let z_col =
            (0..num_col_variables).map(|_| challenger.sample_ext(builder)).collect::<Point<_>>();

        let z_row = point;

        // Collect the claims for the different polynomials.
        let mut column_claims =
            evaluation_claims.iter().flatten().flatten().copied().collect::<Vec<_>>();

        // For each commit, Rizz needed a commitment to a vector of length a multiple of
        // 1 << self.pcs.log_stacking_height, and this is achieved by adding a single column of
        // zeroes as the last matrix of the commitment. We insert these "artificial" zeroes
        // into the evaluation claims.
        let zero_ext: Ext<JC::F, JC::EF> = builder.constant(JC::EF::zero());
        for insertion_point in insertion_points.iter().rev() {
            column_claims.insert(*insertion_point, zero_ext);
        }

        // Pad the column claims to the next power of two.
        column_claims.resize(column_claims.len().next_power_of_two(), zero_ext);

        let column_mle = Mle::from(column_claims);
        let sumcheck_claim: Ext<JC::F, JC::EF> =
            evaluate_mle_ext(builder, column_mle, z_col.clone())[0];

        builder.assert_ext_eq(sumcheck_claim, sumcheck_proof.claimed_sum);

        builder.cycle_tracker_v2_enter("jagged - verify sumcheck");
        verify_sumcheck::<C, SC>(builder, challenger, sumcheck_proof);
        builder.cycle_tracker_v2_exit();

        builder.cycle_tracker_v2_enter("jagged - jagged-eval");
        let (jagged_eval, prefix_sum_felts) = self.jagged_evaluator.jagged_evaluation(
            builder,
            params,
            z_row,
            z_col,
            sumcheck_proof.point_and_eval.0.clone(),
            jagged_eval_proof,
            challenger,
        );
        builder.cycle_tracker_v2_exit();

        // Compute the expected evaluation of the dense trace polynomial.
        let expected_eval: SymbolicExt<BabyBear, BinomialExtensionField<BabyBear, 4>> =
            sumcheck_proof.point_and_eval.1 / jagged_eval;

        // Verify the evaluation proof.
        let evaluation_point = sumcheck_proof.point_and_eval.0.clone();
        self.stacked_pcs_verifier.verify_trusted_evaluation(
            builder,
            commitments,
            &evaluation_point,
            stacked_pcs_proof,
            expected_eval,
            challenger,
        );
        prefix_sum_felts
    }
}

#[allow(dead_code)]
pub struct RecursiveMachineJaggedPcsVerifier<
    'a,
    SC: BabyBearFriConfigVariable<C>,
    C: CircuitConfig<F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>>,
    JC: RecursiveJaggedConfig<
        BatchPcsVerifier = RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
    >,
> {
    pub jagged_pcs_verifier: &'a RecursiveJaggedPcsVerifier<SC, C, JC>,
    pub column_counts_by_round: Vec<Vec<usize>>,
}

impl<
        'a,
        SC: BabyBearFriConfigVariable<C>,
        C: CircuitConfig<F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>>,
        JC: RecursiveJaggedConfig<
            F = C::F,
            EF = C::EF,
            Circuit = C,
            Commitment = SC::DigestVariable,
            Challenger = SC::FriChallengerVariable,
            BatchPcsProof = RecursiveBasefoldProof<RecursiveBasefoldConfigImpl<C, SC>>,
            BatchPcsVerifier = RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
        >,
    > RecursiveMachineJaggedPcsVerifier<'a, SC, C, JC>
{
    #[allow(dead_code)]
    pub fn new(
        jagged_pcs_verifier: &'a RecursiveJaggedPcsVerifier<SC, C, JC>,
        column_counts_by_round: Vec<Vec<usize>>,
    ) -> Self {
        Self { jagged_pcs_verifier, column_counts_by_round }
    }

    #[allow(dead_code)]
    pub fn verify_trusted_evaluations(
        &self,
        builder: &mut Builder<JC::Circuit>,
        commitments: &[JC::Commitment],
        point: Point<Ext<JC::F, JC::EF>>,
        evaluation_claims: &[Evaluations<Ext<JC::F, JC::EF>>],
        proof: &JaggedPcsProofVariable<JC>,
        challenger: &mut JC::Challenger,
    ) -> Vec<Felt<JC::F>> {
        let insertion_points = self
            .column_counts_by_round
            .iter()
            .scan(0, |state, y| {
                *state += y.iter().sum::<usize>();
                Some(*state)
            })
            .collect::<Vec<_>>();

        self.jagged_pcs_verifier.verify_trusted_evaluations(
            builder,
            commitments,
            point,
            evaluation_claims,
            proof,
            &insertion_points,
            challenger,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::{marker::PhantomData, sync::Arc};

    use rand::{thread_rng, Rng};
    use slop_algebra::AbstractField;
    use slop_baby_bear::DiffusionMatrixBabyBear;
    use slop_basefold::{BasefoldVerifier, Poseidon2BabyBear16BasefoldConfig};
    use slop_challenger::CanObserve;
    use slop_commit::Rounds;
    use slop_jagged::{
        JaggedConfig, JaggedPcsProof, JaggedPcsVerifier, JaggedProver, JaggedProverComponents,
        Poseidon2BabyBearJaggedCpuProverComponents,
    };
    use slop_merkle_tree::my_bb_16_perm;
    use slop_multilinear::{Evaluations, Mle, PaddedMle, Point};
    use sp1_core_machine::utils::setup_logger;
    use sp1_recursion_compiler::circuit::{AsmBuilder, AsmCompiler, AsmConfig, CircuitV2Builder};
    use sp1_recursion_executor::Runtime;
    use sp1_stark::BabyBearPoseidon2;

    use crate::{
        basefold::{
            stacked::RecursiveStackedPcsVerifier, tcs::RecursiveMerkleTreeTcs,
            RecursiveBasefoldConfigImpl, RecursiveBasefoldVerifier,
        },
        challenger::{CanObserveVariable, DuplexChallengerVariable},
        jagged::{
            jagged_eval::RecursiveJaggedEvalSumcheckConfig,
            verifier::{
                RecursiveJaggedConfigImpl, RecursiveJaggedPcsVerifier,
                RecursiveMachineJaggedPcsVerifier,
            },
        },
        witness::Witnessable,
    };

    type SC = BabyBearPoseidon2;
    type F = <SC as JaggedConfig>::F;
    type EF = <SC as JaggedConfig>::EF;
    type C = AsmConfig<F, EF>;
    type Prover = JaggedProver<Poseidon2BabyBearJaggedCpuProverComponents>;

    async fn generate_jagged_proof(
        jagged_verifier: &JaggedPcsVerifier<SC>,
        round_mles: Rounds<Vec<PaddedMle<F>>>,
        eval_point: Point<EF>,
    ) -> (
        JaggedPcsProof<SC>,
        Rounds<<Poseidon2BabyBearJaggedCpuProverComponents as JaggedProverComponents>::Commitment>,
        Rounds<Evaluations<EF>>,
    ) {
        let jagged_prover = Prover::from_verifier(jagged_verifier);

        let mut challenger = jagged_verifier.challenger();

        let mut prover_data = Rounds::new();
        let mut commitments = Rounds::new();
        for round in round_mles.iter() {
            let (commit, data) =
                jagged_prover.commit_multilinears(round.clone()).await.ok().unwrap();
            challenger.observe(commit);
            let data_bytes = bincode::serialize(&data).unwrap();
            let data = bincode::deserialize(&data_bytes).unwrap();
            prover_data.push(data);
            commitments.push(commit);
        }

        let mut evaluation_claims = Rounds::new();
        for round in round_mles.iter() {
            let mut evals = Evaluations::default();
            for mle in round.iter() {
                let eval = mle.eval_at(&eval_point).await;
                evals.push(eval);
            }
            evaluation_claims.push(evals);
        }

        let proof = jagged_prover
            .prove_trusted_evaluations(
                eval_point.clone(),
                evaluation_claims.clone(),
                prover_data,
                &mut challenger,
            )
            .await
            .ok()
            .unwrap();

        (proof, commitments, evaluation_claims)
    }

    #[tokio::test]
    async fn test_jagged_verifier() {
        setup_logger();

        let row_counts_rounds = vec![
            vec![
                1 << 13,
                1 << 8,
                1 << 11,
                1 << 7,
                1 << 16,
                1 << 14,
                1 << 20,
                1 << 7,
                1 << 9,
                1 << 11,
                1 << 8,
                1 << 7,
                1 << 14,
                1 << 10,
                1 << 14,
                1 << 8,
            ],
            vec![1 << 8],
        ];
        let column_counts_rounds = vec![
            vec![47, 41, 41, 58, 52, 109, 428, 50, 53, 93, 100, 83, 31, 68, 134, 80],
            vec![512],
        ];

        let log_blowup = 1;
        let log_stacking_height = 21;
        let max_log_row_count = 21;

        let row_counts = row_counts_rounds.into_iter().collect::<Rounds<Vec<usize>>>();
        let column_counts = column_counts_rounds.into_iter().collect::<Rounds<Vec<usize>>>();

        assert!(row_counts.len() == column_counts.len());

        let mut rng = thread_rng();

        let round_mles = row_counts
            .iter()
            .zip(column_counts.iter())
            .map(|(row_counts, col_counts)| {
                row_counts
                    .iter()
                    .zip(col_counts.iter())
                    .map(|(num_rows, num_cols)| {
                        if *num_rows == 0 {
                            PaddedMle::zeros(*num_cols, max_log_row_count)
                        } else {
                            let mle = Mle::<F>::rand(&mut rng, *num_cols, num_rows.ilog(2));
                            PaddedMle::padded_with_zeros(Arc::new(mle), max_log_row_count)
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Rounds<_>>();

        let jagged_verifier = JaggedPcsVerifier::<SC>::new(
            log_blowup,
            log_stacking_height,
            max_log_row_count as usize,
        );

        let eval_point = (0..max_log_row_count).map(|_| rng.gen::<EF>()).collect::<Point<_>>();

        // Generate the jagged proof.
        let (proof, mut commitments, evaluation_claims) =
            generate_jagged_proof(&jagged_verifier, round_mles, eval_point.clone()).await;

        // Define the verification circuit.
        let mut builder = AsmBuilder::<F, EF>::default();
        builder.cycle_tracker_v2_enter("jagged - read input");
        let mut challenger_variable = DuplexChallengerVariable::new(&mut builder);
        let commitments_var = commitments.read(&mut builder);
        let eval_point_var = eval_point.read(&mut builder);
        let evaluation_claims_var = evaluation_claims.read(&mut builder);
        let proof_var = proof.read(&mut builder);
        builder.cycle_tracker_v2_exit();
        builder.cycle_tracker_v2_enter("jagged - observe commitments");
        for commitment_var in commitments_var.iter() {
            challenger_variable.observe_slice(&mut builder, *commitment_var);
        }
        builder.cycle_tracker_v2_exit();
        let verifier = BasefoldVerifier::<Poseidon2BabyBear16BasefoldConfig>::new(log_blowup);
        let recursive_verifier = RecursiveBasefoldVerifier::<RecursiveBasefoldConfigImpl<C, SC>> {
            fri_config: verifier.fri_config,
            tcs: RecursiveMerkleTreeTcs::<C, SC>(PhantomData),
        };
        let recursive_verifier =
            RecursiveStackedPcsVerifier::new(recursive_verifier, log_stacking_height);

        let recursive_jagged_verifier = RecursiveJaggedPcsVerifier::<
            SC,
            C,
            RecursiveJaggedConfigImpl<
                C,
                SC,
                RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>,
            >,
        > {
            stacked_pcs_verifier: recursive_verifier,
            max_log_row_count: max_log_row_count as usize,
            jagged_evaluator: RecursiveJaggedEvalSumcheckConfig::<BabyBearPoseidon2>(PhantomData),
        };

        let recursive_jagged_verifier = RecursiveMachineJaggedPcsVerifier::new(
            &recursive_jagged_verifier,
            vec![column_counts[0].clone(), column_counts[1].clone()],
        );

        builder.cycle_tracker_v2_enter("jagged-verifier");
        recursive_jagged_verifier.verify_trusted_evaluations(
            &mut builder,
            &commitments_var,
            eval_point_var,
            &evaluation_claims_var,
            &proof_var,
            &mut challenger_variable,
        );
        builder.cycle_tracker_v2_exit();

        let block = builder.into_root_block();
        let mut compiler = AsmCompiler::<AsmConfig<F, EF>>::default();

        // Compile the verification circuit.
        let program = compiler.compile_inner(block).validate().unwrap();

        // Run the verification circuit with the proof artifacts.
        let mut witness_stream = Vec::new();
        Witnessable::<AsmConfig<F, EF>>::write(&commitments, &mut witness_stream);
        Witnessable::<AsmConfig<F, EF>>::write(&eval_point, &mut witness_stream);
        Witnessable::<AsmConfig<F, EF>>::write(&evaluation_claims, &mut witness_stream);
        Witnessable::<AsmConfig<F, EF>>::write(&proof, &mut witness_stream);
        let mut runtime = Runtime::<F, EF, DiffusionMatrixBabyBear>::new(
            Arc::new(program.clone()),
            my_bb_16_perm(),
        );
        runtime.witness_stream = witness_stream.into();
        runtime.run().unwrap();

        // Run the verification circuit with the proof artifacts with an expected failure.
        let mut witness_stream = Vec::new();
        commitments.rounds[0][0] += F::one();
        Witnessable::<AsmConfig<F, EF>>::write(&commitments, &mut witness_stream);
        Witnessable::<AsmConfig<F, EF>>::write(&eval_point, &mut witness_stream);
        Witnessable::<AsmConfig<F, EF>>::write(&evaluation_claims, &mut witness_stream);
        Witnessable::<AsmConfig<F, EF>>::write(&proof, &mut witness_stream);
        let mut runtime =
            Runtime::<F, EF, DiffusionMatrixBabyBear>::new(Arc::new(program), my_bb_16_perm());
        runtime.witness_stream = witness_stream.into();
        runtime.run().expect_err("invalid proof should not be verified");
    }
}
