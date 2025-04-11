use std::{fmt::Debug, marker::PhantomData};

use serde::{Deserialize, Serialize};
use slop_algebra::{ExtensionField, Field};
use slop_alloc::Backend;
use slop_challenger::FieldChallenger;
use slop_multilinear::{Mle, Point};
use slop_sumcheck::{
    partially_verify_sumcheck_proof, reduce_sumcheck_to_evaluation, PartialSumcheckProof,
    SumcheckError,
};
use thiserror::Error;

use crate::{
    poly::BranchingProgram, JaggedLittlePolynomialProverParams,
    JaggedLittlePolynomialVerifierParams,
};

use super::{
    sumcheck_poly::JaggedEvalSumcheckPoly, JaggedAssistSumAsPoly, JaggedEvalConfig,
    JaggedEvalProver,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaggedSumcheckEvalProof<F> {
    pub branching_program_evals: Vec<F>,
    pub partial_sumcheck_proof: PartialSumcheckProof<F>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct JaggedEvalSumcheckConfig<F>(PhantomData<F>);

#[derive(Debug, Error)]
pub enum JaggedEvalSumcheckError<F: Field> {
    #[error("sumcheck error: {0}")]
    SumcheckError(SumcheckError),
    #[error("jagged evaluation proof verification failed, expected: {0}, got: {1}")]
    JaggedEvaluationFailed(F, F),
}

impl<F, EF, Challenger> JaggedEvalConfig<F, EF, Challenger> for JaggedEvalSumcheckConfig<F>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    type JaggedEvalProof = JaggedSumcheckEvalProof<EF>;
    type JaggedEvalError = JaggedEvalSumcheckError<EF>;

    fn jagged_evaluation(
        &self,
        params: &JaggedLittlePolynomialVerifierParams<F>,
        z_row: &Point<EF>,
        z_col: &Point<EF>,
        z_trace: &Point<EF>,
        proof: &Self::JaggedEvalProof,
        challenger: &mut Challenger,
    ) -> Result<EF, Self::JaggedEvalError> {
        let JaggedSumcheckEvalProof { branching_program_evals, partial_sumcheck_proof } = proof;
        // Calculate the partial lagrange from z_col point.
        let z_col_partial_lagrange = Mle::blocking_partial_lagrange(z_col);
        let z_col_partial_lagrange = z_col_partial_lagrange.guts().as_slice();

        // Calcuate the jagged eval from the branching program eval claims.
        let jagged_eval = z_col_partial_lagrange
            .iter()
            .zip(branching_program_evals.iter())
            .map(|(partial_lagrange, branching_program_eval)| {
                *partial_lagrange * *branching_program_eval
            })
            .sum::<EF>();

        // Verify the jagged eval proof.
        partially_verify_sumcheck_proof(partial_sumcheck_proof, challenger)
            .map_err(JaggedEvalSumcheckError::SumcheckError)?;

        let (first_half_z_index, second_half_z_index) = partial_sumcheck_proof
            .point_and_eval
            .0
            .split_at(partial_sumcheck_proof.point_and_eval.0.dimension() / 2);
        assert!(first_half_z_index.len() == second_half_z_index.len());

        // Compute the jagged eval sc expected eval and assert it matches the proof's eval.
        let current_column_prefix_sums = params.col_prefix_sums.iter();
        let next_column_prefix_sums = params.col_prefix_sums.iter().skip(1);
        let mut is_first_column = true;
        let mut prev_merged_prefix_sum = Point::<F>::default();
        let mut prev_full_lagrange_eval = EF::zero();
        let mut jagged_eval_sc_expected_eval = current_column_prefix_sums
            .zip(next_column_prefix_sums)
            .zip(z_col_partial_lagrange.iter())
            .map(|((current_column_prefix_sum, next_column_prefix_sum), z_col_eq_val)| {
                let mut merged_prefix_sum = current_column_prefix_sum.clone();
                merged_prefix_sum.extend(next_column_prefix_sum);

                let full_lagrange_eval =
                    if prev_merged_prefix_sum == merged_prefix_sum && !is_first_column {
                        prev_full_lagrange_eval
                    } else {
                        let full_lagrange_eval = Mle::full_lagrange_eval(
                            &merged_prefix_sum,
                            &partial_sumcheck_proof.point_and_eval.0,
                        );
                        prev_full_lagrange_eval = full_lagrange_eval;
                        full_lagrange_eval
                    };

                prev_merged_prefix_sum = merged_prefix_sum;
                is_first_column = false;

                *z_col_eq_val * full_lagrange_eval
            })
            .sum::<EF>();

        let branching_program = BranchingProgram::new(z_row.clone(), z_trace.clone());
        jagged_eval_sc_expected_eval *=
            branching_program.eval(&first_half_z_index, &second_half_z_index);

        if jagged_eval_sc_expected_eval != partial_sumcheck_proof.point_and_eval.1 {
            Err(JaggedEvalSumcheckError::JaggedEvaluationFailed(
                jagged_eval_sc_expected_eval,
                partial_sumcheck_proof.point_and_eval.1,
            ))
        } else {
            Ok(jagged_eval)
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct JaggedEvalSumcheckProver<F, BPE, A>(pub PhantomData<(F, BPE, A)>);

impl<F, BPE, A> Default for JaggedEvalSumcheckProver<F, BPE, A> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<F, EF, Challenger, BPE, A> JaggedEvalProver<F, EF, Challenger>
    for JaggedEvalSumcheckProver<F, BPE, A>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + Send + Sync,
    BPE: JaggedAssistSumAsPoly<F, EF, A> + Send + Sync + Clone + Debug + 'static,
    A: Backend,
{
    type EvalProof = JaggedSumcheckEvalProof<EF>;
    type EvalConfig = JaggedEvalSumcheckConfig<F>;

    async fn prove_jagged_evaluation(
        &self,
        params: &JaggedLittlePolynomialProverParams,
        z_row: &Point<EF>,
        z_col: &Point<EF>,
        z_trace: &Point<EF>,
        challenger: &mut Challenger,
    ) -> Self::EvalProof {
        // Create sumcheck proof for the jagged eval.
        let jagged_eval_sc_poly = JaggedEvalSumcheckPoly::<F, EF, BPE, A>::new(
            z_row.clone(),
            z_col.clone(),
            z_trace.clone(),
            params.col_prefix_sums_usize.clone(),
        )
        .await;

        // Compute the full eval of the jagged poly.
        let verifier_params = params.clone().into_verifier_params();
        let (expected_sum, branching_program_evals) =
            verifier_params.full_jagged_little_polynomial_evaluation(z_row, z_col, z_trace);

        let (partial_sumcheck_proof, _) = reduce_sumcheck_to_evaluation(
            vec![jagged_eval_sc_poly],
            challenger,
            vec![expected_sum],
            1,
            EF::one(),
        )
        .await;

        JaggedSumcheckEvalProof { branching_program_evals, partial_sumcheck_proof }
    }
}
