use serde::{Deserialize, Serialize};
use slop_algebra::AbstractField;
use slop_challenger::FieldChallenger;
use slop_multilinear::{Evaluations, Mle, Point};
use slop_stacked::{StackedPcsProof, StackedPcsVerifier};
use slop_sumcheck::{partially_verify_sumcheck_proof, PartialSumcheckProof, SumcheckError};
use thiserror::Error;

use crate::{BranchingProgram, JaggedConfig, JaggedLittlePolynomialVerifierParams};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JaggedPcsProof<C: JaggedConfig> {
    pub stacked_pcs_proof: StackedPcsProof<C::BatchPcsProof, C::EF>,
    pub sumcheck_proof: PartialSumcheckProof<C::EF>,
    pub branching_program_evals: Vec<C::EF>,
    pub jagged_eval_proof: PartialSumcheckProof<C::EF>,
    pub params: JaggedLittlePolynomialVerifierParams<C::EF>,
}

#[derive(Debug, Clone)]
pub struct JaggedPcsVerifier<C: JaggedConfig> {
    pub stacked_pcs_verifier: StackedPcsVerifier<C::BatchPcsVerifier>,
    pub max_log_row_count: usize,
}

#[derive(Debug, Error)]
pub enum JaggedPcsVerifierError<C: JaggedConfig> {
    #[error("sumcheck claim mismatch: {0} != {1}")]
    SumcheckClaimMismatch(C::EF, C::EF),
    #[error("sumcheck proof verification failed: {0}")]
    SumcheckError(SumcheckError),
    #[error("dense pcs verification failed")]
    DensePcsVerificationFailed,
}

impl<C: JaggedConfig> JaggedPcsVerifier<C> {
    pub fn verify_trusted_evaluations(
        &self,
        commitments: &[C::Commitment],
        point: Point<C::EF>,
        evaluation_claims: &[Evaluations<C::EF>],
        proof: &JaggedPcsProof<C>,
        insertion_points: &[usize],
        challenger: &mut C::Challenger,
    ) -> Result<(), JaggedPcsVerifierError<C>> {
        let JaggedPcsProof {
            stacked_pcs_proof,
            sumcheck_proof,
            branching_program_evals,
            jagged_eval_proof,
            params,
        } = proof;
        let num_col_variables = (params.col_prefix_sums.len() - 1).next_power_of_two().ilog2();
        let z_col = (0..num_col_variables)
            .map(|_| challenger.sample_ext_element::<C::EF>())
            .collect::<Point<_>>();

        let z_row = point;

        // Collect the claims for the different polynomials.
        let mut column_claims =
            evaluation_claims.iter().flatten().flatten().copied().collect::<Vec<_>>();

        // For each commit, Rizz needed a commitment to a vector of length a multiple of
        // 1 << self.pcs.log_stacking_height, and this is achieved by adding a single column of zeroes
        // as the last matrix of the commitment. We insert these "artificial" zeroes into the evaluation
        // claims.
        for insertion_point in insertion_points.iter().rev() {
            column_claims.insert(*insertion_point, C::EF::zero());
        }

        // Pad the column claims to the next power of two.
        column_claims.resize(column_claims.len().next_power_of_two(), C::EF::zero());

        let column_mle = Mle::from(column_claims);
        let sumcheck_claim = column_mle.blocking_eval_at(&z_col)[0];

        if sumcheck_claim != sumcheck_proof.claimed_sum {
            return Err(JaggedPcsVerifierError::SumcheckClaimMismatch(
                sumcheck_claim,
                sumcheck_proof.claimed_sum,
            ));
        }

        partially_verify_sumcheck_proof(sumcheck_proof, challenger)
            .map_err(JaggedPcsVerifierError::SumcheckError)?;

        // Calculate the partial lagrange from z_col point.
        let z_col_partial_lagrange = Mle::blocking_partial_lagrange(&z_col);
        let z_col_partial_lagrange = z_col_partial_lagrange.guts().as_slice();

        // Calcuate the jagged eval from the branching program eval claims.
        let jagged_eval = z_col_partial_lagrange
            .iter()
            .zip(branching_program_evals.iter())
            .map(|(partial_lagrange, branching_program_eval)| {
                *partial_lagrange * *branching_program_eval
            })
            .sum::<C::EF>();

        // Verify the jagged eval proof.
        partially_verify_sumcheck_proof(jagged_eval_proof, challenger)
            .map_err(JaggedPcsVerifierError::SumcheckError)?;

        let (first_half_z_index, second_half_z_index) = jagged_eval_proof
            .point_and_eval
            .0
            .split_at(jagged_eval_proof.point_and_eval.0.dimension() / 2);
        assert!(first_half_z_index.len() == second_half_z_index.len());

        // Compute the jagged eval sc expected eval and assert it matches the proof's eval.
        let current_column_prefix_sums = params.col_prefix_sums.iter();
        let next_column_prefix_sums = params.col_prefix_sums.iter().skip(1);
        let mut prev_merged_prefix_sum = Point::<C::EF>::default();
        let mut prev_full_lagrange_eval = C::EF::zero();
        let mut jagged_eval_sc_expected_eval = current_column_prefix_sums
            .zip(next_column_prefix_sums)
            .zip(z_col_partial_lagrange.iter())
            .map(|((current_column_prefix_sum, next_column_prefix_sum), z_col_eq_val)| {
                let mut merged_prefix_sum = current_column_prefix_sum.clone();
                merged_prefix_sum.extend(next_column_prefix_sum);

                let full_lagrange_eval = if prev_merged_prefix_sum == merged_prefix_sum {
                    prev_full_lagrange_eval
                } else {
                    let full_lagrange_eval = Mle::full_lagrange_eval(
                        &merged_prefix_sum,
                        &jagged_eval_proof.point_and_eval.0,
                    );
                    prev_full_lagrange_eval = full_lagrange_eval;
                    full_lagrange_eval
                };

                prev_merged_prefix_sum = merged_prefix_sum;

                *z_col_eq_val * full_lagrange_eval
            })
            .sum::<C::EF>();

        let branching_program =
            BranchingProgram::new(z_row.clone(), sumcheck_proof.point_and_eval.0.clone());
        jagged_eval_sc_expected_eval *=
            branching_program.eval(&first_half_z_index, &second_half_z_index);

        assert!(jagged_eval_sc_expected_eval == jagged_eval_proof.point_and_eval.1);

        // Compute the expected evaluation of the dense trace polynomial.
        let expected_eval = sumcheck_proof.point_and_eval.1 / jagged_eval;

        // Verify the evaluation proof.
        let evaluation_point = sumcheck_proof.point_and_eval.0.clone();
        self.stacked_pcs_verifier
            .verify_trusted_evaluation(
                commitments,
                &evaluation_point,
                stacked_pcs_proof,
                expected_eval,
                challenger,
            )
            .map_err(|_| JaggedPcsVerifierError::DensePcsVerificationFailed)?;

        Ok(())
    }
}

pub struct MachineJaggedPcsVerifier<'a, C: JaggedConfig> {
    pub jagged_pcs_verifier: &'a JaggedPcsVerifier<C>,
    pub column_counts_by_round: Vec<Vec<usize>>,
}

impl<'a, C: JaggedConfig> MachineJaggedPcsVerifier<'a, C> {
    pub fn new(
        jagged_pcs_verifier: &'a JaggedPcsVerifier<C>,
        column_counts_by_round: Vec<Vec<usize>>,
    ) -> Self {
        Self { jagged_pcs_verifier, column_counts_by_round }
    }

    pub fn verify_trusted_evaluations(
        &self,
        commitments: &[C::Commitment],
        point: Point<C::EF>,
        evaluation_claims: &[Evaluations<C::EF>],
        proof: &JaggedPcsProof<C>,
        challenger: &mut C::Challenger,
    ) -> Result<(), JaggedPcsVerifierError<C>> {
        let insertion_points = self
            .column_counts_by_round
            .iter()
            .scan(0, |state, y| {
                *state += y.iter().sum::<usize>();
                Some(*state)
            })
            .collect::<Vec<_>>();

        self.jagged_pcs_verifier.verify_trusted_evaluations(
            commitments,
            point,
            evaluation_claims,
            proof,
            &insertion_points,
            challenger,
        )
    }
}
