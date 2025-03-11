use serde::{Deserialize, Serialize};
use slop_algebra::{AbstractField, Field};
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
            jagged_eval_proof: branching_program_evals_proof,
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

        let jagged_eval = branching_program_evals_proof.claimed_sum;

        tracing::debug_span!("verifying branching program evals proof").in_scope(|| {
            partially_verify_sumcheck_proof(branching_program_evals_proof, challenger)
                .map_err(JaggedPcsVerifierError::SumcheckError)
        })?;

        let branching_program =
            BranchingProgram::new(z_row.clone(), sumcheck_proof.point_and_eval.0.clone());
        let (first_half_z_index, second_half_z_index) = branching_program_evals_proof
            .point_and_eval
            .0
            .split_at(branching_program_evals_proof.point_and_eval.0.dimension() / 2);
        assert!(first_half_z_index.len() == second_half_z_index.len());

        let num_chunks = 4;
        let chunk_size = (branching_program_evals_proof.point_and_eval.0.dimension() + num_chunks
            - 1)
            / num_chunks;

        let point_chunks = branching_program_evals_proof
            .point_and_eval
            .0
            .values()
            .chunks(chunk_size)
            .map(|chunk| Point::from(chunk.to_vec()))
            .collect::<Vec<_>>();

        let partial_lagranges = point_chunks
            .iter()
            .map(|point_chunk| {
                tracing::debug_span!("computing partial lagrange mle")
                    .in_scope(|| Mle::blocking_partial_lagrange(point_chunk))
            })
            .collect::<Vec<_>>();

        let partial_lagrange_z_col_eq =
            tracing::debug_span!("computing partial lagrange z col eq vals")
                .in_scope(|| Mle::blocking_partial_lagrange(&z_col));
        let partial_lagrange_z_col_eq = partial_lagrange_z_col_eq.guts().as_slice();

        let mut expected_branching_program_batched_eval =
            tracing::debug_span!("computing expected branching eval").in_scope(|| {
                (0..branching_program_evals.len())
                    .map(|i| {
                        let z_col_eq_val = partial_lagrange_z_col_eq[i];
                        let mut merged_prefix_sum = params.col_prefix_sums[i].clone();
                        let next_prefix_sum = params.col_prefix_sums[i + 1].clone();
                        merged_prefix_sum.extend(&next_prefix_sum);

                        let merge_prefix_sum_chunks =
                            merged_prefix_sum.values().chunks(chunk_size).map(|chunk| {
                                chunk
                                    .iter()
                                    .enumerate()
                                    .map(
                                        |(i, val)| {
                                            if val.is_one() {
                                                1 << (chunk.len() - i - 1)
                                            } else {
                                                0
                                            }
                                        },
                                    )
                                    .sum::<usize>()
                            });

                        let full_lagrange_eval = partial_lagranges
                            .iter()
                            .zip(merge_prefix_sum_chunks)
                            .map(|(partial_lagrange, merge_prefix_sum_chunk)| {
                                partial_lagrange.guts().as_slice()[merge_prefix_sum_chunk]
                            })
                            .product::<C::EF>();

                        z_col_eq_val * full_lagrange_eval
                    })
                    .sum::<C::EF>()
            });

        expected_branching_program_batched_eval *=
            branching_program.eval(&first_half_z_index, &second_half_z_index);

        assert!(
            expected_branching_program_batched_eval
                == branching_program_evals_proof.point_and_eval.1
        );

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
