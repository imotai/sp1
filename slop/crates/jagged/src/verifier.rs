use itertools::Itertools;
use serde::{Deserialize, Serialize};
use slop_algebra::AbstractField;
use slop_challenger::{FieldChallenger, IopCtx};
use slop_multilinear::{full_geq, Evaluations, Mle, MultilinearPcsVerifier, Point};
use slop_sumcheck::{partially_verify_sumcheck_proof, PartialSumcheckProof, SumcheckError};
use std::fmt::Debug;
use thiserror::Error;

use crate::{
    JaggedConfig, JaggedError, JaggedEvalSumcheckConfig, JaggedLittlePolynomialVerifierParams,
    JaggedSumcheckEvalProof,
};

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct JaggedPcsProof<GC: IopCtx, C: JaggedConfig<GC>> {
    pub pcs_proof: <C::PcsVerifier as MultilinearPcsVerifier<GC>>::Proof,
    pub sumcheck_proof: PartialSumcheckProof<GC::EF>,
    pub jagged_eval_proof: JaggedSumcheckEvalProof<GC::EF>,
    pub params: JaggedLittlePolynomialVerifierParams<GC::F>,
    pub added_columns: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct JaggedPcsVerifier<GC: IopCtx, C: JaggedConfig<GC>> {
    pub pcs_verifier: C::PcsVerifier,
    pub max_log_row_count: usize,
}

#[derive(Debug, Error)]
pub enum JaggedPcsVerifierError<EF, PcsError> {
    #[error("sumcheck claim mismatch: {0} != {1}")]
    SumcheckClaimMismatch(EF, EF),
    #[error("sumcheck proof verification failed: {0}")]
    SumcheckError(SumcheckError),
    #[error("jagged evaluation proof verification failed")]
    JaggedEvalProofVerificationFailed,
    #[error("dense pcs verification failed")]
    DensePcsVerificationFailed(#[from] PcsError),
    #[error("booleanity check failed")]
    BooleanityCheckFailed,
    #[error("montonicity check failed")]
    MonotonicityCheckFailed,
    #[error("proof has incorrect shape")]
    IncorrectShape,
    #[error("invalid prefix sums")]
    InvalidPrefixSums,
}

impl<GC: IopCtx, C: JaggedConfig<GC>> JaggedPcsVerifier<GC, C> {
    pub fn challenger(&self) -> GC::Challenger {
        self.pcs_verifier.default_challenger()
    }

    fn verify_trusted_evaluations(
        &self,
        commitments: &[GC::Digest],
        point: Point<GC::EF>,
        evaluation_claims: &[Evaluations<GC::EF>],
        proof: &JaggedPcsProof<GC, C>,
        insertion_points: &[usize],
        challenger: &mut GC::Challenger,
    ) -> Result<(), JaggedPcsVerifierError<GC::EF, JaggedError<GC, C>>> {
        let JaggedPcsProof { pcs_proof, sumcheck_proof, jagged_eval_proof, params, added_columns } =
            proof;

        if params.col_prefix_sums.is_empty() || params.max_log_row_count != self.max_log_row_count {
            return Err(JaggedPcsVerifierError::IncorrectShape);
        }

        let num_col_variables = (params.col_prefix_sums.len() - 1).next_power_of_two().ilog2();
        let z_col = (0..num_col_variables)
            .map(|_| challenger.sample_ext_element::<GC::EF>())
            .collect::<Point<_>>();

        let z_row = point;

        if z_row.dimension() != self.max_log_row_count {
            return Err(JaggedPcsVerifierError::IncorrectShape);
        }

        // Collect the claims for the different polynomials.
        let mut column_claims =
            evaluation_claims.iter().flatten().flatten().copied().collect::<Vec<_>>();

        if insertion_points.len() != added_columns.len()
            || insertion_points.len() != commitments.len()
            || insertion_points.len() != evaluation_claims.len()
        /*|| insertion_points.len() != proof.pcs_proof.batch_evaluations.len()*/
        {
            return Err(JaggedPcsVerifierError::IncorrectShape);
        }

        // For each commit, the stacked PCS needed a commitment to a vector of length a multiple of
        // 1 << self.pcs.log_stacking_height, and this is achieved by adding columns of zeroes after
        // the "real" columns. We insert these "artificial" zeroes into the evaluation claims on the
        // verifier side.
        for (insertion_point, num_added_columns) in
            insertion_points.iter().rev().zip_eq(added_columns.iter().rev())
        {
            for _ in 0..*num_added_columns {
                column_claims.insert(*insertion_point, GC::EF::zero());
            }
        }

        if params.col_prefix_sums.len() != column_claims.len() + 1 {
            return Err(JaggedPcsVerifierError::IncorrectShape);
        }

        let prefix_sums: Vec<u32> = params
            .col_prefix_sums
            .iter()
            .map(|bits| {
                bits.iter()
                    .fold(0u32, |acc, &bit| (acc << 1) | if bit == GC::F::one() { 1 } else { 0 })
            })
            .collect();

        // Validate monotonicity and bounds
        for window in prefix_sums.windows(2) {
            let (sum, next_sum) = (window[0], window[1]);
            let max_increment = 1u32 << self.max_log_row_count;

            if sum > next_sum || next_sum > sum.saturating_add(max_increment) {
                return Err(JaggedPcsVerifierError::InvalidPrefixSums);
            }
        }

        // Validate stacked columns alignment
        let mut prefix_sum_index = 0;
        // let mut num_stacked_columns = 0;

        for i in 0..evaluation_claims.len() {
            let count_polys = |evals: &[slop_multilinear::MleEval<_>]| {
                evals.iter().map(slop_multilinear::MleEval::num_polynomials).sum::<usize>()
            };

            prefix_sum_index += count_polys(&evaluation_claims[i].round_evaluations);
            prefix_sum_index += added_columns[i];

            // TODO: Put these in stacked PCS.
            // num_stacked_columns += count_polys(&pcs_proof.batch_evaluations[i].round_evaluations);

            // let expected = (num_stacked_columns as u32)
            //     .saturating_mul(1u32 << self.stacked_pcs_verifier.log_stacking_height);

            // if prefix_sums[prefix_sum_index] != expected {
            //     return Err(JaggedPcsVerifierError::InvalidPrefixSums);
            // }
        }
        if prefix_sum_index != params.col_prefix_sums.len() - 1 {
            return Err(JaggedPcsVerifierError::IncorrectShape);
        }

        // Pad the column claims to the next power of two.
        column_claims.resize(column_claims.len().next_power_of_two(), GC::EF::zero());

        if (1 << z_col.len()) != column_claims.len() {
            return Err(JaggedPcsVerifierError::IncorrectShape);
        }

        let column_mle = Mle::from(column_claims);
        let sumcheck_claim = column_mle.blocking_eval_at(&z_col)[0];

        if sumcheck_claim != sumcheck_proof.claimed_sum {
            return Err(JaggedPcsVerifierError::SumcheckClaimMismatch(
                sumcheck_claim,
                sumcheck_proof.claimed_sum,
            ));
        }

        partially_verify_sumcheck_proof(
            sumcheck_proof,
            challenger,
            params.col_prefix_sums[0].len() - 1,
            2,
        )
        .map_err(JaggedPcsVerifierError::SumcheckError)?;

        // Check the booleanity of the column prefix sums.
        for t_col in params.col_prefix_sums.iter() {
            for &elem in t_col.iter() {
                if elem * (GC::F::one() - elem) != GC::F::zero() {
                    return Err(JaggedPcsVerifierError::BooleanityCheckFailed);
                }
            }
        }

        for (t_col, next_t_col) in
            params.col_prefix_sums.iter().zip(params.col_prefix_sums.iter().skip(1))
        {
            // We bound the prefix sums to be < 2^30. While this function is implemented with
            // `C::F` being any field, this function is intended for use with primes larger than
            // `2^30`. We recommend using this function for Mersenne31, BabyBear, KoalaBear.
            if t_col.len() != next_t_col.len() || t_col.len() >= 31 || t_col.is_empty() {
                return Err(JaggedPcsVerifierError::IncorrectShape);
            }
            // Check monotonicity of the column prefix sums.
            if full_geq(t_col, next_t_col) != GC::F::one() {
                return Err(JaggedPcsVerifierError::MonotonicityCheckFailed);
            }
        }

        let jagged_eval = JaggedEvalSumcheckConfig::jagged_evaluation(
            params,
            &z_row,
            &z_col,
            &sumcheck_proof.point_and_eval.0,
            jagged_eval_proof,
            challenger,
        )
        .map_err(|_| JaggedPcsVerifierError::JaggedEvalProofVerificationFailed)?;

        // Compute the expected evaluation of the dense trace polynomial.
        let expected_eval = sumcheck_proof.point_and_eval.1 / jagged_eval;

        // Verify the evaluation proof using the (dense) stacked PCS verifier.
        let evaluation_point = sumcheck_proof.point_and_eval.0.clone();
        self.pcs_verifier
            .verify_trusted_evaluation(
                commitments,
                evaluation_point,
                expected_eval,
                pcs_proof,
                challenger,
            )
            .map_err(JaggedPcsVerifierError::DensePcsVerificationFailed)
    }
}

pub struct MachineJaggedPcsVerifier<'a, GC: IopCtx, C: JaggedConfig<GC>> {
    pub jagged_pcs_verifier: &'a JaggedPcsVerifier<GC, C>,
    pub column_counts_by_round: Vec<Vec<usize>>,
}

impl<'a, GC: IopCtx, C: JaggedConfig<GC>> MachineJaggedPcsVerifier<'a, GC, C> {
    pub fn new(
        jagged_pcs_verifier: &'a JaggedPcsVerifier<GC, C>,
        column_counts_by_round: Vec<Vec<usize>>,
    ) -> Self {
        Self { jagged_pcs_verifier, column_counts_by_round }
    }

    pub fn verify_trusted_evaluations(
        &self,
        commitments: &[GC::Digest],
        point: Point<GC::EF>,
        evaluation_claims: &[Evaluations<GC::EF>],
        proof: &JaggedPcsProof<GC, C>,
        challenger: &mut GC::Challenger,
    ) -> Result<(), JaggedPcsVerifierError<GC::EF, JaggedError<GC, C>>> {
        if evaluation_claims.len() != self.column_counts_by_round.len() {
            return Err(JaggedPcsVerifierError::IncorrectShape);
        }
        for (claims, expected_counts) in
            evaluation_claims.iter().zip_eq(&self.column_counts_by_round)
        {
            let claim_count: usize = claims
                .round_evaluations
                .iter()
                .map(slop_multilinear::MleEval::num_polynomials)
                .sum();

            let expected_count: usize = expected_counts.iter().sum();

            if claim_count != expected_count {
                return Err(JaggedPcsVerifierError::IncorrectShape);
            }
        }
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
