use std::{convert::Infallible, error::Error, fmt::Debug, future::Future, marker::PhantomData};

use itertools::Itertools;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use slop_algebra::{
    interpolate_univariate_polynomial, ExtensionField, Field, UnivariatePolynomial,
};
use slop_alloc::Buffer;
use slop_challenger::FieldChallenger;
use slop_multilinear::{Mle, Point};
use slop_sumcheck::{
    partially_verify_sumcheck_proof, reduce_sumcheck_to_evaluation, ComponentPoly,
    PartialSumcheckProof, SumcheckError, SumcheckPoly, SumcheckPolyBase, SumcheckPolyFirstRound,
};
use slop_utils::log2_ceil_usize;
use thiserror::Error;

use crate::{
    poly::BranchingProgram, JaggedLittlePolynomialProverParams,
    JaggedLittlePolynomialVerifierParams,
};

pub trait JaggedEvalConfig<F: Field, Challenger>:
    'static + Send + Sync + Serialize + DeserializeOwned + std::fmt::Debug + Clone
{
    type JaggedEvalProof: 'static + Debug + Clone + Send + Sync + Serialize + DeserializeOwned;

    type JaggedEvalError: Error + 'static + Send + Sync;

    fn jagged_evaluation(
        &self,
        params: &JaggedLittlePolynomialVerifierParams<F>,
        z_row: &Point<F>,
        z_col: &Point<F>,
        z_trace: &Point<F>,
        proof: &Self::JaggedEvalProof,
        challenger: &mut Challenger,
    ) -> Result<F, Self::JaggedEvalError>;
}

pub trait JaggedEvalProver<F: Field, Challenger>:
    'static + Send + Sync + std::fmt::Debug + Clone
{
    type EvalProof: 'static + Debug + Clone + Send + Sync + Serialize + DeserializeOwned;
    type EvalConfig: JaggedEvalConfig<F, Challenger, JaggedEvalProof = Self::EvalProof>;

    fn prove_jagged_evaluation(
        &self,
        params: &JaggedLittlePolynomialProverParams,
        z_row: &Point<F>,
        z_col: &Point<F>,
        z_trace: &Point<F>,
        challenger: &mut Challenger,
    ) -> impl Future<Output = Self::EvalProof> + Send;
}

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct TrivialJaggedEvalConfig;

impl<F: Field, C> JaggedEvalConfig<F, C> for TrivialJaggedEvalConfig {
    type JaggedEvalProof = ();
    type JaggedEvalError = Infallible;

    fn jagged_evaluation(
        &self,
        params: &JaggedLittlePolynomialVerifierParams<F>,
        z_row: &Point<F>,
        z_col: &Point<F>,
        z_trace: &Point<F>,
        _proof: &Self::JaggedEvalProof,
        _challenger: &mut C,
    ) -> Result<F, Self::JaggedEvalError> {
        let (result, _) = params.full_jagged_little_polynomial_evaluation(z_row, z_col, z_trace);
        Ok(result)
    }
}

impl<F: Field, C: Send + Sync> JaggedEvalProver<F, C> for TrivialJaggedEvalConfig {
    type EvalProof = ();
    type EvalConfig = TrivialJaggedEvalConfig;
    async fn prove_jagged_evaluation(
        &self,
        _params: &JaggedLittlePolynomialProverParams,
        _z_row: &Point<F>,
        _z_col: &Point<F>,
        _z_trace: &Point<F>,
        _challenger: &mut C,
    ) -> Self::EvalProof {
    }
}

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

impl<F, EF, Challenger> JaggedEvalConfig<EF, Challenger> for JaggedEvalSumcheckConfig<F>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F>,
{
    type JaggedEvalProof = JaggedSumcheckEvalProof<EF>;
    type JaggedEvalError = JaggedEvalSumcheckError<EF>;

    fn jagged_evaluation(
        &self,
        params: &JaggedLittlePolynomialVerifierParams<EF>,
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
        let mut prev_merged_prefix_sum = Point::<EF>::default();
        let mut prev_full_lagrange_eval = EF::zero();
        let mut jagged_eval_sc_expected_eval = current_column_prefix_sums
            .zip(next_column_prefix_sums)
            .zip(z_col_partial_lagrange.iter())
            .map(|((current_column_prefix_sum, next_column_prefix_sum), z_col_eq_val)| {
                let mut merged_prefix_sum = current_column_prefix_sum.clone();
                merged_prefix_sum.extend(next_column_prefix_sum);

                let full_lagrange_eval =
                    if prev_merged_prefix_sum == merged_prefix_sum && is_first_column {
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
            return Err(JaggedEvalSumcheckError::JaggedEvaluationFailed(
                jagged_eval_sc_expected_eval,
                partial_sumcheck_proof.point_and_eval.1,
            ));
        }

        Ok(jagged_eval)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct JaggedEvalSumcheckProver<F>(PhantomData<F>);

impl<F, EF, Challenger> JaggedEvalProver<EF, Challenger> for JaggedEvalSumcheckProver<F>
where
    F: Field,
    EF: ExtensionField<F>,
    Challenger: FieldChallenger<F> + Send + Sync,
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
        let jagged_eval_sc_poly = JaggedEvalSumcheckPoly::new(
            z_row.clone(),
            z_col.clone(),
            z_trace.clone(),
            params.col_prefix_sums_usize.clone(),
        );

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

/// A struct that represents the polynomial that is used to evaluate the sumcheck.
pub(crate) struct JaggedEvalSumcheckPoly<K: Field> {
    /// The branching program that has a constant z_row and z_index.
    h_poly: BranchingProgram<K>,
    /// The random point generated during the sumcheck proving time.
    rho: Point<K>,
    /// The z_col point.
    z_col: Point<K>,
    /// This is a concatenation of the bitstring of t_c and t_{c+1} for every column c.
    merged_prefix_sums: Vec<Point<K>>,
    /// The evaluations of the z_col at the merged prefix sums.
    z_col_eq_vals: Vec<K>,
    /// The sumcheck round number that this poly is used in.
    round_num: usize,
    /// The intermediate full evaluations of the eq polynomials.
    intermediate_eq_full_evals: Vec<K>,
    /// The half value.
    half: K,
}

impl<K: Field + 'static> JaggedEvalSumcheckPoly<K> {
    pub fn new(
        z_row: Point<K>,
        z_col: Point<K>,
        z_index: Point<K>,
        prefix_sums: Vec<usize>,
    ) -> Self {
        let log_m = log2_ceil_usize(*prefix_sums.last().unwrap());
        let col_prefix_sums: Vec<Point<K>> =
            prefix_sums.iter().map(|&x| Point::from_usize(x, log_m + 1)).collect();

        // Generate all of the merged prefix sums.
        let merged_prefix_sums = col_prefix_sums
            .windows(2)
            .map(|prefix_sums| {
                let mut merged_prefix_sum = prefix_sums[0].clone();
                merged_prefix_sum.extend(&prefix_sums[1]);
                merged_prefix_sum
            })
            .collect_vec();

        // Generate all of the z_col partial lagrange mle.
        let z_col_partial_lagrange = Mle::blocking_partial_lagrange(&z_col);

        // Condense the merged_prefix_sums and z_col_eq_vals for empty tables.
        let (merged_prefix_sums, z_col_eq_vals): (Vec<Point<K>>, Vec<K>) = merged_prefix_sums
            .iter()
            .zip(z_col_partial_lagrange.guts().as_slice())
            .group_by(|(merged_prefix_sum, _)| *merged_prefix_sum)
            .into_iter()
            .map(|(merged_prefix_sum, group)| {
                let group_elements =
                    group.into_iter().map(|(_, z_col_eq_val)| *z_col_eq_val).collect_vec();
                (merged_prefix_sum.clone(), group_elements.into_iter().sum::<K>())
            })
            .unzip();

        let merged_prefix_sums_len = merged_prefix_sums.len();
        assert!(merged_prefix_sums_len == z_col_eq_vals.len());

        Self {
            h_poly: BranchingProgram::new(z_row, z_index),
            rho: Point::new(Buffer::default()),
            z_col,
            merged_prefix_sums,
            round_num: 0,
            z_col_eq_vals,
            intermediate_eq_full_evals: vec![K::one(); merged_prefix_sums_len],
            half: K::two().inverse(),
        }
    }

    #[inline]
    fn eval(
        &self,
        x: K,
        merged_prefix_sum: &Point<K>,
        z_col_eq_val: K,
        intermediate_eq_full_eval: K,
    ) -> K {
        // We want to calculate eq(z_col, col_idx) * eq(x_1, (x, rho)) * h(x_2, x, rho) where
        // x_1 || x_2 = merged_prefix_sum and rho is the sumcheck random point.  Note that the
        // eq(x_col, col_idx) is already computed as `z_col_eq_val` and all but one term of eq(x_i, (x, rho))
        // is computed as `intermediate_eq_full_eval`.

        // Split the merged prefix sum so that x_1 || x_2 = merged_prefix_sum.
        let (h_prefix_sum, eq_prefix_sum) =
            merged_prefix_sum.split_at(merged_prefix_sum.dimension() - self.round_num - 1);

        // Compute the remaining eq term for `eq(x_i, (x, rho))`.
        let eq_val = if x == K::zero() {
            K::one() - *eq_prefix_sum.values()[0]
        } else if x == self.half {
            self.half
        } else {
            unreachable!("x must be 0 or 1/2")
        };

        // Compute full eval of eq(x_i, (x, rho))
        let eq_eval = intermediate_eq_full_eval * eq_val;

        // Compute eval of h(x_2, x, rho).
        let mut h_prefix_sum = h_prefix_sum.clone();
        h_prefix_sum.add_dimension_back(x);
        h_prefix_sum.extend(&self.rho);
        let num_dimensions = h_prefix_sum.dimension();
        let (h_left, h_right) = h_prefix_sum.split_at(num_dimensions / 2);
        let h_eval = self.h_poly.eval(&h_left, &h_right);

        z_col_eq_val * h_eval * eq_eval
    }
}

impl<K: Field> SumcheckPolyBase for JaggedEvalSumcheckPoly<K> {
    fn num_variables(&self) -> u32 {
        self.merged_prefix_sums.first().unwrap().dimension() as u32
    }
}

impl<K: Field> ComponentPoly<K> for JaggedEvalSumcheckPoly<K> {
    async fn get_component_poly_evals(&self) -> Vec<K> {
        Vec::new()
    }
}

impl<K: Field> SumcheckPolyFirstRound<K> for JaggedEvalSumcheckPoly<K> {
    type NextRoundPoly = JaggedEvalSumcheckPoly<K>;

    async fn fix_t_variables(self, alpha: K, _t: usize) -> Self::NextRoundPoly {
        self.fix_last_variable(alpha).await
    }

    async fn sum_as_poly_in_last_t_variables(
        &self,
        claim: Option<K>,
        t: usize,
    ) -> UnivariatePolynomial<K> {
        assert!(t == 1);
        self.sum_as_poly_in_last_variable(claim).await
    }
}

impl<K: Field> SumcheckPoly<K> for JaggedEvalSumcheckPoly<K> {
    async fn fix_last_variable(self, alpha: K) -> Self {
        // Add alpha to the rho point.
        let mut rho = self.rho.clone();
        rho.add_dimension(alpha);

        let merged_prefix_sum_dim = self.merged_prefix_sums[0].dimension();

        // Update the intermediate full eq evals.
        let updated_intermediate_eq_full_evals = self
            .merged_prefix_sums
            .iter()
            .zip_eq(self.intermediate_eq_full_evals.iter())
            .map(|(merged_prefix_sum, intermediate_eq_full_eval)| {
                let x_i = merged_prefix_sum
                    .values()
                    .get(merged_prefix_sum_dim - 1 - self.round_num)
                    .unwrap();
                *intermediate_eq_full_eval
                    * ((alpha * *x_i) + (K::one() - alpha) * (K::one() - *x_i))
            })
            .collect_vec();

        Self {
            h_poly: self.h_poly,
            rho,
            merged_prefix_sums: self.merged_prefix_sums,
            z_col: self.z_col,
            round_num: self.round_num + 1,
            z_col_eq_vals: self.z_col_eq_vals,
            intermediate_eq_full_evals: updated_intermediate_eq_full_evals,
            half: self.half,
        }
    }

    async fn sum_as_poly_in_last_variable(&self, claim: Option<K>) -> UnivariatePolynomial<K> {
        // We have a poly that has degree 2, so to produce the univariate poly we need to evaluate
        // at 3 points.  We calculate f(0) and f(1/2) and then infer f(1) from the claim.

        // Calculate the partition chunk size.
        let chunk_size = std::cmp::max(self.z_col_eq_vals.len() / num_cpus::get(), 1);

        // Compute the values at x = 0 and x = 1/2.
        let (y_0, y_half) = self
            .merged_prefix_sums
            .par_chunks(chunk_size)
            .zip_eq(self.z_col_eq_vals.par_chunks(chunk_size))
            .zip_eq(self.intermediate_eq_full_evals.par_chunks(chunk_size))
            .map(
                |(
                    (merged_prefix_sum_chunk, z_col_eq_val_chunk),
                    intermediate_eq_full_eval_chunk,
                )| {
                    merged_prefix_sum_chunk
                        .iter()
                        .zip_eq(z_col_eq_val_chunk.iter())
                        .zip_eq(intermediate_eq_full_eval_chunk.iter())
                        .map(|((merged_prefix_sum, z_col_eq_val), intermediate_eq_full_eval)| {
                            let y_0 = self.eval(
                                K::zero(),
                                merged_prefix_sum,
                                *z_col_eq_val,
                                *intermediate_eq_full_eval,
                            );
                            let y_half = self.eval(
                                self.half,
                                merged_prefix_sum,
                                *z_col_eq_val,
                                *intermediate_eq_full_eval,
                            );

                            (y_0, y_half)
                        })
                        .fold((K::zero(), K::zero()), |(y_0, y_2), (y_0_i, y_2_i)| {
                            (y_0 + y_0_i, y_2 + y_2_i)
                        })
                },
            )
            .reduce(
                || (K::zero(), K::zero()),
                |(y_0, y_2), (y_0_i, y_2_i)| (y_0 + y_0_i, y_2 + y_2_i),
            );

        // Infer the value at x = 1 from the claim.
        let y_1 = claim.unwrap() - y_0;

        interpolate_univariate_polynomial(&[K::zero(), K::one(), self.half], &[y_0, y_1, y_half])
    }
}

#[cfg(test)]
mod tests {

    use crate::JaggedLittlePolynomialProverParams;
    use rand::{thread_rng, Rng};
    use slop_algebra::{extension::BinomialExtensionField, AbstractField};
    use slop_baby_bear::BabyBear;
    use slop_challenger::DuplexChallenger;
    use slop_merkle_tree::{my_bb_16_perm, Perm};
    use slop_sumcheck::{partially_verify_sumcheck_proof, reduce_sumcheck_to_evaluation};
    use slop_utils::log2_ceil_usize;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[tokio::test]
    async fn test_jagged_eval_sumcheck() {
        let row_counts = [12, 1, 0, 0, 17, 0];

        let mut rng = thread_rng();

        let mut prefix_sums = row_counts
            .iter()
            .scan(0, |state, row_count| {
                let result = *state;
                *state += row_count;
                Some(result)
            })
            .collect::<Vec<_>>();

        prefix_sums.push(*prefix_sums.last().unwrap() + row_counts.last().unwrap());
        let log_m = log2_ceil_usize(*prefix_sums.last().unwrap());
        let merged_prefix_sums = prefix_sums
            .windows(2)
            .map(|x| {
                let mut merged_prefix_sum = Point::from_usize(x[0], log_m + 1);
                merged_prefix_sum.extend(&Point::from_usize(x[1], log_m + 1));
                merged_prefix_sum
            })
            .collect::<Vec<_>>();

        let log_max_row_count = 7;

        let z_row: Point<EF> = (0..log_max_row_count).map(|_| rng.gen::<EF>()).collect();
        let z_col: Point<EF> =
            (0..log2_ceil_usize(row_counts.len())).map(|_| rng.gen::<EF>()).collect();
        let z_index: Point<EF> = (0..log_m + 1).map(|_| rng.gen::<EF>()).collect();

        let z_col_eq_vals = (0..row_counts.len())
            .map(|c| {
                let c_point = Point::from_usize(c, z_col.dimension());
                Mle::full_lagrange_eval(&c_point, &z_col)
            })
            .collect_vec();

        let h_poly = BranchingProgram::new(z_row.clone(), z_index.clone());

        let prover_params =
            JaggedLittlePolynomialProverParams::new(row_counts.to_vec(), log_max_row_count);
        let verifier_params = prover_params.clone().into_verifier_params();
        let (expected_sum, _) =
            verifier_params.full_jagged_little_polynomial_evaluation(&z_row, &z_col, &z_index);

        let batch_eval_poly = JaggedEvalSumcheckPoly::new(
            z_row.clone(),
            z_col.clone(),
            z_index.clone(),
            prefix_sums.clone(),
        );

        let default_perm = my_bb_16_perm();
        let mut challenger = DuplexChallenger::<BabyBear, Perm, 16, 8>::new(default_perm.clone());

        let (sc_proof, _) = reduce_sumcheck_to_evaluation(
            vec![batch_eval_poly],
            &mut challenger,
            vec![expected_sum],
            1,
            EF::one(),
        )
        .await;

        assert!(sc_proof.claimed_sum == expected_sum);

        let mut challenger = DuplexChallenger::<BabyBear, Perm, 16, 8>::new(default_perm);
        partially_verify_sumcheck_proof(&sc_proof, &mut challenger).unwrap();

        let out_of_domain_point = sc_proof.point_and_eval.0;

        let expected_eval = merged_prefix_sums
            .iter()
            .zip(z_col_eq_vals.iter())
            .map(|(merged_prefix_sum, z_col_eq_val)| {
                let (lower, upper) =
                    out_of_domain_point.split_at(out_of_domain_point.dimension() / 2);
                let h_eval = h_poly.eval(&lower, &upper);
                *z_col_eq_val
                    * Mle::full_lagrange_eval(merged_prefix_sum, &out_of_domain_point)
                    * h_eval
            })
            .sum::<EF>();

        assert!(expected_eval == sc_proof.point_and_eval.1);
    }
}
