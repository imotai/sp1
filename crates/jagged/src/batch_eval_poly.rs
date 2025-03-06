use itertools::Itertools;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};
use slop_algebra::{interpolate_univariate_polynomial, Field, UnivariatePolynomial};
use slop_alloc::Buffer;
use slop_multilinear::{Mle, Point};
use slop_sumcheck::{ComponentPoly, SumcheckPoly, SumcheckPolyBase, SumcheckPolyFirstRound};

use crate::poly::HPoly;

pub(crate) struct BatchEvalPoly<K: Field + 'static> {
    h_poly: HPoly<K>,
    rho: Point<K>,
    z_col: Point<K>,
    merged_prefix_sums: Vec<Point<K>>,
    is_empty_col: Vec<bool>,
    z_col_eq_vals: Vec<K>,
    round_num: usize,
    eq_adjustments: Vec<K>,
    half: K,
}

impl<K: Field + 'static> BatchEvalPoly<K> {
    pub fn new(
        z_row: Point<K>,
        z_col: Point<K>,
        z_index: Point<K>,
        prefix_sums: Vec<usize>,
    ) -> Self {
        let log_m = z_index.dimension();
        let col_prefix_sums: Vec<Point<K>> =
            prefix_sums.iter().map(|&x| Point::from_usize(x, log_m)).collect();
        let next_col_prefix_sums: Vec<Point<K>> =
            prefix_sums.iter().skip(1).map(|&x| Point::from_usize(x, log_m)).collect();

        let (merged_prefix_sums, is_empty_col): (Vec<Point<K>>, Vec<bool>) = col_prefix_sums
            .iter()
            .zip(next_col_prefix_sums.iter())
            .map(|(a, b)| {
                let mut a = a.clone();
                let is_same = *a == **b;
                a.extend(b);
                (a, is_same)
            })
            .unzip();

        let num_polys = prefix_sums.len();
        let z_col_eq_vals = (0..num_polys - 1)
            .map(|c| {
                let c_point = Point::from_usize(c, z_col.dimension());
                Mle::full_lagrange_eval(&c_point, &z_col)
            })
            .collect_vec();

        // Condense the merged_prefix_sums and z_col_eq_vals.
        let (merged_prefix_sums, z_col_eq_vals): (Vec<Point<K>>, Vec<K>) = merged_prefix_sums
            .iter()
            .zip_eq(z_col_eq_vals.iter())
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
            h_poly: HPoly::new(z_row, z_index),
            rho: Point::new(Buffer::default()),
            z_col,
            merged_prefix_sums,
            round_num: 0,
            is_empty_col,
            z_col_eq_vals,
            eq_adjustments: vec![K::one(); merged_prefix_sums_len],
            half: K::two().inverse(),
        }
    }
}

impl<K: Field + 'static> SumcheckPolyBase for BatchEvalPoly<K> {
    fn num_variables(&self) -> u32 {
        self.h_poly.num_vars as u32 * 2
    }
}

impl<K: Field + 'static> ComponentPoly<K> for BatchEvalPoly<K> {
    async fn get_component_poly_evals(&self) -> Vec<K> {
        Vec::new()
    }
}

impl<K: Field + 'static> SumcheckPolyFirstRound<K> for BatchEvalPoly<K> {
    type NextRoundPoly = BatchEvalPoly<K>;

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

impl<K: Field + 'static> SumcheckPoly<K> for BatchEvalPoly<K> {
    async fn fix_last_variable(self, alpha: K) -> Self {
        let mut rho = self.rho.clone();

        let merged_prefix_sum_dim = self.merged_prefix_sums[0].dimension();

        let new_eq_adjustments = self
            .merged_prefix_sums
            .iter()
            .zip_eq(self.eq_adjustments.iter())
            .map(|(merged_prefix_sum, eq_adjustment)| {
                let x_i = merged_prefix_sum
                    .values()
                    .get(merged_prefix_sum_dim - 1 - self.round_num)
                    .unwrap();
                *eq_adjustment * ((alpha * *x_i) + (K::one() - alpha) * (K::one() - *x_i))
            })
            .collect_vec();

        rho.add_dimension(alpha);
        Self {
            h_poly: self.h_poly,
            rho,
            merged_prefix_sums: self.merged_prefix_sums,
            z_col: self.z_col,
            round_num: self.round_num + 1,
            is_empty_col: self.is_empty_col,
            z_col_eq_vals: self.z_col_eq_vals,
            eq_adjustments: new_eq_adjustments,
            half: self.half,
        }
    }

    async fn sum_as_poly_in_last_variable(&self, claim: Option<K>) -> UnivariatePolynomial<K> {
        // We have a poly that has degree 2, so to produce the univariate poly we need to evaluate
        // at 3 points.
        // We can get a point from the eq root, and since f(0) + f(1) == claim, we can just
        // calculate f(0) and infer f(1).
        let chunk_size = std::cmp::max(self.z_col_eq_vals.len() / num_cpus::get(), 1);

        let (y_0, y_half) = self
            .merged_prefix_sums
            .par_chunks(chunk_size)
            .zip_eq(self.z_col_eq_vals.par_chunks(chunk_size))
            .zip_eq(self.eq_adjustments.par_chunks(chunk_size))
            .map(|((merged_prefix_sum_chunk, z_col_eq_val_chunk), eq_adjustment_chunk)| {
                merged_prefix_sum_chunk
                    .iter()
                    .zip_eq(z_col_eq_val_chunk.iter())
                    .zip_eq(eq_adjustment_chunk.iter())
                    .map(|((merged_prefix_sum, z_col_eq_val), eq_adjustment)| {
                        let (h_prefix_sum, eq_prefix_sum) = merged_prefix_sum
                            .split_at(merged_prefix_sum.dimension() - self.round_num - 1);
                        let eq_prefix_sum_value = *eq_prefix_sum.values()[0];
                        let eq_val_0 = K::one() - eq_prefix_sum_value;
                        let eq_val_half = K::two().inverse();

                        let func = |x: K, eq_val: K| -> K {
                            let eq_eval = *eq_adjustment * eq_val;

                            let mut h_prefix_sum = h_prefix_sum.clone();
                            h_prefix_sum.add_dimension_back(x);
                            h_prefix_sum.extend(&self.rho);
                            let num_dimensions = h_prefix_sum.dimension();
                            let (h_left, h_right) = h_prefix_sum.split_at(num_dimensions / 2);

                            let h_eval = self.h_poly.eval(&h_left, &h_right);

                            *z_col_eq_val * h_eval * eq_eval
                        };

                        (func(K::zero(), eq_val_0), func(K::two().inverse(), eq_val_half))
                    })
                    .fold((K::zero(), K::zero()), |(y_0, y_2), (y_0_i, y_2_i)| {
                        (y_0 + y_0_i, y_2 + y_2_i)
                    })
            })
            .reduce(
                || (K::zero(), K::zero()),
                |(y_0, y_2), (y_0_i, y_2_i)| (y_0 + y_0_i, y_2 + y_2_i),
            );

        let y_1 = claim.unwrap() - y_0;

        interpolate_univariate_polynomial(
            &[K::zero(), K::one(), K::two().inverse()],
            &[y_0, y_1, y_half],
        )
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
    async fn test_batch_eval_poly() {
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

        let h_poly = HPoly::new(z_row.clone(), z_index.clone());

        let prover_params =
            JaggedLittlePolynomialProverParams::new(row_counts.to_vec(), log_max_row_count);
        let verifier_params = prover_params.clone().into_verifier_params();
        let expected_sum =
            verifier_params.full_jagged_little_polynomial_evaluation(&z_row, &z_col, &z_index);

        let batch_eval_poly =
            BatchEvalPoly::new(z_row.clone(), z_col.clone(), z_index.clone(), prefix_sums.clone());

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
