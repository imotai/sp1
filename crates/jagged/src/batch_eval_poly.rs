use itertools::Itertools;
use slop_algebra::{interpolate_univariate_polynomial, Field, UnivariatePolynomial};
use slop_alloc::Buffer;
use slop_multilinear::{Mle, Point};
use slop_sumcheck::{ComponentPoly, SumcheckPoly, SumcheckPolyBase, SumcheckPolyFirstRound};

use crate::poly::HPoly;

struct BatchEvalPoly<K: Field + 'static> {
    h_poly: HPoly<K>,
    rho: Point<K>,
    merged_prefix_sums: Vec<Point<K>>,
    powers_of_beta: Vec<K>,
    is_empty_col: Vec<bool>,
    round_num: usize,
}

impl<K: Field + 'static> BatchEvalPoly<K> {
    pub fn new(
        z_row: Point<K>,
        z_index: Point<K>,
        prefix_sums: Vec<usize>,
        beta: K,
        round_num: usize,
    ) -> Self {
        let log_m = z_index.dimension();
        let col_prefix_sums: Vec<Point<K>> =
            prefix_sums.iter().map(|&x| Point::from_usize(x, log_m + 1)).collect();
        let next_col_prefix_sums: Vec<Point<K>> =
            prefix_sums.iter().skip(1).map(|&x| Point::from_usize(x, log_m + 1)).collect();

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

        let num_polys = merged_prefix_sums.len();
        let powers_of_beta = beta.powers().take(num_polys).collect();

        Self {
            h_poly: HPoly::new(z_row, z_index),
            rho: Point::new(Buffer::default()),
            merged_prefix_sums,
            powers_of_beta,
            round_num,
            is_empty_col,
        }
    }
}

impl<K: Field + 'static> SumcheckPolyBase for BatchEvalPoly<K> {
    fn num_variables(&self) -> u32 {
        (self.h_poly.num_vars as u32 + 1) * 2
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
        rho.add_dimension_back(alpha);
        Self {
            h_poly: self.h_poly,
            rho,
            merged_prefix_sums: self.merged_prefix_sums,
            powers_of_beta: self.powers_of_beta,
            round_num: self.round_num + 1,
            is_empty_col: self.is_empty_col,
        }
    }

    async fn sum_as_poly_in_last_variable(&self, claim: Option<K>) -> UnivariatePolynomial<K> {
        // We have a poly that has degree 2, so to produce the univariate poly we need to evaluate
        // at 3 points.
        // We can get a point from the eq root, and since f(0) + f(1) == claim, we can just
        // calculate f(0) and infer f(1).

        let (y_0, y_2) = self
            .powers_of_beta
            .iter()
            .zip_eq(self.merged_prefix_sums.iter())
            .map(|(beta, merged_prefix_sum)| {
                let func = |x: K| -> K {
                    let (eq_prefix_sum, mut h_prefix_sum) =
                        merged_prefix_sum.split_at(self.round_num + 1);

                    let mut rho = self.rho.clone();
                    rho.add_dimension_back(x);
                    let eq_eval = Mle::full_lagrange_eval(&eq_prefix_sum, &rho);

                    h_prefix_sum.add_dimension_back(x);
                    h_prefix_sum.extend(&self.rho);
                    let num_dimensions = h_prefix_sum.dimension();
                    let (h_left, h_right) = h_prefix_sum.split_at(num_dimensions / 2);

                    let h_eval = self.h_poly.eval(&h_left, &h_right);
                    *beta * h_eval * eq_eval
                };

                (func(K::zero()), func(K::two()))
            })
            .fold((K::zero(), K::zero()), |(y_0, y_1), (y_0_i, y_1_i)| (y_0 + y_0_i, y_1 + y_1_i));

        let y_1 = claim.unwrap() - y_0;

        interpolate_univariate_polynomial(&[K::zero(), K::one(), K::two()], &[y_0, y_1, y_2])
    }
}

#[cfg(test)]
mod tests {

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
        let row_counts = [12, 1, 2, 1, 17, 0];

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

        let log_max_row_count = 7;

        let z_row: Point<EF> = (0..log_max_row_count).map(|_| rng.gen::<EF>()).collect();
        let z_index: Point<EF> = (0..log_m).map(|_| rng.gen::<EF>()).collect();
        let beta = rng.gen::<EF>();
        let beta_powers = beta.powers();

        let h_poly = HPoly::new(z_row.clone(), z_index.clone());
        let expected_sum = prefix_sums
            .windows(2)
            .zip(beta_powers)
            .map(|(prefix_sum_range, beta_power)| {
                let lower = Point::from_usize(prefix_sum_range[0], log_m + 1);
                let upper = Point::from_usize(prefix_sum_range[1], log_m + 1);

                let h_eval = h_poly.eval(&lower, &upper);
                beta_power * h_eval
            })
            .sum::<EF>();

        let batch_eval_poly = BatchEvalPoly::new(z_row, z_index, prefix_sums, beta, 0);

        let default_perm = my_bb_16_perm();
        let mut challenger = DuplexChallenger::<BabyBear, Perm, 16, 8>::new(default_perm.clone());

        let (sc_proof, _) = reduce_sumcheck_to_evaluation(
            vec![batch_eval_poly],
            &mut challenger,
            vec![expected_sum],
            1,
            EF::zero(),
        )
        .await;

        let mut challenger = DuplexChallenger::<BabyBear, Perm, 16, 8>::new(default_perm);
        partially_verify_sumcheck_proof(&sc_proof, &mut challenger).unwrap();

        let (first_half_point, second_half_point) =
            sc_proof.point_and_eval.0.split_at(sc_proof.point_and_eval.0.dimension() / 2);

        let eval = h_poly.eval(&first_half_point, &second_half_point);
        assert!(eval == sc_proof.point_and_eval.1);
    }
}
