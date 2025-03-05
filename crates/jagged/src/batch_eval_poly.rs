use itertools::Itertools;
use slop_algebra::{Field, UnivariatePolynomial};
use slop_alloc::Buffer;
use slop_multilinear::Point;
use slop_sumcheck::{ComponentPoly, SumcheckPoly, SumcheckPolyBase};

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
            prefix_sums.iter().skip(1).map(|&x| Point::from_usize(x - 1, log_m + 1)).collect();

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
        self.h_poly.num_vars as u32 * 2
    }
}

impl<K: Field + 'static> ComponentPoly<K> for BatchEvalPoly<K> {
    async fn get_component_poly_evals(&self) -> Vec<K> {
        unimplemented!()
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
            .zip_eq(self.is_empty_col.iter())
            .zip_eq(self.merged_prefix_sums.iter())
            .map(|((beta, is_empty), merged_prefix_sum)| {
                let func = |x: K| -> K {
                    let (mut h_prefix_sum, eq_prefix_sum) =
                        merged_prefix_sum.split_at(self.round_num + 1);

                    let mut rho = self.rho.clone();
                    rho.add_dimension_back(x);
                    let eq_eval = full_lagrange_eval(eq_prefix_sum, rho);

                    h_prefix_sum.add_dimension_back(s);
                    h_prefix_sum.extend(&self.rho);
                    let num_dimensions = h_prefix_sum.dimension();
                    assert!(num_dimensions == self.h_poly.num_vars * 2);
                    let (h_left, h_right) = h_prefix_sum.split_at(num_dimensions / 2);

                    let h_eval = self.h_poly.eval(&h_left, &h_right);
                    beta * h_eval * eq_eval
                };

                (func(K::zero()), func(K::two()))
            })
            .fold((K::zero(), K::zero()), |(y_0, y_1), (y_0_i, y_1_i)| (y_0 + y_0_i, y_1 + y_1_i));

        let y_1 = *claim.unwrap() - y_0;

        interpolate_univariate_polynomial(&[K::zero(), K::one(), K::two()], &[y_0, y_1, y_2])
    }
}
