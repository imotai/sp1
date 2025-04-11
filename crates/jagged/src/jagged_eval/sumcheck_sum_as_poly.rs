use std::future::Future;
use std::sync::Arc;

use itertools::Itertools;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};
use slop_algebra::{ExtensionField, Field};
use slop_alloc::Backend;
use slop_multilinear::Point;

use crate::BranchingProgram;

/// A trait for the jagged assist's sum as poly.
pub trait JaggedAssistSumAsPoly<F: Field, EF: ExtensionField<F>, A: Backend>: Sized {
    fn new(
        z_row: Point<EF>,
        z_index: Point<EF>,
        merged_prefix_sums: Arc<Vec<Point<F>>>,
        z_col_eq_vals: Vec<EF>,
    ) -> impl Future<Output = Self> + Send;

    fn sum_as_poly(
        &self,
        round_num: usize,
        z_col_eq_vals: &[EF],
        intermediate_eq_full_evals: &[EF],
        rhos: &Point<EF>,
    ) -> impl Future<Output = (EF, EF)> + Send;
}

#[derive(Debug, Clone, Default)]
pub struct JaggedAssistSumAsPolyCPUImpl<F: Field, EF: ExtensionField<F>> {
    branching_program: BranchingProgram<EF>,
    merged_prefix_sums: Arc<Vec<Point<F>>>,
    half: EF,
}

impl<F: Field, EF: ExtensionField<F>> JaggedAssistSumAsPolyCPUImpl<F, EF> {
    #[inline]
    fn eval(
        &self,
        lambda: EF,
        round_num: usize,
        merged_prefix_sum: &Point<F>,
        z_col_eq_val: EF,
        intermediate_eq_full_eval: EF,
        rhos: &Point<EF>,
    ) -> EF {
        // We want to calculate eq(z_col, col_idx) * eq(x_1, (x, rho)) * h(x_2, x, rho) where
        // x_1 || x_2 = merged_prefix_sum and rho is the sumcheck random point.  Note that the
        // eq(x_col, col_idx) is already computed as `z_col_eq_val` and all but one term of eq(x_i, (x, rho))
        // is computed as `intermediate_eq_full_eval`.

        // Split the merged prefix sum so that x_1 || x_2 = merged_prefix_sum.
        let (h_prefix_sum, eq_prefix_sum) =
            merged_prefix_sum.split_at(merged_prefix_sum.dimension() - round_num - 1);

        // Compute the remaining eq term for `eq(x_i, (x, rho))`.
        let eq_val = if lambda == EF::zero() {
            EF::one() - *eq_prefix_sum.values()[0]
        } else if lambda == self.half {
            self.half
        } else {
            unreachable!("lambda must be 0 or 1/2")
        };

        // Compute full eval of eq(x_i, (x, rho))
        let eq_eval = intermediate_eq_full_eval * eq_val;

        // Compute eval of h(x_2, x, rho).
        let mut h_prefix_sum: Point<EF> =
            h_prefix_sum.to_vec().iter().map(|x| (*x).into()).collect::<Vec<_>>().into();
        h_prefix_sum.add_dimension_back(lambda);
        h_prefix_sum.extend(rhos);
        let num_dimensions = h_prefix_sum.dimension();
        let (h_left, h_right) = h_prefix_sum.split_at(num_dimensions / 2);
        let h_eval = self.branching_program.eval(&h_left, &h_right);

        z_col_eq_val * h_eval * eq_eval
    }
}

impl<F: Field, EF: ExtensionField<F>, A: Backend> JaggedAssistSumAsPoly<F, EF, A>
    for JaggedAssistSumAsPolyCPUImpl<F, EF>
{
    async fn new(
        z_row: Point<EF>,
        z_index: Point<EF>,
        merged_prefix_sums: Arc<Vec<Point<F>>>,
        _z_col_eq_vals: Vec<EF>,
    ) -> Self {
        let branching_program = BranchingProgram::new(z_row, z_index);

        Self { branching_program, merged_prefix_sums, half: EF::two().inverse() }
    }

    async fn sum_as_poly(
        &self,
        round_num: usize,
        z_col_eq_vals: &[EF],
        intermediate_eq_full_evals: &[EF],
        rhos: &Point<EF>,
    ) -> (EF, EF) {
        // Calculate the partition chunk size.
        let chunk_size = std::cmp::max(z_col_eq_vals.len() / num_cpus::get(), 1);

        // Compute the values at x = 0 and x = 1/2.
        let (y_0, y_half) = self
            .merged_prefix_sums
            .par_chunks(chunk_size)
            .zip_eq(z_col_eq_vals.par_chunks(chunk_size))
            .zip_eq(intermediate_eq_full_evals.par_chunks(chunk_size))
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
                                EF::zero(),
                                round_num,
                                merged_prefix_sum,
                                *z_col_eq_val,
                                *intermediate_eq_full_eval,
                                rhos,
                            );
                            let y_half = self.eval(
                                self.half,
                                round_num,
                                merged_prefix_sum,
                                *z_col_eq_val,
                                *intermediate_eq_full_eval,
                                rhos,
                            );

                            (y_0, y_half)
                        })
                        .fold((EF::zero(), EF::zero()), |(y_0, y_2), (y_0_i, y_2_i)| {
                            (y_0 + y_0_i, y_2 + y_2_i)
                        })
                },
            )
            .reduce(
                || (EF::zero(), EF::zero()),
                |(y_0, y_2), (y_0_i, y_2_i)| (y_0 + y_0_i, y_2 + y_2_i),
            );

        (y_0, y_half)
    }
}
