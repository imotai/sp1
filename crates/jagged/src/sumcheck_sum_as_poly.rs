use std::sync::Arc;

use csl_cuda::reduce::DeviceSumKernel;
use csl_cuda::TaskScope;
use slop_algebra::{ExtensionField, Field};
use slop_alloc::{Backend, IntoHost};
use slop_jagged::JaggedAssistSumAsPoly;
use slop_multilinear::Point;
use slop_tensor::{ReduceSumBackend, Tensor};

use crate::branching_program;
use crate::BranchingProgramKernel;

#[derive(Debug, Clone)]
pub struct JaggedAssistSumAsPolyGPUImpl<F: Field, EF: ExtensionField<F>> {
    z_row: Point<EF, TaskScope>,
    z_index: Point<EF, TaskScope>,
    current_prefix_sums: Tensor<F, TaskScope>,
    next_prefix_sums: Tensor<F, TaskScope>,
    prefix_sum_length: usize,
    num_columns: usize,
    z_col_eq_vals: Tensor<EF, TaskScope>,
    lambdas: Tensor<EF, TaskScope>,
}

impl<F: Field, EF: ExtensionField<F>> JaggedAssistSumAsPolyGPUImpl<F, EF> {
    async fn into_device(
        current_prefix_sum_rho: Point<EF>,
        next_prefix_sum_rho: Point<EF>,
        intermediate_eq_full_evals: Vec<EF>,
    ) -> (Point<EF, TaskScope>, Point<EF, TaskScope>, Tensor<EF, TaskScope>) {
        csl_cuda::spawn(move |t| async move {
            let current_prefix_sum_rho_device =
                t.into_device(current_prefix_sum_rho).await.unwrap();
            let next_prefix_sum_rho_device = t.into_device(next_prefix_sum_rho).await.unwrap();

            let intermediate_eq_full_evals_tensor: Tensor<EF> =
                intermediate_eq_full_evals.to_vec().into();
            let intermediate_eq_full_evals_device =
                t.into_device(intermediate_eq_full_evals_tensor).await.unwrap();

            (
                current_prefix_sum_rho_device,
                next_prefix_sum_rho_device,
                intermediate_eq_full_evals_device,
            )
        })
        .await
        .unwrap()
        .await
        .unwrap()
    }
}

impl<F: Field, EF: ExtensionField<F>> JaggedAssistSumAsPoly<F, EF, TaskScope>
    for JaggedAssistSumAsPolyGPUImpl<F, EF>
where
    TaskScope: Backend + DeviceSumKernel<EF> + BranchingProgramKernel<F, EF> + ReduceSumBackend<EF>,
{
    async fn new(
        z_row: Point<EF>,
        z_index: Point<EF>,
        merged_prefix_sums: Arc<Vec<Point<F>>>,
        z_col_eq_vals: Vec<EF>,
    ) -> Self
    where
        TaskScope: Backend,
    {
        csl_cuda::spawn(move |t| async move {
            let z_row_device = t.into_device(z_row).await.unwrap();
            let z_index_device = t.into_device(z_index).await.unwrap();

            // Chop up the merged prefix sums into current and next prefix sums.
            let mut flattened_current_prefix_sums = Vec::new();
            let mut flattened_next_prefix_sums = Vec::new();
            for prefix_sum in merged_prefix_sums.iter() {
                let (current, next) = prefix_sum.split_at(prefix_sum.dimension() / 2);
                flattened_current_prefix_sums.extend(current.to_vec());
                flattened_next_prefix_sums.extend(next.to_vec());
            }

            let mut curr_prefix_sum_tensor: Tensor<F> = flattened_current_prefix_sums.into();
            let mut next_prefix_sum_tensor: Tensor<F> = flattened_next_prefix_sums.into();

            let num_columns = merged_prefix_sums.len();
            let prefix_sum_length = merged_prefix_sums[0].dimension() / 2;
            curr_prefix_sum_tensor.reshape_in_place([num_columns, prefix_sum_length]);
            next_prefix_sum_tensor.reshape_in_place([num_columns, prefix_sum_length]);

            let curr_prefix_sums_tensor_transposed = curr_prefix_sum_tensor.transpose();
            let next_prefix_sums_tensor_transposed = next_prefix_sum_tensor.transpose();

            let curr_prefix_sums_device =
                t.into_device(curr_prefix_sums_tensor_transposed).await.unwrap();
            let next_prefix_sums_device =
                t.into_device(next_prefix_sums_tensor_transposed).await.unwrap();

            let z_col_eq_vals_tensor: Tensor<EF> = z_col_eq_vals.into();
            let z_col_eq_vals_device = t.into_device(z_col_eq_vals_tensor).await.unwrap();

            let half = EF::two().inverse();

            let lambdas = vec![EF::zero(), half];
            let lambdas_tensor: Tensor<EF> = lambdas.into();
            let lambdas_device = t.into_device(lambdas_tensor).await.unwrap();

            Self {
                z_row: z_row_device,
                z_index: z_index_device,
                current_prefix_sums: curr_prefix_sums_device,
                next_prefix_sums: next_prefix_sums_device,
                prefix_sum_length,
                num_columns,
                z_col_eq_vals: z_col_eq_vals_device,
                lambdas: lambdas_device,
            }
        })
        .await
        .unwrap()
        .await
        .unwrap()
    }

    async fn sum_as_poly(
        &self,
        round_num: usize,
        _z_col_eq_vals: &[EF],
        intermediate_eq_full_evals: &[EF],
        rhos: &Point<EF>,
    ) -> (EF, EF) {
        let (current_prefix_sum_rho_point, next_prefix_sum_rho_point): (Point<EF>, Point<EF>) =
            if round_num < self.prefix_sum_length {
                (Vec::new().into(), rhos.clone())
            } else {
                let current_prefix_sum_rho_point_dim = round_num - self.prefix_sum_length;
                rhos.split_at(current_prefix_sum_rho_point_dim)
            };

        let (
            current_prefix_sum_rho_device,
            next_prefix_sum_rho_device,
            intermediate_eq_full_evals_device,
        ) = Self::into_device(
            current_prefix_sum_rho_point,
            next_prefix_sum_rho_point,
            intermediate_eq_full_evals.to_vec(),
        )
        .await;

        let bp_results_device = branching_program(
            &self.current_prefix_sums,
            &self.next_prefix_sums,
            self.prefix_sum_length,
            &current_prefix_sum_rho_device,
            &next_prefix_sum_rho_device,
            &self.z_row,
            &self.z_index,
            self.num_columns,
            round_num.try_into().unwrap(),
            &self.lambdas,
            &self.z_col_eq_vals,
            &intermediate_eq_full_evals_device,
        )
        .await;

        let bp_results_device = bp_results_device.storage.into_host().await.unwrap();
        let bp_results = bp_results_device.into_vec();

        let y_0 = bp_results[0];
        let y_half = bp_results[1];

        (y_0, y_half)
    }
}
