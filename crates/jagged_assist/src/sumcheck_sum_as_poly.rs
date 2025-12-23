use std::marker::PhantomData;
use std::sync::Arc;

use csl_challenger::DeviceGrindingChallenger;
use csl_cuda::reduce::DeviceSumKernel;
use csl_cuda::{args, TaskScope};
use slop_algebra::{ExtensionField, Field};
use slop_alloc::{Backend, IntoHost};
use slop_alloc::{Buffer, HasBackend};
use slop_challenger::FieldChallenger;
use slop_jagged::JaggedAssistSumAsPoly;
use slop_jagged::JaggedEvalSumcheckPoly;
use slop_multilinear::Point;
use slop_tensor::{ReduceSumBackend, Tensor, TransposeBackend};

use crate::branching_program_and_sample;
use crate::BranchingProgramKernel;

#[derive(Debug, Clone)]
pub struct JaggedAssistSumAsPolyGPUImpl<F: Field, EF: ExtensionField<F>, Challenger> {
    z_row: Point<EF, TaskScope>,
    z_index: Point<EF, TaskScope>,
    current_prefix_sums: Tensor<F, TaskScope>,
    next_prefix_sums: Tensor<F, TaskScope>,
    prefix_sum_length: usize,
    num_columns: usize,
    lambdas: Tensor<EF, TaskScope>,
    _marker: PhantomData<Challenger>,
}

impl<
        F: Field,
        EF: ExtensionField<F>,
        Challenger: DeviceGrindingChallenger + FieldChallenger<F> + Send + Sync,
    > JaggedAssistSumAsPoly<F, EF, TaskScope, Challenger, Challenger::OnDeviceChallenger>
    for JaggedAssistSumAsPolyGPUImpl<F, EF, Challenger>
where
    TaskScope: Backend
        + DeviceSumKernel<EF>
        + BranchingProgramKernel<F, EF, Challenger::OnDeviceChallenger>
        + ReduceSumBackend<EF>
        + TransposeBackend<F>,
{
    async fn new(
        z_row: Point<EF>,
        z_index: Point<EF>,
        merged_prefix_sums: Arc<Vec<Point<F>>>,
        _z_col_eq_vals: Vec<EF>,
        t: TaskScope,
    ) -> Self
    where
        TaskScope: Backend,
    {
        // *Warning*
        // This is a hack to get around the fact that the TaskScope is not available from the prover
        // and needs to be spawned. The issue here is that the task sciope should not live beyond
        // the execution, but nothing really enforces this. It should be fixed in the future, but
        // for now this hack should suffice since tasks are owned.
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

        let curr_prefix_sums_device = t.into_device(curr_prefix_sum_tensor).await.unwrap();
        let next_prefix_sums_device = t.into_device(next_prefix_sum_tensor).await.unwrap();

        let curr_prefix_sums_device = curr_prefix_sums_device.transpose();
        let next_prefix_sums_device = next_prefix_sums_device.transpose();

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
            lambdas: lambdas_device,
            _marker: PhantomData,
        }
    }

    async fn sum_as_poly_and_sample_into_point(
        &self,
        round_num: usize,
        z_col_eq_vals: &Buffer<EF, TaskScope>,
        intermediate_eq_full_evals: &Buffer<EF, TaskScope>,
        sum_values: &mut Buffer<EF, TaskScope>,
        challenger: &mut Challenger::OnDeviceChallenger,
        claim: EF,
        rhos: Point<EF, TaskScope>,
    ) -> (EF, Point<EF, TaskScope>) {
        let (current_prefix_sum_rho_point, next_prefix_sum_rho_point): (
            Point<EF, TaskScope>,
            Point<EF, TaskScope>,
        ) = if round_num < self.prefix_sum_length {
            (Point::new(Buffer::with_capacity_in(0, rhos.backend().clone())), rhos.clone())
        } else {
            let current_prefix_sum_rho_point_dim = round_num - self.prefix_sum_length;
            let mut current_prefix_sum_rho_point =
                Buffer::with_capacity_in(current_prefix_sum_rho_point_dim, rhos.backend().clone());

            let mut next_prefix_sum_rho_point = Buffer::with_capacity_in(
                rhos.dimension() - current_prefix_sum_rho_point_dim,
                rhos.backend().clone(),
            );

            let (a, b) = rhos.split_at(current_prefix_sum_rho_point_dim);
            current_prefix_sum_rho_point.extend_from_device_slice(a).unwrap();
            next_prefix_sum_rho_point.extend_from_device_slice(b).unwrap();
            assert_eq!(current_prefix_sum_rho_point.len(), current_prefix_sum_rho_point_dim);
            assert_eq!(
                next_prefix_sum_rho_point.len(),
                rhos.dimension() - current_prefix_sum_rho_point_dim
            );
            assert_eq!(current_prefix_sum_rho_point.capacity(), current_prefix_sum_rho_point_dim);
            assert_eq!(
                next_prefix_sum_rho_point.capacity(),
                rhos.dimension() - current_prefix_sum_rho_point_dim
            );
            (Point::new(current_prefix_sum_rho_point), Point::new(next_prefix_sum_rho_point))
        };

        let (bp_results_device, new_randomness) = branching_program_and_sample(
            &self.current_prefix_sums,
            &self.next_prefix_sums,
            self.prefix_sum_length,
            &current_prefix_sum_rho_point,
            &next_prefix_sum_rho_point,
            &self.z_row,
            &self.z_index,
            self.num_columns,
            round_num.try_into().unwrap(),
            &self.lambdas,
            z_col_eq_vals,
            intermediate_eq_full_evals,
            challenger,
            &rhos,
            sum_values,
            claim,
        )
        .await;

        let bp_results_device = bp_results_device.storage.into_host().await.unwrap();

        let bp_results = bp_results_device.into_vec();

        (bp_results[0], Point::new(new_randomness))
    }

    async fn fix_last_variable(
        poly: slop_jagged::JaggedEvalSumcheckPoly<
            F,
            EF,
            Challenger,
            Challenger::OnDeviceChallenger,
            Self,
            TaskScope,
        >,
    ) -> slop_jagged::JaggedEvalSumcheckPoly<
        F,
        EF,
        Challenger,
        Challenger::OnDeviceChallenger,
        Self,
        TaskScope,
    > {
        let merged_prefix_sum_dim = poly.prefix_sum_dimension as usize;

        let mut intermediate_eq_full_evals = poly.intermediate_eq_full_evals;
        let backend = intermediate_eq_full_evals.backend().clone();

        const BLOCK_SIZE: usize = 512;
        const STRIDE: usize = 1;
        let grid_size_x = poly.merged_prefix_sums.len().div_ceil(BLOCK_SIZE * STRIDE);
        let grid_size = (grid_size_x, 1, 1);

        unsafe {
            let args = args!(
                poly.merged_prefix_sums.as_ptr(),
                intermediate_eq_full_evals.as_mut_ptr(),
                poly.rho.as_ptr(),
                merged_prefix_sum_dim,
                intermediate_eq_full_evals.len(),
                poly.round_num,
                poly.rho.dimension()
            );

            backend
                .launch_kernel(
                    <TaskScope as BranchingProgramKernel<F, EF, Challenger::OnDeviceChallenger>>::fix_last_variable(),
                    grid_size,
                    (BLOCK_SIZE, 1, 1),
                    &args,
                    0,
                )
                .unwrap();
        }

        JaggedEvalSumcheckPoly::new(
            poly.bp_batch_eval,
            poly.rho,
            poly.z_col,
            poly.merged_prefix_sums,
            poly.z_col_eq_vals,
            poly.round_num + 1,
            intermediate_eq_full_evals,
            poly.half,
            poly.prefix_sum_dimension,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use csl_challenger::DuplexChallenger;
    use csl_cuda::TaskScope;
    use itertools::Itertools;
    use rand::Rng;
    use slop_algebra::extension::BinomialExtensionField;
    use slop_algebra::AbstractField;
    use slop_alloc::Buffer;
    use slop_koala_bear::KoalaBear;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;

    #[tokio::test]
    async fn test_fix_last_variable() {
        let merged_prefix_sum_dim = 50;

        let num_columns = 1000;

        let mut rng = rand::thread_rng();

        let intermediate_eq_full_evals =
            (0..num_columns).map(|_| rng.gen::<EF>()).collect::<Vec<_>>();

        let merged_prefix_sums =
            (0..num_columns * merged_prefix_sum_dim).map(|_| rng.gen::<F>()).collect::<Vec<_>>();

        let new_randomness_point =
            (0..merged_prefix_sum_dim).map(|_| rng.gen::<EF>()).collect::<Vec<_>>();

        csl_cuda::run_in_place(|backend| async move {
            for round_num in 0..merged_prefix_sum_dim {
                let merged_prefix_sums_device = backend
                    .into_device(Buffer::<F>::from(merged_prefix_sums.clone()))
                    .await
                    .unwrap();

                let mut intermediate_eq_full_evals_device = backend
                    .into_device(Buffer::<EF>::from(intermediate_eq_full_evals.clone()))
                    .await
                    .unwrap();

                let new_randomness_point_device = backend
                    .into_device(Buffer::<EF>::from(new_randomness_point.clone()))
                    .await
                    .unwrap();

                const BLOCK_SIZE: usize = 512;
                const STRIDE: usize = 1;
                let grid_size_x = merged_prefix_sums_device.len().div_ceil(BLOCK_SIZE * STRIDE);
                let grid_size = (grid_size_x, 1, 1);

                unsafe {
                    let time = std::time::Instant::now();
                    let args = args!(
                        merged_prefix_sums_device.as_ptr(),
                        intermediate_eq_full_evals_device.as_mut_ptr(),
                        new_randomness_point_device.as_ptr(),
                        { merged_prefix_sum_dim },
                        { num_columns },
                        { round_num },
                        new_randomness_point_device.len()
                    );

                    backend
                        .launch_kernel(
                            <TaskScope as BranchingProgramKernel<
                                F,
                                EF,
                                DuplexChallenger<F, TaskScope>,
                            >>::fix_last_variable(),
                            grid_size,
                            (BLOCK_SIZE, 1, 1),
                            &args,
                            0,
                        )
                        .unwrap();
                    println!("Kernel execution time: {:?}", time.elapsed());
                }
                let intermediate_eq_full_evals_from_device =
                    intermediate_eq_full_evals_device.into_host().await.unwrap();

                let alpha = *new_randomness_point.first().unwrap();

                let expected_intermediate_eq_full_evals = merged_prefix_sums
                    .to_vec()
                    .chunks(merged_prefix_sum_dim)
                    .zip_eq(intermediate_eq_full_evals.iter())
                    .map(|(merged_prefix_sum, intermediate_eq_full_eval)| {
                        let x_i =
                            merged_prefix_sum.get(merged_prefix_sum_dim - 1 - round_num).unwrap();
                        *intermediate_eq_full_eval
                            * ((alpha * *x_i) + (EF::one() - alpha) * (EF::one() - *x_i))
                    })
                    .collect_vec();

                for (i, (expected, actual)) in expected_intermediate_eq_full_evals
                    .iter()
                    .zip_eq(intermediate_eq_full_evals_from_device.iter())
                    .enumerate()
                {
                    assert_eq!(expected, actual, "Mismatch at index {i}");
                }
            }
        })
        .await
        .await
        .unwrap();
    }
}
