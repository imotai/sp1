use slop_algebra::{ExtensionField, Field};
use slop_multilinear::Point;
use slop_tensor::{ReduceSumBackend, Tensor};

use csl_cuda::{
    args,
    sys::{jagged::branching_program_kernel, runtime::KernelPtr},
    TaskScope,
};

/// # Safety
///
pub unsafe trait BranchingProgramKernel<F: Field, EF: ExtensionField<F>> {
    fn branching_program_kernel() -> KernelPtr;
}

/// # Safety
///
unsafe impl<F: Field, EF: ExtensionField<F>> BranchingProgramKernel<F, EF> for TaskScope {
    fn branching_program_kernel() -> KernelPtr {
        unsafe { branching_program_kernel() }
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn branching_program<F: Field, EF: ExtensionField<F>>(
    curr_prefix_sums_device: &Tensor<F, TaskScope>,
    next_prefix_sums_device: &Tensor<F, TaskScope>,
    prefix_sum_length: usize,
    current_prefix_sum_rho_device: &Point<EF, TaskScope>,
    next_prefix_sum_rho_device: &Point<EF, TaskScope>,
    z_row_device: &Point<EF, TaskScope>,
    z_index_device: &Point<EF, TaskScope>,
    num_columns: usize,
    round_num: i8,
    lambdas_device: &Tensor<EF, TaskScope>,
    z_col_eq_vals_device: &Tensor<EF, TaskScope>,
    intermediate_eq_full_evals_device: &Tensor<EF, TaskScope>,
) -> Tensor<EF, TaskScope>
where
    TaskScope: BranchingProgramKernel<F, EF> + ReduceSumBackend<EF>,
{
    // Right now, we assume there are two points.
    assert!(lambdas_device.total_len() == 2);

    let backend = curr_prefix_sums_device.backend();

    const BLOCK_SIZE: usize = 256;
    const STRIDE: usize = 1;
    let grid_size_x = num_columns.div_ceil(BLOCK_SIZE * STRIDE);
    let grid_size = (grid_size_x, 2, 1);

    let mut bp_results: Tensor<EF, TaskScope> =
        Tensor::with_sizes_in([2, num_columns], backend.clone());

    unsafe {
        let args = args!(
            curr_prefix_sums_device.as_ptr(),
            next_prefix_sums_device.as_ptr(),
            prefix_sum_length,
            z_row_device.as_ptr(),
            z_row_device.dimension(),
            z_index_device.as_ptr(),
            z_index_device.dimension(),
            current_prefix_sum_rho_device.as_ptr(),
            current_prefix_sum_rho_device.dimension(),
            next_prefix_sum_rho_device.as_ptr(),
            next_prefix_sum_rho_device.dimension(),
            num_columns,
            round_num,
            lambdas_device.as_ptr(),
            z_col_eq_vals_device.as_ptr(),
            intermediate_eq_full_evals_device.as_ptr(),
            bp_results.as_mut_ptr()
        );

        bp_results.assume_init();

        backend
            .launch_kernel(
                <TaskScope as BranchingProgramKernel<F, EF>>::branching_program_kernel(),
                grid_size,
                (BLOCK_SIZE, 1, 1),
                &args,
                0,
            )
            .unwrap();
    }

    bp_results.sum(1).await
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::{distributions::Standard, Rng};
    use slop_algebra::{extension::BinomialExtensionField, AbstractField};
    use slop_alloc::{Buffer, CpuBackend, IntoHost};
    use slop_baby_bear::BabyBear;
    use slop_jagged::{
        all_bit_states, all_memory_states, transition_function, BranchingProgram,
        StateOrFail::{Fail, State},
    };
    use slop_multilinear::Point;
    use slop_tensor::Tensor;

    use csl_cuda::{
        args,
        sys::{jagged::transition_kernel, runtime::KernelPtr},
        TaskScope,
    };

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    pub trait TransitionKernel {
        fn transition_kernel() -> KernelPtr;
    }

    impl TransitionKernel for TaskScope {
        fn transition_kernel() -> KernelPtr {
            unsafe { transition_kernel() }
        }
    }

    #[tokio::test]
    async fn test_transition() {
        let bit_states = all_bit_states();
        let memory_states = all_memory_states(); // Note that this doesn't contain the FAIL state.

        let mut cpu_transition_results = Vec::new();
        for bit_state in bit_states.iter() {
            let mut bit_state_results = Vec::new();
            for output_memory_state in memory_states.iter() {
                bit_state_results.push(transition_function(*bit_state, *output_memory_state));
            }
            cpu_transition_results.push(bit_state_results);
        }

        let gpu_transition_results: Buffer<usize, CpuBackend> =
            csl_cuda::run_in_place(|t| async move {
                unsafe {
                    // The +1 is for the FAIL state.
                    let mut gpu_transition_results: Tensor<usize, TaskScope> =
                        Tensor::with_sizes_in(
                            [bit_states.len(), memory_states.len() + 1],
                            t.clone(),
                        );

                    let args = args!(gpu_transition_results.as_mut_ptr());

                    gpu_transition_results.assume_init();

                    t.launch_kernel(
                        <TaskScope as TransitionKernel>::transition_kernel(),
                        (1usize, 1usize, 1usize),
                        (1usize, 1usize, 1usize),
                        &args,
                        0,
                    )
                    .unwrap();

                    gpu_transition_results.storage.into_host().await.unwrap()
                }
            })
            .await
            .await
            .unwrap();

        // Need to retrieve these again, because they are moved into the cuda task.
        let bit_states = all_bit_states();
        let memory_states = all_memory_states();

        let mut gpu_transition_results: Tensor<usize, CpuBackend> = gpu_transition_results.into();
        gpu_transition_results.reshape_in_place([bit_states.len(), memory_states.len() + 1]);
        for (cpu_transition_mem_results, gpu_transition_mem_results) in
            cpu_transition_results.iter().zip(gpu_transition_results.split())
        {
            for (cpu_transition_result, gpu_transition_result) in
                cpu_transition_mem_results.iter().zip(gpu_transition_mem_results.clone().as_slice())
            {
                match cpu_transition_result {
                    State(cpu_transition_result) => {
                        assert_eq!(cpu_transition_result.get_index(), *gpu_transition_result);
                    }
                    Fail => {
                        assert_eq!(*gpu_transition_result, 4);
                    }
                }
            }

            // Verify that the transition from the FAIL state is FAIL.
            assert_eq!(gpu_transition_mem_results.as_slice()[4], 4);
        }
    }

    #[allow(clippy::type_complexity)]
    fn generate_branching_program_test_data(
        num_columns: usize,
        prefix_sum_length: usize,
        max_height: usize,
    ) -> (Vec<Point<F>>, Vec<Point<F>>, Point<EF>, Point<EF>, Vec<EF>, Vec<EF>) {
        let mut rng = rand::thread_rng();

        let mut prefix_sums = vec![];
        let mut cumulative_height = 0;
        let point: Point<F> = Point::from_usize(cumulative_height, prefix_sum_length);
        prefix_sums.push(point);
        for _ in 0..num_columns {
            let height = rng.gen_range(0..=max_height);
            cumulative_height += height;
            let point = Point::from_usize(cumulative_height, prefix_sum_length);
            prefix_sums.push(point);
        }

        let prefix_sum_iter = prefix_sums.iter();
        let next_prefix_sum_iter = prefix_sums.iter().skip(1);

        let mut curr_prefix_sum_points: Vec<Point<F>> = Vec::new();
        let mut next_prefix_sum_points: Vec<Point<F>> = Vec::new();
        for (curr_prefix_sum, next_prefix_sum) in prefix_sum_iter.zip(next_prefix_sum_iter) {
            curr_prefix_sum_points.push(curr_prefix_sum.clone());
            next_prefix_sum_points.push(next_prefix_sum.clone());
        }

        let z_row = (0..prefix_sum_length).map(|_| rng.sample(Standard)).collect::<Vec<_>>();
        let z_row_point: Point<EF> = z_row.into();
        let z_index = (0..prefix_sum_length).map(|_| rng.sample(Standard)).collect::<Vec<_>>();
        let z_index_point: Point<EF> = z_index.into();

        let z_col_eq_vals = (0..num_columns).map(|_| rng.sample(Standard)).collect::<Vec<_>>();
        let intermediate_eq_full_evals =
            (0..num_columns).map(|_| rng.sample(Standard)).collect::<Vec<_>>();

        (
            curr_prefix_sum_points,
            next_prefix_sum_points,
            z_row_point,
            z_index_point,
            z_col_eq_vals,
            intermediate_eq_full_evals,
        )
    }

    #[tokio::test]
    async fn test_naive_branching_program() {
        let num_columns = 1000;
        const PREFIX_SUM_LENGTH: usize = 30;
        const MAX_HEIGHT: usize = 1 << 21;

        let (
            curr_prefix_sum_points,
            next_prefix_sum_points,
            z_row_point,
            z_index_point,
            z_col_eq_vals,
            intermediate_eq_full_evals,
        ) = generate_branching_program_test_data(num_columns, PREFIX_SUM_LENGTH, MAX_HEIGHT);

        // Get the CPU results to compare.
        let bp = BranchingProgram::new(z_row_point.clone(), z_index_point.clone());

        let mut expected_result = EF::zero();
        for (curr_prefix_sum, next_prefix_sum) in
            curr_prefix_sum_points.iter().zip(next_prefix_sum_points.iter())
        {
            let curr_prefix_sum_ef: Point<EF> =
                curr_prefix_sum.values().iter().map(|x| (*x).into()).collect::<Vec<_>>().into();
            let next_prefix_sum_ef: Point<EF> =
                next_prefix_sum.values().iter().map(|x| (*x).into()).collect::<Vec<_>>().into();
            expected_result += bp.eval(&curr_prefix_sum_ef, &next_prefix_sum_ef);
        }

        // The kernel currently assumes there are two points, even if they are not used (e.g. if
        // the round number == -1).
        let lambdas = vec![EF::zero(), EF::two().inverse()];

        let bp_results_host = csl_cuda::run_in_place(|t| async move {
            let curr_prefix_sum_values = curr_prefix_sum_points
                .iter()
                .flat_map(|x| x.values().clone().into_vec())
                .collect::<Vec<_>>();
            let next_prefix_sum_values = next_prefix_sum_points
                .iter()
                .flat_map(|x| x.values().clone().into_vec())
                .collect::<Vec<_>>();

            let mut curr_prefix_sum_tensor: Tensor<F> = curr_prefix_sum_values.into();
            curr_prefix_sum_tensor.reshape_in_place([num_columns, PREFIX_SUM_LENGTH]);
            let curr_prefix_sum_tensor_transposed = curr_prefix_sum_tensor.transpose();

            let mut next_prefix_sum_tensor: Tensor<F> = next_prefix_sum_values.into();
            next_prefix_sum_tensor.reshape_in_place([num_columns, PREFIX_SUM_LENGTH]);
            let next_prefix_sum_tensor_transposed = next_prefix_sum_tensor.transpose();

            let curr_prefix_sums_device =
                t.into_device(curr_prefix_sum_tensor_transposed).await.unwrap();
            let next_prefix_sums_device =
                t.into_device(next_prefix_sum_tensor_transposed).await.unwrap();

            let z_row_device = t.into_device(z_row_point).await.unwrap();
            let z_index_device = t.into_device(z_index_point).await.unwrap();

            let empty_buffer = Buffer::<EF, TaskScope>::with_capacity_in(0, t.clone());
            let empty_point: Point<EF, TaskScope> = Point::new(empty_buffer);

            let lambdas_tensor: Tensor<EF> = lambdas.into();
            let lambdas_device = t.into_device(lambdas_tensor).await.unwrap();

            let z_col_eq_vals_tensor: Tensor<EF> = z_col_eq_vals.into();
            let z_col_eq_vals_device = t.into_device(z_col_eq_vals_tensor).await.unwrap();

            let intermediate_eq_full_evals_tensor: Tensor<EF> = intermediate_eq_full_evals.into();
            let intermediate_eq_full_evals_device =
                t.into_device(intermediate_eq_full_evals_tensor).await.unwrap();

            t.synchronize().await.unwrap();
            let time = std::time::Instant::now();

            let _bp_results_device = branching_program(
                &curr_prefix_sums_device,
                &next_prefix_sums_device,
                PREFIX_SUM_LENGTH,
                &empty_point,
                &empty_point,
                &z_row_device,
                &z_index_device,
                num_columns,
                -1,
                &lambdas_device,
                &z_col_eq_vals_device,
                &intermediate_eq_full_evals_device,
            )
            .await;

            t.synchronize().await.unwrap();
            println!("warmup time: {:?}", time.elapsed());

            t.synchronize().await.unwrap();
            let time = std::time::Instant::now();

            let bp_results_device = branching_program(
                &curr_prefix_sums_device,
                &next_prefix_sums_device,
                PREFIX_SUM_LENGTH,
                &empty_point,
                &empty_point,
                &z_row_device,
                &z_index_device,
                num_columns,
                -1,
                &lambdas_device,
                &z_col_eq_vals_device,
                &intermediate_eq_full_evals_device,
            )
            .await;

            t.synchronize().await.unwrap();
            println!("branching program time: {:?}", time.elapsed());

            let bp_results = bp_results_device.storage.into_host().await.unwrap();
            bp_results.into_vec()
        })
        .await
        .await
        .unwrap();

        assert_eq!(bp_results_host, vec![expected_result, expected_result]);
    }

    #[tokio::test]
    async fn test_branching_program() {
        let mut rng = rand::thread_rng();

        type F = BabyBear;
        type EF = BinomialExtensionField<F, 4>;

        let num_columns = 1000;
        const PREFIX_SUM_LENGTH: usize = 30;
        const MAX_HEIGHT: usize = 1 << 21;

        for round_num in 0..PREFIX_SUM_LENGTH * 2 {
            let (
                curr_prefix_sum_points,
                next_prefix_sum_points,
                z_row_point,
                z_index_point,
                z_col_eq_vals,
                intermediate_eq_full_evals,
            ) = generate_branching_program_test_data(num_columns, PREFIX_SUM_LENGTH, MAX_HEIGHT);

            let rhos: Point<EF> =
                (0..round_num).map(|_| rng.sample(Standard)).collect::<Vec<_>>().into();

            let lambdas = vec![EF::zero(), EF::two().inverse()];

            // Get the CPU results to compare.
            let bp = BranchingProgram::new(z_row_point.clone(), z_index_point.clone());

            let mut expected_results = vec![EF::zero(); 2];
            for (lambda, expected_result) in lambdas.iter().zip(expected_results.iter_mut()) {
                for column in 0..num_columns {
                    let mut merged_prefix_sum = curr_prefix_sum_points[column].clone();
                    merged_prefix_sum.extend(&next_prefix_sum_points[column]);

                    let merged_prefix_sum: Point<EF> = merged_prefix_sum
                        .values()
                        .iter()
                        .map(|x| (*x).into())
                        .collect::<Vec<_>>()
                        .into();

                    let (mut bp_prefix_sum, eq_prefix_sum) =
                        merged_prefix_sum.split_at(merged_prefix_sum.dimension() - round_num - 1);
                    bp_prefix_sum.add_dimension_back(*lambda);
                    bp_prefix_sum.extend(&rhos);
                    let num_dimensions = bp_prefix_sum.dimension();
                    assert!(bp_prefix_sum.dimension() == PREFIX_SUM_LENGTH * 2);
                    let (curr_prefix_sum, next_prefix_sum) =
                        bp_prefix_sum.split_at(num_dimensions / 2);

                    let eq_val = if *lambda == EF::zero() {
                        EF::one() - *eq_prefix_sum.values()[0]
                    } else if *lambda == EF::two().inverse() {
                        EF::two().inverse()
                    } else {
                        unreachable!("lambda must be 0 or 1/2")
                    };

                    let eq_eval = intermediate_eq_full_evals[column] * eq_val;

                    *expected_result += (eq_eval * z_col_eq_vals[column])
                        * bp.eval(&curr_prefix_sum, &next_prefix_sum);
                }
            }

            let bp_results_device = csl_cuda::run_in_place(|t| async move {
                let curr_prefix_sum_values = curr_prefix_sum_points
                    .iter()
                    .flat_map(|x| x.values().clone().into_vec())
                    .collect::<Vec<_>>();
                let next_prefix_sum_values = next_prefix_sum_points
                    .iter()
                    .flat_map(|x| x.values().clone().into_vec())
                    .collect::<Vec<_>>();

                let mut curr_prefix_sum_tensor: Tensor<F> = curr_prefix_sum_values.into();
                curr_prefix_sum_tensor.reshape_in_place([num_columns, PREFIX_SUM_LENGTH]);
                let curr_prefix_sum_tensor_transposed = curr_prefix_sum_tensor.transpose();

                let mut next_prefix_sum_tensor: Tensor<F> = next_prefix_sum_values.into();
                next_prefix_sum_tensor.reshape_in_place([num_columns, PREFIX_SUM_LENGTH]);
                let next_prefix_sum_tensor_transposed = next_prefix_sum_tensor.transpose();

                let curr_prefix_sums_device =
                    t.into_device(curr_prefix_sum_tensor_transposed).await.unwrap();
                let next_prefix_sums_device =
                    t.into_device(next_prefix_sum_tensor_transposed).await.unwrap();

                let z_row_device = t.into_device(z_row_point).await.unwrap();
                let z_index_device = t.into_device(z_index_point).await.unwrap();

                let (current_prefix_sum_rho_point, next_prefix_sum_rho_point): (
                    Point<EF>,
                    Point<EF>,
                ) = if round_num < PREFIX_SUM_LENGTH {
                    (Vec::new().into(), rhos.clone())
                } else {
                    let current_prefix_sum_rho_point_dim = round_num - PREFIX_SUM_LENGTH;
                    rhos.split_at(current_prefix_sum_rho_point_dim)
                };

                let lambdas_tensor: Tensor<EF> = lambdas.into();
                let lambdas_device = t.into_device(lambdas_tensor).await.unwrap();

                let z_col_eq_vals_tensor: Tensor<EF> = z_col_eq_vals.into();
                let z_col_eq_vals_device = t.into_device(z_col_eq_vals_tensor).await.unwrap();

                t.synchronize().await.unwrap();
                let time = std::time::Instant::now();

                let current_prefix_sum_rho_device =
                    t.into_device(current_prefix_sum_rho_point.clone()).await.unwrap();
                let next_prefix_sum_rho_device =
                    t.into_device(next_prefix_sum_rho_point.clone()).await.unwrap();

                let intermediate_eq_full_evals_tensor: Tensor<EF> =
                    intermediate_eq_full_evals.into();
                let intermediate_eq_full_evals_device =
                    t.into_device(intermediate_eq_full_evals_tensor).await.unwrap();

                let bp_results_device = branching_program(
                    &curr_prefix_sums_device,
                    &next_prefix_sums_device,
                    PREFIX_SUM_LENGTH,
                    &current_prefix_sum_rho_device,
                    &next_prefix_sum_rho_device,
                    &z_row_device,
                    &z_index_device,
                    num_columns,
                    round_num.try_into().unwrap(),
                    &lambdas_device,
                    &z_col_eq_vals_device,
                    &intermediate_eq_full_evals_device,
                )
                .await;

                t.synchronize().await.unwrap();
                println!("branching program time: {:?}", time.elapsed());

                let bp_results_device = bp_results_device.storage.into_host().await.unwrap();
                bp_results_device.into_vec()
            })
            .await
            .await
            .unwrap();

            assert_eq!(bp_results_device, expected_results);
        }
    }
}
