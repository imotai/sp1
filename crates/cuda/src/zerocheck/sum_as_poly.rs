use futures::executor::block_on;
use slop_air::Air;
use slop_algebra::{AbstractExtensionField, ExtensionField, Field};
use slop_multilinear::{Mle, MleBaseBackend};
use slop_tensor::{ReduceSumBackend, Tensor};
use sp1_stark::{air::MachineAir, ConstraintSumcheckFolder};
use std::ops::{Add, Mul, Sub};

use crate::{
    args,
    zerocheck::{InterpolateRowKernel, EVAL_PROGRAM_GENERATOR},
    IntoHost, TaskScope,
};

use super::ConstraintPolyEvalKernel;

impl<
        K: Field + From<F> + Add<F, Output = K> + Sub<F, Output = K> + Mul<F, Output = K>,
        F: Field,
        EF: ExtensionField<F> + From<K> + ExtensionField<F> + AbstractExtensionField<K>,
        A: for<'b> Air<ConstraintSumcheckFolder<'b, F, K, EF>> + MachineAir<F>,
        const IS_FIRST_ROUND: bool,
    > SumAsPolyInLastVariableBackend<K, F, EF, A, IS_FIRST_ROUND> for TaskScope
where
    Self: ConstraintPolyEvalKernel<K> + InterpolateRowKernel<K> + ReduceSumBackend<EF>,
{
    fn sum_as_poly_in_last_variable(
        partial_lagrange: &Mle<EF, Self>,
        preprocessed_values: Option<&Mle<K, Self>>,
        main_values: &Mle<K, Self>,
        _num_non_padded_terms: usize,
        public_values: &[F],
        powers_of_alpha: &[EF],
        air: &A,
    ) -> (EF, EF, EF) {
        let interpolated_main_rows = interpolate_rows::<K, F>(main_values);

        let interpolated_preprocessed_rows =
            preprocessed_values.map(|values| interpolate_rows::<K, F>(values));

        let output = constraint_poly_eval(
            partial_lagrange,
            &interpolated_preprocessed_rows,
            &interpolated_main_rows,
            powers_of_alpha,
            public_values,
            &air.name(),
        );

        let y_s_device = output.sum(1);
        let y_s_host = block_on(y_s_device.storage.into_host()).unwrap();

        (*y_s_host[0], *y_s_host[1], *y_s_host[2])
    }
}

fn interpolate_rows<
    K: Field + From<F> + Add<F, Output = K> + Sub<F, Output = K> + Mul<F, Output = K>,
    F: Field,
>(
    values: &Mle<K, TaskScope>,
) -> Tensor<K, TaskScope>
where
    TaskScope: InterpolateRowKernel<K>,
{
    let backend = values.backend();

    let height = (1 << TaskScope::num_variables(values.guts())) as usize;
    let width = TaskScope::num_polynomials(values.guts());
    let interpolated_rows_height = height.div_ceil(2);

    let mut interpolated_rows_device: Tensor<K, TaskScope> =
        Tensor::with_sizes_in([3, width, interpolated_rows_height], backend.clone());

    const BLOCK_SIZE: usize = 256;
    const STRIDE: usize = 1;
    let grid_size_x = interpolated_rows_height.div_ceil(BLOCK_SIZE * STRIDE);
    let grid_size_y = width;
    let grid_size = (grid_size_x, grid_size_y, 1);

    let args = args!(
        values.guts().as_ptr(),
        interpolated_rows_device.as_mut_ptr(),
        interpolated_rows_height,
        width
    );

    unsafe {
        interpolated_rows_device.assume_init();

        backend
            .launch_kernel(
                <TaskScope as InterpolateRowKernel<K>>::interpolate_row_kernel(),
                grid_size,
                (BLOCK_SIZE, 1, 1),
                &args,
                0,
            )
            .unwrap();
    }

    interpolated_rows_device
}

fn constraint_poly_eval<
    K: Field + From<F> + Add<F, Output = K> + Sub<F, Output = K> + Mul<F, Output = K>,
    F: Field,
    EF: ExtensionField<F>,
>(
    partial_lagrange: &Mle<EF, TaskScope>,
    interpolated_preprocessed_rows: &Option<Tensor<K, TaskScope>>,
    interpolated_main_rows: &Tensor<K, TaskScope>,
    powers_of_alpha: &[EF],
    public_values: &[F],
    chip_name: &str,
) -> Tensor<EF, TaskScope>
where
    TaskScope: ConstraintPolyEvalKernel<K>,
{
    let (operations, f_ctr, f_constants, ef_constants) =
        &EVAL_PROGRAM_GENERATOR.get_eval_program(chip_name);
    let operations = operations.clone();
    let f_constants = f_constants.clone();
    let ef_constants = ef_constants.clone();

    let backend = interpolated_main_rows.backend();

    let operations_len = operations.len();
    let main_width = interpolated_main_rows.sizes()[1];
    let interpolated_main_rows_height = interpolated_main_rows.sizes()[2];

    let operations_device = block_on(backend.into_device(operations)).unwrap();
    let f_constants_device = block_on(backend.into_device(f_constants)).unwrap();
    let ef_constants_device = block_on(backend.into_device(ef_constants)).unwrap();
    let powers_of_alpha_device = block_on(backend.into_device(powers_of_alpha.to_vec())).unwrap();
    let public_values_device = block_on(backend.into_device(public_values.to_vec())).unwrap();
    let mut output: Tensor<EF, TaskScope> =
        Tensor::with_sizes_in([3, interpolated_main_rows_height], backend.clone());

    let preprocessed_ptr = if let Some(preprocessed_rows) = interpolated_preprocessed_rows {
        preprocessed_rows.as_ptr()
    } else {
        std::ptr::null()
    };

    // Run the interpolate row kernel.
    let args = args!(
        operations_device.as_ptr(),
        operations_len,
        f_constants_device.as_ptr(),
        ef_constants_device.as_ptr(),
        partial_lagrange.guts().as_ptr(),
        preprocessed_ptr,
        0,
        interpolated_main_rows.as_ptr(),
        main_width,
        interpolated_main_rows_height,
        powers_of_alpha_device.as_ptr(),
        public_values_device.as_ptr(),
        output.as_mut_ptr()
    );

    const BLOCK_SIZE: usize = 256;
    const STRIDE: usize = 1;
    let grid_size_x = interpolated_main_rows_height.div_ceil(BLOCK_SIZE * STRIDE);
    let grid_size = (grid_size_x, 3, 1);

    // Evalulate the constraint polynomial on the interpolated rows.
    unsafe {
        output.assume_init();
        backend
            .launch_kernel(
                <TaskScope as ConstraintPolyEvalKernel<K>>::constraint_poly_eval_kernel(
                    *f_ctr as usize,
                ),
                grid_size,
                (BLOCK_SIZE, 1, 1),
                &args,
                0,
            )
            .unwrap();
    }

    output
}

#[cfg(test)]
mod tests {
    use std::{array, time::Instant};

    use futures::executor::block_on;
    use rand::{distributions::Standard, Rng};
    use slop_air::BaseAir;
    use slop_algebra::{extension::BinomialExtensionField, AbstractField};
    use slop_alloc::CpuBackend;
    use slop_baby_bear::BabyBear;
    use slop_multilinear::{Mle, Point};
    use slop_tensor::Tensor;

    use sp1_core_machine::riscv::RiscvAir;
    use sp1_stark::{
        air::MachineAir, zerocheck::SumAsPolyInLastVariableBackend, PROOF_MAX_NUM_PVS,
    };

    use crate::{zerocheck::sum_as_poly::interpolate_rows, IntoHost, TaskScope};

    #[tokio::test]
    async fn test_interpolate() {
        let mut rng = rand::thread_rng();

        type F = BabyBear;
        type EF = BinomialExtensionField<F, 4>;

        let main_height = 1 << 10;
        let main_width = 1;

        let main_mle = Mle::<EF>::new(Tensor::rand(&mut rng, [main_height, main_width]));
        let main_guts = main_mle.guts().as_slice();

        let (expected_y_0s, (expected_y_2s, expected_y_4s)): (Vec<_>, (Vec<_>, Vec<_>)) = main_guts
            .chunks_exact(2)
            .map(|chunk| {
                (
                    chunk[0],
                    (
                        EF::from_canonical_usize(2) * (chunk[1] - chunk[0]) + chunk[0],
                        EF::from_canonical_usize(4) * (chunk[1] - chunk[0]) + chunk[0],
                    ),
                )
            })
            .unzip();

        let task = crate::task().await.unwrap();
        let main_mle_device = task
            .run(|t| async move { t.into_device(main_mle).await.unwrap() })
            .await
            .await
            .unwrap();
        let interpolated_rows = interpolate_rows::<EF, F>(&main_mle_device);
        let interpolated_rows_host = block_on(interpolated_rows.storage.into_host()).unwrap();
        let mut interpolated_rows_host = Tensor::from(interpolated_rows_host);
        interpolated_rows_host.reshape_in_place([3, main_width, main_height / 2]);

        let calculated_y_0s = interpolated_rows_host.get(0).unwrap().as_slice();
        let calculated_y_2s = interpolated_rows_host.get(1).unwrap().as_slice();
        let calculated_y_4s = interpolated_rows_host.get(2).unwrap().as_slice();
        assert_eq!(calculated_y_0s, expected_y_0s);
        assert_eq!(calculated_y_2s, expected_y_2s);
        assert_eq!(calculated_y_4s, expected_y_4s);
    }

    #[tokio::test]
    async fn test_e2e_cpu_chip() {
        let mut rng = rand::thread_rng();

        type F = BabyBear;
        type EF = BinomialExtensionField<F, 4>;

        let cpu_chip = &RiscvAir::<F>::chips()[0];
        assert!(cpu_chip.name() == "Cpu");
        let cpu_width = cpu_chip.width();
        let cpu_num_constraints = 36; // TODO fill this out.

        let main_mle = Mle::<EF>::new(Tensor::rand(&mut rng, [1 << 21, cpu_width]));

        let alpha: EF = rng.sample(Standard);
        let mut powers_of_alpha = alpha.powers().take(cpu_num_constraints).collect::<Vec<_>>();
        powers_of_alpha.reverse();

        let public_values: [F; PROOF_MAX_NUM_PVS] = array::from_fn(|_| rng.sample(Standard));

        let random_point = Point::<EF>::rand(&mut rng, 20);
        let partial_lagrange_cpu = Mle::<EF>::partial_lagrange(&random_point);

        let start_time = Instant::now();
        let (cpu_y_0, cpu_y_2, cpu_y_4) = <CpuBackend as SumAsPolyInLastVariableBackend<
            EF,
            F,
            EF,
            RiscvAir<F>,
            false,
        >>::sum_as_poly_in_last_variable(
            &partial_lagrange_cpu,
            None,
            &main_mle,
            1 << 20,
            &public_values,
            &powers_of_alpha,
            &cpu_chip.air,
        );
        let cpu_time = start_time.elapsed();
        let task = crate::task().await.unwrap();
        let (partial_lagrange_cpu_device, main_mle_device) = task
            .run(|t| async move {
                let main_mle_device = t.into_device(main_mle).await.unwrap();
                let partial_lagrange_cpu_device =
                    t.into_device(partial_lagrange_cpu).await.unwrap();
                (partial_lagrange_cpu_device, main_mle_device)
            })
            .await
            .await
            .unwrap();

        let start_time = Instant::now();
        let (gpu_y_0, gpu_y_2, gpu_y_4) = <TaskScope as SumAsPolyInLastVariableBackend<
            EF,
            F,
            EF,
            RiscvAir<F>,
            false,
        >>::sum_as_poly_in_last_variable(
            &partial_lagrange_cpu_device,
            None,
            &main_mle_device,
            1 << 20,
            &public_values,
            &powers_of_alpha,
            &cpu_chip.air,
        );
        let gpu_time = start_time.elapsed();

        assert_eq!(cpu_y_0, gpu_y_0);
        assert_eq!(cpu_y_2, gpu_y_2);
        assert_eq!(cpu_y_4, gpu_y_4);

        println!("CPU time: {:?}", cpu_time);
        println!("GPU time: {:?}", gpu_time);
    }
}
