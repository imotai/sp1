use slop_air::Air;
use slop_algebra::{AbstractExtensionField, ExtensionField, Field};
use slop_alloc::{Buffer, IntoHost};
use slop_multilinear::{Mle, MleBaseBackend};
use slop_tensor::{ReduceSumBackend, Tensor};
use sp1_stark::{air::MachineAir, zerocheck::ZerocheckProver, ConstraintSumcheckFolder};
use std::{
    ops::{Add, Mul, Sub},
    sync::Arc,
};

use crate::{EvalProgram, InterpolateRowKernel};
use csl_cuda::{args, TaskScope};

use super::ConstraintPolyEvalKernel;

/// A prover that uses the eval program to evaluate the constraint polynomial.
pub struct ZerocheckEvalProgramProver<F, EF, A> {
    eval_program: Arc<EvalProgram<F, EF>>,
    /// The public values.
    public_values: Arc<Buffer<F>>,
    /// The powers of alpha.
    powers_of_alpha: Arc<Buffer<EF>>,
    /// The AIR that contains the constraint polynomial.
    air: Arc<A>,
}

impl<F, EF, A> ZerocheckEvalProgramProver<F, EF, A> {
    pub fn new(
        eval_program: Arc<EvalProgram<F, EF>>,
        air: Arc<A>,
        public_values: Arc<Buffer<F>>,
        powers_of_alpha: Arc<Buffer<EF>>,
    ) -> Self {
        Self { eval_program, public_values, powers_of_alpha, air }
    }

    async fn constraint_poly_eval<
        K: Field + From<F> + Add<F, Output = K> + Sub<F, Output = K> + Mul<F, Output = K>,
    >(
        &self,
        partial_lagrange: &Mle<EF, TaskScope>,
        interpolated_preprocessed_rows: &Option<Tensor<K, TaskScope>>,
        interpolated_main_rows: &Tensor<K, TaskScope>,
    ) -> Tensor<EF, TaskScope>
    where
        TaskScope: ConstraintPolyEvalKernel<K>,
        F: Field,
        EF: ExtensionField<F>,
    {
        let EvalProgram { operations, f_ctr, f_constants, ef_constants } =
            self.eval_program.as_ref();

        let backend = interpolated_main_rows.backend();

        let operations_len = operations.len();
        let main_width = interpolated_main_rows.sizes()[1];
        let interpolated_main_rows_height = interpolated_main_rows.sizes()[2];

        let operations_device = backend.to_device(operations).await.unwrap();
        let f_constants_device = backend.to_device(f_constants).await.unwrap();
        let ef_constants_device = backend.to_device(ef_constants).await.unwrap();
        let powers_of_alpha_device =
            backend.to_device(self.powers_of_alpha.as_ref()).await.unwrap();
        let public_values_device = backend.to_device(self.public_values.as_ref()).await.unwrap();
        let mut output: Tensor<EF, TaskScope> =
            Tensor::with_sizes_in([3, interpolated_main_rows_height], backend.clone());

        let preprocessed_ptr = if let Some(preprocessed_rows) = interpolated_preprocessed_rows {
            preprocessed_rows.as_ptr()
        } else {
            std::ptr::null()
        };

        // Evalulate the constraint polynomial on the interpolated rows.
        unsafe {
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
}

impl<F, K, EF, A> ZerocheckProver<F, K, EF, TaskScope> for ZerocheckEvalProgramProver<F, EF, A>
where
    F: Field,
    EF: ExtensionField<F> + From<K> + ExtensionField<F> + AbstractExtensionField<K>,
    K: Field + From<F> + Add<F, Output = K> + Sub<F, Output = K> + Mul<F, Output = K>,
    A: for<'b> Air<ConstraintSumcheckFolder<'b, F, K, EF>> + MachineAir<F>,
    TaskScope: InterpolateRowKernel<K> + ConstraintPolyEvalKernel<K> + ReduceSumBackend<EF>,
{
    type Air = A;

    #[inline]
    fn air(&self) -> &Self::Air {
        &self.air
    }

    #[inline]
    fn public_values(&self) -> &[F] {
        &self.public_values
    }

    #[inline]
    fn powers_of_alpha(&self) -> &[EF] {
        &self.powers_of_alpha
    }

    async fn sum_as_poly_in_last_variable<const IS_FIRST_ROUND: bool>(
        &self,
        partial_lagrange: Arc<Mle<EF, TaskScope>>,
        preprocessed_values: Option<Arc<Mle<K, TaskScope>>>,
        main_values: Arc<Mle<K, TaskScope>>,
        _num_non_padded_terms: usize,
    ) -> (EF, EF, EF) {
        let interpolated_main_rows = interpolate_rows(main_values.as_ref());

        let interpolated_preprocessed_rows =
            preprocessed_values.map(|values| interpolate_rows(values.as_ref()));

        let output = self
            .constraint_poly_eval(
                &partial_lagrange,
                &interpolated_preprocessed_rows,
                &interpolated_main_rows,
            )
            .await;

        let y_s_device = output.sum(1).await;
        let y_s_host = y_s_device.into_host().await.unwrap().into_buffer().into_vec();

        (y_s_host[0], y_s_host[1], y_s_host[2])
    }
}

fn interpolate_rows<K: Field>(values: &Mle<K, TaskScope>) -> Tensor<K, TaskScope>
where
    TaskScope: InterpolateRowKernel<K>,
{
    let backend = values.backend();

    let height = (1 << TaskScope::num_variables(values.guts())) as usize;
    let width = TaskScope::num_polynomials(values.guts());
    let interpolated_rows_height = height.div_ceil(2);

    let mut interpolated_rows: Tensor<K, TaskScope> =
        Tensor::with_sizes_in([3, width, interpolated_rows_height], backend.clone());

    const BLOCK_SIZE: usize = 256;
    const STRIDE: usize = 1;
    let grid_size_x = interpolated_rows_height.div_ceil(BLOCK_SIZE * STRIDE);
    let grid_size_y = width;
    let grid_size = (grid_size_x, grid_size_y, 1);

    let args = args!(
        values.guts().as_ptr(),
        interpolated_rows.as_mut_ptr(),
        interpolated_rows_height,
        width
    );

    unsafe {
        interpolated_rows.assume_init();

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

    interpolated_rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{array, time::Instant};

    use rand::{distributions::Standard, Rng};
    use slop_air::BaseAir;
    use slop_algebra::{extension::BinomialExtensionField, AbstractField};
    use slop_baby_bear::BabyBear;
    use slop_multilinear::{Mle, Point};
    use slop_tensor::Tensor;

    use sp1_core_machine::riscv::RiscvAir;
    use sp1_stark::{air::MachineAir, zerocheck::ZerocheckCpuProver, PROOF_MAX_NUM_PVS};

    #[tokio::test]
    async fn test_interpolate_rows() {
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

        let interpolated_rows_host = csl_cuda::task()
            .await
            .unwrap()
            .run(|t| async move {
                let main_mle_device = t.into_device(main_mle).await.unwrap();
                let interpolated_rows = interpolate_rows::<EF>(&main_mle_device);
                let interpolated_rows_host = interpolated_rows.storage.into_host().await.unwrap();
                Tensor::from(interpolated_rows_host).reshape([3, main_width, main_height / 2])
            })
            .await
            .await
            .unwrap();

        let calculated_y_0s = interpolated_rows_host.get(0).unwrap().as_slice();
        let calculated_y_2s = interpolated_rows_host.get(1).unwrap().as_slice();
        let calculated_y_4s = interpolated_rows_host.get(2).unwrap().as_slice();
        assert_eq!(calculated_y_0s, expected_y_0s);
        assert_eq!(calculated_y_2s, expected_y_2s);
        assert_eq!(calculated_y_4s, expected_y_4s);
    }

    #[tokio::test]
    async fn test_zerocheck_cpu_air() {
        let mut rng = rand::thread_rng();

        type F = BabyBear;
        type EF = BinomialExtensionField<F, 4>;

        let cpu_chip = &RiscvAir::<F>::chips()[0];
        assert!(cpu_chip.name() == "Cpu");
        let cpu_width = cpu_chip.width();
        let cpu_num_constraints = 36; // TODO fill this out.
        let num_non_padded_terms = 1 << 20;

        let main_mle = Arc::new(Mle::<EF>::new(Tensor::rand(&mut rng, [1 << 21, cpu_width])));

        let alpha: EF = rng.sample(Standard);
        let mut powers_of_alpha = alpha.powers().take(cpu_num_constraints).collect::<Vec<_>>();
        powers_of_alpha.reverse();

        let public_values: [F; PROOF_MAX_NUM_PVS] = array::from_fn(|_| rng.sample(Standard));

        let random_point = Point::<EF>::rand(&mut rng, 20);
        let partial_lagrange_cpu = Arc::new(Mle::<EF>::partial_lagrange(&random_point).await);

        let host_prover = ZerocheckCpuProver::new(
            cpu_chip.air.clone(),
            Arc::new(public_values.to_vec()),
            Arc::new(powers_of_alpha.clone()),
        );

        let start_time = Instant::now();
        let (cpu_y_0, cpu_y_2, cpu_y_4) = host_prover
            .sum_as_poly_in_last_variable::<false>(
                partial_lagrange_cpu,
                None,
                main_mle.clone(),
                num_non_padded_terms,
            )
            .await;
        let cpu_time = start_time.elapsed();
        println!("CPU time: {:?}", cpu_time);

        // Get the eval program.
        let eval_program = Arc::new(EvalProgram::compile(cpu_chip.air.as_ref()));

        let device_prover = ZerocheckEvalProgramProver::new(
            eval_program,
            cpu_chip.air.clone(),
            Arc::new(Buffer::from(public_values.to_vec())),
            Arc::new(Buffer::from(powers_of_alpha)),
        );

        let (gpu_y_0, gpu_y_2, gpu_y_4) = csl_cuda::task()
            .await
            .unwrap()
            .run(|t| async move {
                let main_mle_device = t.into_device(main_mle).await.unwrap();
                let point = t.to_device(&random_point).await.unwrap();
                let partial_lagrange =
                    Arc::new(Mle::<EF, TaskScope>::partial_lagrange(&point).await);

                t.synchronize().await.unwrap();
                let time = tokio::time::Instant::now();
                let (gpu_y_0, gpu_y_2, gpu_y_4) = device_prover
                    .sum_as_poly_in_last_variable::<false>(
                        partial_lagrange,
                        None,
                        Arc::new(main_mle_device),
                        num_non_padded_terms,
                    )
                    .await;
                t.synchronize().await.unwrap();
                let gpu_time = time.elapsed();
                println!("GPU time: {:?}", gpu_time);

                (gpu_y_0, gpu_y_2, gpu_y_4)
            })
            .await
            .await
            .unwrap();

        assert_eq!(cpu_y_0, gpu_y_0);
        assert_eq!(cpu_y_2, gpu_y_2);
        assert_eq!(cpu_y_4, gpu_y_4);
    }
}
