use std::sync::Arc;

use csl_sys::{
    mle::{
        mle_fix_last_variable_baby_bear_base_base_constant_padding,
        mle_fix_last_variable_baby_bear_base_base_padded,
        mle_fix_last_variable_baby_bear_base_extension_constant_padding,
        mle_fix_last_variable_baby_bear_base_extension_padded,
        mle_fix_last_variable_baby_bear_ext_ext_constant_padding,
        mle_fix_last_variable_baby_bear_ext_ext_padded,
        mle_fix_last_variable_in_place_baby_bear_base,
        mle_fix_last_variable_in_place_baby_bear_extension,
    },
    runtime::KernelPtr,
};
use slop_algebra::{extension::BinomialExtensionField, ExtensionField, Field};
use slop_baby_bear::BabyBear;
use slop_multilinear::{
    MleBaseBackend, MleEval, MleFixLastVariableBackend, MleFixLastVariableInPlaceBackend,
};
use slop_tensor::{Dimensions, Tensor};

use crate::{args, TaskScope};

/// # Safety
pub unsafe trait MleFixLastVariableKernel<F: Field, EF: ExtensionField<F>> {
    fn mle_fix_last_variable_kernel() -> KernelPtr;

    fn mle_fix_last_variable_constant_padding_kernel() -> KernelPtr;
}

impl<F: Field, EF: ExtensionField<F>> MleFixLastVariableBackend<F, EF> for TaskScope
where
    Self: MleFixLastVariableKernel<F, EF>,
{
    async fn mle_fix_last_variable(
        mle: &Tensor<F, Self>,
        alpha: EF,
        padding_values: Arc<MleEval<F, Self>>,
    ) -> Tensor<EF, Self> {
        let num_polynomials = Self::num_polynomials(mle);
        let input_height = mle.sizes()[1];
        assert!(input_height > 0);
        let output_height = input_height.div_ceil(2);
        let mut output: Tensor<EF, Self> = mle.backend().uninit_mle(num_polynomials, output_height);

        const BLOCK_SIZE: usize = 256;
        const STRIDE: usize = 128;
        let grid_size_x = output_height.div_ceil(BLOCK_SIZE * STRIDE);
        let grid_size_y = num_polynomials;
        let grid_size = (grid_size_x, grid_size_y, 1);

        let args = args!(
            mle.as_ptr(),
            output.as_mut_ptr(),
            padding_values.evaluations().as_ptr(),
            alpha,
            input_height,
            num_polynomials
        );

        unsafe {
            output.assume_init();
            mle.backend()
                .launch_kernel(
                    <Self as MleFixLastVariableKernel<F, EF>>::mle_fix_last_variable_kernel(),
                    grid_size,
                    BLOCK_SIZE,
                    &args,
                    0,
                )
                .unwrap();
        }

        output
    }

    async fn mle_fix_last_variable_constant_padding(
        mle: &Tensor<F, Self>,
        alpha: EF,
        padding_value: F,
    ) -> Tensor<EF, Self> {
        let num_polynomials = Self::num_polynomials(mle);
        let input_height = mle.sizes()[1];
        assert!(input_height > 0);
        let output_height = input_height.div_ceil(2);
        let mut output: Tensor<EF, Self> = mle.backend().uninit_mle(num_polynomials, output_height);

        const BLOCK_SIZE: usize = 256;
        const STRIDE: usize = 128;
        let grid_size_x = output_height.div_ceil(BLOCK_SIZE * STRIDE);
        let grid_size_y = num_polynomials;
        let grid_size = (grid_size_x, grid_size_y, 1);

        let args = args!(
            mle.as_ptr(),
            output.as_mut_ptr(),
            padding_value,
            alpha,
            input_height,
            num_polynomials
        );

        unsafe {
            output.assume_init();
            mle.backend()
                .launch_kernel(
                    <Self as MleFixLastVariableKernel<F, EF>>::mle_fix_last_variable_constant_padding_kernel(),
                    grid_size,
                    BLOCK_SIZE,
                    &args,
                    0,
                )
                .unwrap();
        }

        output
    }
}

/// # Safety
pub unsafe trait MleFixLastVariableInPlaceKernel<F: Field> {
    fn mle_fix_last_variable_in_place_kernel() -> KernelPtr;
}

impl<F: Field> MleFixLastVariableInPlaceBackend<F> for TaskScope
where
    TaskScope: MleFixLastVariableInPlaceKernel<F>,
{
    async fn mle_fix_last_variable_in_place(mle: &mut Tensor<F, Self>, alpha: F) {
        let num_polynomials = Self::num_polynomials(mle);
        let input_height = mle.sizes()[1];
        assert!(input_height > 0);
        let output_height = input_height.div_ceil(2);

        const BLOCK_SIZE: usize = 256;
        const STRIDE: usize = 128;
        let grid_size_x = output_height.div_ceil(BLOCK_SIZE * STRIDE);
        let grid_size_y = num_polynomials;
        let grid_size = (grid_size_x, grid_size_y, 1);
        let args = args!(mle.as_mut_ptr(), alpha, output_height, num_polynomials);

        unsafe {
            mle.backend()
                .launch_kernel(
                    <Self as MleFixLastVariableInPlaceKernel<F>>::mle_fix_last_variable_in_place_kernel(),
                    grid_size,
                    BLOCK_SIZE,
                    &args,
                    0,
                )
                .unwrap();
        }

        let dimensions = Dimensions::try_from([num_polynomials, output_height]).unwrap();
        mle.dimensions = dimensions;
    }
}

unsafe impl MleFixLastVariableKernel<BabyBear, BabyBear> for TaskScope {
    fn mle_fix_last_variable_kernel() -> KernelPtr {
        unsafe { mle_fix_last_variable_baby_bear_base_base_padded() }
    }

    fn mle_fix_last_variable_constant_padding_kernel() -> KernelPtr {
        unsafe { mle_fix_last_variable_baby_bear_base_base_constant_padding() }
    }
}

unsafe impl MleFixLastVariableKernel<BabyBear, BinomialExtensionField<BabyBear, 4>> for TaskScope {
    fn mle_fix_last_variable_kernel() -> KernelPtr {
        unsafe { mle_fix_last_variable_baby_bear_base_extension_padded() }
    }

    fn mle_fix_last_variable_constant_padding_kernel() -> KernelPtr {
        unsafe { mle_fix_last_variable_baby_bear_base_extension_constant_padding() }
    }
}

unsafe impl
    MleFixLastVariableKernel<
        BinomialExtensionField<BabyBear, 4>,
        BinomialExtensionField<BabyBear, 4>,
    > for TaskScope
{
    fn mle_fix_last_variable_kernel() -> KernelPtr {
        unsafe { mle_fix_last_variable_baby_bear_ext_ext_padded() }
    }

    fn mle_fix_last_variable_constant_padding_kernel() -> KernelPtr {
        unsafe { mle_fix_last_variable_baby_bear_ext_ext_constant_padding() }
    }
}

unsafe impl MleFixLastVariableInPlaceKernel<BabyBear> for TaskScope {
    fn mle_fix_last_variable_in_place_kernel() -> KernelPtr {
        unsafe { mle_fix_last_variable_in_place_baby_bear_base() }
    }
}

unsafe impl MleFixLastVariableInPlaceKernel<BinomialExtensionField<BabyBear, 4>> for TaskScope {
    fn mle_fix_last_variable_in_place_kernel() -> KernelPtr {
        unsafe { mle_fix_last_variable_in_place_baby_bear_extension() }
    }
}

#[cfg(test)]
mod tests {
    use std::{iter::once, sync::Arc};

    use rand::Rng;
    use slop_algebra::extension::BinomialExtensionField;
    use slop_algebra::AbstractField;
    use slop_alloc::{CanCopyFromRef, CpuBackend, IntoHost, ToHost};
    use slop_baby_bear::BabyBear;
    use slop_commit::Message;
    use slop_multilinear::{Mle, PaddedMle, Padding, Point};
    use slop_tensor::Tensor;

    use crate::{sync::CudaSend, IntoDevice, ToDevice};

    #[tokio::test]
    async fn test_mle_fix_last_variable() {
        let mut rng = rand::thread_rng();

        type F = BabyBear;
        type EF = BinomialExtensionField<F, 4>;

        let mle = Mle::<F>::new(Tensor::rand(&mut rng, [(1 << 16) - 1000, 1]));
        let random_point = Point::<EF>::rand(&mut rng, 15);
        let alpha = rng.gen::<EF>();

        let d_mle = mle.clone();

        let random_point_ref = &random_point;
        let evals = crate::run_in_place(|t| async move {
            let d_mle = t.into_device(d_mle).await.unwrap();
            let restriction = d_mle.fix_last_variable(alpha).await;
            restriction
                .eval_at(&random_point_ref.copy_into(&t))
                .await
                .into_evaluations()
                .into_host()
                .await
                .unwrap()
        })
        .await
        .await
        .unwrap()
        .into_buffer()
        .into_vec();

        let restriction = mle.fix_last_variable(alpha).await;
        let host_evals = restriction.eval_at(&random_point).await.to_vec();

        assert_eq!(evals, host_evals);
    }

    #[tokio::test]
    async fn test_spawned_mle_fix_last_variable() {
        let mut rng = rand::thread_rng();

        type F = BabyBear;
        type EF = BinomialExtensionField<F, 4>;

        let num_variables = 16;
        let num_tasks = 10;

        let mles = (0..num_tasks)
            .map(|_| Mle::<F>::new(Tensor::rand(&mut rng, [1 << num_variables, 1])))
            .collect::<Message<_>>();
        let random_point = Point::<EF>::rand(&mut rng, num_variables - 1);
        let alpha = rng.gen::<EF>();

        let complete_point = random_point.iter().copied().chain(once(alpha)).collect::<Point<_>>();

        // Do all the fix last variables in parallel.

        // let random_point_ref = &random_point;
        crate::run_in_place(|t| async move {
            // let random_point = random_point.to_device_in(&t).await.unwrap();
            let random_point = random_point.clone();
            let mut handles = Vec::new();
            for mle in mles.iter().cloned() {
                let random_point = random_point.clone();
                let mle = mle.into_device_in(&t).await.unwrap();
                let handle = t.spawn(move |s| async move {
                    let mle = unsafe { mle.send_to_scope(&s) };
                    let restriction = mle.fix_last_variable(alpha).await;
                    let random_point = random_point.to_device_in(&s).await.unwrap();
                    let evals = restriction.eval_at(&random_point).await;
                    evals.into_evaluations().into_host().await.unwrap()
                });
                handles.push(handle);
            }
            let mut evals = Vec::new();
            for handle in handles {
                // Get the evals
                let eval = handle.await.unwrap();
                evals.push(eval);
            }
            let complete_point = complete_point.to_device_in(&t).await.unwrap();
            for (eval, mle) in evals.iter().zip(mles) {
                let d_mle = mle.into_device_in(&t).await.unwrap();
                let d_eval = d_mle.eval_at(&complete_point).await;
                let h_eval = d_eval.into_evaluations().into_host().await.unwrap();
                for (e, h_e) in eval.as_slice().iter().zip(h_eval.as_slice().iter()) {
                    assert_eq!(e, h_e);
                }
            }
        })
        .await
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_padded_mle_fix_last_variable() {
        let mut rng = rand::thread_rng();

        type F = BabyBear;
        type EF = BinomialExtensionField<F, 4>;

        let mle = Mle::<F>::new(Tensor::rand(&mut rng, [(1 << 16) - 5, 2]));
        let padded_mle = PaddedMle::padded(
            Arc::new(mle.clone()),
            17,
            Padding::Generic(Arc::new(vec![F::one(), F::two()].into())),
        );
        let alpha = rng.gen::<EF>();

        let mle_evals: Padding<F, CpuBackend> =
            Padding::Generic(Arc::new(vec![F::one(), F::two()].into()));
        let evals = crate::run_in_place(|t| async move {
            let d_padded_mle = PaddedMle::padded(
                Arc::new(t.into_device(mle).await.unwrap()),
                17,
                t.copy_to(&mle_evals).await.unwrap(),
            );
            let restriction = d_padded_mle.fix_last_variable(alpha).await;
            restriction.inner().clone().unwrap().into_host().await.unwrap()
        })
        .await
        .await
        .unwrap();

        let restriction = padded_mle.fix_last_variable(alpha).await.inner().clone();

        for (i, (eval, host_eval)) in evals
            .guts()
            .as_slice()
            .iter()
            .zip(restriction.as_ref().unwrap().guts().as_slice().iter())
            .enumerate()
        {
            assert_eq!(eval, host_eval, "Incorrect values at index {}", i);
        }
    }

    #[tokio::test]
    async fn test_padded_mle_eval_at_device() {
        let mut rng = rand::thread_rng();

        type F = BabyBear;
        type EF = BinomialExtensionField<F, 4>;

        let mle = Mle::<F>::new(Tensor::rand(&mut rng, [(1 << 16) - 5, 2]));
        let padded_mle = PaddedMle::padded(
            Arc::new(mle.clone()),
            17,
            Padding::Generic(Arc::new(vec![F::one(), F::two()].into())),
        );
        let point = (0..17).map(|_| rng.gen::<EF>()).collect::<Point<_>>();
        let point_clone = point.clone();

        let padded_evals: Padding<F, CpuBackend> =
            Padding::Generic(Arc::new(vec![F::one(), F::two()].into()));
        let device_evals = crate::run_in_place(|t| async move {
            let d_padded_mle = PaddedMle::padded(
                Arc::new(t.into_device(mle).await.unwrap()),
                17,
                t.copy_to(&padded_evals).await.unwrap(),
            );
            let restriction = d_padded_mle.eval_at(&point).await;
            restriction.to_host().await.unwrap()
        })
        .await
        .await
        .unwrap();

        let host_evals = padded_mle.eval_at(&point_clone).await;

        assert_eq!(device_evals, host_evals);
    }

    #[tokio::test]
    async fn test_in_place_mle_fix_last_variable() {
        let mut rng = rand::thread_rng();

        type EF = BinomialExtensionField<BabyBear, 4>;

        let mle = Mle::<EF>::new(Tensor::rand(&mut rng, [(1 << 16) - 1000, 1]));
        let random_point = Point::<EF>::rand(&mut rng, 15);
        let alpha = rng.gen::<EF>();

        let d_mle = mle.clone();

        let random_point_ref = &random_point;
        let evals = crate::run_in_place(|t| async move {
            let mut d_mle = t.into_device(d_mle).await.unwrap();
            d_mle.fix_last_variable_in_place(alpha).await;
            d_mle
                .eval_at(&random_point_ref.copy_into(&t))
                .await
                .into_evaluations()
                .into_host()
                .await
                .unwrap()
        })
        .await
        .await
        .unwrap()
        .into_buffer()
        .into_vec();

        let restriction = mle.fix_last_variable(alpha).await;
        let host_evals = restriction.eval_at(&random_point).await.to_vec();

        assert_eq!(evals, host_evals);
    }
}
