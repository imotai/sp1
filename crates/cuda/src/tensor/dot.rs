use csl_sys::{
    reduce::{
        partial_dot_baby_bear_base_extension_kernel, partial_dot_baby_bear_extension_kernel,
        partial_dot_baby_bear_kernel,
    },
    runtime::KernelPtr,
};
use slop_algebra::extension::BinomialExtensionField;
use slop_baby_bear::BabyBear;
use slop_tensor::{DotBackend, Tensor};

use crate::{args, reduce::partial_sum_reduction_into, DeviceCopy, TaskScope};

use super::reduce::DeviceSumKernel;

///
/// # Safety
pub unsafe trait DotKernel<T: DeviceCopy, U: DeviceCopy>: DeviceSumKernel<U> {
    fn partial_dot_kernel() -> KernelPtr;
}

impl<T: DeviceCopy, U: DeviceCopy> DotBackend<T, U> for TaskScope
where
    TaskScope: DotKernel<T, U>,
{
    fn dot_along_dim_into(
        src: &Tensor<T, Self>,
        scalars: &Tensor<U, Self>,
        dst: &mut Tensor<U, Self>,
        dim: usize,
    ) {
        assert!(
            dim == src.sizes().len() - 1,
            "only dot product over the last dimension is supported"
        );
        for scalar_size in scalars.sizes().iter().rev().skip(1) {
            assert_eq!(*scalar_size, 1, "The scalar tensor must be a 1D tensor");
        }

        let height = src.sizes()[dim];
        let width = src.total_len() / height;

        let null_ptr = std::ptr::null::<std::ffi::c_void>();
        let partial_args = args!(null_ptr, src.as_ptr(), scalars.as_ptr(), width, height);
        const BLOCK_SIZE: usize = 256;
        const INTIAL_STRIDE: usize = 4;
        unsafe {
            partial_sum_reduction_into::<U, BLOCK_SIZE, INTIAL_STRIDE, 5>(
                dst.as_view_mut(),
                TaskScope::partial_dot_kernel(),
                partial_args,
                0,
                src.shape(),
                dim,
                src.backend(),
            );
        }
    }

    fn dot_along_dim(
        src: &Tensor<T, Self>,
        scalars: &Tensor<U, Self>,
        dim: usize,
    ) -> Tensor<U, Self> {
        let mut sizes = src.sizes().to_vec();
        sizes.remove(dim);
        let mut dst = Tensor::zeros_in(sizes, src.backend().clone());
        Self::dot_along_dim_into(src, scalars, &mut dst, dim);
        dst
    }
}

unsafe impl DotKernel<BabyBear, BabyBear> for TaskScope {
    fn partial_dot_kernel() -> KernelPtr {
        unsafe { partial_dot_baby_bear_kernel() }
    }
}

unsafe impl DotKernel<BinomialExtensionField<BabyBear, 4>, BinomialExtensionField<BabyBear, 4>>
    for TaskScope
{
    fn partial_dot_kernel() -> KernelPtr {
        unsafe { partial_dot_baby_bear_extension_kernel() }
    }
}

unsafe impl DotKernel<BabyBear, BinomialExtensionField<BabyBear, 4>> for TaskScope {
    fn partial_dot_kernel() -> KernelPtr {
        unsafe { partial_dot_baby_bear_base_extension_kernel() }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use slop_algebra::extension::BinomialExtensionField;
    use slop_baby_bear::BabyBear;
    use slop_tensor::Tensor;

    use crate::IntoHost;

    #[tokio::test]
    async fn test_baby_bear_dot() {
        let num_summands = 100;
        let mut rng = rand::thread_rng();

        for size in [10, 100, 1 << 16] {
            let tensor = Tensor::<BabyBear>::rand(&mut rng, [num_summands, size]);
            let scalars = Tensor::<BabyBear>::rand(&mut rng, [size]);

            let tensor_sent = tensor.clone();
            let scalars_sent = scalars.clone();
            let inner_product = crate::task()
                .await
                .unwrap()
                .run(|t| async move {
                    let device_tensor = t.into_device(tensor_sent).await.unwrap();
                    let device_scalars = t.into_device(scalars_sent).await.unwrap();
                    let inner_product = device_tensor.dot(&device_scalars, 1);
                    inner_product.into_host().await.unwrap()
                })
                .await
                .await
                .unwrap();

            assert_eq!(inner_product.sizes(), [num_summands]);
            for i in 0..num_summands {
                let expected_inner_product: BabyBear = tensor
                    .get(i)
                    .unwrap()
                    .as_slice()
                    .iter()
                    .copied()
                    .zip_eq(scalars.as_buffer().iter().copied())
                    .map(|(a, b)| a * b)
                    .sum();
                assert_eq!(expected_inner_product, *inner_product[[i]]);
            }
        }
    }

    #[tokio::test]
    async fn test_baby_bear_extension_dot() {
        let num_summands = 100;
        let mut rng = rand::thread_rng();

        type EF = BinomialExtensionField<BabyBear, 4>;

        for size in [10, 100, 1 << 16] {
            let tensor = Tensor::<EF>::rand(&mut rng, [num_summands, size]);
            let scalars = Tensor::<EF>::rand(&mut rng, [size]);

            let tensor_sent = tensor.clone();
            let scalars_sent = scalars.clone();
            let inner_product = crate::task()
                .await
                .unwrap()
                .run(|t| async move {
                    let device_tensor = t.into_device(tensor_sent).await.unwrap();
                    let device_scalars = t.into_device(scalars_sent).await.unwrap();
                    let inner_product = device_tensor.dot(&device_scalars, 1);
                    inner_product.into_host().await.unwrap()
                })
                .await
                .await
                .unwrap();

            assert_eq!(inner_product.sizes(), [num_summands]);
            for i in 0..num_summands {
                let expected_inner_product: EF = tensor
                    .get(i)
                    .unwrap()
                    .as_slice()
                    .iter()
                    .copied()
                    .zip_eq(scalars.as_buffer().iter().copied())
                    .map(|(a, b)| a * b)
                    .sum();
                assert_eq!(expected_inner_product, *inner_product[[i]]);
            }
        }
    }

    #[tokio::test]
    async fn test_baby_bear_base_extension_dot() {
        let mut rng = rand::thread_rng();

        type F = BabyBear;
        type EF = BinomialExtensionField<BabyBear, 4>;

        for size in [10, 100, 1 << 10, 1 << 12, 1 << 16] {
            for num_summands in [64, 128] {
                let tensor = Tensor::<F>::rand(&mut rng, [num_summands, size]);
                let scalars = Tensor::<EF>::rand(&mut rng, [size]);

                let tensor_sent = tensor.clone();
                let scalars_sent = scalars.clone();
                let inner_product = crate::task()
                    .await
                    .unwrap()
                    .run(|t| async move {
                        let device_tensor = t.into_device(tensor_sent).await.unwrap();
                        let device_scalars = t.into_device(scalars_sent).await.unwrap();
                        t.synchronize_blocking().unwrap();
                        let time = std::time::Instant::now();
                        let inner_product = device_tensor.dot(&device_scalars, 1);
                        t.synchronize_blocking().unwrap();
                        println!(
                            "Dot time for size {}, num_summands: {}, time: {:?}",
                            size,
                            num_summands,
                            time.elapsed()
                        );
                        inner_product.into_host().await.unwrap()
                    })
                    .await
                    .await
                    .unwrap();

                assert_eq!(inner_product.sizes(), [num_summands]);
                for i in 0..num_summands {
                    let expected_inner_product: EF = tensor
                        .get(i)
                        .unwrap()
                        .as_slice()
                        .iter()
                        .copied()
                        .zip_eq(scalars.as_buffer().iter().copied())
                        .map(|(a, b)| b * a)
                        .sum();
                    assert_eq!(expected_inner_product, *inner_product[[i]]);
                }
            }
        }
    }
}
