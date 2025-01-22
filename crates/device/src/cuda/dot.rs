use csl_sys::{
    reduce::{
        partial_dot_baby_bear_base_extension_kernel, partial_dot_baby_bear_extension_kernel,
        partial_dot_baby_bear_kernel,
    },
    runtime::KernelPtr,
};
use slop_algebra::extension::BinomialExtensionField;
use slop_baby_bear::BabyBear;

use crate::{
    args,
    cuda::reduce::partial_sum_reduction_into,
    mem::DeviceData,
    tensor::{DotBackend, TensorView, TensorViewMut},
};

use super::{reduce::DeviceSumKernel, TaskScope};

///
/// # Safety
pub unsafe trait DotKernel<T: DeviceData, U: DeviceData>: DeviceSumKernel<U> {
    fn partial_dot_kernel() -> KernelPtr;
}

impl<T: DeviceData, U: DeviceData> DotBackend<T, U> for TaskScope
where
    TaskScope: DotKernel<T, U>,
{
    fn dot_along_dim(
        &self,
        src: TensorView<T, Self>,
        scalars: TensorView<U, Self>,
        dst: TensorViewMut<U, Self>,
        dim: usize,
    ) {
        assert!(
            dim == src.sizes().len() - 1,
            "only dot product over the last dimension is supported"
        );

        let height = src.sizes()[dim];
        let width = src.total_len() / height;

        let null_ptr = std::ptr::null::<std::ffi::c_void>();
        let partial_args = args!(null_ptr, src.as_ptr(), scalars.as_ptr(), width, height);
        const BLOCK_SIZE: usize = 256;
        const INTIAL_STRIDE: usize = 4;
        unsafe {
            partial_sum_reduction_into::<U, BLOCK_SIZE, INTIAL_STRIDE, 5>(
                dst,
                TaskScope::partial_dot_kernel(),
                partial_args,
                0,
                src.shape(),
                dim,
                self,
            );
        }
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
    use rand::Rng;
    use slop_algebra::extension::BinomialExtensionField;
    use slop_baby_bear::BabyBear;

    use crate::{DeviceTensor, HostTensor};

    #[tokio::test]
    async fn test_baby_bear_dot() {
        let num_summands = 100;
        let mut rng = rand::thread_rng();

        for size in [10, 100, 1 << 16] {
            let values_1: Vec<BabyBear> = (0..(num_summands * size)).map(|_| rng.gen()).collect();
            let values_2: Vec<BabyBear> = (0..(size)).map(|_| rng.gen()).collect();
            let tensor =
                HostTensor::<BabyBear>::from(values_1).reshape([num_summands, size]).unwrap();
            let scalars = HostTensor::<BabyBear>::from(values_2).reshape([size]).unwrap();

            let tensor_sent = tensor.clone();
            let scalars_sent = scalars.clone();
            let inner_product = crate::cuda::task()
                .await
                .unwrap()
                .run(|t| async move {
                    let device_tensor =
                        DeviceTensor::from_host(tensor_sent, t.clone()).await.unwrap();
                    let device_scalars =
                        DeviceTensor::from_host(scalars_sent, t.clone()).await.unwrap();
                    let inner_product = device_tensor.dot(device_scalars.as_view(), 1);
                    inner_product.into_host().await.unwrap()
                })
                .await
                .await
                .unwrap();

            assert_eq!(inner_product.sizes(), [num_summands]);
            for i in 0..num_summands {
                let expected_inner_product: BabyBear = tensor
                    .index(i)
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
            let values_1: Vec<EF> = (0..(num_summands * size)).map(|_| rng.gen()).collect();
            let values_2: Vec<EF> = (0..(size)).map(|_| rng.gen()).collect();
            let tensor = HostTensor::<EF>::from(values_1).reshape([num_summands, size]).unwrap();
            let scalars = HostTensor::<EF>::from(values_2).reshape([size]).unwrap();

            let tensor_sent = tensor.clone();
            let scalars_sent = scalars.clone();
            let inner_product = crate::cuda::task()
                .await
                .unwrap()
                .run(|t| async move {
                    let device_tensor =
                        DeviceTensor::from_host(tensor_sent, t.clone()).await.unwrap();
                    let device_scalars =
                        DeviceTensor::from_host(scalars_sent, t.clone()).await.unwrap();
                    let inner_product = device_tensor.dot(device_scalars.as_view(), 1);
                    inner_product.into_host().await.unwrap()
                })
                .await
                .await
                .unwrap();

            assert_eq!(inner_product.sizes(), [num_summands]);
            for i in 0..num_summands {
                let expected_inner_product: EF = tensor
                    .index(i)
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
                let values_1: Vec<F> = (0..(num_summands * size)).map(|_| rng.gen()).collect();
                let values_2: Vec<EF> = (0..(size)).map(|_| rng.gen()).collect();
                let tensor = HostTensor::<F>::from(values_1).reshape([num_summands, size]).unwrap();
                let scalars = HostTensor::<EF>::from(values_2).reshape([size]).unwrap();

                let tensor_sent = tensor.clone();
                let scalars_sent = scalars.clone();
                let inner_product = crate::cuda::task()
                    .await
                    .unwrap()
                    .run(|t| async move {
                        let device_tensor =
                            DeviceTensor::from_host(tensor_sent, t.clone()).await.unwrap();
                        let device_scalars =
                            DeviceTensor::from_host(scalars_sent, t.clone()).await.unwrap();
                        t.synchronize_blocking().unwrap();
                        let time = std::time::Instant::now();
                        let inner_product = device_tensor.dot(device_scalars.as_view(), 1);
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
                        .index(i)
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
