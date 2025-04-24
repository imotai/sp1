use csl_sys::{
    runtime::{Dim3, KernelPtr},
    transpose::{
        transpose_kernel_baby_bear, transpose_kernel_baby_bear_digest,
        transpose_kernel_baby_bear_extension, transpose_kernel_u32, transpose_kernel_u32_digest,
    },
};
use slop_algebra::extension::BinomialExtensionField;
use slop_baby_bear::BabyBear;
use slop_tensor::{Tensor, TensorViewMut, TransposeBackend};

use crate::{args, DeviceCopy, TaskScope};

/// # Safety
pub unsafe trait DeviceTransposeKernel<T> {
    fn transpose_kernel() -> KernelPtr;
}

impl<T: DeviceCopy> TransposeBackend<T> for TaskScope
where
    TaskScope: DeviceTransposeKernel<T>,
{
    /// Transposes the tensor into the given destination tensor.
    fn transpose_tensor_into(src: &Tensor<T, Self>, mut dst: TensorViewMut<T, Self>) {
        let num_dims = src.sizes().len();

        let dim_x = src.sizes()[num_dims - 2];
        let dim_y = src.sizes()[num_dims - 1];
        let dim_z: usize = src.sizes().iter().take(num_dims - 2).product();
        assert_eq!(dim_x, dst.sizes()[num_dims - 1]);
        assert_eq!(dim_y, dst.sizes()[num_dims - 2]);

        let block_dim: Dim3 = (32u32, 32u32, 1u32).into();
        let grid_dim: Dim3 = (
            dim_x.div_ceil(block_dim.x as usize),
            dim_y.div_ceil(block_dim.y as usize),
            dim_z.div_ceil(block_dim.z as usize),
        )
            .into();
        let args = args!(src.as_ptr(), dst.as_mut_ptr(), dim_x, dim_y, dim_z);
        unsafe {
            src.backend()
                .launch_kernel(Self::transpose_kernel(), grid_dim, block_dim, &args, 0)
                .unwrap();
        }
    }
}

unsafe impl DeviceTransposeKernel<u32> for TaskScope {
    fn transpose_kernel() -> KernelPtr {
        unsafe { transpose_kernel_u32() }
    }
}

unsafe impl DeviceTransposeKernel<[u32; 8]> for TaskScope {
    fn transpose_kernel() -> KernelPtr {
        unsafe { transpose_kernel_u32_digest() }
    }
}

unsafe impl DeviceTransposeKernel<BabyBear> for TaskScope {
    fn transpose_kernel() -> KernelPtr {
        unsafe { transpose_kernel_baby_bear() }
    }
}

unsafe impl DeviceTransposeKernel<BinomialExtensionField<BabyBear, 4>> for TaskScope {
    fn transpose_kernel() -> KernelPtr {
        unsafe { transpose_kernel_baby_bear_extension() }
    }
}

unsafe impl DeviceTransposeKernel<[BabyBear; 8]> for TaskScope {
    fn transpose_kernel() -> KernelPtr {
        unsafe { transpose_kernel_baby_bear_digest() }
    }
}

#[cfg(test)]
mod tests {

    use slop_alloc::IntoHost;

    use super::*;

    #[tokio::test]
    async fn test_tensor_transpose() {
        let mut rng = rand::thread_rng();

        for (height, width) in [
            (1024, 1024),
            (1024, 6),
            (6, 1024),
            (1024, 6),
            (1024, 2048),
            (2048, 1024),
            (2048, 2048),
            (1 << 22, 100),
        ] {
            let tensor = Tensor::<u32>::rand(&mut rng, [height, width]);
            let transposed_expected = tensor.transpose();
            let transposed = crate::spawn(move |t| async move {
                let tensor = t.into_device(tensor).await.unwrap();
                let transposed = tensor.transpose();
                transposed.into_host().await.unwrap()
            })
            .await
            .unwrap();

            for (val, expected) in
                transposed.as_buffer().iter().zip(transposed_expected.as_buffer().iter())
            {
                assert_eq!(*val, *expected);
            }
        }
    }
}
