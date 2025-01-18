use csl_sys::{
    runtime::{Dim3, KernelPtr},
    transpose::{
        transpose_kernel_baby_bear, transpose_kernel_baby_bear_digest, transpose_kernel_u32,
        transpose_kernel_u32_digest,
    },
};
use slop_baby_bear::BabyBear;

use crate::{
    args,
    mem::DeviceData,
    tensor::{TensorViewMut, TransposeBackend},
    Tensor,
};

use super::TaskScope;

/// # Safety
pub unsafe trait DeviceTransposeKernel<T> {
    fn transpose_kernel() -> KernelPtr;
}

impl<T: DeviceData> TransposeBackend<T> for TaskScope
where
    TaskScope: DeviceTransposeKernel<T>,
{
    /// Transposes the tensor into the given destination tensor.
    fn transpose_tensor_into(src: &Tensor<T, Self>, mut dst: TensorViewMut<T, Self>) {
        let num_dims = src.sizes().len();

        let dim_x = src.sizes()[num_dims - 1];
        let dim_y = src.sizes()[num_dims - 2];
        let dim_z: usize = src.sizes().iter().take(num_dims - 2).product();
        assert_eq!(dim_x, dst.sizes()[num_dims - 2]);
        assert_eq!(dim_y, dst.sizes()[num_dims - 1]);

        let block_dim: Dim3 = (256u32, 256u32, 32u32).into();
        let grid_dim: Dim3 = (
            dim_x.div_ceil(block_dim.x as usize),
            dim_y.div_ceil(block_dim.y as usize),
            dim_z.div_ceil(block_dim.z as usize),
        )
            .into();
        let args = args!(src.as_ptr(), dst.as_mut_ptr(), dim_x, dim_y, dim_z);
        // let args = [
        //     &src.as_ptr() as *const _ as *mut c_void,
        //     &dst.as_mut_ptr() as *const _ as *mut c_void,
        //     &dim_x as *const usize as _,
        //     &dim_y as *const usize as _,
        //     &dim_z as *const usize as _,
        // ];
        unsafe {
            src.scope()
                .launch_kernel(Self::transpose_kernel(), block_dim, grid_dim, &args, 0)
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

unsafe impl DeviceTransposeKernel<[BabyBear; 8]> for TaskScope {
    fn transpose_kernel() -> KernelPtr {
        unsafe { transpose_kernel_baby_bear_digest() }
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use crate::{cuda::DeviceTensor, HostTensor};

    #[tokio::test]
    async fn test_device_transpose() {
        let sizes = [5, 9, 4, 10];
        let dim_x = sizes[sizes.len() - 1];
        let dim_y = sizes[sizes.len() - 2];
        let dim_z: usize = sizes.iter().take(sizes.len() - 2).product();

        let total_elements = dim_x * dim_y * dim_z;

        let mut rng = rand::thread_rng();
        let values: Vec<u32> = (0..total_elements).map(|_| rng.gen()).collect();

        let tensor = HostTensor::<u32>::from(values).reshape(sizes).unwrap();

        let tensor_sent = tensor.clone();
        let transposed_host = crate::cuda::task()
            .await
            .unwrap()
            .run(|t| async move {
                let tensor = DeviceTensor::from_host(tensor_sent, t).await.unwrap();
                let transposed = tensor.transpose();
                transposed.into_host().await.unwrap()
            })
            .await
            .await
            .unwrap();

        for i in 0..sizes[0] {
            for j in 0..sizes[1] {
                for k in 0..sizes[2] {
                    for l in 0..sizes[3] {
                        assert_eq!(*tensor[[i, j, k, l]], *transposed_host[[i, j, l, k]],);
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_device_digest_transpose() {
        let sizes = [5, 9, 4, 10];
        let dim_x = sizes[sizes.len() - 1];
        let dim_y = sizes[sizes.len() - 2];
        let dim_z: usize = sizes.iter().take(sizes.len() - 2).product();

        let total_elements = dim_x * dim_y * dim_z;

        let mut rng = rand::thread_rng();
        let values: Vec<[u32; 8]> = (0..total_elements).map(|_| rng.gen()).collect();

        let tensor = HostTensor::<[u32; 8]>::from(values).reshape(sizes).unwrap();

        let tensor_sent = tensor.clone();
        let transposed_host = crate::cuda::task()
            .await
            .unwrap()
            .run(|t| async move {
                let tensor = DeviceTensor::from_host(tensor_sent, t).await.unwrap();
                let transposed = tensor.transpose();
                transposed.into_host().await.unwrap()
            })
            .await
            .await
            .unwrap();

        for i in 0..sizes[0] {
            for j in 0..sizes[1] {
                for k in 0..sizes[2] {
                    for l in 0..sizes[3] {
                        assert_eq!(*tensor[[i, j, k, l]], *transposed_host[[i, j, l, k]],);
                    }
                }
            }
        }
    }
}
