use csl_sys::{
    algebra::{add_baby_bear_base_ext_kernel, add_baby_bear_ext_ext_kernel, add_baby_bear_kernel},
    runtime::KernelPtr,
};
use slop_algebra::extension::BinomialExtensionField;
use slop_baby_bear::BabyBear;
use slop_tensor::{AddBackend, Tensor};

use crate::{args, DeviceCopy, TaskScope};

///
/// # Safety
pub unsafe trait AddKernel<U: DeviceCopy, T: DeviceCopy> {
    fn add_kernel() -> KernelPtr;
}

impl<T: DeviceCopy, U: DeviceCopy> AddBackend<U, T> for TaskScope
where
    TaskScope: AddKernel<U, T>,
{
    type AddOutput = U;

    fn add_into(
        lhs: &Tensor<U, Self>,
        rhs: &Tensor<T, Self>,
        dst: &mut Tensor<Self::AddOutput, Self>,
    ) {
        const BLOCK_SIZE: usize = 256;
        const GRID_STRIDE: usize = 1;
        unsafe {
            let grid_dim = lhs.total_len().div_ceil(BLOCK_SIZE).div_ceil(GRID_STRIDE);
            let args = args!(lhs.as_ptr(), rhs.as_ptr(), dst.as_ptr(), lhs.total_len());
            lhs.backend()
                .launch_kernel(Self::add_kernel(), grid_dim, BLOCK_SIZE, &args, 0)
                .unwrap();
        }
    }

    fn add(lhs: &Tensor<U, Self>, rhs: &Tensor<T, Self>) -> Tensor<Self::AddOutput, Self> {
        let mut dst = Tensor::with_sizes_in(lhs.sizes(), lhs.backend().clone());
        unsafe {
            dst.assume_init();
        }
        Self::add_into(lhs, rhs, &mut dst);
        dst
    }
}

unsafe impl AddKernel<BabyBear, BabyBear> for TaskScope {
    fn add_kernel() -> KernelPtr {
        unsafe { add_baby_bear_kernel() }
    }
}

unsafe impl AddKernel<BinomialExtensionField<BabyBear, 4>, BabyBear> for TaskScope {
    fn add_kernel() -> KernelPtr {
        unsafe { add_baby_bear_base_ext_kernel() }
    }
}

unsafe impl AddKernel<BinomialExtensionField<BabyBear, 4>, BinomialExtensionField<BabyBear, 4>>
    for TaskScope
{
    fn add_kernel() -> KernelPtr {
        unsafe { add_baby_bear_ext_ext_kernel() }
    }
}

#[cfg(test)]
mod tests {
    use crate::IntoHost;

    use super::*;

    #[tokio::test]
    async fn test_baby_bear_add() {
        let mut rng = rand::thread_rng();
        let a = Tensor::<BabyBear>::rand(&mut rng, [10, 450, 60]);
        let b = Tensor::<BabyBear>::rand(&mut rng, [10, 450, 60]);
        let c = &a + &b;

        let a_ext = Tensor::<BinomialExtensionField<BabyBear, 4>>::rand(&mut rng, [10, 450, 60]);
        let c_ext = &a_ext + &b;

        let (c_d, c_ext_d) = crate::task()
            .await
            .unwrap()
            .run(|t| async move {
                let a = t.into_device(a).await.unwrap();
                let b = t.into_device(b).await.unwrap();
                let c = &a + &b;
                let a_ext = t.into_device(a_ext).await.unwrap();
                let c_ext = &a_ext + &b;
                (c.into_host().await.unwrap(), c_ext.into_host().await.unwrap())
            })
            .await
            .await
            .unwrap();

        for (val, exp) in c.as_buffer().iter().zip(c_d.as_buffer().iter()) {
            assert_eq!(val, exp)
        }

        for (val, exp) in c_ext.as_buffer().iter().zip(c_ext_d.as_buffer().iter()) {
            assert_eq!(val, exp)
        }
    }
}
