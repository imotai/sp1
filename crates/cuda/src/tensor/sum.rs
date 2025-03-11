use csl_sys::{
    algebra::{
        add_assign_baby_bear_ext_kernel, add_assign_baby_bear_kernel,
        add_baby_bear_base_ext_kernel, add_baby_bear_ext_ext_kernel, add_baby_bear_kernel,
    },
    runtime::KernelPtr,
};
use slop_algebra::extension::BinomialExtensionField;
use slop_baby_bear::BabyBear;
use slop_tensor::{AddAssignBackend, AddBackend, Tensor};

use crate::{args, DeviceCopy, TaskScope};

///
/// # Safety
pub unsafe trait AddKernel<U: DeviceCopy, T: DeviceCopy> {
    fn add_kernel() -> KernelPtr;
}

///
/// # Safety
pub unsafe trait AddAssignKernel<T: DeviceCopy> {
    fn add_assign_kernel() -> KernelPtr;
}

impl<T: DeviceCopy, U: DeviceCopy> AddBackend<U, T> for TaskScope
where
    TaskScope: AddKernel<U, T>,
{
    type AddOutput = U;

    async fn add(lhs: &Tensor<U, Self>, rhs: &Tensor<T, Self>) -> Tensor<Self::AddOutput, Self> {
        let mut dst = Tensor::with_sizes_in(lhs.sizes(), lhs.backend().clone());
        unsafe {
            dst.assume_init();
        }
        const BLOCK_SIZE: usize = 256;
        const GRID_STRIDE: usize = 1;
        unsafe {
            let grid_dim = lhs.total_len().div_ceil(BLOCK_SIZE).div_ceil(GRID_STRIDE);
            let args = args!(lhs.as_ptr(), rhs.as_ptr(), dst.as_ptr(), lhs.total_len());
            lhs.backend()
                .launch_kernel(Self::add_kernel(), grid_dim, BLOCK_SIZE, &args, 0)
                .unwrap();
        }
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

impl<T: DeviceCopy> AddAssignBackend<T> for TaskScope
where
    TaskScope: AddAssignKernel<T>,
{
    async fn add_assign(lhs: &mut Tensor<T, Self>, rhs: T) {
        const BLOCK_SIZE: usize = 256;
        const GRID_STRIDE: usize = 1;
        unsafe {
            let grid_dim = lhs.total_len().div_ceil(BLOCK_SIZE).div_ceil(GRID_STRIDE);
            let args = args!(lhs.as_mut_ptr(), rhs, lhs.total_len());
            lhs.backend()
                .launch_kernel(Self::add_assign_kernel(), grid_dim, BLOCK_SIZE, &args, 0)
                .unwrap();
        }
    }
}

unsafe impl AddAssignKernel<BabyBear> for TaskScope {
    fn add_assign_kernel() -> KernelPtr {
        unsafe { add_assign_baby_bear_kernel() }
    }
}

unsafe impl AddAssignKernel<BinomialExtensionField<BabyBear, 4>> for TaskScope {
    fn add_assign_kernel() -> KernelPtr {
        unsafe { add_assign_baby_bear_ext_kernel() }
    }
}
