use csl_sys::runtime::KernelPtr;
use slop_koala_bear::KoalaBear;

use crate::TaskScope;

/// # Safety
pub unsafe trait ScanKernel<F> {
    fn single_block_scan_kernel_large_bb31_septic_curve() -> KernelPtr;
    fn scan_kernel_large_bb31_septic_curve() -> KernelPtr;
}

unsafe impl ScanKernel<KoalaBear> for TaskScope {
    fn single_block_scan_kernel_large_bb31_septic_curve() -> KernelPtr {
        unsafe { csl_sys::scan::single_block_scan_kernel_large_bb31_septic_curve() }
    }
    fn scan_kernel_large_bb31_septic_curve() -> KernelPtr {
        unsafe { csl_sys::scan::scan_kernel_large_bb31_septic_curve() }
    }
}
