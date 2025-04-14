use csl_sys::runtime::KernelPtr;
use slop_baby_bear::BabyBear;

use crate::TaskScope;

/// # Safety
pub unsafe trait TracegenCoreGlobalKernel<F> {
    fn tracegen_core_global_decompress_kernel() -> KernelPtr;
    fn tracegen_core_global_finalize_kernel() -> KernelPtr;
}

unsafe impl TracegenCoreGlobalKernel<BabyBear> for TaskScope {
    fn tracegen_core_global_decompress_kernel() -> KernelPtr {
        unsafe { csl_sys::tracegen::core_global_generate_trace_decompress_kernel() }
    }
    fn tracegen_core_global_finalize_kernel() -> KernelPtr {
        unsafe { csl_sys::tracegen::core_global_generate_trace_finalize_kernel() }
    }
}
