use csl_sys::runtime::KernelPtr;
use slop_baby_bear::BabyBear;

use crate::TaskScope;

/// # Safety
pub unsafe trait TracegenRiscvGlobalKernel<F> {
    fn tracegen_riscv_global_decompress_kernel() -> KernelPtr;
    fn tracegen_riscv_global_finalize_kernel() -> KernelPtr;
}

unsafe impl TracegenRiscvGlobalKernel<BabyBear> for TaskScope {
    fn tracegen_riscv_global_decompress_kernel() -> KernelPtr {
        unsafe { csl_sys::tracegen::riscv_global_generate_trace_decompress_kernel() }
    }
    fn tracegen_riscv_global_finalize_kernel() -> KernelPtr {
        unsafe { csl_sys::tracegen::riscv_global_generate_trace_finalize_kernel() }
    }
}

/// # Safety
pub unsafe trait TracegenPreprocessedRecursionBaseAluKernel<F> {
    fn tracegen_preprocessed_recursion_base_alu_kernel() -> KernelPtr;
}

unsafe impl TracegenPreprocessedRecursionBaseAluKernel<BabyBear> for TaskScope {
    fn tracegen_preprocessed_recursion_base_alu_kernel() -> KernelPtr {
        unsafe {
            csl_sys::tracegen::recursion_base_alu_generate_preprocessed_trace_baby_bear_kernel()
        }
    }
}

/// # Safety
pub unsafe trait TracegenRecursionBaseAluKernel<F> {
    fn tracegen_recursion_base_alu_kernel() -> KernelPtr;
}

unsafe impl TracegenRecursionBaseAluKernel<BabyBear> for TaskScope {
    fn tracegen_recursion_base_alu_kernel() -> KernelPtr {
        unsafe { csl_sys::tracegen::recursion_base_alu_generate_trace_baby_bear_kernel() }
    }
}

/// # Safety
pub unsafe trait TracegenPreprocessedRecursionExtAluKernel<F> {
    fn tracegen_preprocessed_recursion_ext_alu_kernel() -> KernelPtr;
}

unsafe impl TracegenPreprocessedRecursionExtAluKernel<BabyBear> for TaskScope {
    fn tracegen_preprocessed_recursion_ext_alu_kernel() -> KernelPtr {
        unsafe {
            csl_sys::tracegen::recursion_ext_alu_generate_preprocessed_trace_baby_bear_kernel()
        }
    }
}

/// # Safety
pub unsafe trait TracegenRecursionExtAluKernel<F> {
    fn tracegen_recursion_ext_alu_kernel() -> KernelPtr;
}

unsafe impl TracegenRecursionExtAluKernel<BabyBear> for TaskScope {
    fn tracegen_recursion_ext_alu_kernel() -> KernelPtr {
        unsafe { csl_sys::tracegen::recursion_ext_alu_generate_trace_baby_bear_kernel() }
    }
}

/// # Safety
pub unsafe trait TracegenPreprocessedRecursionPoseidon2WideKernel<F> {
    fn tracegen_preprocessed_recursion_poseidon2_wide_kernel() -> KernelPtr;
}

unsafe impl TracegenPreprocessedRecursionPoseidon2WideKernel<BabyBear> for TaskScope {
    fn tracegen_preprocessed_recursion_poseidon2_wide_kernel() -> KernelPtr {
        unsafe {
            csl_sys::tracegen::recursion_poseidon2_wide_generate_preprocessed_trace_baby_bear_kernel(
            )
        }
    }
}

/// # Safety
pub unsafe trait TracegenRecursionPoseidon2WideKernel<F> {
    fn tracegen_recursion_poseidon2_wide_kernel() -> KernelPtr;
}

unsafe impl TracegenRecursionPoseidon2WideKernel<BabyBear> for TaskScope {
    fn tracegen_recursion_poseidon2_wide_kernel() -> KernelPtr {
        unsafe { csl_sys::tracegen::recursion_poseidon2_wide_generate_trace_baby_bear_kernel() }
    }
}

/// # Safety
pub unsafe trait TracegenPreprocessedRecursionSelectKernel<F> {
    fn tracegen_preprocessed_recursion_select_kernel() -> KernelPtr;
}

unsafe impl TracegenPreprocessedRecursionSelectKernel<BabyBear> for TaskScope {
    fn tracegen_preprocessed_recursion_select_kernel() -> KernelPtr {
        unsafe {
            csl_sys::tracegen::recursion_select_generate_preprocessed_trace_baby_bear_kernel()
        }
    }
}

/// # Safety
pub unsafe trait TracegenRecursionSelectKernel<F> {
    fn tracegen_recursion_select_kernel() -> KernelPtr;
}

unsafe impl TracegenRecursionSelectKernel<BabyBear> for TaskScope {
    fn tracegen_recursion_select_kernel() -> KernelPtr {
        unsafe { csl_sys::tracegen::recursion_select_generate_trace_baby_bear_kernel() }
    }
}

/// # Safety
pub unsafe trait TracegenRecursionPrefixSumChecksKernel<F> {
    fn tracegen_recursion_prefix_sum_checks_kernel() -> KernelPtr;
}

unsafe impl TracegenRecursionPrefixSumChecksKernel<BabyBear> for TaskScope {
    fn tracegen_recursion_prefix_sum_checks_kernel() -> KernelPtr {
        unsafe { csl_sys::tracegen::recursion_prefix_sum_checks_generate_trace_baby_bear_kernel() }
    }
}
