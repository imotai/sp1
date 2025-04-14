use crate::runtime::KernelPtr;

extern "C" {
    pub fn core_global_generate_trace_decompress_kernel() -> KernelPtr;
    pub fn core_global_generate_trace_finalize_kernel() -> KernelPtr;
}
