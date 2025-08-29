use std::ffi::c_void;

use crate::runtime::{CudaRustError, CudaStreamHandle, KernelPtr};

extern "C" {

    // Sum kernels

    pub fn sum_kernel_u32() -> KernelPtr;
    pub fn sum_kernel_felt() -> KernelPtr;
    pub fn sum_kernel_ext() -> KernelPtr;

    // Reduce kernels

    pub fn reduce_kernel_felt() -> KernelPtr;
    pub fn reduce_kernel_ext() -> KernelPtr;

    // Zerocheck kernels
    pub fn zerocheck_sum_as_poly_base_ext_kernel() -> KernelPtr;
    pub fn zerocheck_sum_as_poly_ext_ext_kernel() -> KernelPtr;

    pub fn zerocheck_fix_last_variable_and_sum_as_poly_base_ext_kernel() -> KernelPtr;
    pub fn zerocheck_fix_last_variable_and_sum_as_poly_ext_ext_kernel() -> KernelPtr;

    // Hadamard kernels
    pub fn hadamard_sum_as_poly_base_ext_kernel() -> KernelPtr;
    pub fn hadamard_sum_as_poly_ext_ext_kernel() -> KernelPtr;

    pub fn hadamard_fix_last_variable_and_sum_as_poly_base_ext_kernel() -> KernelPtr;
    pub fn hadamard_fix_last_variable_and_sum_as_poly_ext_ext_kernel() -> KernelPtr;

    pub fn fix_last_variable_ext_ext_kernel() -> KernelPtr;

    // Populate restrict eq
    pub fn populate_restrict_eq_host(
        src: *const c_void,
        len: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;
    pub fn populate_restrict_eq_device(
        src: *const c_void,
        len: usize,
        stream: CudaStreamHandle,
    ) -> CudaRustError;

    // Look ahead kernels - FIX_TILE=32
    pub fn round_kernel_1_32_2_2_false() -> KernelPtr;
    pub fn round_kernel_2_32_2_2_true() -> KernelPtr;
    pub fn round_kernel_2_32_2_2_false() -> KernelPtr;
    pub fn round_kernel_4_32_2_2_true() -> KernelPtr;
    pub fn round_kernel_4_32_2_2_false() -> KernelPtr;
    pub fn round_kernel_8_32_2_2_true() -> KernelPtr;
    pub fn round_kernel_8_32_2_2_false() -> KernelPtr;

    // Look ahead kernels - FIX_TILE=64
    pub fn round_kernel_1_64_2_2_false() -> KernelPtr;
    pub fn round_kernel_2_64_2_2_true() -> KernelPtr;
    pub fn round_kernel_2_64_2_2_false() -> KernelPtr;
    pub fn round_kernel_4_64_2_2_true() -> KernelPtr;
    pub fn round_kernel_4_64_2_2_false() -> KernelPtr;
    pub fn round_kernel_8_64_2_2_true() -> KernelPtr;
    pub fn round_kernel_8_64_2_2_false() -> KernelPtr;

    // Look ahead kernels - NUM_POINTS=3, FIX_TILE=32
    pub fn round_kernel_1_32_2_3_false() -> KernelPtr;
    pub fn round_kernel_2_32_2_3_true() -> KernelPtr;
    pub fn round_kernel_2_32_2_3_false() -> KernelPtr;
    pub fn round_kernel_4_32_2_3_true() -> KernelPtr;
    pub fn round_kernel_4_32_2_3_false() -> KernelPtr;
    pub fn round_kernel_8_32_2_3_true() -> KernelPtr;
    pub fn round_kernel_8_32_2_3_false() -> KernelPtr;

    // Look ahead kernels - NUM_POINTS=3, FIX_TILE=64
    pub fn round_kernel_1_64_2_3_false() -> KernelPtr;
    pub fn round_kernel_1_64_4_8_false() -> KernelPtr;
    pub fn round_kernel_2_64_2_3_true() -> KernelPtr;
    pub fn round_kernel_2_64_2_3_false() -> KernelPtr;
    pub fn round_kernel_4_64_2_3_true() -> KernelPtr;
    pub fn round_kernel_4_64_2_3_false() -> KernelPtr;
    pub fn round_kernel_4_64_4_8_true() -> KernelPtr;
    pub fn round_kernel_4_64_4_8_false() -> KernelPtr;
    pub fn round_kernel_8_64_2_3_true() -> KernelPtr;
    pub fn round_kernel_8_64_2_3_false() -> KernelPtr;

    // Look ahead kernels - FIX_TILE=128
    pub fn round_kernel_1_128_4_8_false() -> KernelPtr;
}
