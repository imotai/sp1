use crate::runtime::KernelPtr;

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

}
