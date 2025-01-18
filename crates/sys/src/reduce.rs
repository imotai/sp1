use crate::runtime::KernelPtr;

extern "C" {
    pub fn baby_bear_sum_block_reduce_kernel() -> KernelPtr;
    pub fn baby_bear_sum_partial_block_reduce_kernel() -> KernelPtr;
    pub fn baby_bear_extension_sum_block_reduce_kernel() -> KernelPtr;
    pub fn baby_bear_extension_sum_partial_block_reduce_kernel() -> KernelPtr;

    pub fn partial_inner_product_baby_bear_kernel() -> KernelPtr;
    pub fn partial_inner_product_baby_bear_extension_kernel() -> KernelPtr;
    pub fn partial_inner_product_baby_bear_base_extension_kernel() -> KernelPtr;

    pub fn partial_dot_baby_bear_kernel() -> KernelPtr;
    pub fn partial_dot_baby_bear_extension_kernel() -> KernelPtr;
    pub fn partial_dot_baby_bear_base_extension_kernel() -> KernelPtr;
}
