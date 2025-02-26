use crate::runtime::KernelPtr;

extern "C" {
    pub fn interpolate_row_baby_bear_kernel() -> KernelPtr;
    pub fn interpolate_row_baby_bear_extension_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_32_baby_bear_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_64_baby_bear_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_128_baby_bear_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_256_baby_bear_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_512_baby_bear_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_1024_baby_bear_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_32_baby_bear_extension_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_64_baby_bear_extension_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_128_baby_bear_extension_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_256_baby_bear_extension_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_512_baby_bear_extension_kernel() -> KernelPtr;
    pub fn constraint_poly_eval_1024_baby_bear_extension_kernel() -> KernelPtr;
}
