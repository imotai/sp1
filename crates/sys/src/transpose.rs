use crate::runtime::KernelPtr;

#[link_name = "transpose"]
#[allow(unused_attributes)]
extern "C" {
    pub fn transpose_kernel_baby_bear() -> KernelPtr;
    pub fn transpose_kernel_baby_bear_digest() -> KernelPtr;
    pub fn transpose_kernel_u32() -> KernelPtr;
    pub fn transpose_kernel_u32_digest() -> KernelPtr;
    pub fn transpose_kernel_baby_bear_extension() -> KernelPtr;
}
