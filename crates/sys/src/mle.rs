use crate::runtime::KernelPtr;

extern "C" {
    pub fn partial_lagrange_baby_bear() -> KernelPtr;
    pub fn partial_lagrange_baby_bear_extension() -> KernelPtr;
}
