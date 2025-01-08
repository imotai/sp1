use crate::runtime::KernelPtr;

extern "C" {
    pub fn addKernelu32Ptr() -> KernelPtr;
}
