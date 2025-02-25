use crate::runtime::KernelPtr;

extern "C" {
    pub fn jagged_baby_bear_extension_populate() -> KernelPtr;
}
