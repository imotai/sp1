use crate::runtime::KernelPtr;

extern "C" {
    pub fn jagged_baby_bear_extension_populate() -> KernelPtr;

    pub fn jagged_baby_bear_base_ext_sum_as_poly() -> KernelPtr;
    pub fn jagged_baby_bear_extension_virtual_fix_last_variable() -> KernelPtr;
}
