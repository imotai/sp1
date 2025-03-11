use crate::runtime::KernelPtr;

extern "C" {
    pub fn partial_lagrange_baby_bear() -> KernelPtr;
    pub fn partial_lagrange_baby_bear_extension() -> KernelPtr;

    pub fn mle_fix_last_variable_baby_bear_base_base_constant_padding() -> KernelPtr;
    pub fn mle_fix_last_variable_baby_bear_base_extension_constant_padding() -> KernelPtr;
    pub fn mle_fix_last_variable_baby_bear_ext_ext_constant_padding() -> KernelPtr;

    pub fn mle_fix_last_variable_baby_bear_base_base_padded() -> KernelPtr;
    pub fn mle_fix_last_variable_baby_bear_base_extension_padded() -> KernelPtr;
    pub fn mle_fix_last_variable_baby_bear_ext_ext_padded() -> KernelPtr;

    pub fn mle_fold_baby_bear_base_base() -> KernelPtr;
    pub fn mle_fold_baby_bear_base_extension() -> KernelPtr;
    pub fn mle_fold_baby_bear_ext_ext() -> KernelPtr;

    pub fn mle_fix_last_variable_in_place_baby_bear_base() -> KernelPtr;
    pub fn mle_fix_last_variable_in_place_baby_bear_extension() -> KernelPtr;
}
