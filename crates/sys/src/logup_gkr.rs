use crate::runtime::KernelPtr;

#[link_name = "logup_gkr"]
#[allow(unused_attributes)]
extern "C" {
    pub fn gkr_circuit_transition_baby_bear_kernel() -> KernelPtr;
    pub fn gkr_circuit_transition_baby_bear_extension_kernel() -> KernelPtr;
}

#[link_name = "logup_gkr_sum"]
#[allow(unused_attributes)]
extern "C" {
    pub fn logup_gkr_poly_baby_bear_kernel() -> KernelPtr;
    pub fn logup_gkr_poly_baby_bear_extension_kernel() -> KernelPtr;
}

#[link_name = "logup_tracegen"]
#[allow(unused_attributes)]
extern "C" {
    pub fn gkr_tracegen_kernel() -> KernelPtr;
}
