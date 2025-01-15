use crate::runtime::KernelPtr;

extern "C" {
    pub fn leaf_hash_poseidon_2_baby_bear_16_kernel() -> KernelPtr;

    pub fn compress_poseidon_2_baby_bear_16_kernel() -> KernelPtr;

    pub fn compute_paths_poseidon_2_baby_bear_16_kernel() -> KernelPtr;

    pub fn compute_openings_poseidon_2_baby_bear_16_kernel() -> KernelPtr;
}
