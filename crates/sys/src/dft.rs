use slop_baby_bear::BabyBear;

use crate::runtime::{CudaRustError, CudaStreamHandle, DEFAULT_STREAM};

/// # Safety
///
/// TODO
pub unsafe fn sppark_init_default_stream() -> CudaRustError {
    sppark_init(DEFAULT_STREAM)
}

extern "C" {
    pub fn sppark_init(stream: CudaStreamHandle) -> CudaRustError;

    pub fn batch_coset_dft(
        d_out: *mut BabyBear,
        d_in: *mut BabyBear,
        lg_domain_size: u32,
        lg_blowup: u32,
        shift: BabyBear,
        poly_count: u32,
        is_bit_rev: bool,
        stream: CudaStreamHandle,
    ) -> CudaRustError;

    pub fn batch_lde_shift_in_place(
        d_inout: *mut BabyBear,
        lg_domain_size: u32,
        lg_blowup: u32,
        shift: BabyBear,
        poly_count: u32,
        is_bit_rev: bool,
        stream: CudaStreamHandle,
    ) -> CudaRustError;

    pub fn batch_coset_dft_in_place(
        d_inout: *mut BabyBear,
        lg_domain_size: u32,
        lg_blowup: u32,
        shift: BabyBear,
        poly_count: u32,
        is_bit_rev: bool,
        stream: CudaStreamHandle,
    ) -> CudaRustError;

    pub fn batch_NTT(
        d_inout: *mut BabyBear,
        lg_domain_size: u32,
        poly_count: u32,
        stream: CudaStreamHandle,
    ) -> CudaRustError;

    pub fn batch_iNTT(
        d_inout: *mut BabyBear,
        lg_domain_size: u32,
        poly_count: u32,
        stream: CudaStreamHandle,
    ) -> CudaRustError;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sppark_init() {
        unsafe { sppark_init_default_stream() };
    }
}
