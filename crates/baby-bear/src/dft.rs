use csl_device::cuda::CudaError;
use csl_dft::{SpparkCudaDftSys, SpparkDft};
use csl_sys::dft::{batch_coset_dft, sppark_init_default_stream};
use slop_baby_bear::BabyBear;

#[derive(Copy, Clone, Debug)]
pub struct SpparkB31Kernels;

pub type SpparkDftBabyBear = SpparkDft<SpparkB31Kernels>;

impl Default for SpparkB31Kernels {
    fn default() -> Self {
        unsafe { sppark_init_default_stream() };
        Self
    }
}

impl SpparkCudaDftSys<BabyBear> for SpparkB31Kernels {
    unsafe fn dft_unchecked(
        &self,
        d_out: *mut BabyBear,
        d_in: *mut BabyBear,
        lg_domain_size: u32,
        lg_blowup: u32,
        shift: BabyBear,
        batch_size: u32,
        bit_rev_output: bool,
        scope: &csl_device::cuda::TaskScope,
    ) -> Result<(), CudaError> {
        CudaError::result_from_ffi(batch_coset_dft(
            d_out,
            d_in,
            lg_domain_size,
            lg_blowup,
            shift,
            batch_size,
            bit_rev_output,
            scope.handle(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use csl_device::{DeviceBuffer, DeviceTensor};
    use csl_dft::{Dft, DftOrdering};
    use rand::thread_rng;
    use slop_algebra::AbstractField;
    use slop_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
    use slop_matrix::{bitrev::BitReversableMatrix, dense::RowMajorMatrix, Matrix};

    use super::*;

    #[tokio::test]
    async fn test_batch_coset_dft() {
        let mut rng = thread_rng();

        let log_degrees = 10..21;
        let log_blowup = 1;
        let batch_size = 16;

        let dft = SpparkDftBabyBear::default();
        let p3_dft = Radix2DitParallel;

        for log_d in log_degrees.clone() {
            let d = 1 << log_d;
            let lde_d = d << log_blowup;

            let mat_h = RowMajorMatrix::rand(&mut rng, d, batch_size);
            let mat_h_transposed = mat_h.clone().transpose();

            let mut mat_h_vals = mat_h.values;
            mat_h_vals.resize(lde_d * batch_size, BabyBear::zero());
            let mat_h_clone = RowMajorMatrix::new(mat_h_vals, batch_size);

            let expected_value = p3_dft.coset_dft_batch(mat_h_clone, BabyBear::generator());
            let expected_value = expected_value.to_row_major_matrix();
            let expected_value_bit_rev =
                expected_value.clone().bit_reverse_rows().to_row_major_matrix();

            let expected_value_transposed = expected_value.transpose();
            let expected_values_bit_rev_transposed = expected_value_bit_rev.transpose();

            csl_device::cuda::task()
                .await
                .unwrap()
                .run(|t| async move {
                    let src_values_host = mat_h_transposed.values;

                    let src =
                        DeviceBuffer::from_host_vec(src_values_host, t.clone()).await.unwrap();
                    let src = DeviceTensor::from(src).reshape([batch_size, d]).unwrap();
                    let mut dst = t.tensor::<BabyBear>([batch_size, lde_d]);
                    let mut dst_bit_rev = t.tensor::<BabyBear>([batch_size, lde_d]);
                    unsafe {
                        dst.assume_init();
                        dst_bit_rev.assume_init();
                    }

                    dft.dft(&src, &mut dst, BabyBear::generator(), DftOrdering::Normal, &t)
                        .unwrap();
                    dft.dft(
                        &src,
                        &mut dst_bit_rev,
                        BabyBear::generator(),
                        DftOrdering::BitReversed,
                        &t,
                    )
                    .unwrap();
                    let dst = dst.into_host().await.unwrap().into_buffer().into_vec();
                    let dst_bit_rev =
                        dst_bit_rev.into_host().await.unwrap().into_buffer().into_vec();

                    for (d_v, d_exp) in dst.into_iter().zip(expected_value_transposed.values) {
                        assert_eq!(d_v, d_exp);
                    }

                    for (d_v, d_exp) in
                        dst_bit_rev.into_iter().zip(expected_values_bit_rev_transposed.values)
                    {
                        assert_eq!(d_v, d_exp);
                    }
                })
                .await;
        }
    }
}
