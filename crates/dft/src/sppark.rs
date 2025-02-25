use csl_cuda::{
    sys::dft::{batch_coset_dft, sppark_init_default_stream},
    CudaError, DeviceCopy, TaskScope,
};
use slop_algebra::Field;

use slop_baby_bear::BabyBear;
use slop_dft::{Dft, DftOrdering};
use slop_tensor::Tensor;

pub trait SpparkCudaDftSys<T: DeviceCopy>: 'static + Send + Sync {
    /// # Safety
    ///
    /// The caller must ensure the validity of pointers, allocation size, and lifetimes.
    #[allow(clippy::too_many_arguments)]
    unsafe fn dft_unchecked(
        &self,
        d_out: *mut T,
        d_in: *mut T,
        lg_domain_size: u32,
        lg_blowup: u32,
        shift: T,
        batch_size: u32,
        bit_rev_output: bool,
        backend: &TaskScope,
    ) -> Result<(), CudaError>;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpparkDft<F>(pub F);

impl<T: Field, F: SpparkCudaDftSys<T>> Dft<T, TaskScope> for SpparkDft<F> {
    type Error = CudaError;

    /// Perofrms a discrete Fourier transform along the last dimension of the input tensor.
    fn coset_dft_into(
        &self,
        src: &Tensor<T, TaskScope>,
        dst: &mut Tensor<T, TaskScope>,
        shift: T,
        log_blowup: usize,
        ordering: DftOrdering,
        dim: usize,
    ) -> Result<(), CudaError> {
        let backend = src.backend();
        let d_in = src.as_ptr() as *mut T;
        let d_out = dst.as_mut_ptr();
        let src_dimensions = src.sizes();
        let dst_dimensions = dst.sizes();

        let shift = shift / T::generator();

        assert_eq!(
            src_dimensions[0], dst_dimensions[0],
            "dimension mismatch along the first dimension"
        );
        assert_eq!(src.sizes().len(), 2);
        assert_eq!(dst.sizes().len(), 2);
        assert_eq!(dim, 1);

        let lg_domain_size = src_dimensions[1].ilog2();
        let lg_blowup = dst_dimensions[1].ilog2() - lg_domain_size;
        assert_eq!(log_blowup, lg_blowup as usize);
        let batch_size = src_dimensions[0] as u32;
        let bit_rev_output = ordering == DftOrdering::BitReversed;

        unsafe {
            // Set the correct length for the output tensor
            dst.assume_init();
            // Call the function.
            self.0.dft_unchecked(
                d_out,
                d_in,
                lg_domain_size,
                lg_blowup,
                shift,
                batch_size,
                bit_rev_output,
                backend,
            )
        }
    }
}

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
        scope: &TaskScope,
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
    use rand::thread_rng;
    use slop_algebra::AbstractField;
    use slop_alloc::IntoHost;
    use slop_dft::p3::Radix2DitParallel;

    use super::*;

    #[tokio::test]
    async fn test_batch_coset_dft() {
        let mut rng = thread_rng();

        let log_degrees = [1, 2, 3, 4, 10, 11, 12, 13, 14, 15];
        let log_blowup = 1;
        let shift = BabyBear::generator();
        let batch_size = 16;

        let p3_dft = Radix2DitParallel;

        for log_d in log_degrees.iter() {
            let d = 1 << log_d;

            let tensor_h = Tensor::<BabyBear>::rand(&mut rng, [d, batch_size]);

            let tensor_h_sent = tensor_h.clone();
            let result = csl_cuda::task()
                .await
                .unwrap()
                .run(|t| async move {
                    let tensor = t.into_device(tensor_h_sent).await.unwrap().transpose();
                    let dft = SpparkDftBabyBear::default();
                    let result = dft
                        .coset_dft(&tensor, shift, log_blowup, DftOrdering::BitReversed, 1)
                        .unwrap();
                    let result = result.transpose();
                    result.into_host().await.unwrap()
                })
                .await
                .await
                .unwrap();

            let expected_result = p3_dft
                .coset_dft(&tensor_h, shift, log_blowup, DftOrdering::BitReversed, 0)
                .unwrap();

            for (r, e) in result.as_slice().iter().zip(expected_result.as_slice()) {
                assert_eq!(r, e);
            }
        }
    }
}
