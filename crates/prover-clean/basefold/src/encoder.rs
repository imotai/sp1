use csl_cuda::TaskScope;
use cslpc_utils::Felt;
use slop_dft::DftOrdering;

use csl_cuda::{
    sys::dft::{batch_coset_dft, sppark_init_default_stream},
    CudaError, DeviceCopy,
};
use slop_algebra::Field;
use slop_koala_bear::KoalaBear;
use slop_tensor::{Tensor, TensorView};

pub fn encode_batch<'a>(
    dft: SpparkDftKoalaBear,
    log_blowup: u32,
    data: TensorView<'a, Felt, TaskScope>,
) -> Result<Tensor<Felt, TaskScope>, CudaError> {
    let dft = dft.dft(data, log_blowup as usize, DftOrdering::BitReversed, 1).unwrap();
    Ok(dft)
}

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
pub struct SpparkDft<F, T>(pub F, std::marker::PhantomData<T>);

impl<T: Field, F: SpparkCudaDftSys<T>> SpparkDft<F, T> {
    /// Performs a discrete Fourier transform along the last dimension of the input tensor.
    fn coset_dft_into<'a>(
        &self,
        src: TensorView<'a, T, TaskScope>,
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

    fn coset_dft<'a>(
        &self,
        src: TensorView<'a, T, TaskScope>,
        shift: T,
        log_blowup: usize,
        ordering: DftOrdering,
        dim: usize,
    ) -> Result<Tensor<T, TaskScope>, CudaError> {
        let mut sizes = src.sizes().to_vec();
        sizes[dim] <<= log_blowup;
        let mut dst = Tensor::with_sizes_in(sizes, src.backend().clone());
        self.coset_dft_into(src, &mut dst, shift, log_blowup, ordering, dim)?;
        Ok(dst)
    }

    fn dft<'a>(
        &self,
        src: TensorView<'a, T, TaskScope>,
        log_blowup: usize,
        ordering: DftOrdering,
        dim: usize,
    ) -> Result<Tensor<T, TaskScope>, CudaError> {
        self.coset_dft(src, T::one(), log_blowup, ordering, dim)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SpparkB31Kernels;

pub type SpparkDftKoalaBear = SpparkDft<SpparkB31Kernels, Felt>;

impl Default for SpparkB31Kernels {
    fn default() -> Self {
        unsafe { sppark_init_default_stream() };
        Self
    }
}

impl SpparkCudaDftSys<KoalaBear> for SpparkB31Kernels {
    unsafe fn dft_unchecked(
        &self,
        d_out: *mut KoalaBear,
        d_in: *mut KoalaBear,
        lg_domain_size: u32,
        lg_blowup: u32,
        shift: KoalaBear,
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
    use itertools::Itertools;
    use rand::thread_rng;
    use slop_algebra::AbstractField;
    use slop_alloc::IntoHost;
    use slop_dft::{p3::Radix2DitParallel, Dft};

    use super::*;

    #[tokio::test]
    async fn test_batch_coset_dft() {
        let mut rng = thread_rng();

        let log_degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let log_blowup = 1;
        let shift = KoalaBear::generator();
        let batch_size = 16;

        let p3_dft = Radix2DitParallel;

        for log_d in log_degrees.iter() {
            let d = 1 << log_d;

            let tensor_h = Tensor::<KoalaBear>::rand(&mut rng, [d, batch_size]);

            let tensor_h_sent = tensor_h.clone();
            let result = csl_cuda::spawn(move |t| async move {
                let tensor = t.into_device(tensor_h_sent).await.unwrap().transpose();
                let dft = SpparkDftKoalaBear::default();
                let result = dft
                    .coset_dft(tensor.as_view(), shift, log_blowup, DftOrdering::BitReversed, 1)
                    .unwrap();

                let result = result.transpose();
                result.into_host().await.unwrap()
            })
            .await
            .unwrap();

            let expected_result = p3_dft
                .coset_dft(&tensor_h, shift, log_blowup, DftOrdering::BitReversed, 0)
                .unwrap();

            for (i, (r, e)) in
                result.as_slice().iter().zip_eq(expected_result.as_slice()).enumerate()
            {
                assert_eq!(r, e, "Mismatch at index {i}");
            }
        }
    }
}
