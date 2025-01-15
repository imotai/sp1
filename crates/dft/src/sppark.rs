use csl_device::{
    cuda::{CudaError, TaskScope},
    mem::DeviceData,
    tensor::{TensorView, TensorViewMut},
};
use slop_algebra::Field;

use crate::{Dft, DftOrdering};

pub trait SpparkCudaDftSys<T: DeviceData> {
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
        scope: &TaskScope,
    ) -> Result<(), CudaError>;
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpparkDft<F>(pub F);

impl<T: DeviceData + Field, F: SpparkCudaDftSys<T>> Dft<T, TaskScope> for SpparkDft<F> {
    type Error = CudaError;

    /// Perofrms a discrete Fourier transform along the last dimension of the input tensor.
    fn dft<'a>(
        &self,
        src: impl Into<TensorView<'a, T, TaskScope>>,
        dst: impl Into<TensorViewMut<'a, T, TaskScope>>,
        shift: T,
        ordering: DftOrdering,
        scope: &TaskScope,
    ) -> Result<(), CudaError> {
        let src = src.into();
        let d_in = src.as_ptr() as *mut T;
        let mut dst = dst.into();
        let d_out = dst.as_mut_ptr();
        let src_dimensions = src.sizes();
        let dst_dimensions = dst.sizes();

        let shift = shift / T::generator();

        assert_eq!(
            src_dimensions[0], dst_dimensions[0],
            "dimension mismatch along the first dimension"
        );

        let lg_domain_size = src_dimensions[1].ilog2();
        let lg_blowup = dst_dimensions[1].ilog2() - lg_domain_size;
        let batch_size = src_dimensions[0] as u32;
        let bit_rev_output = ordering == DftOrdering::BitReversed;

        unsafe {
            self.0.dft_unchecked(
                d_out,
                d_in,
                lg_domain_size,
                lg_blowup,
                shift,
                batch_size,
                bit_rev_output,
                scope,
            )
        }
    }
}
