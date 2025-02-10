use std::error::Error;

use slop_algebra::TwoAdicField;
use slop_alloc::{Backend, CpuBackend};
use slop_basefold::{FriConfig, RsCodeWord};
use slop_dft::{Dft, DftOrdering};
use slop_tensor::Tensor;

pub trait ReedSolomonEncoder<F: TwoAdicField, A: Backend = CpuBackend> {
    /// The error type returned by the encoder.
    type Error: Error;

    /// Encodes the input into a new codeword.
    fn encode(&self, data: &Tensor<F, A>) -> Result<RsCodeWord<F, A>, Self::Error>;
}

pub struct CpuDftEncoder<F: TwoAdicField, D> {
    config: FriConfig<F>,
    dft: D,
}

impl<F: TwoAdicField, D: Dft<F>> ReedSolomonEncoder<F> for CpuDftEncoder<F, D> {
    type Error = D::Error;

    fn encode(&self, data: &Tensor<F>) -> Result<RsCodeWord<F>, Self::Error> {
        assert_eq!(data.sizes().len(), 2, "Expected a 2D tensor");
        // Perform a DFT along the first axis of the tensor (assumed to be the long dimension).
        let dft = self.dft.dft(data, self.config.log_blowup(), DftOrdering::BitReversed, 0)?;
        Ok(RsCodeWord { data: dft })
    }
}
