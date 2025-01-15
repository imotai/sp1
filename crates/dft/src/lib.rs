mod sppark;

use std::error::Error;

pub use sppark::*;

use csl_device::{
    mem::DeviceData,
    tensor::{TensorView, TensorViewMut},
    DeviceScope,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DftOrdering {
    Normal,
    BitReversed,
}

pub trait Dft<T: DeviceData, A: DeviceScope> {
    type Error: Error;
    /// Perofrms a discrete Fourier transform along the last dimension of the input tensor.
    fn dft<'a>(
        &self,
        src: impl Into<TensorView<'a, T, A>>,
        dst: impl Into<TensorViewMut<'a, T, A>>,
        shift: T,
        ordering: DftOrdering,
        scope: &A,
    ) -> Result<(), Self::Error>
    where
        A: 'a;
}
