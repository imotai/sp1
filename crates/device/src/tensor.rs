use std::ops::{Deref, Index};

use csl_alloc::{Allocator, TryReserveError};
use thiserror::Error;

use crate::{
    mem::{DeviceData, Init},
    Buffer,
};

#[derive(Copy, Clone, Debug)]
pub struct Dimensions([usize; 2]);

#[derive(Copy, Clone, Debug)]
pub struct TensorIndex([usize; 2]);

pub struct Tensor<T: DeviceData, A: Allocator> {
    pub(crate) storage: Buffer<T, A>,
    pub(crate) dimensions: Dimensions,
}

#[derive(Debug, Clone, Copy, Error)]
#[error("total number of elements must match, expected {0}, got {1}")]
pub struct ShapeError(usize, usize);

impl<T: DeviceData, A: Allocator> Tensor<T, A> {
    #[inline]
    pub fn with_dimensions_in(dimensions: impl Into<Dimensions>, allocator: A) -> Self {
        let dimensions = dimensions.into();
        Self { storage: Buffer::with_capacity_in(dimensions.total_len(), allocator), dimensions }
    }

    #[inline]
    pub fn try_with_dimensions_in(
        dimensions: impl Into<Dimensions>,
        allocator: A,
    ) -> Result<Self, TryReserveError> {
        let dimensions = dimensions.into();
        Ok(Self {
            storage: Buffer::try_with_capacity_in(dimensions.total_len(), allocator)?,
            dimensions,
        })
    }

    pub fn reshape_in_place(
        &mut self,
        dimensions: impl Into<Dimensions>,
    ) -> Result<(), ShapeError> {
        let dimensions = dimensions.into();
        self.dimensions.compatible(&dimensions)?;
        self.dimensions = dimensions;
        Ok(())
    }

    #[inline]
    pub fn reshape(mut self, dimensions: impl Into<Dimensions>) -> Result<Self, ShapeError> {
        self.reshape_in_place(dimensions)?;
        Ok(self)
    }

    #[inline]
    pub fn flatten_in_place(&mut self) -> Result<(), ShapeError> {
        self.reshape_in_place(self.dimensions.total_len())?;
        Ok(())
    }

    #[inline]
    pub fn flatten(mut self) -> Result<Self, ShapeError> {
        self.flatten_in_place()?;
        Ok(self)
    }

    pub fn into_buffer(self) -> Buffer<T, A> {
        self.storage
    }

    /// Returns the dimensions of the tensor.
    #[inline]
    pub fn dimensions(&self) -> &Dimensions {
        &self.dimensions
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.storage.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.storage.as_mut_ptr()
    }

    #[inline]
    pub fn view(&self) -> TensorView<T, A> {
        TensorView { view: self, dimensions: self.dimensions }
    }

    #[inline]
    pub fn view_mut(&mut self) -> TensorViewMut<T, A> {
        TensorViewMut { dimensions: self.dimensions, view: self }
    }

    /// # Safety
    ///
    /// See [std::mem::MaybeUninit::assume_init].
    #[inline]
    pub unsafe fn assume_init(&mut self) {
        self.storage.set_len(self.storage.capacity());
    }

    #[inline]
    pub fn index_map(&self, index: impl Into<TensorIndex>) -> usize {
        let index = index.into();
        index[0] * self.dimensions[0] + index[1]
    }
}

impl Dimensions {
    #[inline]
    fn total_len(&self) -> usize {
        self.0.iter().product()
    }

    #[inline]
    fn compatible(&self, other: &Dimensions) -> Result<(), ShapeError> {
        if self.total_len() != other.total_len() {
            return Err(ShapeError(self.total_len(), other.total_len()));
        }
        Ok(())
    }
}

impl Deref for Dimensions {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<usize> for Dimensions {
    fn from(value: usize) -> Self {
        Dimensions([value, 1])
    }
}

impl From<(usize, usize)> for Dimensions {
    fn from(value: (usize, usize)) -> Self {
        Dimensions([value.0, value.1])
    }
}

impl From<[usize; 2]> for Dimensions {
    fn from(value: [usize; 2]) -> Self {
        Dimensions(value)
    }
}

pub struct TensorView<'a, T: DeviceData, A: Allocator> {
    view: &'a Tensor<T, A>,
    dimensions: Dimensions,
}

impl<'a, T: DeviceData, A: Allocator> TensorView<'a, T, A> {
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.view.as_ptr()
    }

    #[inline]
    pub fn dimensions(&self) -> &Dimensions {
        &self.dimensions
    }

    #[inline]
    pub fn reshape(
        self,
        dimensions: impl Into<Dimensions>,
    ) -> Result<TensorView<'a, T, A>, ShapeError> {
        let dimensions = dimensions.into();
        self.dimensions.compatible(&dimensions)?;
        Ok(TensorView { view: self.view, dimensions })
    }
}

pub struct TensorViewMut<'a, T: DeviceData, A: Allocator> {
    view: &'a mut Tensor<T, A>,
    dimensions: Dimensions,
}

impl<'a, T: DeviceData, A: Allocator> TensorViewMut<'a, T, A> {
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.view.as_mut_ptr()
    }

    #[inline]
    pub fn dimensions(&self) -> &Dimensions {
        &self.dimensions
    }

    #[inline]
    pub fn reshape(
        self,
        dimensions: impl Into<Dimensions>,
    ) -> Result<TensorViewMut<'a, T, A>, ShapeError> {
        let dimensions = dimensions.into();
        self.dimensions.compatible(&dimensions)?;
        Ok(TensorViewMut { view: self.view, dimensions })
    }
}

impl<'a, T: DeviceData, A: Allocator> From<&'a Tensor<T, A>> for TensorView<'a, T, A> {
    fn from(tensor: &'a Tensor<T, A>) -> Self {
        tensor.view()
    }
}

impl<'a, T: DeviceData, A: Allocator> From<&'a mut Tensor<T, A>> for TensorViewMut<'a, T, A> {
    fn from(tensor: &'a mut Tensor<T, A>) -> Self {
        tensor.view_mut()
    }
}

impl<T: DeviceData, A: Allocator> From<Buffer<T, A>> for Tensor<T, A> {
    fn from(buffer: Buffer<T, A>) -> Self {
        let len = buffer.len();
        Self { storage: buffer, dimensions: Dimensions([len, 1]) }
    }
}

impl From<(usize, usize)> for TensorIndex {
    fn from(value: (usize, usize)) -> Self {
        TensorIndex([value.0, value.1])
    }
}

impl From<[usize; 2]> for TensorIndex {
    fn from(value: [usize; 2]) -> Self {
        TensorIndex(value)
    }
}

impl Deref for TensorIndex {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: DeviceData, A: Allocator, I: Into<TensorIndex>> Index<I> for Tensor<T, A> {
    type Output = Init<T, A>;

    fn index(&self, index: I) -> &Self::Output {
        let index = self.index_map(index.into());
        &self.storage[index]
    }
}
