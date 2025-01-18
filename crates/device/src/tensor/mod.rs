mod dot;
mod sum;
mod transpose;

pub use dot::*;
pub use sum::*;
pub use transpose::*;

use std::{
    marker::PhantomData,
    ops::{Index, IndexMut, Range, RangeFrom, RangeFull, RangeTo},
};

use arrayvec::ArrayVec;
use csl_alloc::TryReserveError;
use itertools::Itertools;
use thiserror::Error;

use crate::{mem::DeviceData, Buffer, DeviceScope, Init, Slice};

use crate::cuda::sync::CudaSend;
use crate::cuda::TaskScope;
use csl_derive::CudaSend;

const MAX_DIMENSIONS: usize = 10;

#[derive(CudaSend, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(C)]
pub struct Dimensions {
    sizes: ArrayVec<usize, MAX_DIMENSIONS>,
    strides: ArrayVec<usize, MAX_DIMENSIONS>,
}

#[derive(Debug, Clone, CudaSend)]
pub struct Tensor<T: DeviceData, A: DeviceScope> {
    pub(crate) storage: Buffer<T, A>,
    pub(crate) dimensions: Dimensions,
}

#[derive(Debug, Clone, Copy, Error)]
pub enum DimensionsError {
    #[error("Too many dimensions {0}, maximum number allowed is {MAX_DIMENSIONS}")]
    TooManyDimensions(usize),
    #[error("total number of elements must match, expected {0}, got {1}")]
    NumElementsMismatch(usize, usize),
}

impl<T: DeviceData, A: DeviceScope> Tensor<T, A> {
    #[inline]
    pub fn with_sizes_in(sizes: impl AsRef<[usize]>, allocator: A) -> Self {
        Self::try_with_sizes_in(sizes, allocator).unwrap()
    }

    #[inline]
    pub fn zeros_in(sizes: impl AsRef<[usize]>, allocator: A) -> Self {
        let mut tensor = Self::with_sizes_in(sizes, allocator);
        unsafe {
            tensor
                .storage
                .write_bytes_uncheked(0, tensor.total_len() * std::mem::size_of::<T>())
                .unwrap()
        };
        tensor
    }

    #[inline]
    pub fn try_with_sizes_in(
        sizes: impl AsRef<[usize]>,
        allocator: A,
    ) -> Result<Self, TryReserveError> {
        let dimensions = Dimensions::try_from(sizes.as_ref()).unwrap();
        Ok(Self {
            storage: Buffer::try_with_capacity_in(dimensions.total_len(), allocator)?,
            dimensions,
        })
    }

    pub fn reshape_in_place(&mut self, sizes: impl AsRef<[usize]>) -> Result<(), DimensionsError> {
        let dimensions: Dimensions = sizes.as_ref().try_into().unwrap();
        self.dimensions.compatible(&dimensions)?;
        self.dimensions = dimensions;
        Ok(())
    }

    #[inline]
    pub fn reshape(mut self, sizes: impl AsRef<[usize]>) -> Result<Self, DimensionsError> {
        self.reshape_in_place(sizes)?;
        Ok(self)
    }

    #[inline]
    pub fn flatten_in_place(&mut self) -> Result<(), DimensionsError> {
        self.reshape_in_place([self.dimensions.total_len()])?;
        Ok(())
    }

    #[inline]
    pub fn flatten(mut self) -> Result<Self, DimensionsError> {
        self.flatten_in_place()?;
        Ok(self)
    }

    #[inline]
    pub fn into_buffer(self) -> Buffer<T, A> {
        self.storage
    }

    #[inline]
    pub fn as_buffer(&self) -> &Buffer<T, A> {
        &self.storage
    }

    #[inline]
    pub fn scope(&self) -> &A {
        self.storage.allocator()
    }

    /// Returns the dimensions of the tensor.
    #[inline]
    pub fn sizes(&self) -> &[usize] {
        self.dimensions.sizes()
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        self.dimensions.strides()
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.storage.as_ptr()
    }

    #[inline]
    pub fn total_len(&self) -> usize {
        self.dimensions.total_len()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.storage.as_mut_ptr()
    }

    #[inline]
    pub fn as_view(&self) -> TensorView<T, A> {
        TensorView { ptr: self.as_ptr(), dimensions: self.dimensions.clone(), _marker: PhantomData }
    }

    #[inline]
    pub fn as_view_mut(&mut self) -> TensorViewMut<T, A> {
        TensorViewMut {
            ptr: self.as_mut_ptr(),
            dimensions: self.dimensions.clone(),
            _marker: PhantomData,
        }
    }

    #[inline]
    pub fn index(&self, index: impl TensorIndex) -> TensorView<T, A> {
        self.as_view().index(index)
    }

    #[inline]
    pub fn index_mut(&mut self, index: impl TensorIndex) -> TensorViewMut<T, A> {
        self.as_view_mut().index(index)
    }

    /// # Safety
    ///
    /// See [std::mem::MaybeUninit::assume_init].
    #[inline]
    pub unsafe fn assume_init(&mut self) {
        self.storage.set_len(self.storage.capacity());
    }

    #[inline]
    pub fn split(&self) -> impl Iterator<Item = TensorView<T, A>> {
        self.as_view().split()
    }

    #[inline]
    pub fn split_mut(&mut self) -> impl Iterator<Item = TensorViewMut<T, A>> {
        self.as_view_mut().split_mut()
    }
}

impl Dimensions {
    fn new(sizes: ArrayVec<usize, MAX_DIMENSIONS>) -> Self {
        let mut strides = ArrayVec::new();
        let mut stride = 1;
        for size in sizes.iter().rev() {
            strides.push(stride);
            stride *= size;
        }
        strides.reverse();
        Self { sizes, strides }
    }

    #[inline]
    pub fn total_len(&self) -> usize {
        self.sizes.iter().product()
    }

    #[inline]
    fn compatible(&self, other: &Dimensions) -> Result<(), DimensionsError> {
        if self.total_len() != other.total_len() {
            return Err(DimensionsError::NumElementsMismatch(self.total_len(), other.total_len()));
        }
        Ok(())
    }

    #[inline]
    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    #[inline]
    fn index_map(&self, index: impl AsRef<[usize]>) -> usize {
        index.as_ref().iter().zip_eq(self.strides.iter()).map(|(i, s)| i * s).sum()
    }
}

impl TryFrom<&[usize]> for Dimensions {
    type Error = DimensionsError;

    fn try_from(value: &[usize]) -> Result<Self, Self::Error> {
        let sizes = ArrayVec::try_from(value)
            .map_err(|_| DimensionsError::TooManyDimensions(value.len()))?;
        Ok(Self::new(sizes))
    }
}

impl FromIterator<usize> for Dimensions {
    #[inline]
    fn from_iter<T: IntoIterator<Item = usize>>(iter: T) -> Self {
        let sizes = ArrayVec::from_iter(iter);
        Self::new(sizes)
    }
}

impl<T: DeviceData, A: DeviceScope, I: AsRef<[usize]>> Index<I> for Tensor<T, A> {
    type Output = Init<T, A>;

    fn index(&self, index: I) -> &Self::Output {
        let index = self.dimensions.index_map(index);
        &self.storage[index]
    }
}

impl<T: DeviceData, A: DeviceScope, I: AsRef<[usize]>> IndexMut<I> for Tensor<T, A> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        let index = self.dimensions.index_map(index);
        &mut self.storage[index]
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
pub struct TensorView<'a, T: DeviceData, A: DeviceScope> {
    ptr: *const T,
    dimensions: Dimensions,
    /// Marker to ensure that the view is not used after the original tensor is freed.
    _marker: PhantomData<&'a Tensor<T, A>>,
}

impl<'a, T: DeviceData, A: DeviceScope> TensorView<'a, T, A> {
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    #[inline]
    pub fn sizes(&self) -> &[usize] {
        self.dimensions.sizes()
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        self.dimensions.strides()
    }

    #[inline]
    pub fn total_len(&self) -> usize {
        self.dimensions.total_len()
    }

    #[inline]
    pub fn shape(&self) -> &Dimensions {
        &self.dimensions
    }

    #[inline]
    pub fn flatten(self) -> TensorView<'a, T, A> {
        let total_len = self.total_len();
        self.reshape([total_len]).unwrap()
    }

    #[inline]
    pub fn reshape(
        self,
        sizes: impl AsRef<[usize]>,
    ) -> Result<TensorView<'a, T, A>, DimensionsError> {
        let dimensions: Dimensions = sizes.as_ref().try_into().unwrap();
        self.dimensions.compatible(&dimensions)?;
        Ok(TensorView { ptr: self.ptr, dimensions, _marker: PhantomData })
    }

    #[inline]
    pub fn index(mut self, index: impl TensorIndex) -> Self {
        let offset = index.dim_map(&mut self.dimensions);
        let ptr = unsafe { self.ptr.add(offset) };
        Self { ptr, dimensions: self.dimensions, _marker: PhantomData }
    }

    pub fn split(self) -> impl Iterator<Item = Self> {
        (0..self.dimensions.sizes()[0]).map(move |i| self.clone().index(i))
    }

    #[inline]
    pub fn as_slice(self) -> &'a Slice<T, A> {
        unsafe { Slice::from_raw_parts(self.ptr, self.dimensions.total_len()) }
    }
}

#[derive(Debug)]
pub struct TensorViewMut<'a, T: DeviceData, A: DeviceScope> {
    ptr: *mut T,
    dimensions: Dimensions,
    /// Marker to ensure that we get an exlusive reference, and that the view is not used after the
    /// original tensor is freed.
    _marker: PhantomData<&'a mut Tensor<T, A>>,
}

impl<'a, T: DeviceData, A: DeviceScope> TensorViewMut<'a, T, A> {
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    #[inline]
    pub fn sizes(&self) -> &[usize] {
        self.dimensions.sizes()
    }

    #[inline]
    pub fn shape(&self) -> &Dimensions {
        &self.dimensions
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        self.dimensions.strides()
    }

    #[inline]
    pub fn flatten(self) -> TensorViewMut<'a, T, A> {
        let total_len = self.total_len();
        self.reshape([total_len]).unwrap()
    }

    #[inline]
    pub fn reshape(
        self,
        sizes: impl AsRef<[usize]>,
    ) -> Result<TensorViewMut<'a, T, A>, DimensionsError> {
        let dimensions: Dimensions = sizes.as_ref().try_into().unwrap();
        self.dimensions.compatible(&dimensions)?;
        Ok(TensorViewMut { ptr: self.ptr, dimensions, _marker: PhantomData })
    }

    pub fn index(mut self, index: impl TensorIndex) -> Self {
        let offset = index.dim_map(&mut self.dimensions);
        let ptr = unsafe { self.ptr.add(offset) };
        Self { ptr, dimensions: self.dimensions, _marker: PhantomData }
    }

    pub fn split_mut(self) -> impl Iterator<Item = Self> {
        (0..self.dimensions.sizes()[0]).map(move |i| {
            let mut dimensions = self.dimensions.clone();
            let offset = i.dim_map(&mut dimensions);
            let ptr = unsafe { self.ptr.add(offset) };
            Self { ptr, dimensions, _marker: PhantomData }
        })
    }

    #[inline]
    pub fn as_slice(self) -> &'a Slice<T, A> {
        unsafe { Slice::from_raw_parts(self.ptr, self.dimensions.total_len()) }
    }

    #[inline]
    pub fn total_len(&self) -> usize {
        self.dimensions.total_len()
    }

    #[inline]
    pub fn as_mut_slice(self) -> &'a mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.dimensions.total_len()) }
    }
}

impl<'a, T: DeviceData, A: DeviceScope> From<&'a Tensor<T, A>> for TensorView<'a, T, A> {
    fn from(tensor: &'a Tensor<T, A>) -> Self {
        tensor.as_view()
    }
}

impl<'a, T: DeviceData, A: DeviceScope> From<&'a mut Tensor<T, A>> for TensorViewMut<'a, T, A> {
    fn from(tensor: &'a mut Tensor<T, A>) -> Self {
        tensor.as_view_mut()
    }
}

impl<T: DeviceData, A: DeviceScope> From<Buffer<T, A>> for Tensor<T, A> {
    fn from(buffer: Buffer<T, A>) -> Self {
        let dims = [buffer.len()].into_iter().collect();
        Self { storage: buffer, dimensions: dims }
    }
}

/// # Safety
///
/// This trait is unsafe because it allows for out of bounds access to the tensor, it is the
/// responsibility of the implementor to ensure that the indices are valid.
pub unsafe trait TensorIndex: Sized {
    fn dim_map(self, dimensions: &mut Dimensions) -> usize;
}

unsafe impl TensorIndex for usize {
    fn dim_map(self, dimensions: &mut Dimensions) -> usize {
        #[inline(never)]
        #[cold]
        #[track_caller]
        fn len_mismatch_fail(dst_len: usize, src_len: usize) -> ! {
            panic!("source dimension ({}) too small for index ({})", src_len, dst_len,);
        }
        let size = dimensions.sizes.remove(0);
        if self >= size {
            len_mismatch_fail(size, self);
        }
        let stride = dimensions.strides.remove(0);
        self * stride
    }
}

unsafe impl TensorIndex for Range<usize> {
    fn dim_map(self, dimensions: &mut Dimensions) -> usize {
        #[inline(never)]
        #[cold]
        #[track_caller]
        fn range_mismatch_fail(size: usize, start: usize, end: usize) -> ! {
            panic!(
                "source dimension ({}) too small for range ({})",
                size,
                (start..end).format(", ")
            );
        }

        if !(0..dimensions.sizes[0]).contains(&self.start)
            || !(0..dimensions.sizes[0]).contains(&self.end)
        {
            range_mismatch_fail(dimensions.sizes[0], self.start, self.end);
        }

        let shift = self.start * dimensions.strides[0];
        dimensions.sizes[0] = self.len();
        shift
    }
}

unsafe impl TensorIndex for RangeFull {
    #[inline]
    fn dim_map(self, _dimensions: &mut Dimensions) -> usize {
        0
    }
}

unsafe impl TensorIndex for RangeTo<usize> {
    fn dim_map(self, dimensions: &mut Dimensions) -> usize {
        #[inline(never)]
        #[cold]
        #[track_caller]
        fn range_to_mismatch_fail(size: usize, end: usize) -> ! {
            panic!("source dimension ({}) too small for range (..{})", size, end,);
        }

        if !(0..dimensions.sizes[0]).contains(&self.end) {
            range_to_mismatch_fail(dimensions.sizes[0], self.end);
        }
        dimensions.sizes[0] = self.end;
        0
    }
}

unsafe impl TensorIndex for RangeFrom<usize> {
    fn dim_map(self, dimensions: &mut Dimensions) -> usize {
        #[inline(never)]
        #[cold]
        #[track_caller]
        fn range_from_mismatch_fail(size: usize, start: usize) -> ! {
            panic!("source dimension ({}) too small for range ({}..)", size, start);
        }

        if !(0..dimensions.sizes[0]).contains(&self.start) {
            range_from_mismatch_fail(dimensions.sizes[0], self.start);
        }
        dimensions.sizes[0] -= self.start;
        // The starting index is the offset of the first element in the range.
        self.start * dimensions.strides[0]
    }
}

// unsafe impl<const N: usize>

impl<'a, T: DeviceData, A: DeviceScope, I: AsRef<[usize]>> Index<I> for TensorView<'a, T, A> {
    type Output = Init<T, A>;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        let index = self.dimensions.index_map(index);
        unsafe {
            let ptr = self.ptr.add(index) as *const Init<T, A>;
            ptr.as_ref().unwrap()
        }
    }
}

impl<'a, T: DeviceData, A: DeviceScope, I: AsRef<[usize]>> Index<I> for TensorViewMut<'a, T, A> {
    type Output = Init<T, A>;

    #[inline]
    fn index(&self, index: I) -> &Self::Output {
        let index = self.dimensions.index_map(index);
        unsafe {
            let ptr = self.ptr.add(index) as *const T as *const Init<T, A>;
            ptr.as_ref().unwrap()
        }
    }
}

impl<'a, T: DeviceData, A: DeviceScope, I: AsRef<[usize]>> IndexMut<I> for TensorViewMut<'a, T, A> {
    #[inline]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        let index = self.dimensions.index_map(index);
        unsafe {
            let ptr = self.ptr.add(index) as *mut Init<T, A>;
            ptr.as_mut().unwrap()
        }
    }
}
