use std::marker::PhantomData;
use std::ops::{
    Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use csl_alloc::Allocator;

use crate::mem::{DeviceData, Init};

/// A slice of data associated with a specific allocator type.
///
/// This type is enssentially a wrapper around a slice and has an indicator for the type of the
/// allocator to induicate where the memory resides but it
#[repr(transparent)]
pub struct Slice<T, A> {
    allocator: PhantomData<A>,
    slice: [T],
}

impl<T: DeviceData, A: Allocator> Slice<T, A> {
    #[inline]
    pub const fn len(&self) -> usize {
        self.slice.len()
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.slice.is_empty()
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.slice.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.slice.as_mut_ptr()
    }

    #[inline(always)]
    pub(crate) unsafe fn from_slice(src: &[T]) -> &Self {
        &*(src as *const [T] as *const Self)
    }

    #[inline(always)]
    pub(crate) unsafe fn from_slice_mut(src: &mut [T]) -> &mut Self {
        &mut *(src as *mut [T] as *mut Self)
    }

    /// # Safety
    pub unsafe fn from_raw_parts_mut<'a>(data: *mut T, len: usize) -> &'a mut Self {
        Self::from_slice_mut(std::slice::from_raw_parts_mut(data, len))
    }

    #[inline]
    pub fn split_at_mut(&mut self, mid: usize) -> (&mut Self, &mut Self) {
        let (left, right) = self.slice.split_at_mut(mid);
        unsafe { (Self::from_slice_mut(left), Self::from_slice_mut(right)) }
    }

    #[inline]
    pub fn split_at(&self, mid: usize) -> (&Self, &Self) {
        let (left, right) = self.slice.split_at(mid);
        unsafe { (Self::from_slice(left), Self::from_slice(right)) }
    }
}

macro_rules! impl_index {
    ($($t:ty)*) => {
        $(
            impl<T : DeviceData, A: Allocator> Index<$t> for Slice<T, A>
            {
                type Output = Slice<T, A>;

                fn index(&self, index: $t) -> &Self {
                    unsafe { Slice::from_slice(self.slice.index(index)) }
                }
            }

            impl<T : DeviceData, A: Allocator> IndexMut<$t> for Slice<T, A>
            {
                fn index_mut(&mut self, index: $t) -> &mut Self {
                    unsafe { Slice::from_slice_mut( self.slice.index_mut(index)) }
                }
            }
        )*
    }
}

impl_index! {
    Range<usize>
    RangeFull
    RangeFrom<usize>
    RangeInclusive<usize>
    RangeTo<usize>
    RangeToInclusive<usize>
}

impl<T: DeviceData, A: Allocator> Index<usize> for Slice<T, A> {
    type Output = Init<T, A>;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        let element = &self[index..(index + 1)];
        let raw = element.as_ptr() as *mut T as *mut Init<T, A>;
        unsafe { &*raw }
    }
}
