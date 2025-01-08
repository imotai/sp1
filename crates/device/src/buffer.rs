use crate::mem::{CopyDirection, CopyError, DeviceData, DeviceMemory};
use crate::slice::Slice;
use csl_alloc::{Allocator, RawBuffer, TryReserveError};
use std::{
    alloc::Layout,
    mem::MaybeUninit,
    ops::{
        Deref, DerefMut, Index, IndexMut, Range, RangeFrom, RangeFull, RangeInclusive, RangeTo,
        RangeToInclusive,
    },
};

/// Fixed-size device-side buffer.
#[derive(Debug)]
#[repr(C)]
pub struct Buffer<T: DeviceData, A: Allocator> {
    buf: RawBuffer<T, A>,
    len: usize,
}

unsafe impl<T: DeviceData, A: Allocator> Send for Buffer<T, A> {}
unsafe impl<T: DeviceData, A: Allocator> Sync for Buffer<T, A> {}

impl<T, A> Buffer<T, A>
where
    T: DeviceData,
    A: Allocator,
{
    #[inline]
    #[must_use]
    pub fn with_capcity_in(capacity: usize, allocator: A) -> Self {
        let buf = RawBuffer::with_capacity_in(capacity, allocator);
        Self { buf, len: 0 }
    }

    #[inline]
    pub fn try_with_capacity_in(capacity: usize, allocator: A) -> Result<Self, TryReserveError> {
        let buf = RawBuffer::try_with_capacity_in(capacity, allocator)?;
        Ok(Self { buf, len: 0 })
    }

    /// Returns a new buffer from a pointer, length, and capacity.
    ///
    /// # Safety
    ///
    /// The pointer must be valid, it must have allocated memory in the size of
    /// capacity * size_of<T>, and the first `len` elements of the buffer must be initialized or
    /// about to be initialized in a foreign call.
    pub unsafe fn from_raw_parts(ptr: *mut T, length: usize, capacity: usize, alloc: A) -> Self {
        Self { buf: RawBuffer::from_raw_parts_in(ptr, capacity, alloc), len: length }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.buf.capacity()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.buf.ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.buf.ptr()
    }

    /// # Safety
    ///
    /// TODO
    #[inline]
    pub unsafe fn set_len(&mut self, new_len: usize) {
        self.len = new_len;
    }

    /// Copies all elements from `src` into `self`, using a cudaMemcpy.
    ///
    /// The length of `src` must be the same as `self`.
    ///
    /// # Panics
    ///
    /// This function will panic if the two slices have different lengths or if cudaMalloc
    /// returned an error.
    ///
    /// # Safety
    /// This operation is potentially asynchronous. The caller must insure the memory of the source
    /// is valid for the duration of the operation.
    pub unsafe fn copy_from_host_slice(&mut self, src: &[T]) -> Result<(), CopyError>
    where
        A: DeviceMemory,
    {
        // The panic code path was put into a cold function to not bloat the
        // call site.
        #[inline(never)]
        #[cold]
        #[track_caller]
        fn len_mismatch_fail(dst_len: usize, src_len: usize) -> ! {
            panic!(
                "source slice length ({}) does not match destination slice length ({})",
                src_len, dst_len,
            );
        }

        if self.len() != src.len() {
            len_mismatch_fail(self.len(), src.len());
        }

        let layout = Layout::array::<T>(src.len()).unwrap();

        unsafe {
            self.buf.allocator().copy_nonoverlapping(
                src.as_ptr() as *const u8,
                self.buf.ptr() as *mut u8,
                layout.size(),
                CopyDirection::HostToDevice,
            )
        }
    }

    #[inline]
    pub fn allocator(&self) -> &A {
        self.buf.allocator()
    }

    /// # Safety
    #[inline]
    pub unsafe fn allocator_mut(&mut self) -> &mut A {
        self.buf.allocator_mut()
    }

    /// Appends all the elements from `src` into `self`, using a cudaMemcpy.
    ///
    /// # Panics
    ///
    /// This function will panic if the resulting length will extend the buffer's capacity or if
    /// cudaMalloc returned an error.
    ///
    ///  # Safety
    /// This operation is potentially asynchronous. The caller must insure the memory of the source
    /// is valid for the duration of the operation.
    pub unsafe fn extend_from_host_slice(&mut self, src: &[T]) -> Result<(), CopyError>
    where
        A: DeviceMemory,
    {
        // The panic code path was put into a cold function to not bloat the
        // call site.
        #[inline(never)]
        #[cold]
        #[track_caller]
        fn capacity_fail(dst_len: usize, src_len: usize, cap: usize) -> ! {
            panic!(
                "source slice length ({}) too long for buffer of length ({}) and capacity ({})",
                src_len, dst_len, cap
            );
        }

        if self.len() + src.len() > self.capacity() {
            capacity_fail(self.len(), src.len(), self.capacity());
        }

        let layout = Layout::array::<T>(src.len()).unwrap();

        unsafe {
            self.buf.allocator().copy_nonoverlapping(
                src.as_ptr() as *const u8,
                self.buf.ptr().add(self.len()) as *mut u8,
                layout.size(),
                CopyDirection::HostToDevice,
            )?;
        }

        // Extend the length of the buffer to include the new elements.
        self.len += src.len();

        Ok(())
    }

    /// Copies all elements from `self` into `dst`, using a cudaMemcpy.
    ///
    /// The length of `dst` must be the same as `self`.
    ///
    /// **Note**: This function might be blocking.
    ///
    /// # Safety
    ///
    /// This operation is potentially asynchronous. The caller must insure the memory of the
    /// destination is valid for the duration of the operation.
    pub unsafe fn copy_into_host(&mut self, dst: &mut [MaybeUninit<T>]) -> Result<(), CopyError>
    where
        A: DeviceMemory,
    {
        // The panic code path was put into a cold function to not bloat the
        // call site.
        #[inline(never)]
        #[cold]
        #[track_caller]
        fn len_mismatch_fail(dst_len: usize, src_len: usize) -> ! {
            panic!(
                "source slice length ({}) does not match destination slice length ({})",
                src_len, dst_len,
            );
        }

        if self.len() != dst.len() {
            len_mismatch_fail(dst.len(), self.len());
        }

        let layout = Layout::array::<T>(dst.len()).unwrap();

        self.buf.allocator().copy_nonoverlapping(
            self.buf.ptr() as *const u8,
            dst.as_mut_ptr() as *mut u8,
            layout.size(),
            CopyDirection::DeviceToHost,
        )
    }

    /// # Safety
    ///
    /// This operation is potentially asynchronous.
    pub unsafe fn write_bytes_uncheked(&mut self, value: u8, len: usize) -> Result<(), CopyError>
    where
        A: DeviceMemory,
    {
        // The panic code path was put into a cold function to not bloat the
        // call site.
        #[inline(never)]
        #[cold]
        #[track_caller]
        fn capacity_fail(dst_len: usize, len: usize, cap: usize) -> ! {
            panic!(
                "Cannot write {} bytes to buffer of length {} and capacity {}",
                len, dst_len, cap
            );
        }

        // The panic code path was put into a cold function to not bloat the
        // call site.
        #[inline(never)]
        #[cold]
        #[track_caller]
        fn align_fail(len: usize, size: usize) -> ! {
            panic!("Number of bytes ({}) does not match the size of the type ({})", len, size);
        }

        // Check that the number of bytes matches the size of the type.
        if len % std::mem::size_of::<T>() != 0 {
            align_fail(len, std::mem::size_of::<T>());
        }

        // Check that the buffer has enough capacity.
        if self.len() * std::mem::size_of::<T>() + len > self.capacity() * std::mem::size_of::<T>()
        {
            capacity_fail(self.len(), len, self.capacity());
        }

        // Write the bytes to the buffer.
        self.buf.allocator().write_bytes(self.buf.ptr().add(self.len()) as *mut u8, value, len)?;

        // Extend the length of the buffer to include the new elements.
        self.len += len / std::mem::size_of::<T>();

        Ok(())
    }
}

macro_rules! impl_index {
    ($($t:ty)*) => {
        $(
            impl<T : DeviceData, A: Allocator> Index<$t> for Buffer<T, A>
            {
                type Output = Slice<T, A>;

                fn index(&self, index: $t) -> &Slice<T, A> {
                    unsafe {
                        Slice::from_slice(
                         std::slice::from_raw_parts(self.as_ptr(), self.len).index(index)
                    )
                  }
                }
            }

            impl<T : DeviceData, A: Allocator> IndexMut<$t> for Buffer<T, A>
            {
                fn index_mut(&mut self, index: $t) -> &mut Slice<T, A> {
                    unsafe {
                        Slice::from_slice_mut(
                            std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len).index_mut(index)
                        )
                    }
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

impl<T: DeviceData, A: Allocator> Deref for Buffer<T, A> {
    type Target = Slice<T, A>;

    fn deref(&self) -> &Self::Target {
        &self[..]
    }
}

impl<T: DeviceData, A: Allocator> DerefMut for Buffer<T, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self[..]
    }
}
