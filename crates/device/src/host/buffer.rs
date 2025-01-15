use std::mem::{ManuallyDrop, MaybeUninit};

use csl_alloc::TryReserveError;

use crate::{mem::DeviceData, Buffer};

use super::{GlobalAllocator, PinnedAllocator, GLOBAL_ALLOCATOR, PINNED_ALLOCATOR};

pub type HostBuffer<T> = Buffer<T, GlobalAllocator>;
pub type PinnedBuffer<T> = Buffer<T, PinnedAllocator>;

impl<T: DeviceData> PinnedBuffer<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, PINNED_ALLOCATOR)
    }

    pub fn try_with_capacity(capacity: usize) -> Result<Self, TryReserveError> {
        Self::try_with_capacity_in(capacity, PINNED_ALLOCATOR)
    }
}

impl<T: DeviceData> HostBuffer<T> {
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, GLOBAL_ALLOCATOR)
    }

    #[inline]
    pub fn into_vec(self) -> Vec<T> {
        let mut self_undropped = ManuallyDrop::new(self);
        unsafe {
            Vec::from_raw_parts(
                self_undropped.as_mut_ptr(),
                self_undropped.len(),
                self_undropped.capacity(),
            )
        }
    }

    #[inline]
    pub fn from_vec(vec: Vec<T>) -> Self {
        unsafe {
            let mut vec = ManuallyDrop::new(vec);
            Buffer::from_raw_parts(vec.as_mut_ptr(), vec.len(), vec.capacity(), GlobalAllocator)
        }
    }

    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        let mut vec = ManuallyDrop::new(unsafe {
            Vec::from_raw_parts(self.as_mut_ptr(), self.len(), self.capacity())
        });
        let slice = vec.spare_capacity_mut();
        let len = slice.len();
        let ptr = slice.as_mut_ptr();
        unsafe { std::slice::from_raw_parts_mut(ptr, len) }
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        // The panic code path is put into a cold function to not bloat the
        // call site.
        #[inline(never)]
        #[cold]
        #[track_caller]
        fn buffer_is_full(len: usize, capacity: usize) -> ! {
            panic!("buffer is full: len: {}, capacity: {}", len, capacity);
        }

        if self.len() == self.capacity() {
            buffer_is_full(self.len(), self.capacity());
        }

        // This is safe because we have just checked that the buffer is not full.
        unsafe {
            let ptr = self.as_mut_ptr().add(self.len());
            ptr.write(value);
            self.set_len(self.len() + 1);
        }
    }

    #[inline]
    pub fn pop(&mut self) -> T {
        // The panic code path is put into a cold function to not bloat the
        // call site.
        #[inline(never)]
        #[cold]
        #[track_caller]
        fn buffer_is_empty() -> ! {
            panic!("buffer is empty");
        }

        if self.is_empty() {
            buffer_is_empty();
        }

        // This is safe because we have just checked that the buffer is not empty.
        unsafe {
            let ptr = self.as_mut_ptr().add(self.len() - 1);
            let value = ptr.read();
            self.set_len(self.len() - 1);
            value
        }
    }

    #[inline]
    pub fn extend_from_slice(&mut self, slice: &[T]) {
        unsafe { self.extend_from_host_slice(slice).unwrap() }
    }
}

impl<T: DeviceData> From<Vec<T>> for HostBuffer<T> {
    #[inline]
    fn from(vec: Vec<T>) -> Self {
        Self::from_vec(vec)
    }
}

impl<T: DeviceData> From<HostBuffer<T>> for Vec<T> {
    #[inline]
    fn from(buffer: HostBuffer<T>) -> Self {
        buffer.into_vec()
    }
}

// A macro host_buffer!() that just uses the vec!() macro and then converts it to a HostBuffer
#[macro_export]
macro_rules! host_buffer {
    ($($x:expr),*) => {
       $crate::host::HostBuffer::from(vec![$($x),*])
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_host_buffer() {
        let mut buffer = HostBuffer::<u32>::with_capacity(10);
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.capacity(), 10);

        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        assert_eq!(buffer.len(), 3);

        let as_slice: &[u32] = &buffer[..];
        assert_eq!(as_slice, &[1, 2, 3]);

        let val = *buffer[0];
        assert_eq!(val, 1);

        let val = *buffer[1];
        assert_eq!(val, 2);

        let val = *buffer[2];
        assert_eq!(val, 3);

        let value = buffer.pop();
        assert_eq!(value, 3);
        assert_eq!(buffer.len(), 2);

        buffer.extend_from_slice(&[4, 5, 6]);
        let host_vec = buffer.into_vec();
        assert_eq!(host_vec, [1, 2, 4, 5, 6]);

        // Test the host_buffer!() macro
        let buffer = host_buffer![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(buffer.len(), 10);
        assert_eq!(buffer.capacity(), 10);
        assert_eq!(*buffer[0], 1);
        assert_eq!(*buffer[1], 2);
        assert_eq!(*buffer[2], 3);
        assert_eq!(*buffer[3], 4);
        assert_eq!(*buffer[4], 5);
        assert_eq!(*buffer[5], 6);
        assert_eq!(*buffer[6], 7);
        assert_eq!(*buffer[7], 8);
        assert_eq!(*buffer[8], 9);
        assert_eq!(*buffer[9], 10);
    }
}
