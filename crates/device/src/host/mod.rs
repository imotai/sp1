mod tensor;

pub use tensor::{HostTensor, PinnedTensor};

use std::{
    alloc::Layout,
    ffi::c_void,
    mem::{ManuallyDrop, MaybeUninit},
    ptr::{self, NonNull},
};

use csl_alloc::{AllocError, Allocator, TryReserveError};
use csl_sys::runtime::{cuda_free_host, cuda_malloc_host};

use crate::{
    cuda::CudaError,
    mem::{CopyDirection, CopyError, DeviceData, DeviceMemory},
    Buffer,
};

const GLOBAL_ALLOCATOR: GlobalAllocator = GlobalAllocator;

const PINNED_ALLOCATOR: PinnedAllocator = PinnedAllocator;

pub struct GlobalAllocator;

pub struct PinnedAllocator;

pub type HostBuffer<T> = Buffer<T, GlobalAllocator>;
pub type PinnedBuffer<T> = Buffer<T, PinnedAllocator>;

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
}

impl<T: DeviceData> PinnedBuffer<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, PINNED_ALLOCATOR)
    }

    pub fn try_with_capacity(capacity: usize) -> Result<Self, TryReserveError> {
        Self::try_with_capacity_in(capacity, PINNED_ALLOCATOR)
    }
}

unsafe impl Allocator for GlobalAllocator {
    #[inline]
    unsafe fn allocate(&self, layout: Layout) -> Result<ptr::NonNull<[u8]>, AllocError> {
        let ptr = std::alloc::alloc(layout);
        Ok(NonNull::slice_from_raw_parts(NonNull::new_unchecked(ptr), layout.size()))
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        std::alloc::dealloc(ptr.as_ptr(), layout);
    }
}

impl DeviceMemory for GlobalAllocator {
    #[inline]
    unsafe fn copy_nonoverlapping(
        &self,
        src: *const u8,
        dst: *mut u8,
        size: usize,
        _direction: CopyDirection,
    ) -> Result<(), CopyError> {
        src.copy_to_nonoverlapping(dst, size);
        Ok(())
    }

    #[inline]
    unsafe fn write_bytes(&self, dst: *mut u8, value: u8, size: usize) -> Result<(), CopyError> {
        dst.write_bytes(value, size);
        Ok(())
    }
}

unsafe impl Allocator for PinnedAllocator {
    #[inline]
    unsafe fn allocate(&self, layout: Layout) -> Result<ptr::NonNull<[u8]>, AllocError> {
        let mut ptr: *mut c_void = ptr::null_mut();
        unsafe {
            CudaError::result_from_ffi(cuda_malloc_host(
                &mut ptr as *mut *mut c_void,
                layout.size(),
            ))
            .map_err(|_| AllocError)?;
        };
        let ptr = ptr as *mut u8;
        Ok(NonNull::slice_from_raw_parts(NonNull::new_unchecked(ptr), layout.size()))
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, _layout: Layout) {
        unsafe { CudaError::result_from_ffi(cuda_free_host(ptr.as_ptr() as *mut c_void)).unwrap() }
    }
}

impl DeviceMemory for PinnedAllocator {
    #[inline]
    unsafe fn copy_nonoverlapping(
        &self,
        src: *const u8,
        dst: *mut u8,
        size: usize,
        _direction: CopyDirection,
    ) -> Result<(), CopyError> {
        src.copy_to_nonoverlapping(dst, size);
        Ok(())
    }

    #[inline]
    unsafe fn write_bytes(&self, dst: *mut u8, value: u8, size: usize) -> Result<(), CopyError> {
        dst.write_bytes(value, size);
        Ok(())
    }
}
