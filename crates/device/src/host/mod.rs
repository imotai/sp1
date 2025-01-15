mod buffer;
mod tensor;

pub use tensor::{
    HostTensor, HostTensorView, HostTensorViewMut, PinnedTensor, PinnedTensorView,
    PinnedTensorViewMut,
};

pub use buffer::{HostBuffer, PinnedBuffer};

use std::{
    alloc::Layout,
    ffi::c_void,
    ops::{Deref, DerefMut},
    ptr::{self, NonNull},
};

use csl_alloc::{AllocError, Allocator};
use csl_sys::runtime::{cuda_free_host, cuda_malloc_host};

use crate::{
    cuda::CudaError,
    mem::{CopyDirection, CopyError, DeviceData, DeviceMemory},
    DeviceScope, Init, Slice,
};

const GLOBAL_ALLOCATOR: GlobalAllocator = GlobalAllocator;

const PINNED_ALLOCATOR: PinnedAllocator = PinnedAllocator;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GlobalAllocator;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PinnedAllocator;

unsafe impl DeviceScope for GlobalAllocator {}
unsafe impl DeviceScope for PinnedAllocator {}

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

pub type HostSlice<T> = Slice<T, GlobalAllocator>;

impl<T: DeviceData> Deref for HostSlice<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
}

impl<T: DeviceData> DerefMut for HostSlice<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }
}

pub type PinnedSlice<T> = Slice<T, PinnedAllocator>;

impl<T: DeviceData> Deref for PinnedSlice<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
}

impl<T: DeviceData> DerefMut for PinnedSlice<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }
}

impl<T: DeviceData> Deref for Init<T, GlobalAllocator> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<T: DeviceData> DerefMut for Init<T, GlobalAllocator> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}
