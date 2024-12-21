use thiserror::Error;

/// The [AllocError] error indicates an allocation failure that may be due to resource exhaustion
/// or to something wrong when combining the given input arguments with this allocator.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Error)]
#[error("allocation error")]
pub struct AllocError;

/// A trait that defines memory operations for a device.
pub trait DeviceMemory {
    /// # Safety
    ///
    unsafe fn copy_nonoverlapping(
        &self,
        src: *const u8,
        dst: *mut u8,
        size: usize,
    ) -> Result<(), AllocError>;

    /// TODO
    ///
    /// # Safety
    unsafe fn write_bytes(&self, dst: *mut u8, value: u8, size: usize) -> Result<(), AllocError>;
}

pub struct DeviceMemoryImpl;
