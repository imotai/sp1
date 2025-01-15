mod buffer;
pub mod cuda;
pub mod host;
pub mod mem;
pub mod slice;
pub mod tensor;

pub use buffer::Buffer;
use csl_alloc::Allocator;
pub use cuda::{DeviceBuffer, DeviceTensor};
pub use host::*;
pub use mem::DeviceMemory;
pub use mem::Init;
pub use slice::Slice;
pub use tensor::Tensor;

pub use csl_alloc as alloc;
pub use csl_sys as sys;

pub use csl_sys::runtime::KernelPtr;

pub use crate::cuda::sync::CudaSend;
pub use csl_derive::CudaSend;

/// # Safety
///
/// TODO
pub unsafe trait DeviceScope: Sized + Allocator + DeviceMemory + Clone {}
