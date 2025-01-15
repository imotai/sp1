mod buffer;
mod error;
mod event;
mod global;
mod stream;
pub mod sync;
pub mod task;
mod tensor;
mod transpose;

use std::{alloc::Layout, mem::MaybeUninit};

pub use error::CudaError;
pub use event::CudaEvent;
pub use stream::{CudaStream, StreamCallbackFuture, UnsafeCudaStream};

pub use task::*;
pub use tensor::DeviceTensor;
pub use transpose::*;

pub use buffer::DeviceBuffer;

use crate::{
    mem::{CopyDirection, CopyError, DeviceData},
    DeviceMemory, Init,
};

impl<T: DeviceData> Init<T, TaskScope> {
    pub fn to_host_blocking(&self, scope: &TaskScope) -> Result<T, CopyError> {
        let mut host_val = MaybeUninit::uninit();
        let layout = Layout::new::<T>();
        unsafe {
            scope.copy_nonoverlapping(
                self.as_ptr() as *const u8,
                host_val.as_mut_ptr() as *mut u8,
                layout.size(),
                CopyDirection::DeviceToHost,
            )?;

            Ok(host_val.assume_init())
        }
    }
}
