use crate::TaskScope;
use slop_alloc::{mem::CopyError, CopyIntoBackend, CopyToBackend, CpuBackend};
use std::future::Future;

pub trait DeviceCopy: Copy + 'static + Sized + Send + Sync {}

impl<T: Copy + 'static + Sized + Send + Sync> DeviceCopy for T {}

pub trait IntoDevice: CopyIntoBackend<TaskScope, CpuBackend> + Sized {
    fn into_device_in(
        self,
        backend: &TaskScope,
    ) -> impl Future<Output = Result<Self::Output, CopyError>> + Send {
        self.copy_into_backend(backend)
    }
}

impl<T> IntoDevice for T where T: CopyIntoBackend<TaskScope, CpuBackend> + Sized {}

pub trait ToDevice: CopyToBackend<TaskScope, CpuBackend> + Sized {
    fn to_device_in(
        &self,
        backend: &TaskScope,
    ) -> impl Future<Output = Result<Self::Output, CopyError>> + Send {
        self.copy_to_backend(backend)
    }
}

impl<T> ToDevice for T where T: CopyToBackend<TaskScope, CpuBackend> + Sized {}

#[macro_export]
macro_rules! args {
    ($($arg:expr),*) => {
        [
            $(
                &$arg as *const _ as *mut std::ffi::c_void
            ),*
        ]
    };
}
