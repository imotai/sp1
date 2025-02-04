use std::future::Future;

use slop_alloc::{mem::CopyError, Buffer, CpuBackend};
use tokio::sync::oneshot;

use crate::TaskScope;

pub trait DeviceCopy: Copy + 'static + Sized + Send + Sync {}

pub trait IntoDevice {
    type DeviceData;

    fn into_device_in(
        self,
        scope: &TaskScope,
    ) -> impl Future<Output = Result<Self::DeviceData, CopyError>> + Send;
}

impl<T: Copy + 'static + Sized + Send + Sync> DeviceCopy for T {}

pub type DeviceBuffer<T> = Buffer<T, TaskScope>;

impl<T: DeviceCopy> IntoDevice for Buffer<T, CpuBackend> {
    type DeviceData = Buffer<T, TaskScope>;

    fn into_device_in(
        self,
        scope: &TaskScope,
    ) -> impl Future<Output = Result<Self::DeviceData, CopyError>> + Send {
        let scope = scope.clone();
        let (tx, rx) = oneshot::channel::<Result<Self::DeviceData, CopyError>>();
        async move {
            tokio::task::spawn_blocking(move || {
                let buf = self;
                let mut buffer = Buffer::with_capacity_in(buf.len(), scope);
                let res = buffer.extend_from_host_slice(&buf);
                tx.send(res.map(move |_| buffer)).ok();
            });
            rx.await.unwrap()
        }
    }
}

impl<T: DeviceCopy> IntoDevice for Vec<T> {
    type DeviceData = Buffer<T, TaskScope>;

    fn into_device_in(
        self,
        scope: &TaskScope,
    ) -> impl Future<Output = Result<Self::DeviceData, CopyError>> + Send {
        Buffer::into_device_in(Buffer::from(self), scope)
    }
}

pub trait IntoHost {
    type HostData;

    fn into_host(self) -> impl Future<Output = Result<Self::HostData, CopyError>> + Send;
}

impl<T: DeviceCopy> IntoHost for Buffer<T, TaskScope> {
    type HostData = Buffer<T, CpuBackend>;

    async fn into_host(self) -> Result<Self::HostData, CopyError> {
        let (tx, rx) = oneshot::channel::<Result<Self::HostData, CopyError>>();
        tokio::task::spawn_blocking(move || {
            let buffer = self;
            let mut vec = Vec::with_capacity(buffer.len());
            unsafe {
                let res = buffer.copy_into_host(vec.spare_capacity_mut());
                vec.set_len(buffer.len());
                tx.send(res.map(move |_| Buffer::from(vec)))
            }
        });
        rx.await.unwrap()
    }
}

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
