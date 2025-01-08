use crate::{
    mem::{CopyError, DeviceData},
    Buffer,
};

use super::{sync::CudaSend, TaskScope};

pub type DeviceBuffer<T> = Buffer<T, TaskScope>;

impl<T: DeviceData> Buffer<T, TaskScope> {
    pub async fn from_host_vec(buf: Vec<T>, scope: TaskScope) -> Result<Self, CopyError>
    where
        T: Send,
    {
        tokio::task::spawn_blocking(move || Self::from_host_slice_blocking(&buf, scope))
            .await
            .unwrap()
    }

    pub fn from_host_slice_blocking(buf: &[T], scope: TaskScope) -> Result<Self, CopyError> {
        let mut buffer = scope.alloc(buf.len());
        unsafe {
            buffer.extend_from_host_slice(buf)?;
        }
        Ok(buffer)
    }

    pub async fn to_vec(self) -> Result<Vec<T>, CopyError>
    where
        T: Send,
    {
        tokio::task::spawn_blocking(move || self.to_vec_blocking()).await.unwrap()
    }

    pub fn to_vec_blocking(mut self) -> Result<Vec<T>, CopyError> {
        let mut vec = Vec::with_capacity(self.len());
        unsafe {
            self.copy_into_host(vec.spare_capacity_mut())?;
            vec.set_len(self.len());
        }
        Ok(vec)
    }

    #[inline]
    pub fn write_bytes(&mut self, value: u8, len: usize) -> Result<(), CopyError> {
        // This is safe because getting a Buffer with a TaskScope makes sure that the values remain
        // valid.
        unsafe { self.write_bytes_uncheked(value, len) }
    }
}

unsafe impl<T: DeviceData> CudaSend for Buffer<T, TaskScope> {
    fn change_scope(&mut self, scope: &TaskScope) {
        unsafe {
            *self.allocator_mut() = scope.clone();
        }
    }
}
