use slop_alloc::{mem::CopyError, Backend, Buffer, CopyIntoBackend, CpuBackend, HasBackend};
use tokio::sync::oneshot;

use crate::{DeviceCopy, TaskScope};

impl<T: DeviceCopy> CopyIntoBackend<CpuBackend, TaskScope> for Buffer<T, TaskScope> {
    type Output = Buffer<T, CpuBackend>;
    async fn copy_into_backend(self, _backend: &CpuBackend) -> Result<Self::Output, CopyError> {
        let (tx, rx) = oneshot::channel::<Result<Buffer<T, CpuBackend>, CopyError>>();
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

impl<T: DeviceCopy> CopyIntoBackend<TaskScope, CpuBackend> for Buffer<T, CpuBackend> {
    type Output = Buffer<T, TaskScope>;
    async fn copy_into_backend(self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        let scope = backend.clone();
        let (tx, rx) = oneshot::channel::<Result<Self::Output, CopyError>>();
        tokio::task::spawn_blocking(move || {
            let buf = self;
            let mut buffer = Buffer::with_capacity_in(buf.len(), scope);
            let res = buffer.extend_from_host_slice(&buf);
            tx.send(res.map(move |_| buffer)).ok();
        });
        rx.await.unwrap()
    }
}

pub struct SmallBuffer<'a, T, A: Backend> {
    buffer: &'a Buffer<T, A>,
}

impl<'a, T, A: Backend> SmallBuffer<'a, T, A> {
    /// # Safety
    /// The buffer must be small enough so that copying from it to or from host does not block.
    pub unsafe fn new(buffer: &'a Buffer<T, A>) -> Self {
        Self { buffer }
    }
}

impl<'a, T, A: Backend> HasBackend for SmallBuffer<'a, T, A> {
    type Backend = A;
    fn backend(&self) -> &A {
        self.buffer.backend()
    }
}

impl<'a, T: DeviceCopy> CopyIntoBackend<CpuBackend, TaskScope> for SmallBuffer<'a, T, TaskScope> {
    type Output = Buffer<T, CpuBackend>;
    async fn copy_into_backend(self, _backend: &CpuBackend) -> Result<Self::Output, CopyError> {
        let mut vec = Vec::with_capacity(self.buffer.len());
        unsafe {
            self.buffer.copy_into_host(vec.spare_capacity_mut())?;
            vec.set_len(self.buffer.len());
            let host_buffer = Buffer::from(vec);

            self.buffer.backend().synchronize().await.unwrap();
            Ok(host_buffer)
        }
    }
}

impl<'a, T: DeviceCopy> CopyIntoBackend<TaskScope, CpuBackend> for SmallBuffer<'a, T, CpuBackend> {
    type Output = Buffer<T, TaskScope>;
    async fn copy_into_backend(self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        let mut buffer = Buffer::with_capacity_in(self.buffer.len(), backend.clone());
        buffer.extend_from_host_slice(self.buffer)?;
        backend.synchronize().await.unwrap();
        Ok(buffer)
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};
    use slop_alloc::Backend;
    use slop_baby_bear::BabyBear;

    use slop_alloc::IntoHost;

    use super::*;

    #[tokio::test]
    async fn test_copy_buffer_into_backend() {
        let mut rng = thread_rng();
        let buffer: Buffer<BabyBear> = (0..10000).map(|_| rng.gen::<BabyBear>()).collect();

        let cloned_buffer = buffer.clone();
        let buffer_back = crate::spawn(|t| async move {
            let device_buffer = t.copy_from(cloned_buffer).await.unwrap();
            device_buffer.into_host().await.unwrap()
        })
        .await
        .unwrap()
        .await
        .unwrap();

        assert_eq!(buffer_back, buffer);
    }

    #[tokio::test]
    async fn test_copy_small_buffer_into_backend() {
        let mut rng = thread_rng();
        let buffer: Buffer<BabyBear> = (0..10).map(|_| rng.gen::<BabyBear>()).collect();

        let cloned_buffer = buffer.clone();
        let buffer_back = crate::spawn(|t| async move {
            let small_buffer = unsafe { SmallBuffer::new(&cloned_buffer) };
            let device_buffer = t.copy_from(small_buffer).await.unwrap();
            device_buffer.into_host().await.unwrap()
        })
        .await
        .unwrap()
        .await
        .unwrap();

        assert_eq!(buffer_back, buffer);
    }
}
