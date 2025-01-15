use crate::{
    mem::{CopyError, DeviceData},
    tensor::{TensorView, TensorViewMut},
    HostTensor, Tensor,
};

use super::{DeviceBuffer, TaskScope};

pub type DeviceTensor<T> = Tensor<T, TaskScope>;
pub type DeviceTensorView<'a, T> = TensorView<'a, T, TaskScope>;
pub type DeviceTensorViewMut<'a, T> = TensorViewMut<'a, T, TaskScope>;

impl<T: DeviceData> DeviceTensor<T> {
    pub async fn into_host(self) -> Result<HostTensor<T>, CopyError>
    where
        T: Send,
    {
        let storage = self.storage.into_host().await?;
        Ok(HostTensor { storage, dimensions: self.dimensions })
    }

    pub fn into_host_blocking(self) -> Result<HostTensor<T>, CopyError> {
        let storage = self.storage.into_to_host_blocking()?;
        Ok(HostTensor { storage, dimensions: self.dimensions })
    }

    pub async fn from_host(host: HostTensor<T>, scope: TaskScope) -> Result<Self, CopyError>
    where
        T: Send,
    {
        let HostTensor { storage, dimensions } = host;
        let storage = DeviceBuffer::from_host(storage, scope).await?;
        Ok(Self { storage, dimensions })
    }
}
