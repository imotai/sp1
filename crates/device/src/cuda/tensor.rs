use crate::{
    mem::{CopyError, DeviceData},
    HostTensor, Tensor,
};

use super::TaskScope;

pub type DeviceTensor<T> = Tensor<T, TaskScope>;

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
}
