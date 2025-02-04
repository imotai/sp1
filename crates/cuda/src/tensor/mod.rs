mod dot;
pub mod reduce;
mod sum;
mod transpose;

use slop_alloc::{mem::CopyError, CpuBackend};
use slop_tensor::{Tensor, TensorView, TensorViewMut};

use crate::{sync::CudaSend, DeviceCopy, IntoDevice, IntoHost};

use super::TaskScope;

pub type DeviceTensor<T> = Tensor<T, TaskScope>;
pub type DeviceTensorView<'a, T> = TensorView<'a, T, TaskScope>;
pub type DeviceTensorViewMut<'a, T> = TensorViewMut<'a, T, TaskScope>;

impl<T: DeviceCopy> IntoDevice for Tensor<T, CpuBackend> {
    type DeviceData = Tensor<T, TaskScope>;

    async fn into_device_in(self, scope: &TaskScope) -> Result<Self::DeviceData, CopyError> {
        let Tensor { storage, dimensions } = self;
        let storage = scope.into_device(storage).await?;
        Ok(Tensor { storage, dimensions })
    }
}

impl<T: DeviceCopy> IntoHost for Tensor<T, TaskScope> {
    type HostData = Tensor<T, CpuBackend>;

    async fn into_host(self) -> Result<Self::HostData, CopyError> {
        let Tensor { storage, dimensions } = self;
        let storage = storage.into_host().await?;
        Ok(Tensor { storage, dimensions })
    }
}

unsafe impl<T: DeviceCopy> CudaSend for Tensor<T, TaskScope> {
    fn change_scope(&mut self, scope: &TaskScope) {
        self.storage.change_scope(scope);
    }
}

unsafe impl<T: DeviceCopy> CudaSend for Tensor<T, CpuBackend> {
    fn change_scope(&mut self, _scope: &TaskScope) {}
}
