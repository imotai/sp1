mod dot;
pub mod reduce;
mod sum;
pub mod transpose;

pub use dot::dot_along_dim_view;
use slop_alloc::{mem::CopyError, Backend, CopyIntoBackend, CopyToBackend, CpuBackend, HasBackend};
use slop_tensor::Tensor;

use crate::{DeviceCopy, SmallBuffer};

use super::TaskScope;

impl<T: DeviceCopy> CopyIntoBackend<CpuBackend, TaskScope> for Tensor<T, TaskScope> {
    type Output = Tensor<T, CpuBackend>;
    async fn copy_into_backend(self, backend: &CpuBackend) -> Result<Self::Output, CopyError> {
        let Tensor { storage, dimensions } = self;
        let storage = storage.copy_into_backend(backend).await?;
        Ok(Tensor { storage, dimensions })
    }
}

impl<T: DeviceCopy> CopyIntoBackend<TaskScope, CpuBackend> for Tensor<T, CpuBackend> {
    type Output = Tensor<T, TaskScope>;
    async fn copy_into_backend(self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        let Tensor { storage, dimensions } = self;
        let storage = storage.copy_into_backend(backend).await?;
        Ok(Tensor { storage, dimensions })
    }
}

impl<T: DeviceCopy> CopyToBackend<CpuBackend, TaskScope> for Tensor<T, TaskScope> {
    type Output = Tensor<T, CpuBackend>;
    async fn copy_to_backend(&self, backend: &CpuBackend) -> Result<Self::Output, CopyError> {
        let Tensor { storage, dimensions } = self;
        let storage = storage.copy_to_backend(backend).await?;
        Ok(Tensor { storage, dimensions: dimensions.clone() })
    }
}

impl<T: DeviceCopy> CopyToBackend<TaskScope, CpuBackend> for Tensor<T, CpuBackend> {
    type Output = Tensor<T, TaskScope>;
    async fn copy_to_backend(&self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        let Tensor { storage, dimensions } = self;
        let storage = storage.copy_to_backend(backend).await?;
        Ok(Tensor { storage, dimensions: dimensions.clone() })
    }
}

pub struct SmallTensor<'a, T, A: Backend> {
    tensor: &'a Tensor<T, A>,
}

impl<'a, T, A: Backend> SmallTensor<'a, T, A> {
    /// # Safety
    /// The buffer must be small enough so that copying from it to or from host does not block.
    pub unsafe fn new(tensor: &'a Tensor<T, A>) -> Self {
        Self { tensor }
    }
}

impl<T, A: Backend> HasBackend for SmallTensor<'_, T, A> {
    type Backend = A;
    fn backend(&self) -> &A {
        self.tensor.backend()
    }
}

impl<T: DeviceCopy> CopyIntoBackend<CpuBackend, TaskScope> for SmallTensor<'_, T, TaskScope> {
    type Output = Tensor<T, CpuBackend>;
    async fn copy_into_backend(self, backend: &CpuBackend) -> Result<Self::Output, CopyError> {
        let dimensions = self.tensor.dimensions.clone();
        let storage = unsafe { SmallBuffer::new(&self.tensor.storage) };
        let storage = storage.copy_into_backend(backend).await?;
        Ok(Tensor { storage, dimensions })
    }
}

impl<T: DeviceCopy> CopyIntoBackend<TaskScope, CpuBackend> for SmallTensor<'_, T, CpuBackend> {
    type Output = Tensor<T, TaskScope>;
    async fn copy_into_backend(self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        let dimensions = self.tensor.dimensions.clone();
        let storage = unsafe { SmallBuffer::new(&self.tensor.storage) };
        let storage = storage.copy_into_backend(backend).await?;
        Ok(Tensor { storage, dimensions })
    }
}
