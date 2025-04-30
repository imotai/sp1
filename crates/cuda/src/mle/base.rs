use futures::prelude::*;
use std::sync::Arc;

use slop_algebra::Field;
use slop_alloc::{mem::CopyError, Buffer, CopyIntoBackend, CopyToBackend, CpuBackend, ToHost};
use slop_multilinear::{Evaluations, Mle, MleEval, Padding, Point};
use slop_tensor::{Tensor, TransposeBackend};
use tokio::sync::oneshot;

use crate::{sync::CudaSend, DeviceCopy, SmallBuffer, SmallTensor, TaskScope};

impl<F: Field> CopyIntoBackend<CpuBackend, TaskScope> for Mle<F, TaskScope>
where
    TaskScope: TransposeBackend<F>,
{
    type Output = Mle<F, CpuBackend>;
    async fn copy_into_backend(self, backend: &CpuBackend) -> Result<Self::Output, CopyError> {
        // Transpose the values in the device since it's usually faster.
        let tensor = self.into_guts().transpose();
        let guts = tensor.copy_into_backend(backend).await?;
        Ok(Mle::new(guts))
    }
}

impl<F: Field> CopyIntoBackend<CpuBackend, TaskScope> for Arc<Mle<F, TaskScope>>
where
    TaskScope: TransposeBackend<F>,
{
    type Output = Mle<F, CpuBackend>;
    async fn copy_into_backend(self, backend: &CpuBackend) -> Result<Self::Output, CopyError> {
        let tensor = self.guts().transpose();
        let guts = tensor.copy_into_backend(backend).await?;
        Ok(Mle::new(guts))
    }
}

impl<F: Field> CopyToBackend<CpuBackend, TaskScope> for Mle<F, TaskScope>
where
    TaskScope: TransposeBackend<F>,
{
    type Output = Mle<F, CpuBackend>;
    async fn copy_to_backend(&self, backend: &CpuBackend) -> Result<Self::Output, CopyError> {
        let tensor = self.guts().transpose();
        let guts = tensor.copy_into_backend(backend).await?;
        Ok(Mle::new(guts))
    }
}

impl<F: DeviceCopy> CopyIntoBackend<TaskScope, CpuBackend> for Mle<F, CpuBackend>
where
    TaskScope: TransposeBackend<F>,
{
    type Output = Mle<F, TaskScope>;
    async fn copy_into_backend(self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        // Transfer to device and then do trasnspose since it's usually faster.
        let tensor = self.into_guts();
        let guts = tensor.copy_into_backend(backend).await?;
        let guts = guts.transpose();
        Ok(Mle::new(guts))
    }
}

impl<F: DeviceCopy> CopyIntoBackend<TaskScope, CpuBackend> for Arc<Mle<F, CpuBackend>>
where
    TaskScope: TransposeBackend<F>,
{
    type Output = Mle<F, TaskScope>;
    async fn copy_into_backend(self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        // Transfer to device and then do trasnspose since it's usually faster.
        let (tx, rx) = oneshot::channel();
        let backend = backend.clone();
        tokio::task::spawn_blocking(move || {
            let dimensions = self.guts().dimensions.clone();
            let storage_buf = self.guts().as_buffer();
            let mut storage = Buffer::with_capacity_in(storage_buf.len(), backend);
            storage.extend_from_host_slice(storage_buf).unwrap();
            let guts = Tensor { storage, dimensions };
            tx.send(guts).ok();
        });
        let guts = rx.await.unwrap();
        let guts = guts.transpose();
        Ok(Mle::new(guts))
    }
}

impl<F: DeviceCopy> CopyIntoBackend<CpuBackend, TaskScope> for MleEval<F, TaskScope> {
    type Output = MleEval<F, CpuBackend>;
    async fn copy_into_backend(self, backend: &CpuBackend) -> Result<Self::Output, CopyError> {
        let tensor = self.into_evaluations();
        let evaluations = tensor.copy_into_backend(backend).await?;
        Ok(MleEval::new(evaluations))
    }
}

impl<F: DeviceCopy> CopyIntoBackend<CpuBackend, TaskScope> for Padding<F, TaskScope> {
    type Output = Padding<F, CpuBackend>;
    async fn copy_into_backend(self, backend: &CpuBackend) -> Result<Self::Output, CopyError> {
        match self {
            Padding::Generic(padding_values) => padding_values
                .as_ref()
                .copy_to_backend(backend)
                .await
                .map(Arc::new)
                .map(Padding::Generic),
            Padding::Constant((value, num_polys, _)) => {
                Ok(Padding::Constant((value, num_polys, *backend)))
            }
            Padding::Zero((num_polys, _)) => Ok(Padding::Zero((num_polys, *backend))),
        }
    }
}

impl<F: DeviceCopy> CopyIntoBackend<TaskScope, CpuBackend> for MleEval<F, CpuBackend> {
    type Output = MleEval<F, TaskScope>;
    async fn copy_into_backend(self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        let tensor = self.into_evaluations();
        let evaluations = tensor.copy_into_backend(backend).await?;
        Ok(MleEval::new(evaluations))
    }
}

impl<F: DeviceCopy> CopyIntoBackend<TaskScope, CpuBackend> for Padding<F, CpuBackend> {
    type Output = Padding<F, TaskScope>;
    async fn copy_into_backend(self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        match self {
            Padding::Generic(padding_values) => padding_values
                .as_ref()
                .copy_to_backend(backend)
                .await
                .map(Arc::new)
                .map(Padding::Generic),
            Padding::Constant((value, num_polys, _)) => {
                Ok(Padding::Constant((value, num_polys, backend.clone())))
            }
            Padding::Zero((num_polys, _)) => Ok(Padding::Zero((num_polys, backend.clone()))),
        }
    }
}

impl<F: DeviceCopy> CopyToBackend<CpuBackend, TaskScope> for MleEval<F, TaskScope> {
    type Output = MleEval<F, CpuBackend>;
    async fn copy_to_backend(&self, backend: &CpuBackend) -> Result<Self::Output, CopyError> {
        let tensor = unsafe { SmallTensor::new(self.evaluations()) };
        let evaluations = tensor.copy_into_backend(backend).await?;
        Ok(MleEval::new(evaluations))
    }
}

impl<F: DeviceCopy> CopyToBackend<CpuBackend, TaskScope> for Padding<F, TaskScope> {
    type Output = Padding<F, CpuBackend>;
    async fn copy_to_backend(&self, backend: &CpuBackend) -> Result<Self::Output, CopyError> {
        match self {
            Padding::Generic(padding_values) => {
                let padding_values = padding_values.as_ref().copy_to_backend(backend).await?;
                let padding_values = Arc::new(padding_values);
                Ok(Padding::Generic(padding_values))
            }
            Padding::Constant((value, num_polys, _)) => {
                Ok(Padding::Constant((*value, *num_polys, *backend)))
            }
            Padding::Zero((num_polys, _)) => Ok(Padding::Zero((*num_polys, *backend))),
        }
    }
}

impl<F: DeviceCopy> CopyToBackend<TaskScope, CpuBackend> for MleEval<F, CpuBackend> {
    type Output = MleEval<F, TaskScope>;
    async fn copy_to_backend(&self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        // Transfer to device and then do trasnspose since it's usually faster.
        let tensor = unsafe { SmallTensor::new(self.evaluations()) };
        let evaluations = tensor.copy_into_backend(backend).await?;
        Ok(MleEval::new(evaluations))
    }
}

impl<F: DeviceCopy> CopyToBackend<TaskScope, CpuBackend> for Padding<F, CpuBackend> {
    type Output = Padding<F, TaskScope>;
    async fn copy_to_backend(&self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        match self {
            Padding::Generic(padding_values) => {
                let padding_values = padding_values.as_ref().copy_to_backend(backend).await?;
                let padding_values = Arc::new(padding_values);
                Ok(Padding::Generic(padding_values))
            }
            Padding::Constant((value, num_polys, _)) => {
                Ok(Padding::Constant((*value, *num_polys, backend.clone())))
            }
            Padding::Zero((num_polys, _)) => Ok(Padding::Zero((*num_polys, backend.clone()))),
        }
    }
}

impl<F: DeviceCopy> CopyIntoBackend<CpuBackend, TaskScope> for Point<F, TaskScope> {
    type Output = Point<F, CpuBackend>;
    async fn copy_into_backend(self, backend: &CpuBackend) -> Result<Self::Output, CopyError> {
        let tensor = self.into_values();
        let values = tensor.copy_into_backend(backend).await?;
        Ok(Point::new(values))
    }
}

impl<F: DeviceCopy> CopyIntoBackend<TaskScope, CpuBackend> for Point<F, CpuBackend> {
    type Output = Point<F, TaskScope>;
    async fn copy_into_backend(self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        let tensor = self.into_values();
        let values = tensor.copy_into_backend(backend).await?;
        Ok(Point::new(values))
    }
}

impl<F: DeviceCopy> CopyToBackend<CpuBackend, TaskScope> for Point<F, TaskScope> {
    type Output = Point<F, CpuBackend>;
    async fn copy_to_backend(&self, backend: &CpuBackend) -> Result<Self::Output, CopyError> {
        let tensor = unsafe { SmallBuffer::new(self.values()) };
        let values = tensor.copy_into_backend(backend).await?;
        Ok(Point::new(values))
    }
}

impl<F: DeviceCopy> CopyToBackend<TaskScope, CpuBackend> for Point<F, CpuBackend> {
    type Output = Point<F, TaskScope>;
    async fn copy_to_backend(&self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        // Transfer to device and then do trasnspose since it's usually faster.
        let tensor = unsafe { SmallBuffer::new(self.values()) };
        let values = tensor.copy_into_backend(backend).await?;
        Ok(Point::new(values))
    }
}

impl<F: DeviceCopy> CopyIntoBackend<TaskScope, CpuBackend> for Evaluations<F, CpuBackend> {
    type Output = Evaluations<F, TaskScope>;
    async fn copy_into_backend(self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        let evaluations = stream::iter(self.into_iter())
            .then(|evals| async { backend.into_device(evals).await.unwrap() })
            .collect::<Vec<_>>()
            .await;
        Ok(Evaluations::new(evaluations))
    }
}

impl<F: DeviceCopy> CopyIntoBackend<CpuBackend, TaskScope> for Evaluations<F, TaskScope> {
    type Output = Evaluations<F, CpuBackend>;
    async fn copy_into_backend(self, _backend: &CpuBackend) -> Result<Self::Output, CopyError> {
        let evaluations = stream::iter(self.into_iter())
            .then(|evals| async move { evals.to_host().await.unwrap() })
            .collect::<Vec<_>>()
            .await;
        Ok(Evaluations::new(evaluations))
    }
}

impl<T> CudaSend for Mle<T, TaskScope> {
    #[inline]
    unsafe fn send_to_scope(self, scope: &TaskScope) -> Self {
        let guts = self.into_guts().send_to_scope(scope);
        Mle::new(guts)
    }
}

impl<T> CudaSend for MleEval<T, TaskScope> {
    #[inline]
    unsafe fn send_to_scope(self, scope: &TaskScope) -> Self {
        let evaluations = self.into_evaluations().send_to_scope(scope);
        MleEval::new(evaluations)
    }
}

impl<T> CudaSend for Point<T, TaskScope> {
    #[inline]
    unsafe fn send_to_scope(self, scope: &TaskScope) -> Self {
        let values = self.into_values().send_to_scope(scope);
        Point::new(values)
    }
}
