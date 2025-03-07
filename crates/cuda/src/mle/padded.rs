use std::sync::Arc;

use futures::future::OptionFuture;
use slop_algebra::Field;
use slop_alloc::{mem::CopyError, Buffer, CopyIntoBackend, CopyToBackend, CpuBackend, ToHost};
use slop_multilinear::{Mle, PaddedMle};
use slop_tensor::{Tensor, TransposeBackend};
use tokio::sync::oneshot;

use crate::TaskScope;

impl<F: Field> CopyIntoBackend<CpuBackend, TaskScope> for PaddedMle<F, TaskScope>
where
    TaskScope: TransposeBackend<F>,
{
    type Output = PaddedMle<F, CpuBackend>;
    async fn copy_into_backend(self, backend: &CpuBackend) -> Result<Self::Output, CopyError> {
        self.copy_to_backend(backend).await
    }
}

impl<F: Field> CopyToBackend<CpuBackend, TaskScope> for PaddedMle<F, TaskScope>
where
    TaskScope: TransposeBackend<F>,
{
    type Output = PaddedMle<F, CpuBackend>;
    async fn copy_to_backend(&self, backend: &CpuBackend) -> Result<Self::Output, CopyError> {
        let mle =
            OptionFuture::from(self.inner().clone().map(|mle| mle.copy_into_backend(backend)))
                .await
                .transpose()?;
        let mle = mle.map(Arc::new);
        let padded_values = self.padding_values().to_host().await?;
        let num_variables = self.num_variables();
        Ok(PaddedMle::new(mle, num_variables, padded_values))
    }
}

impl<F: Field> CopyIntoBackend<TaskScope, CpuBackend> for PaddedMle<F, CpuBackend>
where
    TaskScope: TransposeBackend<F>,
{
    type Output = PaddedMle<F, TaskScope>;
    async fn copy_into_backend(self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        self.copy_to_backend(backend).await
    }
}

impl<F: Field> CopyToBackend<TaskScope, CpuBackend> for PaddedMle<F, CpuBackend>
where
    TaskScope: TransposeBackend<F>,
{
    type Output = PaddedMle<F, TaskScope>;
    async fn copy_to_backend(&self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        // Transfer to device and then do trasnspose since it's usually faster.
        let (tx, rx) = oneshot::channel();
        let padded_values = self.padding_values().copy_to_backend(backend).await?;
        let num_variables = self.num_variables();
        let backend = backend.clone();
        let inner = self.inner().clone();
        tokio::task::spawn_blocking(move || {
            if let Some(mle) = inner {
                let dimensions = mle.guts().dimensions.clone();
                let storage_buf = mle.guts().as_buffer();
                let mut storage = Buffer::with_capacity_in(storage_buf.len(), backend);
                storage.extend_from_host_slice(storage_buf).unwrap();
                let guts = Tensor { storage, dimensions };
                tx.send(Some(guts)).ok();
            } else {
                tx.send(None).ok();
            }
        });
        let guts = rx.await.unwrap();
        let inner = guts.map(|guts| Arc::new(Mle::new(guts.transpose())));
        Ok(PaddedMle::new(inner, num_variables, padded_values))
    }
}
