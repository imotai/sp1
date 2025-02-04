use slop_algebra::Field;
use slop_alloc::{mem::CopyError, CpuBackend};
use slop_multilinear::{Mle, Point};
use slop_tensor::TransposeBackend;

use crate::{IntoDevice, IntoHost, TaskScope};

impl<F: Field> IntoHost for Mle<F, TaskScope>
where
    TaskScope: TransposeBackend<F>,
{
    type HostData = Mle<F, CpuBackend>;

    async fn into_host(self) -> Result<Self::HostData, CopyError> {
        // Transpose the values in the device since it's usually faster.
        let tensor = self.into_guts().transpose();
        let guts = tensor.into_host().await?;
        Ok(Mle::new(guts))
    }
}

impl<F: Field> IntoDevice for Mle<F, CpuBackend>
where
    TaskScope: TransposeBackend<F>,
{
    type DeviceData = Mle<F, TaskScope>;

    async fn into_device_in(self, scope: &TaskScope) -> Result<Self::DeviceData, CopyError> {
        // Transfer to device and then do trasnspose since it's usually faster.
        let guts = self.into_guts().into_device_in(scope).await?;
        let guts = guts.transpose();
        Ok(Mle::new(guts))
    }
}

impl<F: Field> IntoDevice for Point<F, CpuBackend> {
    type DeviceData = Point<F, TaskScope>;

    async fn into_device_in(self, scope: &TaskScope) -> Result<Self::DeviceData, CopyError> {
        let values = self.into_values().into_device_in(scope).await?;
        Ok(Point::new(values))
    }
}

impl<F: Field> IntoHost for Point<F, TaskScope> {
    type HostData = Point<F, CpuBackend>;

    async fn into_host(self) -> Result<Self::HostData, CopyError> {
        let values = self.into_values().into_host().await?;
        Ok(Point::new(values))
    }
}
