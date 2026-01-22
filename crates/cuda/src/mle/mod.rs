mod base;
mod eval;
mod fold;
mod restrict;

use slop_algebra::Field;
use slop_alloc::{mem::CopyError, CpuBackend};
use slop_multilinear::Mle;
use slop_tensor::Tensor;

pub use eval::{DeviceMleEval, DevicePoint, PartialGeqKernel, PartialLagrangeKernel};
pub use fold::FoldKernel;
pub use restrict::MleFixLastVariableKernel;

use crate::tensor::transpose::DeviceTransposeKernel;
use crate::{DeviceCopy, DeviceTensor, TaskScope};

/// A multilinear extension (MLE) stored on the GPU device.
pub struct DeviceMle<F> {
    raw: Mle<F, TaskScope>,
}

impl<F: DeviceCopy + Field> DeviceMle<F> {
    /// Creates a new DeviceMle from an Mle.
    pub fn new(mle: Mle<F, TaskScope>) -> Self {
        Self { raw: mle }
    }

    /// Creates a new DeviceMle from a DeviceTensor of guts.
    pub fn from_guts(guts: DeviceTensor<F>) -> Self {
        Self { raw: Mle::new(guts.into_inner()) }
    }

    /// Returns a reference to the underlying guts tensor.
    pub fn guts(&self) -> &Tensor<F, TaskScope> {
        self.raw.guts()
    }

    /// Consumes self and returns the underlying Mle.
    pub fn into_inner(self) -> Mle<F, TaskScope> {
        self.raw
    }

    /// Consumes self and returns the underlying guts tensor as a DeviceTensor.
    pub fn into_guts(self) -> DeviceTensor<F> {
        DeviceTensor::from_raw(self.raw.into_guts())
    }

    /// Returns the number of polynomials in this MLE.
    /// MLE guts shape is [num_polynomials, num_entries] for TaskScope convention
    pub fn num_polynomials(&self) -> usize {
        self.raw.guts().sizes()[0]
    }

    /// Returns the number of variables in this MLE.
    pub fn num_variables(&self) -> u32 {
        self.raw.guts().sizes()[1].next_power_of_two().ilog2()
    }

    /// Returns the number of non-zero entries in this MLE.
    pub fn num_non_zero_entries(&self) -> usize {
        self.raw.guts().sizes()[1]
    }

    /// Returns the backend (TaskScope) for this MLE.
    pub fn backend(&self) -> &TaskScope {
        self.raw.guts().backend()
    }

    /// Copies a host MLE to the device.
    ///
    /// The host MLE uses CpuBackend convention [num_entries, num_polynomials].
    /// The device MLE uses TaskScope convention [num_polynomials, num_entries].
    /// This method transposes the data during copy to convert between conventions.
    pub fn from_host(host_mle: &Mle<F, CpuBackend>, scope: &TaskScope) -> Result<Self, CopyError>
    where
        TaskScope: DeviceTransposeKernel<F>,
    {
        let host_guts = host_mle.guts();
        // Host shape is [num_entries, num_polynomials]
        let device_guts_untransposed = DeviceTensor::from_host(host_guts, scope)?;
        // Transpose to [num_polynomials, num_entries] for TaskScope convention
        let device_guts = device_guts_untransposed.transpose();
        Ok(Self::from_guts(device_guts))
    }

    /// Copies this MLE back to the host.
    ///
    /// The device MLE uses TaskScope convention [num_polynomials, num_entries].
    /// The host MLE uses CpuBackend convention [num_entries, num_polynomials].
    /// This method transposes the data during copy to convert between conventions.
    pub fn to_host(&self) -> Result<Mle<F, CpuBackend>, CopyError>
    where
        TaskScope: DeviceTransposeKernel<F>,
    {
        let guts = DeviceTensor::from_raw(self.guts().clone());
        // Device shape is [num_polynomials, num_entries], transpose to [num_entries, num_polynomials]
        let transposed = guts.transpose();
        let host_guts = transposed.to_host()?;
        Ok(Mle::new(host_guts))
    }
}
