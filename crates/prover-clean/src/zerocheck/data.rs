use crate::config::{Ext, Felt};
use crate::{DenseData, JaggedMle};
use csl_cuda::IntoDevice;
use csl_cuda::TaskScope;
use slop_alloc::Buffer;
use slop_alloc::HasBackend;
use slop_alloc::ToHost;
use slop_alloc::{Backend, CpuBackend};
use thiserror::Error;

pub type JaggedDenseMle<F, B> = JaggedMle<DenseBuffer<F, B>, B>;

#[derive(Error, Debug)]
pub enum TransferError {
    #[error("Failed to transfer data from host to device: {0}")]
    HostToDeviceTransferError(String),

    #[error("Failed to transfer data from device to host: {0}")]
    DeviceToHostTransferError(String),
}

/// A dense buffer for `JaggedDenseMle`, a wrapper for `Buffer`.
#[derive(Clone)]
pub struct DenseBuffer<F, B: Backend = TaskScope> {
    pub data: Buffer<F, B>,
}

/// We allow dead code here because this is just a wrapper for a c struct.
#[allow(dead_code)]
pub struct DenseBufferRaw<F> {
    data: *const F,
}

#[allow(dead_code)]
pub struct DenseBufferMutRaw<F> {
    data: *mut F,
}

impl<F, A: Backend> DenseBuffer<F, A> {
    #[inline]
    pub fn new(data: Buffer<F, A>) -> Self {
        Self { data }
    }

    #[allow(clippy::missing_safety_doc)]
    #[inline]
    pub unsafe fn assume_init(&mut self) {
        self.data.assume_init();
    }

    #[inline]
    pub fn into_parts(self) -> Buffer<F, A> {
        self.data
    }
}

impl<F, A: Backend> HasBackend for DenseBuffer<F, A> {
    type Backend = A;

    fn backend(&self) -> &Self::Backend {
        self.data.backend()
    }
}

impl<F, A: Backend> DenseData<A> for DenseBuffer<F, A> {
    type DenseDataRaw = DenseBufferRaw<F>;
    type DenseDataMutRaw = DenseBufferMutRaw<F>;
    fn as_ptr(&self) -> Self::DenseDataRaw {
        DenseBufferRaw { data: self.data.as_ptr() }
    }
    fn as_mut_ptr(&mut self) -> Self::DenseDataMutRaw {
        DenseBufferMutRaw { data: self.data.as_mut_ptr() }
    }
}

impl JaggedDenseMle<Felt, CpuBackend> {
    pub async fn into_device(
        self,
        backend: &TaskScope,
    ) -> Result<JaggedDenseMle<Felt, TaskScope>, TransferError> {
        let JaggedMle { dense_data, col_index, start_indices, column_heights } = self;

        let data = dense_data
            .data
            .into_device_in(backend)
            .await
            .map_err(|e| TransferError::HostToDeviceTransferError(e.to_string()))?;

        let jagged_dense_mle_device = DenseBuffer::new(data);

        let col_index = col_index
            .into_device_in(backend)
            .await
            .map_err(|e| TransferError::HostToDeviceTransferError(e.to_string()))?;

        let start_indices = start_indices
            .into_device_in(backend)
            .await
            .map_err(|e| TransferError::HostToDeviceTransferError(e.to_string()))?;

        Ok(JaggedMle::new(jagged_dense_mle_device, col_index, start_indices, column_heights))
    }
}

impl JaggedDenseMle<Ext, CpuBackend> {
    pub async fn into_device(
        self,
        backend: &TaskScope,
    ) -> Result<JaggedDenseMle<Ext, TaskScope>, TransferError> {
        let JaggedMle { dense_data, col_index, start_indices, column_heights } = self;

        let data = dense_data
            .data
            .into_device_in(backend)
            .await
            .map_err(|e| TransferError::HostToDeviceTransferError(e.to_string()))?;

        let jagged_dense_mle_device = DenseBuffer::new(data);

        let col_index = col_index
            .into_device_in(backend)
            .await
            .map_err(|e| TransferError::HostToDeviceTransferError(e.to_string()))?;

        let start_indices = start_indices
            .into_device_in(backend)
            .await
            .map_err(|e| TransferError::HostToDeviceTransferError(e.to_string()))?;

        Ok(JaggedMle::new(jagged_dense_mle_device, col_index, start_indices, column_heights))
    }
}

impl JaggedDenseMle<Ext, TaskScope> {
    pub async fn into_host(self) -> Result<JaggedDenseMle<Ext, CpuBackend>, TransferError> {
        let JaggedMle { dense_data, col_index, start_indices, column_heights } = self;

        let data = dense_data
            .data
            .to_host()
            .await
            .map_err(|e| TransferError::DeviceToHostTransferError(e.to_string()))?;
        let jagged_dense_mle_host = DenseBuffer::new(data);

        let col_index = col_index
            .to_host()
            .await
            .map_err(|e| TransferError::DeviceToHostTransferError(e.to_string()))?;

        let start_indices = start_indices
            .to_host()
            .await
            .map_err(|e| TransferError::DeviceToHostTransferError(e.to_string()))?;

        Ok(JaggedMle::new(jagged_dense_mle_host, col_index, start_indices, column_heights))
    }
}
