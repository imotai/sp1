use std::ops::{Deref, DerefMut};

use csl_cuda::IntoDevice;
use csl_cuda::TaskScope;
use csl_utils::DenseDataMut;
use csl_utils::{DenseData, Ext, Felt, JaggedMle};
use slop_algebra::Field;
use slop_alloc::Buffer;
use slop_alloc::HasBackend;
use slop_alloc::ToHost;
use slop_alloc::{Backend, CpuBackend};
use thiserror::Error;

pub struct JaggedDenseMle<F, B: Backend>(pub JaggedMle<DenseBuffer<F, B>, B>);
pub struct JaggedDenseInfo<B: Backend>(pub JaggedMle<InfoBuffer<B>, B>);

impl<F: Field, B: Backend> Deref for JaggedDenseMle<F, B> {
    type Target = JaggedMle<DenseBuffer<F, B>, B>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<F: Field, B: Backend> DerefMut for JaggedDenseMle<F, B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<B: Backend> Deref for JaggedDenseInfo<B> {
    type Target = JaggedMle<InfoBuffer<B>, B>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<B: Backend> DerefMut for JaggedDenseInfo<B> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

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

#[derive(Clone)]
pub struct InfoBuffer<B: Backend = TaskScope> {
    pub data: Buffer<u32, B>,
}

/// The raw pointer equivalent of [`DenseBuffer`] for use in cuda kernels.
#[repr(C)]
pub struct DenseBufferRaw<F> {
    data: *const F,
}

/// The raw pointer equivalent of [`InfoBuffer`] for use in cuda kernels.
#[repr(C)]
pub struct InfoBufferRaw {
    data: *const u32,
}

/// The mutable raw pointer equivalent of [`DenseBuffer`] for use in cuda kernels.
#[repr(C)]
pub struct DenseBufferMutRaw<F> {
    data: *mut F,
}

/// The mutable raw pointer equivalent of [`InfoBuffer`] for use in cuda kernels.
#[repr(C)]
pub struct InfoBufferMutRaw {
    data: *mut u32,
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

impl<A: Backend> InfoBuffer<A> {
    #[inline]
    pub fn new(data: Buffer<u32, A>) -> Self {
        Self { data }
    }

    #[allow(clippy::missing_safety_doc)]
    #[inline]
    pub unsafe fn assume_init(&mut self) {
        self.data.assume_init();
    }

    #[inline]
    pub fn into_parts(self) -> Buffer<u32, A> {
        self.data
    }
}

impl<F, A: Backend> HasBackend for DenseBuffer<F, A> {
    type Backend = A;

    fn backend(&self) -> &Self::Backend {
        self.data.backend()
    }
}

impl<A: Backend> HasBackend for InfoBuffer<A> {
    type Backend = A;

    fn backend(&self) -> &Self::Backend {
        self.data.backend()
    }
}

impl<F, A: Backend> DenseData<A> for DenseBuffer<F, A> {
    type DenseDataRaw = DenseBufferRaw<F>;
    fn as_ptr(&self) -> Self::DenseDataRaw {
        DenseBufferRaw { data: self.data.as_ptr() }
    }
}

impl<A: Backend> DenseData<A> for InfoBuffer<A> {
    type DenseDataRaw = InfoBufferRaw;
    fn as_ptr(&self) -> Self::DenseDataRaw {
        InfoBufferRaw { data: self.data.as_ptr() }
    }
}

impl<A: Backend> DenseDataMut<A> for InfoBuffer<A> {
    type DenseDataMutRaw = InfoBufferMutRaw;
    fn as_mut_ptr(&mut self) -> Self::DenseDataMutRaw {
        InfoBufferMutRaw { data: self.data.as_mut_ptr() }
    }
}

impl JaggedDenseMle<Felt, CpuBackend> {
    pub async fn into_device(
        self,
        backend: &TaskScope,
    ) -> Result<JaggedDenseMle<Felt, TaskScope>, TransferError> {
        let JaggedMle { dense_data, col_index, start_indices, column_heights } = self.0;

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

        Ok(JaggedDenseMle::new(jagged_dense_mle_device, col_index, start_indices, column_heights))
    }
}

impl JaggedDenseMle<Ext, CpuBackend> {
    pub async fn into_device(
        self,
        backend: &TaskScope,
    ) -> Result<JaggedDenseMle<Ext, TaskScope>, TransferError> {
        let JaggedMle { dense_data, col_index, start_indices, column_heights } = self.0;

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

        Ok(JaggedDenseMle::new(jagged_dense_mle_device, col_index, start_indices, column_heights))
    }
}

impl JaggedDenseInfo<CpuBackend> {
    pub async fn into_device(
        self,
        backend: &TaskScope,
    ) -> Result<JaggedDenseInfo<TaskScope>, TransferError> {
        let JaggedMle { dense_data, col_index, start_indices, column_heights } = self.0;

        let data = dense_data
            .data
            .into_device_in(backend)
            .await
            .map_err(|e| TransferError::HostToDeviceTransferError(e.to_string()))?;

        let jagged_dense_info_device = InfoBuffer::new(data);

        let col_index = col_index
            .into_device_in(backend)
            .await
            .map_err(|e| TransferError::HostToDeviceTransferError(e.to_string()))?;

        let start_indices = start_indices
            .into_device_in(backend)
            .await
            .map_err(|e| TransferError::HostToDeviceTransferError(e.to_string()))?;

        Ok(JaggedDenseInfo::new(jagged_dense_info_device, col_index, start_indices, column_heights))
    }
}

impl JaggedDenseMle<Felt, TaskScope> {
    pub async fn into_host(self) -> Result<JaggedDenseMle<Felt, CpuBackend>, TransferError> {
        let JaggedMle { dense_data, col_index, start_indices, column_heights } = self.0;

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

        Ok(JaggedDenseMle::new(jagged_dense_mle_host, col_index, start_indices, column_heights))
    }
}

impl JaggedDenseMle<Ext, TaskScope> {
    pub async fn into_host(self) -> Result<JaggedDenseMle<Ext, CpuBackend>, TransferError> {
        let JaggedMle { dense_data, col_index, start_indices, column_heights } = self.0;

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

        Ok(JaggedDenseMle::new(jagged_dense_mle_host, col_index, start_indices, column_heights))
    }
}

impl JaggedDenseInfo<TaskScope> {
    pub async fn into_host(self) -> Result<JaggedDenseInfo<CpuBackend>, TransferError> {
        let JaggedMle { dense_data, col_index, start_indices, column_heights } = self.0;

        let data = dense_data
            .data
            .to_host()
            .await
            .map_err(|e| TransferError::DeviceToHostTransferError(e.to_string()))?;
        let jagged_dense_info_host = InfoBuffer::new(data);

        let col_index = col_index
            .to_host()
            .await
            .map_err(|e| TransferError::DeviceToHostTransferError(e.to_string()))?;

        let start_indices = start_indices
            .to_host()
            .await
            .map_err(|e| TransferError::DeviceToHostTransferError(e.to_string()))?;

        Ok(JaggedDenseInfo::new(jagged_dense_info_host, col_index, start_indices, column_heights))
    }
}

impl<B: Backend> JaggedDenseInfo<B> {
    pub fn new(
        info_buffer: InfoBuffer<B>,
        col_index: Buffer<u32, B>,
        start_indices: Buffer<u32, B>,
        column_heights: Vec<u32>,
    ) -> Self {
        Self(JaggedMle::new(info_buffer, col_index, start_indices, column_heights))
    }
}

impl<F: Field, B: Backend> JaggedDenseMle<F, B> {
    pub fn new(
        dense_data: DenseBuffer<F, B>,
        col_index: Buffer<u32, B>,
        start_indices: Buffer<u32, B>,
        column_heights: Vec<u32>,
    ) -> Self {
        Self(JaggedMle::new(dense_data, col_index, start_indices, column_heights))
    }
}
