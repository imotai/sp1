use csl_cuda::{IntoDevice, TaskScope};
use slop_alloc::{Backend, CpuBackend, HasBackend, ToHost};
use slop_tensor::Tensor;

use crate::{
    config::{Ext, Felt},
    jagged::DenseData,
    JaggedMle,
};

use thiserror::Error;

/// Type alias for a jagged MLE with a GKR layer
pub type JaggedGkrMle<B> = JaggedMle<JaggedGkrLayer<B>, B>;

/// Type alias for a jagged MLE with a first GKR layer
pub type JaggedFirstGkrMle<B> = JaggedMle<JaggedFirstGkrLayer<B>, B>;

#[derive(Error, Debug)]
pub enum LayerError {
    #[error("Failed to transfer data from host to device: {0}")]
    HostToDeviceTransferError(String),

    #[error("Failed to transfer data from device to host: {0}")]
    DeviceToHostTransferError(String),
}

/// A layer of the GKR circuit.
///
/// This layer contains the polynomials p_0, p_1, q_0, q_1 in evaluation form.
#[derive(Clone)]
pub struct JaggedGkrLayer<A: Backend = TaskScope> {
    /// The layer data, stored as a tensor of shape [4, 1, 2 * height].
    pub layer: Tensor<Ext, A>,
    /// Half of the height of the layer.
    pub height: usize,
}

/// We allow dead code here because this is just a wrapper for a c struct. Rust never needs to read these fields.
#[allow(dead_code)]
pub struct JaggedGkrLayerRaw {
    layer: *const Ext,
    height: usize,
}

#[allow(dead_code)]
pub struct JaggedGkrLayerMutRaw {
    layer: *mut Ext,
    height: usize,
}

impl<A: Backend> JaggedGkrLayer<A> {
    #[inline]
    pub fn new(layer: Tensor<Ext, A>, height: usize) -> Self {
        Self { layer, height }
    }

    #[inline]
    pub unsafe fn assume_init(&mut self) {
        self.layer.assume_init();
    }

    #[inline]
    pub fn into_parts(self) -> (Tensor<Ext, A>, usize) {
        (self.layer, self.height)
    }
}

impl<A: Backend> HasBackend for JaggedGkrLayer<A> {
    type Backend = A;

    fn backend(&self) -> &Self::Backend {
        self.layer.backend()
    }
}

impl<A: Backend> DenseData<A> for JaggedGkrLayer<A> {
    type DenseDataRaw = JaggedGkrLayerRaw;
    type DenseDataMutRaw = JaggedGkrLayerMutRaw;
    fn as_ptr(&self) -> Self::DenseDataRaw {
        JaggedGkrLayerRaw { layer: self.layer.as_ptr(), height: self.height }
    }
    fn as_mut_ptr(&mut self) -> Self::DenseDataMutRaw {
        JaggedGkrLayerMutRaw { layer: self.layer.as_mut_ptr(), height: self.height }
    }
}

impl JaggedGkrMle<CpuBackend> {
    pub async fn into_device(
        self,
        backend: &TaskScope,
    ) -> Result<JaggedGkrMle<TaskScope>, LayerError> {
        let JaggedMle { dense_data, col_index, start_indices, column_heights } = self;

        let layer = dense_data
            .layer
            .into_device_in(backend)
            .await
            .map_err(|e| LayerError::HostToDeviceTransferError(e.to_string()))?;
        let jagged_gkr_layer_device = JaggedGkrLayer::new(layer, dense_data.height);

        let col_index = col_index
            .into_device_in(backend)
            .await
            .map_err(|e| LayerError::HostToDeviceTransferError(e.to_string()))?;

        let start_indices = start_indices
            .into_device_in(backend)
            .await
            .map_err(|e| LayerError::HostToDeviceTransferError(e.to_string()))?;

        Ok(JaggedMle::new(jagged_gkr_layer_device, col_index, start_indices, column_heights))
    }
}

impl JaggedGkrMle<TaskScope> {
    pub async fn into_host(self) -> Result<JaggedGkrMle<CpuBackend>, LayerError> {
        let JaggedMle { dense_data, col_index, start_indices, column_heights } = self;

        let layer = dense_data
            .layer
            .to_host()
            .await
            .map_err(|e| LayerError::DeviceToHostTransferError(e.to_string()))?;
        let jagged_gkr_layer_host = JaggedGkrLayer::new(layer, dense_data.height);

        let col_index = col_index
            .to_host()
            .await
            .map_err(|e| LayerError::DeviceToHostTransferError(e.to_string()))?;

        let start_indices = start_indices
            .to_host()
            .await
            .map_err(|e| LayerError::DeviceToHostTransferError(e.to_string()))?;

        Ok(JaggedMle::new(jagged_gkr_layer_host, col_index, start_indices, column_heights))
    }
}

/// The first layer of the GKR circuit. This is a special case because the numerator is Felts, and the denominator is Ext.
pub struct JaggedFirstGkrLayer<A: Backend> {
    /// The numerator of the first layer. Has sizes [2, 1, 2 * height].
    pub numerator: Tensor<Felt, A>,
    /// The denominator of the first layer. Has sizes [2, 1, 2 * height].
    pub denominator: Tensor<Ext, A>,
    /// Half of the real height of the layer.
    pub height: usize,
}

#[allow(dead_code)]
pub struct JaggedFirstGkrLayerRaw {
    numerator: *const Felt,
    denominator: *const Ext,
    height: usize,
}

#[allow(dead_code)]
pub struct JaggedFirstGkrLayerMutRaw {
    numerator: *mut Felt,
    denominator: *mut Ext,
    height: usize,
}

impl<A: Backend> JaggedFirstGkrLayer<A> {
    #[inline]
    pub fn new(numerator: Tensor<Felt, A>, denominator: Tensor<Ext, A>, height: usize) -> Self {
        Self { numerator, denominator, height }
    }

    #[inline]
    pub unsafe fn assume_init(&mut self) {
        self.numerator.assume_init();
        self.denominator.assume_init();
    }

    #[inline]
    pub fn into_parts(self) -> (Tensor<Felt, A>, Tensor<Ext, A>, usize) {
        (self.numerator, self.denominator, self.height)
    }
}

impl<A: Backend> HasBackend for JaggedFirstGkrLayer<A> {
    type Backend = A;

    fn backend(&self) -> &Self::Backend {
        self.numerator.backend()
    }
}

impl<A: Backend> DenseData<A> for JaggedFirstGkrLayer<A> {
    type DenseDataRaw = JaggedFirstGkrLayerRaw;
    type DenseDataMutRaw = JaggedFirstGkrLayerMutRaw;
    fn as_ptr(&self) -> Self::DenseDataRaw {
        JaggedFirstGkrLayerRaw {
            numerator: self.numerator.as_ptr(),
            denominator: self.denominator.as_ptr(),
            height: self.height,
        }
    }
    fn as_mut_ptr(&mut self) -> Self::DenseDataMutRaw {
        JaggedFirstGkrLayerMutRaw {
            numerator: self.numerator.as_mut_ptr(),
            denominator: self.denominator.as_mut_ptr(),
            height: self.height,
        }
    }
}
impl JaggedFirstGkrMle<CpuBackend> {
    pub async fn into_device(
        self,
        backend: &TaskScope,
    ) -> Result<JaggedFirstGkrMle<TaskScope>, LayerError> {
        let JaggedMle { dense_data, col_index, start_indices, column_heights } = self;

        let numerator = dense_data
            .numerator
            .into_device_in(backend)
            .await
            .map_err(|e| LayerError::HostToDeviceTransferError(e.to_string()))?;

        let denominator = dense_data
            .denominator
            .into_device_in(backend)
            .await
            .map_err(|e| LayerError::HostToDeviceTransferError(e.to_string()))?;

        let jagged_gkr_layer_device =
            JaggedFirstGkrLayer::new(numerator, denominator, dense_data.height);

        let col_index = col_index
            .into_device_in(backend)
            .await
            .map_err(|e| LayerError::HostToDeviceTransferError(e.to_string()))?;

        let start_indices = start_indices
            .into_device_in(backend)
            .await
            .map_err(|e| LayerError::HostToDeviceTransferError(e.to_string()))?;

        Ok(JaggedMle::new(jagged_gkr_layer_device, col_index, start_indices, column_heights))
    }
}

impl JaggedFirstGkrMle<TaskScope> {
    pub async fn into_host(self) -> Result<JaggedFirstGkrMle<CpuBackend>, LayerError> {
        let JaggedMle { dense_data, col_index, start_indices, column_heights } = self;

        let numerator = dense_data
            .numerator
            .to_host()
            .await
            .map_err(|e| LayerError::DeviceToHostTransferError(e.to_string()))?;

        let denominator = dense_data
            .denominator
            .to_host()
            .await
            .map_err(|e| LayerError::DeviceToHostTransferError(e.to_string()))?;

        let jagged_gkr_layer_host =
            JaggedFirstGkrLayer::new(numerator, denominator, dense_data.height);

        let col_index = col_index
            .to_host()
            .await
            .map_err(|e| LayerError::DeviceToHostTransferError(e.to_string()))?;

        let start_indices = start_indices
            .to_host()
            .await
            .map_err(|e| LayerError::DeviceToHostTransferError(e.to_string()))?;

        Ok(JaggedMle::new(jagged_gkr_layer_host, col_index, start_indices, column_heights))
    }
}
