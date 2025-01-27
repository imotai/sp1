use csl_device::{
    cuda::TaskScope, mem::DeviceData, CudaSend, DeviceScope, GlobalAllocator, Tensor,
};

use crate::backend::PointBackend;

#[derive(Debug, Clone, CudaSend)]
pub struct Point<T: DeviceData, B: DeviceScope = GlobalAllocator> {
    values: Tensor<T, B>,
}

impl<T: DeviceData, B: PointBackend<T>> Point<T, B> {
    #[inline]
    pub fn new(values: Tensor<T, B>) -> Self {
        Self { values }
    }

    #[inline]
    pub fn into_values(self) -> Tensor<T, B> {
        self.values
    }

    /// # Safety
    ///
    #[inline]
    pub unsafe fn assume_init(&mut self) {
        self.values.assume_init();
    }

    #[inline]
    pub fn values(&self) -> &Tensor<T, B> {
        &self.values
    }

    #[inline]
    pub fn dimension(&self) -> usize {
        B::dimension(&self.values)
    }

    #[inline]
    pub fn scope(&self) -> &B {
        self.values.scope()
    }
}
