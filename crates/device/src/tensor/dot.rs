use crate::{mem::DeviceData, DeviceScope};

use super::{SumBackend, Tensor, TensorView, TensorViewMut};

pub trait DotBackend<T: DeviceData, U: DeviceData>: SumBackend<U> {
    fn dot_along_dim(
        &self,
        src: TensorView<T, Self>,
        scalars: TensorView<U, Self>,
        dst: TensorViewMut<U, Self>,
        dim: usize,
    );
}

impl<T: DeviceData, A: DeviceScope> Tensor<T, A> {
    /// Computes the dot product of the tensor against a 1-dimensional tensor along the specified
    /// dimension.
    pub fn dot<U: DeviceData>(&self, scalars: TensorView<U, A>, dim: usize) -> Tensor<U, A>
    where
        A: DotBackend<T, U>,
    {
        assert_eq!(scalars.sizes().len(), 1, "Scalars must be a 1-dimensional tensor");
        assert_eq!(
            self.sizes()[dim],
            scalars.sizes()[0],
            "The dimension to dot along must have the same size as the scalars"
        );
        let mut sizes = self.sizes().to_vec();
        sizes.remove(dim);
        let mut dst = Tensor::zeros_in(sizes, self.scope().clone());
        self.scope().dot_along_dim(self.as_view(), scalars, dst.as_view_mut(), dim);
        dst
    }
}
