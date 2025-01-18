use crate::{mem::DeviceData, DeviceScope};

use super::{Tensor, TensorViewMut};

pub trait SumBackend<T: DeviceData>: DeviceScope {
    fn sum_tensor_dim(src: &Tensor<T, Self>, dst: TensorViewMut<T, Self>, dim: usize);
}

impl<T: DeviceData, A: SumBackend<T>> Tensor<T, A> {
    pub fn sum_into(&self, dst: TensorViewMut<T, A>, dim: usize) {
        A::sum_tensor_dim(self, dst, dim);
    }

    pub fn sum(&self, dim: usize) -> Tensor<T, A> {
        let mut sizes = self.sizes().to_vec();
        sizes.remove(dim);
        let mut dst = Tensor::zeros_in(sizes, self.scope().clone());

        A::sum_tensor_dim(self, dst.as_view_mut(), dim);
        dst
    }
}

pub trait InnerProductBackend<T: DeviceData, K: DeviceData>: SumBackend<T> {
    /// Compute the inner product of two tensors along a given dimension.
    fn inner_product_tensor_dim(
        src1: &Tensor<T, Self>,
        src2: &Tensor<K, Self>,
        dst: TensorViewMut<T, Self>,
        dim: usize,
    );
}

impl<T: DeviceData, A: DeviceScope> Tensor<T, A> {
    /// Compute the inner product of two tensors along a given dimension and store the result in
    /// `dst`.
    #[inline]
    pub fn inner_product_into<K: DeviceData>(
        &self,
        src2: &Tensor<K, A>,
        dst: TensorViewMut<T, A>,
        dim: usize,
    ) where
        A: InnerProductBackend<T, K>,
    {
        A::inner_product_tensor_dim(self, src2, dst, dim);
    }

    /// Compute the inner product of two tensors along a given dimension.
    #[inline]
    pub fn inner_product<K: DeviceData>(&self, src2: &Tensor<K, A>, dim: usize) -> Tensor<T, A>
    where
        A: InnerProductBackend<T, K>,
    {
        assert_eq!(
            self.sizes(),
            src2.sizes(),
            "inner product only supported for tensors of the same shape"
        );
        let mut sizes = self.sizes().to_vec();
        sizes.remove(dim);
        let mut dst = Tensor::zeros_in(sizes, self.scope().clone());

        A::inner_product_tensor_dim(self, src2, dst.as_view_mut(), dim);
        dst
    }
}
