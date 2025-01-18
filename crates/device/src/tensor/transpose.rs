use crate::{mem::DeviceData, DeviceScope};

use super::{Tensor, TensorViewMut};

pub trait TransposeBackend<T: DeviceData>: DeviceScope {
    fn transpose_tensor_into(src: &Tensor<T, Self>, dst: TensorViewMut<T, Self>);

    /// Returns a new tensor with the last two dimensions transposed.
    fn transpose(src: &Tensor<T, Self>) -> Tensor<T, Self> {
        let mut sizes = src.sizes().to_vec();
        let len = sizes.len();
        sizes.swap(len - 1, len - 2);
        let mut dst = Tensor::with_sizes_in(sizes, src.scope().clone());

        unsafe {
            dst.assume_init();
        }
        Self::transpose_tensor_into(src, dst.as_view_mut());

        dst
    }
}

impl<T: DeviceData, A: TransposeBackend<T>> Tensor<T, A> {
    #[inline]
    pub fn transpose_into(&self, dst: TensorViewMut<T, A>) {
        A::transpose_tensor_into(self, dst);
    }

    #[inline]
    pub fn transpose(&self) -> Tensor<T, A> {
        A::transpose(self)
    }
}
