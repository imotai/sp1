use crate::{
    mem::DeviceData,
    tensor::{TensorView, TensorViewMut},
    Tensor,
};

use super::{GlobalAllocator, HostBuffer, PinnedAllocator, GLOBAL_ALLOCATOR};

pub type HostTensor<T> = Tensor<T, GlobalAllocator>;
pub type HostTensorView<'a, T> = TensorView<'a, T, GlobalAllocator>;
pub type HostTensorViewMut<'a, T> = TensorViewMut<'a, T, GlobalAllocator>;

pub type PinnedTensor<T> = Tensor<T, PinnedAllocator>;
pub type PinnedTensorView<'a, T> = TensorView<'a, T, PinnedAllocator>;
pub type PinnedTensorViewMut<'a, T> = TensorViewMut<'a, T, PinnedAllocator>;

impl<T: DeviceData> From<Vec<T>> for HostTensor<T> {
    #[inline]
    fn from(vec: Vec<T>) -> Self {
        Self::from(HostBuffer::from(vec))
    }
}

impl<T: DeviceData> HostTensor<T> {
    pub fn with_sizes(sizes: impl AsRef<[usize]>) -> Self {
        Self::with_sizes_in(sizes, GLOBAL_ALLOCATOR)
    }
}

#[cfg(test)]
mod tests {
    use crate::host_buffer;

    use super::*;

    #[test]
    fn test_tensor_element_index() {
        let tensor = HostTensor::<u32>::from(host_buffer![1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            .reshape([2, 5])
            .unwrap();
        assert_eq!(*tensor[[0, 0]], 1);
        assert_eq!(*tensor[[0, 1]], 2);
        assert_eq!(*tensor[[0, 2]], 3);
        assert_eq!(*tensor[[0, 3]], 4);
        assert_eq!(*tensor[[0, 4]], 5);
        assert_eq!(*tensor[[1, 0]], 6);
        assert_eq!(*tensor[[1, 1]], 7);
        assert_eq!(*tensor[[1, 2]], 8);
        assert_eq!(*tensor[[1, 3]], 9);
        assert_eq!(*tensor[[1, 4]], 10);
    }

    #[test]
    fn test_tensor_slice_index() {
        let tensor = HostTensor::<u32>::from(host_buffer![1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            .reshape([2, 5])
            .unwrap();

        let first_row = tensor.index(0);
        assert_eq!(first_row.sizes(), [5]);
        assert_eq!(first_row.strides(), [1]);
        assert_eq!(*first_row[[0]], 1);
        assert_eq!(*first_row[[1]], 2);
        assert_eq!(*first_row[[2]], 3);
        assert_eq!(*first_row[[3]], 4);
        assert_eq!(*first_row[[4]], 5);

        let second_row = tensor.index(1);
        assert_eq!(*second_row[[0]], 6);
        assert_eq!(*second_row[[1]], 7);
        assert_eq!(*second_row[[2]], 8);
        assert_eq!(*second_row[[3]], 9);
        assert_eq!(*second_row[[4]], 10);

        let tensor =
            HostTensor::<u32>::from((0..24).collect::<Vec<_>>()).reshape([2, 3, 4]).unwrap();
        assert_eq!(*tensor[[0, 0, 0]], 0);
        assert_eq!(*tensor[[0, 0, 1]], 1);
        assert_eq!(*tensor[[0, 0, 2]], 2);
        assert_eq!(*tensor[[0, 0, 3]], 3);
        assert_eq!(*tensor[[0, 1, 0]], 4);
        assert_eq!(*tensor[[0, 1, 1]], 5);
        assert_eq!(*tensor[[0, 1, 2]], 6);
        assert_eq!(*tensor[[0, 1, 3]], 7);
        assert_eq!(*tensor[[0, 2, 0]], 8);
        assert_eq!(*tensor[[0, 2, 1]], 9);
        assert_eq!(*tensor[[0, 2, 2]], 10);
        assert_eq!(*tensor[[0, 2, 3]], 11);
        assert_eq!(*tensor[[1, 0, 0]], 12);
        assert_eq!(*tensor[[1, 0, 1]], 13);
        assert_eq!(*tensor[[1, 0, 2]], 14);
        assert_eq!(*tensor[[1, 0, 3]], 15);
        assert_eq!(*tensor[[1, 1, 0]], 16);
        assert_eq!(*tensor[[1, 1, 1]], 17);
        assert_eq!(*tensor[[1, 1, 2]], 18);
        assert_eq!(*tensor[[1, 1, 3]], 19);
        assert_eq!(*tensor[[1, 2, 0]], 20);
        assert_eq!(*tensor[[1, 2, 1]], 21);
        assert_eq!(*tensor[[1, 2, 2]], 22);
        assert_eq!(*tensor[[1, 2, 3]], 23);

        let first_row = tensor.index(0);
        let first_two_rows = first_row.index(..2);
        assert_eq!(first_two_rows.sizes(), [2, 4]);
    }
}
