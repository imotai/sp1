use crate::Tensor;

use super::{GlobalAllocator, PinnedAllocator};

pub type HostTensor<T> = Tensor<T, GlobalAllocator>;

pub type PinnedTensor<T> = Tensor<T, PinnedAllocator>;
