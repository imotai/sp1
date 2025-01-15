mod p3;
mod single_layer;
mod tcs;

pub use single_layer::*;
pub use tcs::*;

use csl_device::{mem::DeviceData, Buffer, DeviceScope};

/// A binary Merkle tree.
#[derive(Debug, Clone)]
pub struct MerkleTree<T: DeviceData, A: DeviceScope> {
    /// The digests of the tree.
    ///
    /// The digests are stored so that the root is an index `0`, and for each node `i`, its left
    /// child is at index `2i + 1` and the right child is at index `2i + 2`.
    pub digests: Buffer<T, A>,
    /// The total height of the tree.
    pub height: usize,
}

impl<T: DeviceData, A: DeviceScope> MerkleTree<T, A> {
    pub fn uninit(height: usize, allocator: A) -> Self {
        Self { digests: Buffer::with_capacity_in(Self::digests_len(height), allocator), height }
    }

    pub(crate) unsafe fn new_from_digests(digests: Buffer<T, A>) -> Self {
        let height = digests.len().ilog2() as usize;

        Self { digests, height }
    }

    #[inline]
    const fn digests_len(height: usize) -> usize {
        (1 << (height + 1)) - 1
    }

    /// # Safety
    ///
    /// Todo
    pub unsafe fn assume_init(&mut self) {
        self.digests.set_len(Self::digests_len(self.height));
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    #[inline]
    pub fn digests(&self) -> &Buffer<T, A> {
        &self.digests
    }
}
