use std::ops::{Deref, DerefMut, Index, IndexMut};

use derive_where::derive_where;
use rand::{distributions::Standard, prelude::Distribution};
use serde::{Deserialize, Serialize};
use slop_algebra::AbstractField;
use slop_alloc::{buffer, Backend, Buffer, CpuBackend, Init, Slice};
use slop_tensor::Tensor;

#[derive(Debug, Serialize, Deserialize)]
#[derive_where(PartialEq, Eq, Clone; Buffer<T, A>)]
#[serde(bound(
    serialize = "Buffer<T, A>: Serialize",
    deserialize = "Buffer<T, A>: Deserialize<'de>"
))]
pub struct Point<T, A: Backend = CpuBackend> {
    values: Buffer<T, A>,
}

impl<T, A: Backend> Point<T, A> {
    #[inline]
    pub const fn new(values: Buffer<T, A>) -> Self {
        Self { values }
    }

    #[inline]
    pub fn values(&self) -> &Buffer<T, A> {
        &self.values
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.len() == 0
    }

    #[inline]
    pub fn into_values(self) -> Buffer<T, A> {
        self.values
    }

    #[inline]
    pub fn dimension(&self) -> usize {
        self.values.len()
    }

    /// # Safety
    #[inline]
    pub unsafe fn assume_init(&mut self) {
        self.values.assume_init();
    }

    #[inline]
    pub fn backend(&self) -> &A {
        self.values.allocator()
    }

    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.values.as_ptr()
    }

    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.values.as_mut_ptr()
    }

    #[inline]
    pub fn copy_into_host(&self) -> Point<T, CpuBackend> {
        Point::new(Buffer::from(self.values.copy_into_host_vec()))
    }
}

impl<T, A: Backend> Index<usize> for Point<T, A> {
    type Output = Init<T, A>;

    fn index(&self, index: usize) -> &Self::Output {
        self.values.index(index)
    }
}

impl<T, A: Backend> IndexMut<usize> for Point<T, A> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.values.index_mut(index)
    }
}

impl<T> Point<T, CpuBackend> {
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.values.iter()
    }

    pub fn from_usize(num: usize, dimension: usize) -> Self
    where
        T: AbstractField,
    {
        Point::from(
            (0..dimension)
                .rev()
                .map(|i| T::from_canonical_usize((num >> i) & 1))
                .collect::<Vec<_>>(),
        )
    }

    pub fn remove_last_coordinate(&mut self) -> T {
        self.values.pop().expect("Point is empty")
    }

    pub fn rand<R: rand::Rng>(rng: &mut R, dimension: u32) -> Self
    where
        Standard: Distribution<T>,
    {
        Self::new(Tensor::rand(rng, [dimension as usize]).into_buffer())
    }

    pub fn split_at(&self, k: usize) -> (Self, Self)
    where
        T: Clone,
    {
        let (left, right) = self.values.split_at(k);
        let left_values = Buffer::from(left.to_vec());
        let right_values = Buffer::from(right.to_vec());
        (Self::new(left_values), Self::new(right_values))
    }

    #[inline]
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.values.to_vec()
    }

    #[inline]
    pub fn reverse(&mut self) {
        self.values.reverse();
    }

    #[inline]
    pub fn reversed(&self) -> Self {
        let mut point = self.clone();
        point.reverse();
        point
    }

    #[inline]
    pub fn last_k(&self, k: usize) -> Self
    where
        T: Clone,
    {
        Point::new(Buffer::from(self.to_vec()[self.values.len() - k..].to_vec()))
    }

    pub fn copy_into<A: Backend>(&self, alloc: &A) -> Point<T, A> {
        let mut buffer = Buffer::with_capacity_in(self.values.len(), alloc.clone());
        buffer.extend_from_host_slice(&self.values).unwrap();
        Point::new(buffer)
    }

    #[inline]
    pub fn add_dimension(&mut self, dim_val: T) {
        self.values.insert(0, dim_val);
    }
}

impl<T> From<Vec<T>> for Point<T, CpuBackend> {
    fn from(values: Vec<T>) -> Self {
        Self::new(Buffer::from(values))
    }
}

impl<T> FromIterator<T> for Point<T, CpuBackend> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from(iter.into_iter().collect::<Vec<_>>())
    }
}

impl<T: Default> Default for Point<T, CpuBackend> {
    fn default() -> Self {
        Self::new(buffer![])
    }
}

impl<T, A: Backend> Deref for Point<T, A> {
    type Target = Slice<T, A>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl<T, A: Backend> DerefMut for Point<T, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.values
    }
}
