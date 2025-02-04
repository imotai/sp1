use rayon::prelude::*;

use slop_algebra::{AbstractExtensionField, AbstractField};
use slop_alloc::{buffer, Backend, Buffer, CpuBackend};

use crate::{Dimensions, Tensor};

pub trait DotBackend<T, U>: Backend {
    fn dot_along_dim_into(
        src: &Tensor<T, Self>,
        scalars: &Tensor<U, Self>,
        dst: &mut Tensor<U, Self>,
        dim: usize,
    );

    fn dot_along_dim(
        src: &Tensor<T, Self>,
        scalars: &Tensor<U, Self>,
        dim: usize,
    ) -> Tensor<U, Self>;
}

impl<T, A: Backend> Tensor<T, A> {
    pub fn dot_into<U>(&self, scalars: &Tensor<U, A>, dst: &mut Tensor<U, A>, dim: usize)
    where
        A: DotBackend<T, U>,
    {
        A::dot_along_dim_into(self, scalars, dst, dim);
    }

    pub fn dot<U>(&self, scalars: &Tensor<U, A>, dim: usize) -> Tensor<U, A>
    where
        A: DotBackend<T, U>,
    {
        A::dot_along_dim(self, scalars, dim)
    }
}

impl<T: AbstractField + Sync, U: AbstractExtensionField<T> + Send + Sync> DotBackend<T, U>
    for CpuBackend
{
    fn dot_along_dim_into(
        src: &Tensor<T, Self>,
        scalars: &Tensor<U, Self>,
        dst: &mut Tensor<U, Self>,
        dim: usize,
    ) {
        assert_eq!(dim, 0, "Only dot along the first dimension is supported");
        let total_len = dst.total_len();
        let dot_products = src
            .as_buffer()
            .par_chunks_exact(src.strides()[dim])
            .zip(scalars.as_buffer().par_iter())
            .map(|(chunk, scalar)| chunk.iter().map(|a| scalar.clone() * a.clone()).collect())
            .reduce(
                || vec![U::zero(); total_len],
                |mut a, b| {
                    a.iter_mut().zip(b.iter()).for_each(|(a, b)| *a += b.clone());
                    a
                },
            );

        let dot_products = Buffer::from(dot_products);
        dst.storage = dot_products;
    }

    fn dot_along_dim(
        src: &Tensor<T, Self>,
        scalars: &Tensor<U, Self>,
        dim: usize,
    ) -> Tensor<U, Self> {
        let mut sizes = src.sizes().to_vec();
        sizes.remove(dim);
        let dimensions = Dimensions::try_from(sizes).unwrap();
        let mut dst = Tensor { storage: buffer![], dimensions };
        Self::dot_along_dim_into(src, scalars, &mut dst, dim);
        dst
    }
}

#[cfg(test)]
mod tests {
    use slop_algebra::AbstractField;
    use slop_baby_bear::BabyBear;

    use super::*;

    #[test]
    fn test_dot() {
        let mut rng = rand::thread_rng();
        let tensor = Tensor::<BabyBear, CpuBackend>::rand(&mut rng, [1500, 10]);
        let scalars = Tensor::<BabyBear, CpuBackend>::rand(&mut rng, [1500]);
        let dot = tensor.dot(&scalars, 0);
        for j in 0..10 {
            let mut dot_product = BabyBear::zero();
            for i in 0..1500 {
                dot_product += *scalars[[i]] * *tensor[[i, j]];
            }
            assert_eq!(*dot[[j]], dot_product);
        }
    }
}
