use csl_device::{GlobalAllocator, Tensor};
use slop_algebra::Field;

use super::{MleBaseBackend, PointBackend};

impl<F: Field> PointBackend<F> for GlobalAllocator {
    #[inline]
    fn dimension(values: &Tensor<F, Self>) -> usize {
        values.sizes()[0]
    }
}

impl<F: Field> MleBaseBackend<F> for GlobalAllocator {
    fn num_polynomials(guts: &Tensor<F, Self>) -> usize {
        guts.sizes()[1]
    }

    fn num_variables(guts: &Tensor<F, Self>) -> u32 {
        guts.sizes()[0].ilog2()
    }

    fn uninit_mle(&self, num_polynomials: usize, num_variables: usize) -> Tensor<F, Self> {
        Tensor::with_sizes_in([1 << num_variables, num_polynomials], *self)
    }
}
