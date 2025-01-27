use std::future::Future;

use csl_device::{
    cuda::{IntoDevice, TaskScope},
    tensor::TransposeBackend,
    GlobalAllocator, Tensor,
};
use slop_algebra::Field;

mod point;

use crate::Mle;

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

impl<F: Field> IntoDevice for Mle<F, GlobalAllocator>
where
    TaskScope: TransposeBackend<F>,
{
    type DeviceData = Mle<F, TaskScope>;

    #[inline]
    fn into_device_in(
        self,
        scope: &TaskScope,
    ) -> impl Future<Output = Result<Self::DeviceData, csl_device::mem::CopyError>> + Send {
        let scope = scope.clone();
        async move {
            let guts = self.into_guts().into_device_in(&scope).await?;
            let guts = guts.transpose();
            Ok(Mle::new(guts))
        }
    }
}
