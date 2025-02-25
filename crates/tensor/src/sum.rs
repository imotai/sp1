use std::future::Future;

use slop_alloc::Backend;

use crate::Tensor;

pub trait AddBackend<T, U>: Backend {
    type AddOutput;

    fn add(
        lhs: &Tensor<T, Self>,
        rhs: &Tensor<U, Self>,
    ) -> impl Future<Output = Tensor<Self::AddOutput, Self>> + Send + Sync;
}

pub trait AddAssignBackend<T, U>: Backend {
    fn add_assign(
        lhs: &mut Tensor<T, Self>,
        rhs: &Tensor<U, Self>,
    ) -> impl Future<Output = ()> + Send + Sync;
}
