mod cuda;
mod host;

use csl_device::{mem::DeviceData, DeviceScope, Tensor};
use slop_algebra::{ExtensionField, Field};

pub trait PointBackend<T: DeviceData>: DeviceScope {
    fn dimension(values: &Tensor<T, Self>) -> usize;
}

pub trait MleBaseBackend<F: Field>: DeviceScope {
    /// Returns the number of polynomials in the batch.
    fn num_polynomials(guts: &Tensor<F, Self>) -> usize;

    /// Returns the number of variables in the polynomials.
    fn num_variables(guts: &Tensor<F, Self>) -> u32;

    fn uninit_mle(&self, num_polynomials: usize, num_variables: usize) -> Tensor<F, Self>;
}

pub trait PartialLargangeBackend<F: Field>: PointBackend<F> {
    fn partial_lagrange(point: &Tensor<F, Self>) -> Tensor<F, Self>;
}

pub trait MleEvaluationBackend<F: Field, EF: ExtensionField<F>>: PointBackend<EF> {
    fn eval_mle_at_point(mle: &Tensor<F, Self>, point: &Tensor<EF, Self>) -> Tensor<EF, Self>;
}
