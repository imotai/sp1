use rayon::prelude::*;

use slop_algebra::{AbstractExtensionField, AbstractField, ExtensionField, Field};
use slop_alloc::{Backend, CpuBackend};
use slop_tensor::{AddBackend, Tensor};

use crate::Point;

pub trait MleBaseBackend<F: AbstractField>: Backend {
    /// Returns the number of polynomials in the batch.
    fn num_polynomials(guts: &Tensor<F, Self>) -> usize;

    /// Returns the number of variables in the polynomials.
    fn num_variables(guts: &Tensor<F, Self>) -> u32;

    fn uninit_mle(&self, num_polynomials: usize, num_non_zero_entries: usize) -> Tensor<F, Self>;
}

pub trait PartialLagrangeBackend<F: AbstractField>: MleBaseBackend<F> {
    fn partial_lagrange(point: &Point<F, Self>) -> Tensor<F, Self>;
}

pub trait MleEvaluationBackend<F: AbstractField, EF: AbstractExtensionField<F>>:
    MleBaseBackend<F>
{
    fn eval_mle_at_point(mle: &Tensor<F, Self>, point: &Point<EF, Self>) -> Tensor<EF, Self>;
}

pub trait MleFixLastVariableBackend<F: AbstractField, EF: AbstractExtensionField<F>>:
    MleBaseBackend<F>
{
    fn mle_fix_last_variable(mle: &Tensor<F, Self>, alpha: EF) -> Tensor<EF, Self>;
}

pub trait AddMleBackend<F: AbstractField, EF: AbstractExtensionField<F>>:
    MleBaseBackend<EF> + AddBackend<EF, F, AddOutput = EF>
{
    /// Adds two MLEs together.
    ///
    /// The default implementation assumes that addition of the underlying guts is equivalent to
    /// addition of the MLEs. Backends for which this is not the case should override this method.
    fn add_mle_into(lhs: &Tensor<EF, Self>, rhs: &Tensor<F, Self>, dst: &mut Tensor<EF, Self>) {
        Self::add_into(lhs, rhs, dst);
    }

    fn add_mle(lhs: &Tensor<EF, Self>, rhs: &Tensor<F, Self>) -> Tensor<EF, Self> {
        Self::add(lhs, rhs)
    }
}

impl<F: AbstractField> MleBaseBackend<F> for CpuBackend {
    fn num_polynomials(guts: &Tensor<F, Self>) -> usize {
        guts.sizes()[1]
    }

    fn num_variables(guts: &Tensor<F, Self>) -> u32 {
        guts.sizes()[0].ilog2()
    }

    fn uninit_mle(&self, num_polynomials: usize, num_non_zero_entries: usize) -> Tensor<F, Self> {
        Tensor::with_sizes_in([num_non_zero_entries, num_polynomials], *self)
    }
}

impl<F: AbstractField> PartialLagrangeBackend<F> for CpuBackend {
    fn partial_lagrange(point: &Point<F, Self>) -> Tensor<F, Self> {
        let one = F::one();
        let mut evals = Vec::with_capacity(1 << point.dimension());
        evals.push(one);

        // Build evals in n_variables rounds. In each round, we consider one more entry of `point`,
        // hence the zip.
        point.iter().for_each(|coordinate| {
            evals = evals
                .iter()
                // For each value in the previous round, multiply by (1-coordinate) and coordinate,
                // and collect all these values into a new vec.
                .flat_map(|val| {
                    let prod = val.clone() * coordinate.clone();
                    [val.clone() - prod.clone(), prod]
                })
                .collect();
        });
        Tensor::from(evals).reshape([1 << point.dimension(), 1])
    }
}

impl<F: AbstractField + Sync, EF: AbstractExtensionField<F> + Send + Sync>
    MleEvaluationBackend<F, EF> for CpuBackend
{
    fn eval_mle_at_point(mle: &Tensor<F, Self>, point: &Point<EF, Self>) -> Tensor<EF, Self> {
        let partial_lagrange = Self::partial_lagrange(point);
        mle.dot(&partial_lagrange, 0)
    }
}

impl<F: AbstractField + Send + Sync, EF: AbstractExtensionField<F> + Send + Sync>
    AddMleBackend<F, EF> for CpuBackend
{
}

// Compute the random linear combination of the even and odd coefficients of `vals`. This is used
// to reduce the two evaluation claims for new_point into a single evaluation claim.
pub trait MleFoldBackend<F: AbstractField>: MleBaseBackend<F> {
    fn fold_mle(guts: &Tensor<F, Self>, beta: F) -> Tensor<F, Self>;
}

impl<F: Field> MleFoldBackend<F> for CpuBackend {
    fn fold_mle(guts: &Tensor<F, Self>, beta: F) -> Tensor<F, Self> {
        // Compute the random linear combination of the even and odd coefficients of `vals`. This is
        // used to reduce the two evaluation claims for new_point into a single evaluation claim.
        assert_eq!(guts.sizes()[1], 1, "this is only supported for a single polynomial");
        let fold_guts = guts
            .as_buffer()
            .par_iter()
            .step_by(2)
            .copied()
            .zip(guts.as_buffer().par_iter().skip(1).step_by(2).copied())
            .map(|(a, b)| a + beta * b)
            .collect::<Vec<_>>();
        let dim = fold_guts.len();
        Tensor::from(fold_guts).reshape([dim, 1])
    }
}

impl<F, EF> MleFixLastVariableBackend<F, EF> for CpuBackend
where
    F: Field,
    EF: ExtensionField<F>,
{
    fn mle_fix_last_variable(mle: &Tensor<F, Self>, alpha: EF) -> Tensor<EF, Self> {
        let num_polynomials = CpuBackend::num_polynomials(mle);
        assert_eq!(num_polynomials, 1, "this is only supported for a single polynomial");
        let num_non_zero_elements_out = mle.sizes()[0].div_ceil(2);
        let mut result = Vec::with_capacity(num_non_zero_elements_out);
        mle.as_buffer()
            .par_iter()
            .chunks(2)
            .map(|chunk| {
                let [x, y] = chunk.try_into().unwrap();
                // Computes alpha * y + (1 - alpha) * x
                alpha * (*y - *x) + (*x)
            })
            .collect_into_vec(&mut result);

        Tensor::from(result).reshape([num_non_zero_elements_out, 1])
    }
}
