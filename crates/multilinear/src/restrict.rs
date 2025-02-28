use rayon::prelude::*;
use std::future::Future;
use tokio::sync::oneshot;

use slop_algebra::{AbstractExtensionField, AbstractField, ExtensionField, Field};
use slop_alloc::CpuBackend;
use slop_tensor::Tensor;

use crate::{MleBaseBackend, MleEval, MleEvaluationBackend, Point};

pub trait MleFixLastVariableBackend<F: AbstractField, EF: AbstractExtensionField<F>>:
    MleBaseBackend<F>
{
    fn mle_fix_last_variable(
        mle: &Tensor<F, Self>,
        alpha: EF,
        padding_values: Option<&MleEval<F, Self>>,
    ) -> impl Future<Output = Tensor<EF, Self>> + Send + Sync;
}

pub trait MleFixLastVariableInPlaceBackend<F: AbstractField>: MleBaseBackend<F> {
    fn mle_fix_last_variable_in_place(
        mle: &mut Tensor<F, Self>,
        alpha: F,
    ) -> impl Future<Output = ()> + Send + Sync;
}

impl<F, EF> MleFixLastVariableBackend<F, EF> for CpuBackend
where
    F: Field,
    EF: ExtensionField<F>,
{
    async fn mle_fix_last_variable(
        mle: &Tensor<F, Self>,
        alpha: EF,
        padding_values: Option<&MleEval<F, Self>>,
    ) -> Tensor<EF, Self> {
        let width = CpuBackend::num_polynomials(mle);
        let mle = unsafe { mle.owned_unchecked() };
        let padding_values = match padding_values {
            Some(p) => unsafe { p.owned_unchecked() },
            None => {
                let zeros: MleEval<_> = vec![F::zero(); width].into();
                unsafe { zeros.owned_unchecked() }
            }
        };

        let (tx, rx) = oneshot::channel();
        slop_futures::rayon::spawn(move || {
            let mle = mle;
            let num_polynomials = CpuBackend::num_polynomials(&mle);
            let num_non_zero_elements_out = mle.sizes()[0].div_ceil(2);
            // let mut result = Vec::with_capacity(num_non_zero_elements_out * num_polynomials);
            let result = mle
                .as_buffer()
                .par_chunks(2 * num_polynomials)
                .flat_map_iter(|chunk| {
                    (0..num_polynomials).map(|i| {
                        let x = chunk[i];
                        let y = chunk
                            .get(i + num_polynomials)
                            .copied()
                            .unwrap_or_else(|| padding_values[i]);
                        // return alpha * y + (EF::one() - alpha) * x, but in a more efficient way
                        // that minimizes extension field multiplications.
                        alpha * (y - x) + x
                    })
                })
                .collect::<Vec<_>>();

            let result = Tensor::from(result).reshape([num_non_zero_elements_out, num_polynomials]);
            tx.send(result).unwrap();
        });
        rx.await.unwrap()
    }
}

pub trait MleFixedAtZeroBackend<F: AbstractField, EF: AbstractExtensionField<F>>:
    MleEvaluationBackend<F, EF>
{
    fn fixed_at_zero(
        mle: &Tensor<F, Self>,
        point: &Point<EF>,
    ) -> impl Future<Output = Tensor<EF, CpuBackend>> + Send;
}

impl<F: Field, EF: ExtensionField<F>> MleFixedAtZeroBackend<F, EF> for CpuBackend {
    async fn fixed_at_zero(mle: &Tensor<F, Self>, point: &Point<EF, Self>) -> Tensor<EF, Self> {
        // TODO: A smarter way to do this is pre-cache the partial_lagrange_evals that are implicit
        // in `eval_at_point` so we don't recompute it at every step of BaseFold.
        let mle = unsafe { mle.owned_unchecked() };
        let (tx, rx) = oneshot::channel();
        slop_futures::rayon::spawn(move || {
            let even_values = mle.as_slice().par_iter().step_by(2).copied().collect::<Vec<_>>();
            tx.send(even_values).unwrap();
        });
        let even_values = rx.await.unwrap();
        CpuBackend::eval_mle_at_point(&Tensor::from(even_values), point).await
    }
}
