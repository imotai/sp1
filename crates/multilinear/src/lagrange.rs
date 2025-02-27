use std::future::Future;

use slop_algebra::AbstractField;
use slop_alloc::CpuBackend;
use slop_tensor::Tensor;
use tokio::sync::oneshot;

use crate::{MleBaseBackend, Point};

pub trait PartialLagrangeBackend<F: AbstractField>: MleBaseBackend<F> {
    fn partial_lagrange(
        point: &Point<F, Self>,
    ) -> impl Future<Output = Tensor<F, Self>> + Send + Sync;
}

impl<F: AbstractField + 'static> PartialLagrangeBackend<F> for CpuBackend {
    async fn partial_lagrange(point: &Point<F, Self>) -> Tensor<F, Self> {
        let (tx, rx) = oneshot::channel();
        let point = point.clone();
        rayon::spawn(move || {
            let result = partial_lagrange_blocking(&point);
            tx.send(result).unwrap();
        });
        rx.await.unwrap()
    }
}

pub(crate) fn partial_lagrange_blocking<F: AbstractField>(
    point: &Point<F, CpuBackend>,
) -> Tensor<F, CpuBackend> {
    let one = F::one();
    let mut evals = Vec::with_capacity(1 << point.dimension());
    evals.push(one);

    // Build evals in num_variables rounds. In each round, we consider one more entry of `point`,
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
