use std::sync::Arc;

use futures::future::OptionFuture;
use slop_algebra::Field;
use slop_alloc::{mem::CopyError, Buffer, CopyIntoBackend, CopyToBackend, CpuBackend, ToHost};
use slop_multilinear::{ManuallyDroppedPaddedMle, Mle, PaddedMle, Padding};
use slop_tensor::{Tensor, TransposeBackend};
use tokio::sync::oneshot;

use crate::{sync::CudaSend, TaskScope};

impl<F: Field> CopyIntoBackend<CpuBackend, TaskScope> for PaddedMle<F, TaskScope>
where
    TaskScope: TransposeBackend<F>,
{
    type Output = PaddedMle<F, CpuBackend>;
    async fn copy_into_backend(self, backend: &CpuBackend) -> Result<Self::Output, CopyError> {
        self.copy_to_backend(backend).await
    }
}

impl<F: Field> CopyToBackend<CpuBackend, TaskScope> for PaddedMle<F, TaskScope>
where
    TaskScope: TransposeBackend<F>,
{
    type Output = PaddedMle<F, CpuBackend>;
    async fn copy_to_backend(&self, backend: &CpuBackend) -> Result<Self::Output, CopyError> {
        let mle =
            OptionFuture::from(self.inner().clone().map(|mle| mle.copy_into_backend(backend)))
                .await
                .transpose()?;
        let mle = mle.map(Arc::new);
        let padded_values = self.padding_values().to_host().await?;
        let num_variables = self.num_variables();
        Ok(PaddedMle::new(mle, num_variables, padded_values))
    }
}

impl<F: Field> CopyIntoBackend<TaskScope, CpuBackend> for PaddedMle<F, CpuBackend>
where
    TaskScope: TransposeBackend<F>,
{
    type Output = PaddedMle<F, TaskScope>;
    async fn copy_into_backend(self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        self.copy_to_backend(backend).await
    }
}

impl<F: Field> CopyToBackend<TaskScope, CpuBackend> for PaddedMle<F, CpuBackend>
where
    TaskScope: TransposeBackend<F>,
{
    type Output = PaddedMle<F, TaskScope>;
    async fn copy_to_backend(&self, backend: &TaskScope) -> Result<Self::Output, CopyError> {
        // Transfer to device and then do trasnspose since it's usually faster.
        let (tx, rx) = oneshot::channel();
        let padded_values = match self.padding_values() {
            Padding::Generic(padding_values) => {
                let padding_values = padding_values.as_ref().copy_to_backend(backend).await?;
                let padding_values = Arc::new(padding_values);
                Padding::Generic(padding_values)
            }
            Padding::Constant((value, num_polys, _)) => {
                Padding::Constant((*value, *num_polys, backend.clone()))
            }
        };
        let num_variables = self.num_variables();
        let backend = backend.clone();
        let inner = self.inner().clone();
        tokio::task::spawn_blocking(move || {
            if let Some(mle) = inner {
                let dimensions = mle.guts().dimensions.clone();
                let storage_buf = mle.guts().as_buffer();
                let mut storage = Buffer::with_capacity_in(storage_buf.len(), backend);
                storage.extend_from_host_slice(storage_buf).unwrap();
                let guts = Tensor { storage, dimensions };
                tx.send(Some(guts)).ok();
            } else {
                tx.send(None).ok();
            }
        });
        let guts = rx.await.unwrap();
        let inner = guts.map(|guts| Arc::new(Mle::new(guts.transpose())));
        Ok(PaddedMle::new(inner, num_variables, padded_values))
    }
}

impl<F> CudaSend for ManuallyDroppedPaddedMle<F, TaskScope>
where
    F: Field,
{
    #[inline]
    unsafe fn send_to_scope(self, scope: &TaskScope) -> Self {
        self.owned_unchecked_in(scope.clone())
    }
}

#[cfg(test)]
mod tests {
    use std::{iter::once, sync::Arc};

    use rand::Rng;
    use slop_algebra::extension::BinomialExtensionField;
    use slop_alloc::IntoHost;
    use slop_baby_bear::BabyBear;
    use slop_multilinear::{Mle, MleEval, PaddedMle, Padding, Point};

    use crate::ToDevice;

    #[tokio::test]
    async fn test_padded_mle() {
        let mut rng = rand::thread_rng();

        type F = BabyBear;
        type EF = BinomialExtensionField<F, 4>;

        let mle = Mle::<F>::rand(&mut rng, 100, 16);
        let point = Point::<EF>::rand(&mut rng, 18);

        let padding_values: MleEval<F> = vec![rng.gen::<F>(); 100].into();

        let padded_mle =
            PaddedMle::padded(Arc::new(mle), 18, Padding::Generic(Arc::new(padding_values)));

        let alpha = rng.gen::<EF>();

        let d_mle = padded_mle.clone();

        let point_ref = &point;
        let (evals, fixed_evals) = crate::task()
            .await
            .unwrap()
            .run(|t| async move {
                let d_mle = t.into_device(d_mle).await.unwrap();
                let eval = d_mle.eval_at(point_ref).await;
                (
                    eval.into_evaluations().into_host().await.unwrap(),
                    d_mle.fix_last_variable(alpha).await.into_host().await.unwrap(),
                )
            })
            .await
            .await
            .unwrap();

        let evals = evals.as_slice().to_vec();

        let host_evals = padded_mle.eval_at(&point).await.to_vec();
        assert_eq!(evals, host_evals);

        let host_fixed_evals = padded_mle.fix_last_variable(alpha).await;

        assert_eq!(
            fixed_evals.clone().into_inner().unwrap().guts().as_slice().to_vec(),
            host_fixed_evals.clone().into_inner().unwrap().guts().as_slice().to_vec()
        );

        assert_eq!(fixed_evals.num_variables(), host_fixed_evals.num_variables());
        assert_eq!(fixed_evals.num_polynomials(), host_fixed_evals.num_polynomials());
        assert_eq!(fixed_evals.num_real_entries(), host_fixed_evals.num_real_entries());
    }

    #[tokio::test]
    async fn test_spawned_padded_mle_fix_last_variable() {
        let mut rng = rand::thread_rng();

        type F = BabyBear;
        type EF = BinomialExtensionField<F, 4>;

        let num_variables = 16;
        let num_tasks = 10;

        let mles = (0..num_tasks)
            .map(|_| {
                let mle = Mle::<F>::rand(&mut rng, 1, num_variables >> 1);
                PaddedMle::padded_with_zeros(Arc::new(mle), num_variables)
            })
            .collect::<Vec<_>>();
        let random_point = Point::<EF>::rand(&mut rng, num_variables - 1);
        let alpha = rng.gen::<EF>();

        let complete_point = random_point.iter().copied().chain(once(alpha)).collect::<Point<_>>();

        // Do all the fix last variables in parallel.
        crate::task()
            .await
            .unwrap()
            .run(|t| async move {
                let random_point = random_point.clone();
                let mut handles = Vec::new();
                for mle in mles.iter() {
                    let random_point = random_point.clone();
                    let mle = mle.to_device_in(&t).await.unwrap();
                    let handle = t.spawn(move |s| async move {
                        let mle = unsafe { mle.owned_unchecked_in(s.clone()) };
                        let restriction = mle.fix_last_variable(alpha).await;
                        let evals = restriction.eval_at(&random_point).await;
                        evals.into_evaluations().into_host().await.unwrap()
                    });
                    handles.push(handle);
                }
                let mut evals = Vec::new();
                for handle in handles {
                    // Get the evals
                    let eval = handle.await.unwrap().unwrap();
                    evals.push(eval);
                }
                for (eval, mle) in evals.iter().zip(mles) {
                    let d_mle = mle.to_device_in(&t).await.unwrap();
                    let d_eval = d_mle.eval_at(&complete_point).await;
                    let h_eval = d_eval.into_evaluations().into_host().await.unwrap();
                    for (e, h_e) in eval.as_slice().iter().zip(h_eval.as_slice().iter()) {
                        assert_eq!(e, h_e);
                    }
                }
            })
            .await
            .await
            .unwrap();
    }
}
