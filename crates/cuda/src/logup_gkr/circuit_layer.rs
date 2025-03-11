use std::sync::Arc;

use crate::{
    args,
    sys::{
        logup_gkr::{
            gkr_circuit_transition_baby_bear_extension_kernel,
            gkr_circuit_transition_baby_bear_kernel, logup_gkr_poly_baby_bear_extension_kernel,
            logup_gkr_poly_baby_bear_kernel,
        },
        runtime::KernelPtr,
    },
    TaskScope,
};
use slop_algebra::{extension::BinomialExtensionField, ExtensionField, Field};
use slop_alloc::HasBackend;
use slop_baby_bear::BabyBear;
use slop_multilinear::{Mle, PaddedMle, Padding};
use slop_tensor::Tensor;

pub trait GKRCircuitKernel<K, EK> {
    fn gkr_circuit_transition() -> KernelPtr;
}

impl GKRCircuitKernel<BabyBear, BinomialExtensionField<BabyBear, 4>> for TaskScope {
    fn gkr_circuit_transition() -> KernelPtr {
        unsafe { gkr_circuit_transition_baby_bear_kernel() }
    }
}

impl GKRCircuitKernel<BinomialExtensionField<BabyBear, 4>, BinomialExtensionField<BabyBear, 4>>
    for TaskScope
{
    fn gkr_circuit_transition() -> KernelPtr {
        unsafe { gkr_circuit_transition_baby_bear_extension_kernel() }
    }
}

pub async fn gkr_circuit_transition<K: Field, EK: ExtensionField<K>>(
    p: &PaddedMle<K, TaskScope>,
    q: &PaddedMle<EK, TaskScope>,
) -> (PaddedMle<EK, TaskScope>, PaddedMle<EK, TaskScope>)
where
    TaskScope: GKRCircuitKernel<K, EK>,
{
    let backend = p.padding_values().backend();

    let height = p.num_real_entries();
    let width = p.num_polynomials();

    assert_eq!(q.num_real_entries(), height);
    assert_eq!(q.num_polynomials(), width);

    if p.inner().is_none() {
        assert!(q.inner().is_none());

        return (
            PaddedMle::new(
                None,
                p.num_variables() - 1,
                Padding::Constant((EK::zero(), p.num_polynomials(), backend.clone())),
            ),
            PaddedMle::new(
                None,
                q.num_variables() - 1,
                Padding::Constant((EK::one(), p.num_polynomials(), backend.clone())),
            ),
        );
    }

    let output_height = height.div_ceil(2);

    let mut new_p: Tensor<EK, TaskScope> =
        Tensor::with_sizes_in([width, output_height], backend.clone());

    let mut new_q: Tensor<EK, TaskScope> =
        Tensor::with_sizes_in([width, output_height], backend.clone());

    const BLOCK_SIZE: usize = 256;
    const STRIDE: usize = 1;
    let grid_size_x = output_height.div_ceil(BLOCK_SIZE * STRIDE);
    let grid_size_y = width;
    let grid_size = (grid_size_x, grid_size_y, 1);

    unsafe {
        new_p.assume_init();
        new_q.assume_init();

        let args = args!(
            p.inner().as_ref().unwrap().guts().as_ptr(),
            q.inner().as_ref().unwrap().guts().as_ptr(),
            new_p.as_mut_ptr(),
            new_q.as_mut_ptr(),
            height,
            width
        );

        backend
            .launch_kernel(
                <TaskScope as GKRCircuitKernel<K, EK>>::gkr_circuit_transition(),
                grid_size,
                (BLOCK_SIZE, 1, 1),
                &args,
                0,
            )
            .unwrap();
    }

    (
        PaddedMle::new(
            Some(Arc::new(Mle::new(new_p))),
            p.num_variables() - 1,
            Padding::Constant((EK::zero(), p.num_polynomials(), backend.clone())),
        ),
        PaddedMle::new(
            Some(Arc::new(Mle::new(new_q))),
            q.num_variables() - 1,
            Padding::Constant((EK::one(), p.num_polynomials(), backend.clone())),
        ),
    )
}

pub trait GKRPolyKernel<K, EK> {
    fn gkr_poly() -> KernelPtr;
}

impl GKRPolyKernel<BabyBear, BinomialExtensionField<BabyBear, 4>> for TaskScope {
    fn gkr_poly() -> KernelPtr {
        unsafe { logup_gkr_poly_baby_bear_kernel() }
    }
}

impl GKRPolyKernel<BinomialExtensionField<BabyBear, 4>, BinomialExtensionField<BabyBear, 4>>
    for TaskScope
{
    fn gkr_poly() -> KernelPtr {
        unsafe { logup_gkr_poly_baby_bear_extension_kernel() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::transpose::DeviceTransposeKernel;
    use crate::IntoDevice;
    use itertools::Itertools;
    use rand::{distributions::Standard, prelude::Distribution, Rng};
    use slop_algebra::extension::BinomialExtensionField;
    use slop_algebra::AbstractField;
    use slop_alloc::{CpuBackend, IntoHost};
    use slop_baby_bear::BabyBear;
    use slop_matrix::dense::RowMajorMatrix;
    use slop_multilinear::{Mle, Padding};
    use sp1_stark::circuit_layer;

    pub async fn test_gkr_circuit<F, EF>(p_vals: PaddedMle<F>, q_vals: PaddedMle<EF>)
    where
        F: Field,
        EF: ExtensionField<F>,
        Standard: Distribution<F> + Distribution<EF>,
        TaskScope: GKRCircuitKernel<F, EF> + DeviceTransposeKernel<F> + DeviceTransposeKernel<EF>,
        Mle<F, TaskScope>: IntoHost<Output = Mle<F>>,
        Mle<EF, TaskScope>: IntoHost<Output = Mle<EF>>,
        Mle<F>: IntoDevice<Output = Mle<F, TaskScope>>,
        Mle<EF>: IntoDevice<Output = Mle<EF, TaskScope>>,
    {
        let (expected_ps, expected_qs) = circuit_layer(&p_vals, &q_vals);

        let (new_p_host, new_q_host) = crate::task()
            .await
            .unwrap()
            .run(|t| async move {
                let p_mle_device = t.into_device(p_vals).await.unwrap();
                let q_mle_device = t.into_device(q_vals).await.unwrap();

                let (new_p, new_q) =
                    gkr_circuit_transition::<F, EF>(&p_mle_device, &q_mle_device).await;
                let new_p_host = new_p.into_host().await.unwrap();
                let new_q_host = new_q.into_host().await.unwrap();

                (new_p_host, new_q_host)
            })
            .await
            .await
            .unwrap();

        for (i, ((new_p, new_q), (expected_p, expected_q))) in new_p_host
            .inner()
            .as_ref()
            .map(|x| x.guts().as_slice())
            .unwrap_or(&[])
            .iter()
            .zip_eq(new_q_host.inner().as_ref().map(|x| x.guts().as_slice()).unwrap_or(&[]).iter())
            .zip_eq(
                expected_ps
                    .inner()
                    .as_ref()
                    .map(|x| x.guts().as_slice())
                    .unwrap_or(&[])
                    .iter()
                    .zip_eq(
                        expected_qs
                            .inner()
                            .as_ref()
                            .map(|x| x.guts().as_slice())
                            .unwrap_or(&[])
                            .iter(),
                    ),
            )
            .enumerate()
        {
            assert_eq!(new_p, expected_p, "Mismatch in entry {}", i);
            assert_eq!(new_q, expected_q, "Mismatch in entry {}", i);
        }

        assert_eq!(new_p_host.num_variables(), expected_ps.num_variables());
        assert_eq!(new_q_host.num_variables(), expected_qs.num_variables());
    }

    #[tokio::test]
    async fn test_gkr_circuit_baby_bear_extension() {
        type EF = BinomialExtensionField<BabyBear, 4>;
        let mut rng = rand::thread_rng();

        let height = (1 << 10) + 7;
        let width = 2;

        let p_vals =
            RowMajorMatrix::new((0..height * width).map(|_| rng.gen::<EF>()).collect(), width);
        let q_vals =
            RowMajorMatrix::new((0..height * width).map(|_| rng.gen::<EF>()).collect(), width);

        let p = Mle::<EF>::new(p_vals.into());

        let q = Mle::<EF>::new(q_vals.into());

        let p_vals = PaddedMle::new(
            Some(Arc::new(p)),
            11,
            Padding::Constant((EF::zero(), width, CpuBackend)),
        );
        let q_vals = PaddedMle::new(
            Some(Arc::new(q)),
            11,
            Padding::Constant((EF::one(), width, CpuBackend)),
        );

        test_gkr_circuit::<BinomialExtensionField<BabyBear, 4>, BinomialExtensionField<BabyBear, 4>>(p_vals, q_vals).await;

        let p_vals = PaddedMle::new(None, 11, Padding::Constant((EF::zero(), width, CpuBackend)));
        let q_vals = PaddedMle::new(None, 11, Padding::Constant((EF::one(), width, CpuBackend)));

        test_gkr_circuit::<BinomialExtensionField<BabyBear, 4>, BinomialExtensionField<BabyBear, 4>>(p_vals, q_vals).await;
    }

    #[tokio::test]
    async fn test_gkr_circuit_baby_bear() {
        type F = BabyBear;
        type EF = BinomialExtensionField<BabyBear, 4>;
        let mut rng = rand::thread_rng();

        let height = (1 << 10) + 7;
        let width = 2;

        let p_vals =
            RowMajorMatrix::new((0..height * width).map(|_| rng.gen::<F>()).collect(), width);
        let q_vals =
            RowMajorMatrix::new((0..height * width).map(|_| rng.gen::<EF>()).collect(), width);

        let p = Mle::<F>::new(p_vals.into());

        let q = Mle::<EF>::new(q_vals.into());

        let p_vals = PaddedMle::new(
            Some(Arc::new(p)),
            11,
            Padding::Constant((F::zero(), width, CpuBackend)),
        );
        let q_vals = PaddedMle::new(
            Some(Arc::new(q)),
            11,
            Padding::Constant((EF::one(), width, CpuBackend)),
        );
        test_gkr_circuit::<BabyBear, BinomialExtensionField<BabyBear, 4>>(p_vals, q_vals).await;

        let p_vals = PaddedMle::new(None, 11, Padding::Constant((F::zero(), width, CpuBackend)));
        let q_vals = PaddedMle::new(None, 11, Padding::Constant((EF::one(), width, CpuBackend)));
        test_gkr_circuit::<BabyBear, BinomialExtensionField<BabyBear, 4>>(p_vals, q_vals).await;
    }
}
