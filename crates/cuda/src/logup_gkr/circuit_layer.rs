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
use sp1_stark::GkrMle;

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
    input_mle: &GkrMle<K, EK, TaskScope>,
) -> GkrMle<EK, EK, TaskScope>
where
    TaskScope: GKRCircuitKernel<K, EK>,
{
    let GkrMle { numerator_0: p_0, numerator_1: p_1, denom_0: q_0, denom_1: q_1 } = input_mle;
    let backend = p_0.padding_values().backend();

    let height = p_0.num_real_entries();
    let width = p_0.num_polynomials();

    assert_eq!(q_0.num_real_entries(), height);
    assert_eq!(q_0.num_polynomials(), width);
    assert_eq!(q_1.num_real_entries(), height);
    assert_eq!(q_1.num_polynomials(), width);
    assert_eq!(p_1.num_real_entries(), height);
    assert_eq!(p_1.num_polynomials(), width);

    if p_0.inner().is_none() {
        assert!(q_0.inner().is_none());
        assert!(p_1.inner().is_none());
        assert!(q_1.inner().is_none());

        let numerator_0 = PaddedMle::new(
            None,
            p_0.num_variables() - 1,
            Padding::Constant((EK::zero(), p_0.num_polynomials(), backend.clone())),
        );
        let numerator_1 = numerator_0.clone();
        let denom_0 = PaddedMle::new(
            None,
            q_0.num_variables() - 1,
            Padding::Constant((EK::one(), p_0.num_polynomials(), backend.clone())),
        );
        let denom_1 = denom_0.clone();
        return GkrMle { numerator_0, numerator_1, denom_0, denom_1 };
    }

    let output_height = height.div_ceil(2);

    let mut new_p_0: Tensor<EK, TaskScope> =
        Tensor::with_sizes_in([width, output_height], backend.clone());

    let mut new_p_1: Tensor<EK, TaskScope> =
        Tensor::with_sizes_in([width, output_height], backend.clone());

    let mut new_q_0: Tensor<EK, TaskScope> =
        Tensor::with_sizes_in([width, output_height], backend.clone());

    let mut new_q_1: Tensor<EK, TaskScope> =
        Tensor::with_sizes_in([width, output_height], backend.clone());

    const BLOCK_SIZE: usize = 256;
    const STRIDE: usize = 1;
    let grid_size_x = output_height.div_ceil(BLOCK_SIZE * STRIDE);
    let grid_size_y = width;
    let grid_size = (grid_size_x, grid_size_y, 1);

    unsafe {
        new_p_0.assume_init();
        new_p_1.assume_init();
        new_q_0.assume_init();
        new_q_1.assume_init();

        let args = args!(
            p_0.inner().as_ref().unwrap().guts().as_ptr(),
            p_1.inner().as_ref().unwrap().guts().as_ptr(),
            q_0.inner().as_ref().unwrap().guts().as_ptr(),
            q_1.inner().as_ref().unwrap().guts().as_ptr(),
            new_p_0.as_mut_ptr(),
            new_p_1.as_mut_ptr(),
            new_q_0.as_mut_ptr(),
            new_q_1.as_mut_ptr(),
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

    let numerator_0 = PaddedMle::new(
        Some(Arc::new(Mle::new(new_p_0))),
        p_0.num_variables() - 1,
        Padding::Constant((EK::zero(), p_0.num_polynomials(), backend.clone())),
    );
    let numerator_1 = PaddedMle::new(
        Some(Arc::new(Mle::new(new_p_1))),
        p_1.num_variables() - 1,
        Padding::Constant((EK::zero(), p_1.num_polynomials(), backend.clone())),
    );
    let denom_0 = PaddedMle::new(
        Some(Arc::new(Mle::new(new_q_0))),
        q_0.num_variables() - 1,
        Padding::Constant((EK::one(), q_0.num_polynomials(), backend.clone())),
    );
    let denom_1 = PaddedMle::new(
        Some(Arc::new(Mle::new(new_q_1))),
        q_1.num_variables() - 1,
        Padding::Constant((EK::one(), q_1.num_polynomials(), backend.clone())),
    );

    GkrMle { numerator_0, numerator_1, denom_0, denom_1 }
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

    pub async fn test_gkr_circuit<F, EF>(
        p_0_vals: PaddedMle<F>,
        p_1_vals: PaddedMle<F>,
        q_0_vals: PaddedMle<EF>,
        q_1_vals: PaddedMle<EF>,
    ) where
        F: Field,
        EF: ExtensionField<F>,
        Standard: Distribution<F> + Distribution<EF>,
        TaskScope: GKRCircuitKernel<F, EF> + DeviceTransposeKernel<F> + DeviceTransposeKernel<EF>,
        Mle<F, TaskScope>: IntoHost<Output = Mle<F>>,
        Mle<EF, TaskScope>: IntoHost<Output = Mle<EF>>,
        Mle<F>: IntoDevice<Output = Mle<F, TaskScope>>,
        Mle<EF>: IntoDevice<Output = Mle<EF, TaskScope>>,
    {
        let mle = GkrMle {
            numerator_0: p_0_vals,
            numerator_1: p_1_vals,
            denom_0: q_0_vals,
            denom_1: q_1_vals,
        };
        let GkrMle {
            numerator_0: new_numerator_0,
            numerator_1: new_numerator_1,
            denom_0: new_denom_0,
            denom_1: new_denom_1,
        } = circuit_layer(&mle);

        let new_mle_host = crate::task()
            .await
            .unwrap()
            .run(|t| async move {
                let p_0_mle_device = t.into_device(mle.numerator_0).await.unwrap();
                let p_1_mle_device = t.into_device(mle.numerator_1).await.unwrap();
                let q_0_mle_device = t.into_device(mle.denom_0).await.unwrap();
                let q_1_mle_device = t.into_device(mle.denom_1).await.unwrap();

                let mle_device = GkrMle {
                    numerator_0: p_0_mle_device,
                    numerator_1: p_1_mle_device,
                    denom_0: q_0_mle_device,
                    denom_1: q_1_mle_device,
                };

                let new_mle = gkr_circuit_transition::<F, EF>(&mle_device).await;
                let new_p_0_host = new_mle.numerator_0.into_host().await.unwrap();
                let new_p_1_host = new_mle.numerator_1.into_host().await.unwrap();
                let new_q_0_host = new_mle.denom_0.into_host().await.unwrap();
                let new_q_1_host = new_mle.denom_1.into_host().await.unwrap();

                GkrMle {
                    numerator_0: new_p_0_host,
                    numerator_1: new_p_1_host,
                    denom_0: new_q_0_host,
                    denom_1: new_q_1_host,
                }
            })
            .await
            .await
            .unwrap();

        for (i, ((new_p, new_q), (expected_p, expected_q))) in new_mle_host
            .numerator_0
            .inner()
            .as_ref()
            .map(|x| x.guts().as_slice())
            .unwrap_or(&[])
            .iter()
            .zip_eq(
                new_mle_host
                    .denom_0
                    .inner()
                    .as_ref()
                    .map(|x| x.guts().as_slice())
                    .unwrap_or(&[])
                    .iter(),
            )
            .zip_eq(
                new_numerator_0
                    .inner()
                    .as_ref()
                    .map(|x| x.guts().as_slice())
                    .unwrap_or(&[])
                    .iter()
                    .zip_eq(
                        new_denom_0
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

        for (i, ((new_p, new_q), (expected_p, expected_q))) in new_mle_host
            .numerator_1
            .inner()
            .as_ref()
            .map(|x| x.guts().as_slice())
            .unwrap_or(&[])
            .iter()
            .zip_eq(
                new_mle_host
                    .denom_1
                    .inner()
                    .as_ref()
                    .map(|x| x.guts().as_slice())
                    .unwrap_or(&[])
                    .iter(),
            )
            .zip_eq(
                new_numerator_1
                    .inner()
                    .as_ref()
                    .map(|x| x.guts().as_slice())
                    .unwrap_or(&[])
                    .iter()
                    .zip_eq(
                        new_denom_1
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

        assert_eq!(new_mle_host.numerator_0.num_variables(), new_numerator_0.num_variables());
        assert_eq!(new_mle_host.numerator_1.num_variables(), new_numerator_1.num_variables());
        assert_eq!(new_mle_host.denom_0.num_variables(), new_denom_0.num_variables());
        assert_eq!(new_mle_host.denom_1.num_variables(), new_denom_1.num_variables());

        assert_eq!(new_mle_host.numerator_0.num_polynomials(), new_numerator_0.num_polynomials());
        assert_eq!(new_mle_host.numerator_1.num_polynomials(), new_numerator_1.num_polynomials());
        assert_eq!(new_mle_host.denom_0.num_polynomials(), new_denom_0.num_polynomials());
        assert_eq!(new_mle_host.denom_1.num_polynomials(), new_denom_1.num_polynomials());

        assert_eq!(new_mle_host.numerator_0.num_real_entries(), new_numerator_0.num_real_entries());
        assert_eq!(new_mle_host.numerator_1.num_real_entries(), new_numerator_1.num_real_entries());
        assert_eq!(new_mle_host.denom_0.num_real_entries(), new_denom_0.num_real_entries());
        assert_eq!(new_mle_host.denom_1.num_real_entries(), new_denom_1.num_real_entries());
    }

    #[tokio::test]
    async fn test_gkr_circuit_baby_bear_extension() {
        type EF = BinomialExtensionField<BabyBear, 4>;
        let mut rng = rand::thread_rng();

        let height = 1 << 10;
        let width = 2;

        let p_0_vals =
            RowMajorMatrix::new((0..height * width).map(|_| rng.gen::<EF>()).collect(), width);
        let q_0_vals =
            RowMajorMatrix::new((0..height * width).map(|_| rng.gen::<EF>()).collect(), width);

        let p_1_vals =
            RowMajorMatrix::new((0..height * width).map(|_| rng.gen::<EF>()).collect(), width);
        let q_1_vals =
            RowMajorMatrix::new((0..height * width).map(|_| rng.gen::<EF>()).collect(), width);

        let p_0 = Mle::<EF>::new(p_0_vals.into());
        let q_0 = Mle::<EF>::new(q_0_vals.into());
        let p_1 = Mle::<EF>::new(p_1_vals.into());
        let q_1 = Mle::<EF>::new(q_1_vals.into());

        let p_0_vals = PaddedMle::new(
            Some(Arc::new(p_0)),
            10,
            Padding::Constant((EF::zero(), width, CpuBackend)),
        );
        let q_0_vals = PaddedMle::new(
            Some(Arc::new(q_0)),
            10,
            Padding::Constant((EF::one(), width, CpuBackend)),
        );
        let p_1_vals = PaddedMle::new(
            Some(Arc::new(p_1)),
            10,
            Padding::Constant((EF::zero(), width, CpuBackend)),
        );
        let q_1_vals = PaddedMle::new(
            Some(Arc::new(q_1)),
            10,
            Padding::Constant((EF::one(), width, CpuBackend)),
        );

        test_gkr_circuit::<BinomialExtensionField<BabyBear, 4>, BinomialExtensionField<BabyBear, 4>>(p_0_vals, p_1_vals, q_0_vals, q_1_vals).await;

        let p_0_vals = PaddedMle::new(None, 11, Padding::Constant((EF::zero(), width, CpuBackend)));
        let p_1_vals = PaddedMle::new(None, 11, Padding::Constant((EF::zero(), width, CpuBackend)));
        let q_0_vals = PaddedMle::new(None, 11, Padding::Constant((EF::one(), width, CpuBackend)));
        let q_1_vals = PaddedMle::new(None, 11, Padding::Constant((EF::one(), width, CpuBackend)));

        test_gkr_circuit::<BinomialExtensionField<BabyBear, 4>, BinomialExtensionField<BabyBear, 4>>(p_0_vals, p_1_vals, q_0_vals, q_1_vals).await;
    }

    #[tokio::test]
    async fn test_gkr_circuit_baby_bear() {
        type F = BabyBear;
        type EF = BinomialExtensionField<BabyBear, 4>;
        let mut rng = rand::thread_rng();

        let height = 1 << 10;
        let width = 2;

        let p_0_vals =
            RowMajorMatrix::new((0..height * width).map(|_| rng.gen::<F>()).collect(), width);
        let p_1_vals =
            RowMajorMatrix::new((0..height * width).map(|_| rng.gen::<F>()).collect(), width);
        let q_0_vals =
            RowMajorMatrix::new((0..height * width).map(|_| rng.gen::<EF>()).collect(), width);

        let q_1_vals =
            RowMajorMatrix::new((0..height * width).map(|_| rng.gen::<EF>()).collect(), width);

        let p_0 = Mle::<F>::new(p_0_vals.into());
        let p_1 = Mle::<F>::new(p_1_vals.into());
        let q_0 = Mle::<EF>::new(q_0_vals.into());
        let q_1 = Mle::<EF>::new(q_1_vals.into());

        let p_0_vals = PaddedMle::new(
            Some(Arc::new(p_0)),
            10,
            Padding::Constant((F::zero(), width, CpuBackend)),
        );
        let p_1_vals = PaddedMle::new(
            Some(Arc::new(p_1)),
            10,
            Padding::Constant((F::zero(), width, CpuBackend)),
        );
        let q_0_vals = PaddedMle::new(
            Some(Arc::new(q_0)),
            10,
            Padding::Constant((EF::one(), width, CpuBackend)),
        );
        let q_1_vals = PaddedMle::new(
            Some(Arc::new(q_1)),
            10,
            Padding::Constant((EF::one(), width, CpuBackend)),
        );
        test_gkr_circuit::<BabyBear, BinomialExtensionField<BabyBear, 4>>(
            p_0_vals, p_1_vals, q_0_vals, q_1_vals,
        )
        .await;

        let p_0_vals = PaddedMle::new(None, 11, Padding::Constant((F::zero(), width, CpuBackend)));
        let p_1_vals = PaddedMle::new(None, 11, Padding::Constant((F::zero(), width, CpuBackend)));
        let q_0_vals = PaddedMle::new(None, 11, Padding::Constant((EF::one(), width, CpuBackend)));
        let q_1_vals = PaddedMle::new(None, 11, Padding::Constant((EF::one(), width, CpuBackend)));
        test_gkr_circuit::<BabyBear, BinomialExtensionField<BabyBear, 4>>(
            p_0_vals, p_1_vals, q_0_vals, q_1_vals,
        )
        .await;
    }
}
