use std::{future::Future, sync::Arc};

use crate::{args, PartialLagrangeKernel, TaskScope};
use futures::future::{join4, OptionFuture};
use itertools::izip;
use slop_algebra::{
    interpolate_univariate_polynomial, ExtensionField, Field, UnivariatePolynomial,
};
use slop_alloc::{Buffer, CanCopyFromRef, CopyIntoBackend, CpuBackend, HasBackend, ToHost};
use slop_multilinear::{Mle, MleFixLastVariableBackend, Point};
use slop_sumcheck::{
    ComponentPolyEvalBackend, SumCheckPolyFirstRoundBackend, SumcheckPolyBackend, SumcheckPolyBase,
};
use slop_tensor::{DotBackend, ReduceSumBackend, Tensor};

// use rayon::prelude::*;
use sp1_stark::{log2_ceil_usize, LogupGkrPoly};

use super::GKRPolyKernel;

async fn logup_sum_as_poly_in_last_variable<NumerType: Field, K: ExtensionField<NumerType>>(
    poly: &LogupGkrPoly<NumerType, K, TaskScope>,
    claim: Option<K>,
) -> UnivariatePolynomial<K>
where
    TaskScope: PartialLagrangeKernel<K>
        + GKRPolyKernel<NumerType, K>
        + ReduceSumBackend<K>
        + DotBackend<K, K>,
{
    let claim = claim.expect("Claim must be provided for LogupGkrPoly");

    let mut rest = poly.point.clone();

    let last_coordinate = rest.remove_last_coordinate();
    let one_minus_first_coord = K::one() - last_coordinate;

    let num_real_variables = if poly.numerator_0.num_real_entries() != 0 {
        log2_ceil_usize(poly.numerator_0.num_real_entries().div_ceil(2))
    } else {
        0
    };

    assert!(
        num_real_variables <= rest.dimension(),
        "There are {} real variables but only {} remaining variables",
        num_real_variables,
        rest.dimension()
    );

    let (most_sig, least_sig) = rest.split_at(rest.dimension() - num_real_variables);
    let least_sig_d =
        least_sig.clone().copy_into_backend(poly.numerator_0.backend()).await.unwrap();
    let partial_lagrange = Mle::partial_lagrange(&least_sig_d);

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let padding_adjustment = most_sig.mle_eval_zero();

    let (mut y_0, mut y_half, host_sums_adjustment) = if poly.numerator_0.inner().is_some() {
        let mapped_vals = logup_poly_sum::<NumerType, K>(
            (
                poly.numerator_0.inner().as_ref().unwrap().clone(),
                poly.numerator_1.inner().as_ref().unwrap().clone(),
            ),
            (
                poly.denom_0.inner().as_ref().unwrap().clone(),
                poly.denom_1.inner().as_ref().unwrap().clone(),
            ),
            poly.lambda,
            partial_lagrange,
        );

        let interactionwise_sums = mapped_vals.await.sum(2).await;

        let sums = interactionwise_sums.dot(&poly.batching_randomness_powers_device, 1).await;

        let host_sums = sums.as_buffer().to_host().await.unwrap().to_vec();
        (
            padding_adjustment * host_sums[0],
            padding_adjustment * host_sums[1],
            padding_adjustment * host_sums[2],
        )
    } else {
        (K::zero(), K::zero(), K::zero())
    };

    let half_point = (0..rest.dimension()).map(|_| K::two().inverse()).collect::<Point<_>>();
    let rest_eq_guts_sum = Mle::full_lagrange_eval(&rest, &half_point)
        * NumerType::from_canonical_u32(1 << rest.dimension())
        - host_sums_adjustment;

    let powers_sum = poly
        .batching_randomness_powers_device
        .sum(0)
        .await
        .as_buffer()
        .to_host()
        .await
        .unwrap()
        .as_slice()[0];

    // Compute the sum over the padding via the "closed-form" formula instead of summing over
    // the padding values.
    y_0 += rest_eq_guts_sum * poly.lambda * powers_sum;
    y_half += K::from_canonical_u64(4) * rest_eq_guts_sum * poly.lambda * powers_sum;

    xs.push(K::zero());
    y_0 *= one_minus_first_coord;
    ys.push(y_0);

    y_half *= poly.one_eighth;
    xs.push(poly.one_half);
    ys.push(y_half);

    let y_1 = (claim / poly.eq_adjustment) - y_0;
    xs.push(K::one());
    ys.push(y_1);

    let point_elements = poly.point.to_vec();
    let point_last = point_elements.last().unwrap();
    let b_const = (K::one() - *point_last) / (K::one() - point_last.double());
    xs.push(b_const);
    ys.push(K::zero());

    let ys = ys.iter().map(|y| *y * poly.eq_adjustment).collect::<Vec<_>>();

    interpolate_univariate_polynomial(&xs, &ys)
}

type MlePair<T> = (Arc<Mle<T, TaskScope>>, Arc<Mle<T, TaskScope>>);

pub async fn logup_poly_sum<K: Field, EK: ExtensionField<K>>(
    p_s: MlePair<K>,
    q_s: MlePair<EK>,
    lambda: EK,
    eq_mle: impl Future<Output = Mle<EK, TaskScope>>,
) -> Tensor<EK, TaskScope>
where
    TaskScope: GKRPolyKernel<K, EK>,
{
    let (p_0, p_1) = p_s;
    let (q_0, q_1) = q_s;
    let backend = p_0.backend();

    let height = p_0.num_non_zero_entries();
    let width = p_0.num_polynomials();

    // println!("Height: {}, Width: {}", height, width);

    assert_eq!(p_1.num_non_zero_entries(), height);
    assert_eq!(p_1.num_polynomials(), width);

    assert_eq!(q_0.num_non_zero_entries(), height);
    assert_eq!(q_0.num_polynomials(), width);

    assert_eq!(q_1.num_non_zero_entries(), height);
    assert_eq!(q_1.num_polynomials(), width);

    let output_height = height.div_ceil(2);

    let mut output: Tensor<EK, TaskScope> =
        Tensor::zeros_in([3, width, output_height], backend.clone());

    const BLOCK_SIZE: usize = 256;
    const STRIDE: usize = 1;
    let grid_size_x = output_height.div_ceil(BLOCK_SIZE * STRIDE);
    let grid_size_y = width;
    let grid_size = (grid_size_x, grid_size_y, 1);
    let eq_mle = eq_mle.await;

    unsafe {
        output.assume_init();

        let args = args!(
            p_0.guts().as_ptr(),
            p_1.guts().as_ptr(),
            q_0.guts().as_ptr(),
            q_1.guts().as_ptr(),
            output.as_mut_ptr(),
            eq_mle.guts().as_ptr(),
            lambda,
            height,
            width
        );

        backend
            .launch_kernel(
                <TaskScope as GKRPolyKernel<K, EK>>::gkr_poly(),
                grid_size,
                (BLOCK_SIZE, 1, 1),
                &args,
                0,
            )
            .unwrap();
    }

    output
}

impl<NumerType: Field, K: ExtensionField<NumerType>>
    ComponentPolyEvalBackend<LogupGkrPoly<NumerType, K, TaskScope>, K> for TaskScope
{
    async fn get_component_poly_evals(poly: &LogupGkrPoly<NumerType, K, TaskScope>) -> Vec<K> {
        assert!(poly.num_variables() == 0);
        let numerator_0 = OptionFuture::from(
            poly.numerator_0
                .inner()
                .as_ref()
                .map(|x| async move { x.guts().as_buffer().to_host().await.unwrap() }),
        )
        .await
        .unwrap_or(vec![NumerType::zero(); poly.numerator_0.num_polynomials()].into());
        // .unwrap_or(vec![NumerType::zero(); poly.numerator_0.num_polynomials()]);
        let numerator_1 = OptionFuture::from(
            poly.numerator_1
                .inner()
                .as_ref()
                .map(|x| async move { x.guts().as_buffer().to_host().await.unwrap() }),
        )
        .await
        .unwrap_or(vec![NumerType::zero(); poly.numerator_1.num_polynomials()].into());

        let denom_0 = OptionFuture::from(
            poly.denom_0
                .inner()
                .as_ref()
                .map(|x| async move { x.guts().as_buffer().to_host().await.unwrap() }),
        )
        .await
        .unwrap_or(vec![K::one(); poly.denom_0.num_polynomials()].into());
        let denom_1 = OptionFuture::from(
            poly.denom_1
                .inner()
                .as_ref()
                .map(|x| async move { x.guts().as_buffer().to_host().await.unwrap() }),
        )
        .await
        .unwrap_or(vec![K::one(); poly.denom_1.num_polynomials()].into());

        izip!(
            numerator_0.iter().copied().map(K::from_base),
            numerator_1.iter().copied().map(K::from_base),
            denom_0.iter().copied(),
            denom_1.iter().copied()
        )
        .flat_map(|(numerator_0_val, numerator_1_val, denom_0_val, denom_1_val)| {
            vec![numerator_0_val, numerator_1_val, denom_0_val, denom_1_val]
        })
        .collect::<Vec<_>>()
    }
}

impl<K: Field> SumcheckPolyBackend<LogupGkrPoly<K, K, TaskScope>, K> for TaskScope
where
    TaskScope: MleFixLastVariableBackend<K, K>
        + PartialLagrangeKernel<K>
        + GKRPolyKernel<K, K>
        + DotBackend<K, K>
        + ReduceSumBackend<K>,
{
    async fn fix_last_variable(
        poly: LogupGkrPoly<K, K, TaskScope>,
        alpha: K,
    ) -> LogupGkrPoly<K, K, TaskScope> {
        let mut new_point = poly.point.clone();
        let last_coord = new_point.remove_last_coordinate();
        let prod = alpha * last_coord;

        let eq_adjustment = poly.eq_adjustment * (K::one() - alpha - last_coord + prod + prod);

        let (numerator_0, numerator_1, denom_0, denom_1) = join4(
            poly.numerator_0.fix_last_variable(alpha),
            poly.numerator_1.fix_last_variable(alpha),
            poly.denom_0.fix_last_variable(alpha),
            poly.denom_1.fix_last_variable(alpha),
        )
        .await;
        LogupGkrPoly::<K, K, TaskScope>::new_with_batching_randomness_powers(
            new_point,
            numerator_0,
            numerator_1,
            denom_0,
            denom_1,
            poly.lambda,
            eq_adjustment,
            poly.batching_randomness,
            poly.batching_randomness_powers,
            poly.batching_randomness_powers_device,
        )
    }

    async fn sum_as_poly_in_last_variable(
        poly: &LogupGkrPoly<K, K, TaskScope>,
        claim: Option<K>,
    ) -> UnivariatePolynomial<K> {
        logup_sum_as_poly_in_last_variable(poly, claim).await
    }
}

impl<NumerType: Field, K: ExtensionField<NumerType>>
    SumCheckPolyFirstRoundBackend<LogupGkrPoly<NumerType, K, TaskScope>, K> for TaskScope
where
    TaskScope: MleFixLastVariableBackend<NumerType, K>
        + MleFixLastVariableBackend<K, K>
        + CanCopyFromRef<Buffer<K>, CpuBackend, Output = Buffer<K, TaskScope>>
        + PartialLagrangeKernel<K>
        + GKRPolyKernel<NumerType, K>
        + DotBackend<K, K>
        + ReduceSumBackend<K>
        + GKRPolyKernel<K, K>,
{
    type NextRoundPoly = LogupGkrPoly<K, K, TaskScope>;

    async fn fix_t_variables(
        poly: LogupGkrPoly<NumerType, K, TaskScope>,
        alpha: K,
        t: usize,
    ) -> Self::NextRoundPoly {
        assert!(t == 1);

        let mut new_point = poly.point.clone();
        let last_coord = new_point.remove_last_coordinate();
        let prod = alpha * last_coord;

        let eq_adjustment = poly.eq_adjustment * (K::one() - alpha - last_coord + prod + prod);

        let (numerator_0, numerator_1, denom_0, denom_1) = join4(
            poly.numerator_0.fix_last_variable(alpha),
            poly.numerator_1.fix_last_variable(alpha),
            poly.denom_0.fix_last_variable(alpha),
            poly.denom_1.fix_last_variable(alpha),
        )
        .await;
        LogupGkrPoly::<K, K, TaskScope>::new_with_batching_randomness_powers(
            new_point,
            numerator_0,
            numerator_1,
            denom_0,
            denom_1,
            poly.lambda,
            eq_adjustment,
            poly.batching_randomness,
            poly.batching_randomness_powers,
            poly.batching_randomness_powers_device,
        )
    }

    async fn sum_as_poly_in_last_t_variables(
        poly: &LogupGkrPoly<NumerType, K, TaskScope>,
        claim: Option<K>,
        t: usize,
    ) -> UnivariatePolynomial<K> {
        assert!(t == 1);
        logup_sum_as_poly_in_last_variable::<NumerType, K>(poly, claim).await
    }
}

#[cfg(test)]
pub mod sum_tests {

    use std::sync::Arc;

    use crate::{
        transpose::DeviceTransposeKernel, IntoDevice, MleFixLastVariableKernel,
        PartialLagrangeKernel, TaskScope,
    };
    use itertools::Itertools;
    use rand::{distributions::Standard, prelude::Distribution, Rng};
    use slop_algebra::{extension::BinomialExtensionField, AbstractField, ExtensionField, Field};
    use slop_alloc::{Buffer, CanCopyFrom, CpuBackend, IntoHost};
    use slop_baby_bear::BabyBear;
    use slop_matrix::dense::RowMajorMatrix;
    use slop_multilinear::{
        Mle, MleBaseBackend, PaddedMle, Padding, PartialLagrangeBackend, Point,
    };
    use slop_sumcheck::SumcheckPolyBackend;
    use slop_tensor::{DotBackend, ReduceSumBackend};

    use super::{GKRPolyKernel, LogupGkrPoly};

    async fn test_gkr_poly<F, EF>(
        p_0_padded: PaddedMle<F>,
        p_1_padded: PaddedMle<F>,
        q_0_padded: PaddedMle<EF>,
        q_1_padded: PaddedMle<EF>,
        lambda: EF,
        batching_randomness: EF,
        point: Point<EF>,
    ) where
        F: Field,
        EF: ExtensionField<F>,
        Standard: Distribution<F> + Distribution<EF>,
        TaskScope: GKRPolyKernel<F, EF>
            + DeviceTransposeKernel<F>
            + DeviceTransposeKernel<EF>
            + PartialLagrangeBackend<F>
            + PartialLagrangeBackend<EF>
            + DotBackend<EF, EF>
            + ReduceSumBackend<EF>
            + SumcheckPolyBackend<LogupGkrPoly<EF, EF, TaskScope>, EF>
            + MleFixLastVariableKernel<F, EF>
            + MleBaseBackend<EF>
            + MleFixLastVariableKernel<EF, EF>
            + PartialLagrangeKernel<EF>
            + ReduceSumBackend<EF>
            + DeviceTransposeKernel<EF>
            + ReduceSumBackend<F>
            + DeviceTransposeKernel<F>
            + DotBackend<EF, EF>
            + DotBackend<F, F>
            + DotBackend<F, EF>
            + MleFixLastVariableKernel<F, F>
            + CanCopyFrom<Buffer<EF>, CpuBackend, Output = Buffer<EF, TaskScope>>
            + PartialLagrangeKernel<F>,

        // Tensor<EF>: IntoDevice<Output = Tensor<EF, TaskScope>>
        // + CopyToBackend<TaskScope, CpuBackend, Output = Tensor<EF, TaskScope>>,
        Mle<F, TaskScope>: IntoHost,
        Mle<EF, TaskScope>: IntoHost,
        Mle<F>: IntoDevice<Output = Mle<F, TaskScope>>,
        Mle<EF>: IntoDevice<Output = Mle<EF, TaskScope>>,
    {
        let cpu_logup_poly = LogupGkrPoly::new(
            point.clone(),
            (p_0_padded.clone(), p_1_padded.clone()),
            (q_0_padded.clone(), q_1_padded.clone()),
            lambda,
            EF::one(),
            batching_randomness,
        )
        .await;

        let cpu_sum = cpu_logup_poly.sum_as_poly_in_last_variable(Some(EF::one())).await;

        let cuda_sums = crate::task()
            .await
            .unwrap()
            .run(|t| async move {
                let p_0_padded_device = t.into_device(p_0_padded).await.unwrap();
                let p_1_padded_device = t.into_device(p_1_padded).await.unwrap();
                let q_0_padded_device = t.into_device(q_0_padded).await.unwrap();
                let q_1_padded_device = t.into_device(q_1_padded).await.unwrap();
                let cuda_logup_poly = LogupGkrPoly::<_, _, TaskScope>::new(
                    point,
                    (p_0_padded_device, p_1_padded_device),
                    (q_0_padded_device, q_1_padded_device),
                    lambda,
                    EF::one(),
                    batching_randomness,
                )
                .await;
                super::logup_sum_as_poly_in_last_variable(&cuda_logup_poly, Some(EF::one())).await
            })
            .await
            .await
            .unwrap();

        for (cpu, cuda) in cpu_sum.coefficients.iter().zip_eq(cuda_sums.coefficients.iter()) {
            println!("CPU Coefficient: {}, CUDA Coefficient: {}", cpu, cuda);
            assert_eq!(cpu, cuda);
        }

        // let cuda_numerator_0: PaddedMle<EF> =
        //     cuda_logup_poly.clone().numerator_0.into_host().await.unwrap();

        // let cuda_numerator_1 = cuda_logup_poly.clone().numerator_1.into_host().await.unwrap();

        // let cuda_denom_0 = cuda_logup_poly.denom_0.into_host().await.unwrap();
        // let cuda_denom_1 = cuda_logup_poly.denom_1.into_host().await.unwrap();

        // for (cpu_val, cuda_val) in cpu_logup_poly
        //     .numerator_0
        //     .inner()
        //     .as_ref()
        //     .map(|x| x.guts().as_slice())
        //     .unwrap_or(&[])
        //     .iter()
        //     .zip_eq(
        //         cuda_numerator_0.inner().as_ref().map(|x| x.guts().as_slice()).unwrap_or(&[]).iter(),
        //     )
        // {
        //     println!("CPU Val: {}, CUDA Val: {}", cpu_val, cuda_val);
        //     assert_eq!(cpu_val, cuda_val);
        // }
    }

    #[tokio::test]
    async fn test_gkr_poly_baby_bear() {
        type F = BabyBear;
        type EF = BinomialExtensionField<F, 4>;
        let mut rng = rand::thread_rng();

        let height = (1 << 10) + 7;
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

        let lambda = rng.gen::<EF>();

        let p_padding_val: Padding<F, CpuBackend> =
            Padding::Constant((F::zero(), width, CpuBackend));
        let q_padding_val: Padding<EF, CpuBackend> =
            Padding::Constant((EF::one(), width, CpuBackend));

        let p_0_padded = PaddedMle::new(Some(Arc::new(p_0.clone())), 11, p_padding_val.clone());
        let p_1_padded = PaddedMle::new(Some(Arc::new(p_1.clone())), 11, p_padding_val.clone());
        let q_0_padded = PaddedMle::new(Some(Arc::new(q_0.clone())), 11, q_padding_val.clone());
        let q_1_padded = PaddedMle::new(Some(Arc::new(q_1.clone())), 11, q_padding_val.clone());

        let point = (0..11).map(|_| rng.gen::<EF>()).collect::<Point<_>>();

        let batching_randomness = rng.gen::<EF>();
        test_gkr_poly::<BabyBear, BinomialExtensionField<BabyBear, 4>>(
            p_0_padded,
            p_1_padded,
            q_0_padded,
            q_1_padded,
            lambda,
            batching_randomness,
            point.clone(),
        )
        .await;

        let p_0_padded = PaddedMle::new(None, 11, p_padding_val.clone());
        let p_1_padded = PaddedMle::new(None, 11, p_padding_val.clone());
        let q_0_padded = PaddedMle::new(None, 11, q_padding_val.clone());
        let q_1_padded = PaddedMle::new(None, 11, q_padding_val.clone());

        test_gkr_poly::<BabyBear, BinomialExtensionField<BabyBear, 4>>(
            p_0_padded,
            p_1_padded,
            q_0_padded,
            q_1_padded,
            lambda,
            batching_randomness,
            point,
        )
        .await;
    }

    #[tokio::test]
    async fn test_gkr_poly_baby_bear_extension() {
        type F = BinomialExtensionField<BabyBear, 4>;
        type EF = BinomialExtensionField<BabyBear, 4>;
        let mut rng = rand::thread_rng();

        let height = (1 << 10) + 7;
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

        let lambda = rng.gen::<EF>();

        let p_padding_val: Padding<F, CpuBackend> =
            Padding::Constant((F::zero(), width, CpuBackend));
        let q_padding_val: Padding<EF, CpuBackend> =
            Padding::Constant((EF::one(), width, CpuBackend));

        let p_0_padded = PaddedMle::new(Some(Arc::new(p_0.clone())), 11, p_padding_val.clone());
        let p_1_padded = PaddedMle::new(Some(Arc::new(p_1.clone())), 11, p_padding_val.clone());
        let q_0_padded = PaddedMle::new(Some(Arc::new(q_0.clone())), 11, q_padding_val.clone());
        let q_1_padded = PaddedMle::new(Some(Arc::new(q_1.clone())), 11, q_padding_val.clone());

        let point = (0..11).map(|_| rng.gen::<EF>()).collect::<Point<_>>();

        let batching_randomness = rng.gen::<EF>();
        test_gkr_poly::<F, EF>(
            p_0_padded,
            p_1_padded,
            q_0_padded,
            q_1_padded,
            lambda,
            batching_randomness,
            point.clone(),
        )
        .await;

        let p_0_padded = PaddedMle::new(None, 11, p_padding_val.clone());
        let p_1_padded = PaddedMle::new(None, 11, p_padding_val.clone());
        let q_0_padded = PaddedMle::new(None, 11, q_padding_val.clone());
        let q_1_padded = PaddedMle::new(None, 11, q_padding_val.clone());

        test_gkr_poly(
            p_0_padded,
            p_1_padded,
            q_0_padded,
            q_1_padded,
            lambda,
            batching_randomness,
            point,
        )
        .await;
    }
}
