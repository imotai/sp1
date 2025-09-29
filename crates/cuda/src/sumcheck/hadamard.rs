use csl_sys::{
    runtime::{Dim3, KernelPtr},
    sumcheck::{
        hadamard_univariate_poly_eval_koala_bear_base_ext_kernel,
        hadamard_univariate_poly_eval_koala_bear_ext_kernel,
    },
};
use slop_algebra::{
    extension::BinomialExtensionField, interpolate_univariate_polynomial, ExtensionField, Field,
    UnivariatePolynomial,
};
use slop_alloc::{HasBackend, IntoHost};
use slop_jagged::HadamardProduct;
use slop_koala_bear::KoalaBear;
use slop_multilinear::{MleFixLastVariableBackend, MleFixLastVariableInPlaceBackend};
use slop_sumcheck::{ComponentPolyEvalBackend, SumCheckPolyFirstRoundBackend, SumcheckPolyBackend};
use slop_tensor::{ReduceSumBackend, Tensor};

use crate::{args, SmallTensor, TaskScope};

impl<F, EF> ComponentPolyEvalBackend<HadamardProduct<F, EF, TaskScope>, EF> for TaskScope
where
    F: Field,
    EF: ExtensionField<F>,
{
    async fn get_component_poly_evals(poly: &HadamardProduct<F, EF, TaskScope>) -> Vec<EF> {
        let base_eval = unsafe { SmallTensor::new(poly.base.first_component_mle().guts()) };
        let base_eval = EF::from_base(base_eval.into_host().await.unwrap().as_slice()[0]);
        let ext_eval = unsafe { SmallTensor::new(poly.ext.first_component_mle().guts()) };
        let ext_eval = ext_eval.into_host().await.unwrap().as_slice()[0];
        vec![base_eval, ext_eval]
    }
}

/// # Safety
pub unsafe trait HadamardUnivariatePolyEvalKernel<F, EF> {
    fn hadamard_sum_as_poly_kernel() -> KernelPtr;
}

async fn sum_as_poly_in_last_variable<F, EF>(
    poly: &HadamardProduct<F, EF, TaskScope>,
    claim: EF,
) -> UnivariatePolynomial<EF>
where
    F: Field,
    EF: ExtensionField<F>,
    TaskScope: MleFixLastVariableBackend<F, EF>
        + HadamardUnivariatePolyEvalKernel<F, EF>
        + ReduceSumBackend<EF>,
{
    let mut eval_zero = EF::zero();
    let mut eval_half = EF::zero();

    for (poly_base, poly_ext) in poly.base.components().iter().zip(poly.ext.components().iter()) {
        let num_variables = poly_base.num_variables();
        let num_polys = poly_base.num_polynomials();
        let scope = poly_base.backend();

        debug_assert!(num_variables >= 1);
        const BLOCK_SIZE: usize = 256;
        const STRIDE: usize = 8;

        let output_height = 1usize << (num_variables - 1);
        let grid_dim: Dim3 =
            (output_height.div_ceil(BLOCK_SIZE).div_ceil(STRIDE), num_polys, 1).into();
        let mut univariate_evals = Tensor::<EF, TaskScope>::with_sizes_in(
            [2, grid_dim.y as usize, grid_dim.x as usize],
            scope.clone(),
        );
        // Tensor::<EF, TaskScope>::with_sizes_in([2, grid_dim], scope.clone());
        let num_tiles = BLOCK_SIZE.checked_div(32).unwrap_or(1);
        let shared_mem = num_tiles * std::mem::size_of::<EF>();
        let num_variables_minus_one: usize = num_variables as usize - 1;
        unsafe {
            let args = args!(
                univariate_evals.as_mut_ptr(),
                poly_base.guts().as_ptr(),
                poly_ext.guts().as_ptr(),
                num_variables_minus_one,
                num_polys
            );
            univariate_evals.assume_init();
            scope
                .launch_kernel(
                    TaskScope::hadamard_sum_as_poly_kernel(),
                    grid_dim,
                    BLOCK_SIZE,
                    &args,
                    shared_mem,
                )
                .unwrap();
        }

        let univariate_evals = univariate_evals.sum(2).await.sum(1).await;
        let host_evals = unsafe { univariate_evals.into_buffer().copy_into_host_vec() };

        let [component_eval_zero, component_eval_half] = host_evals.try_into().unwrap();
        eval_zero += component_eval_zero;
        eval_half += component_eval_half;
    }
    let eval_one = claim - eval_zero;

    interpolate_univariate_polynomial(
        &[
            EF::from_canonical_u16(0),
            EF::from_canonical_u16(1),
            EF::from_canonical_u16(2).inverse(),
        ],
        &[eval_zero, eval_one, eval_half * F::from_canonical_u16(4).inverse()],
    )
}

impl<F> SumcheckPolyBackend<HadamardProduct<F, F, TaskScope>, F> for TaskScope
where
    F: Field,
    TaskScope: MleFixLastVariableBackend<F, F>
        + HadamardUnivariatePolyEvalKernel<F, F>
        + ReduceSumBackend<F>,
{
    async fn fix_last_variable(
        poly: HadamardProduct<F, F, Self>,
        alpha: F,
    ) -> HadamardProduct<F, F, TaskScope> {
        let ext = poly.ext.fix_last_variable(alpha).await;
        let base = poly.base.fix_last_variable(alpha).await;
        HadamardProduct { base, ext }
    }

    #[inline]
    async fn sum_as_poly_in_last_variable(
        poly: &HadamardProduct<F, F, Self>,
        claim: Option<F>,
    ) -> UnivariatePolynomial<F> {
        sum_as_poly_in_last_variable(poly, claim.unwrap()).await
    }
}

impl<F, EF> SumCheckPolyFirstRoundBackend<HadamardProduct<F, EF, TaskScope>, EF> for TaskScope
where
    F: Field,
    EF: ExtensionField<F>,
    TaskScope: MleFixLastVariableBackend<F, EF>
        + MleFixLastVariableBackend<EF, EF>
        + MleFixLastVariableInPlaceBackend<EF>
        + HadamardUnivariatePolyEvalKernel<F, EF>
        + HadamardUnivariatePolyEvalKernel<EF, EF>
        + ReduceSumBackend<EF>,
{
    type NextRoundPoly = HadamardProduct<EF, EF, TaskScope>;
    async fn fix_t_variables(
        poly: HadamardProduct<F, EF, TaskScope>,
        alpha: EF,
        _t: usize,
    ) -> Self::NextRoundPoly {
        let base = poly.base.fix_last_variable(alpha).await;
        let ext = poly.ext.fix_last_variable(alpha).await;
        HadamardProduct { base, ext }
    }

    async fn sum_as_poly_in_last_t_variables(
        poly: &HadamardProduct<F, EF, TaskScope>,
        claim: Option<EF>,
        t: usize,
    ) -> UnivariatePolynomial<EF> {
        assert_eq!(t, 1);
        sum_as_poly_in_last_variable(poly, claim.unwrap()).await
    }
}

unsafe impl HadamardUnivariatePolyEvalKernel<KoalaBear, BinomialExtensionField<KoalaBear, 4>>
    for TaskScope
{
    fn hadamard_sum_as_poly_kernel() -> KernelPtr {
        unsafe { hadamard_univariate_poly_eval_koala_bear_base_ext_kernel() }
    }
}

unsafe impl
    HadamardUnivariatePolyEvalKernel<
        BinomialExtensionField<KoalaBear, 4>,
        BinomialExtensionField<KoalaBear, 4>,
    > for TaskScope
{
    #[inline]
    fn hadamard_sum_as_poly_kernel() -> KernelPtr {
        unsafe { hadamard_univariate_poly_eval_koala_bear_ext_kernel() }
    }
}

#[cfg(test)]
mod tests {
    use futures::prelude::*;
    use itertools::Itertools;
    use rand::thread_rng;
    use slop_algebra::extension::BinomialExtensionField;

    use slop_challenger::{CanSample, IopCtx};
    use slop_jagged::LongMle;
    use slop_koala_bear::{KoalaBear, KoalaBearDegree4Duplex};
    use slop_multilinear::Mle;
    use slop_sumcheck::{partially_verify_sumcheck_proof, reduce_sumcheck_to_evaluation};

    use super::*;

    #[tokio::test]
    async fn test_hadamard_product_sumcheck() {
        let mut rng = thread_rng();

        type F = KoalaBear;
        type EF = BinomialExtensionField<KoalaBear, 4>;

        let log_batch_size = 5;
        let batch_size = 1 << log_batch_size;

        for (num_variables, log_stacking_height) in [(19, 10), (21, 11), (24, 18), (27, 21)] {
            assert!(num_variables > log_stacking_height + log_batch_size);
            let num_components = 1 << (num_variables - log_stacking_height - log_batch_size);
            let base = (0..num_components)
                .map(|_| Mle::<F>::rand(&mut rng, batch_size, log_stacking_height))
                .collect::<Vec<_>>();
            let ext = (0..num_components)
                .map(|_| Mle::<EF>::rand(&mut rng, batch_size, log_stacking_height))
                .collect::<Vec<_>>();

            // Get the sumcheck claim, namely compute sum_i (base_i * ext_i)
            let claim = base
                .iter()
                .zip_eq(ext.iter())
                .map(|(p_b, p_e)| {
                    p_e.guts()
                        .as_slice()
                        .iter()
                        .zip_eq(p_b.guts().as_slice().iter())
                        .map(|(e_i, b_i)| *e_i * *b_i)
                        .sum::<EF>()
                })
                .sum::<EF>();

            crate::run_in_place(|t| async move {
                // let base = t.into_device(base).await.unwrap();
                // let ext = t.into_device(ext).await.unwrap();
                let base = stream::iter(base)
                    .then(|mle| async { t.into_device(mle).await.unwrap() })
                    .collect::<Vec<_>>()
                    .await;
                let ext = stream::iter(ext)
                    .then(|mle| async { t.into_device(mle).await.unwrap() })
                    .collect::<Vec<_>>()
                    .await;
                let base = LongMle::from_components(base, log_stacking_height);
                let ext = LongMle::from_components(ext, log_stacking_height);

                let product = HadamardProduct { base, ext };
                println!("product num_variables: {num_variables}");

                let mut challenger = KoalaBearDegree4Duplex::default_challenger();

                let lambda: EF = challenger.sample();

                t.synchronize().await.unwrap();
                let time = tokio::time::Instant::now();
                let (proof, mut eval_claims) = reduce_sumcheck_to_evaluation::<F, EF, _>(
                    vec![product.clone()],
                    &mut challenger,
                    vec![claim],
                    1,
                    lambda,
                )
                .await;
                t.synchronize().await.unwrap();
                println!(
                    "time for partial sumcheck proof for {num_variables}: {:?}",
                    time.elapsed()
                );

                let point = &proof.point_and_eval.0;
                let [exp_eval_base, exp_eval_ext] = eval_claims.pop().unwrap().try_into().unwrap();

                let eval_ext = product.ext.eval_at(point).await;
                let eval_base = product.base.eval_at(point).await;

                assert_eq!(eval_ext, exp_eval_ext);
                assert_eq!(eval_base, exp_eval_base);

                // Check that the final claimed evaluation is the product of the two evaluations
                let claimed_eval = proof.point_and_eval.1;
                assert_eq!(claimed_eval, exp_eval_ext * exp_eval_base);

                let mut challenger = KoalaBearDegree4Duplex::default_challenger();
                let _lambda: EF = challenger.sample();
                assert!(partially_verify_sumcheck_proof::<F, EF, _>(
                    &proof,
                    &mut challenger,
                    num_variables as usize,
                    2
                )
                .is_ok());
                assert_eq!(proof.univariate_polys.len(), num_variables as usize);
            })
            .await
            .await
            .unwrap();
        }
    }
}
