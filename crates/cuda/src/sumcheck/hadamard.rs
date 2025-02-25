use csl_sys::{
    runtime::KernelPtr,
    sumcheck::{
        hadamard_univariate_poly_eval_baby_bear_base_ext_kernel,
        hadamard_univariate_poly_eval_baby_bear_ext_kernel,
    },
};
use slop_algebra::{
    extension::BinomialExtensionField, interpolate_univariate_polynomial, ExtensionField, Field,
    UnivariatePolynomial,
};
use slop_alloc::IntoHost;
use slop_baby_bear::BabyBear;
use slop_jagged::HadamardProduct;
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
    assert!(poly.base.num_components() == 1);
    assert!(poly.ext.num_components() == 1);

    let poly_base = poly.base.first_component_mle();
    let poly_ext = poly.ext.first_component_mle();

    let num_variables = poly_base.num_variables();
    let num_polys = poly_base.num_polynomials();
    assert_eq!(num_polys, 1);
    let scope = poly_base.backend();

    const BLOCK_SIZE: usize = 512;
    const STRIDE: usize = 128;

    let grid_dim = (1usize << (num_variables - 1)).div_ceil(BLOCK_SIZE).div_ceil(STRIDE);
    let mut univariate_evals = Tensor::<EF, TaskScope>::with_sizes_in([2, grid_dim], scope.clone());
    let num_tiles = BLOCK_SIZE.checked_div(32).unwrap_or(1);
    let shared_mem = num_tiles * std::mem::size_of::<EF>();
    unsafe {
        let args = args!(
            univariate_evals.as_mut_ptr(),
            poly_base.guts().as_ptr(),
            poly_ext.guts().as_ptr(),
            num_variables - 1
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

    let univariate_evals = univariate_evals.sum(1).await;
    let host_evals = unsafe { univariate_evals.into_buffer().copy_into_host_vec() };

    let [eval_zero, eval_half] = host_evals.try_into().unwrap();
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
    async fn fix_t_variables(
        poly: HadamardProduct<F, EF, TaskScope>,
        alpha: EF,
        _t: usize,
    ) -> impl slop_sumcheck::SumcheckPoly<EF> {
        let base = poly.base.fix_last_variable(alpha).await;
        let ext = poly.ext.fix_last_variable(alpha).await;
        // let HadamardProduct { base, ext } = poly;
        // let base = base.fix_last_variable(alpha).await;
        // let ext = ext.fix_last_variable_in_place(alpha).await;
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

unsafe impl HadamardUnivariatePolyEvalKernel<BabyBear, BinomialExtensionField<BabyBear, 4>>
    for TaskScope
{
    fn hadamard_sum_as_poly_kernel() -> KernelPtr {
        unsafe { hadamard_univariate_poly_eval_baby_bear_base_ext_kernel() }
    }
}

unsafe impl
    HadamardUnivariatePolyEvalKernel<
        BinomialExtensionField<BabyBear, 4>,
        BinomialExtensionField<BabyBear, 4>,
    > for TaskScope
{
    #[inline]
    fn hadamard_sum_as_poly_kernel() -> KernelPtr {
        unsafe { hadamard_univariate_poly_eval_baby_bear_ext_kernel() }
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;
    use slop_algebra::extension::BinomialExtensionField;
    use slop_alloc::ToHost;
    use slop_baby_bear::BabyBear;
    use slop_basefold::{BasefoldVerifier, Poseidon2BabyBear16BasefoldConfig};
    use slop_challenger::CanSample;
    use slop_jagged::LongMle;
    use slop_multilinear::Mle;
    use slop_sumcheck::{partially_verify_sumcheck_proof, reduce_sumcheck_to_evaluation};

    use super::*;

    #[tokio::test]
    async fn test_hadamard_product_sumcheck() {
        let mut rng = thread_rng();

        type C = Poseidon2BabyBear16BasefoldConfig;

        type F = BabyBear;
        type EF = BinomialExtensionField<BabyBear, 4>;

        for num_variables in [19, 21, 24, 27, 28] {
            let base = Mle::<F>::rand(&mut rng, 1, num_variables);
            let ext = Mle::<EF>::rand(&mut rng, 1, num_variables);

            crate::task()
                .await
                .unwrap()
                .run(|t| async move {
                    let base = t.into_device(base).await.unwrap();
                    let ext = t.into_device(ext).await.unwrap();
                    let base = LongMle::from_components(vec![base], num_variables);
                    let ext = LongMle::from_components(vec![ext], num_variables);

                    let product = HadamardProduct { base, ext };

                    let verifier = BasefoldVerifier::<C>::new(1);

                    let mut challenger = verifier.challenger();

                    t.synchronize().await.unwrap();
                    let time = tokio::time::Instant::now();
                    let claim = product
                        .base
                        .first_component_mle()
                        .guts()
                        .dot(product.ext.first_component_mle().guts(), 1)
                        .await;

                    let claim = claim.into_host().await.unwrap().as_slice()[0];
                    t.synchronize().await.unwrap();
                    println!("time for getting claim for {num_variables}: {:?}", time.elapsed());

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
                    let [exp_eval_base, exp_eval_ext] =
                        eval_claims.pop().unwrap().try_into().unwrap();

                    let point = t.to_device(point).await.unwrap();

                    let eval_ext = product
                        .ext
                        .first_component_mle()
                        .eval_at(&point)
                        .await
                        .to_host()
                        .await
                        .unwrap()[0];
                    let eval_base = product
                        .base
                        .first_component_mle()
                        .eval_at(&point)
                        .await
                        .to_host()
                        .await
                        .unwrap()[0];

                    assert_eq!(eval_ext, exp_eval_ext);
                    assert_eq!(eval_base, exp_eval_base);

                    // Check that the final claimed evaluation is the product of the two evaluations
                    let claimed_eval = proof.point_and_eval.1;
                    assert_eq!(claimed_eval, exp_eval_ext * exp_eval_base);

                    let mut challenger = verifier.challenger();
                    let _lambda: EF = challenger.sample();
                    assert!(partially_verify_sumcheck_proof::<F, EF, _>(&proof, &mut challenger)
                        .is_ok());
                    assert_eq!(proof.univariate_polys.len(), num_variables as usize);
                })
                .await
                .await
                .unwrap();
        }
    }
}
