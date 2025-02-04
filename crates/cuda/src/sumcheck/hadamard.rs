use std::borrow::Cow;

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
use slop_alloc::HasBackend;
use slop_baby_bear::BabyBear;
use slop_multilinear::MleFixLastVariableBackend;
use slop_sumcheck::{
    ComponentPolyEvalBackend, HadamardProduct, SumCheckPolyFirstRoundBackend, SumcheckPolyBackend,
};
use slop_tensor::{ReduceSumBackend, Tensor};

use crate::{args, TaskScope};

impl<'a, F, EF> ComponentPolyEvalBackend<EF, HadamardProduct<'a, F, EF, TaskScope>> for TaskScope
where
    F: Field,
    EF: ExtensionField<F>,
{
    fn get_component_poly_evals(poly: &HadamardProduct<'a, F, EF, TaskScope>) -> Vec<EF> {
        let backend = poly.backend();
        let base_eval: EF = (poly.base.guts()[[0, 0]]).copy_into_host(backend).into();
        let ext_eval: EF = (poly.ext.guts()[[0, 0]]).copy_into_host(backend);
        vec![ext_eval, base_eval]
    }
}

/// # Safety
pub unsafe trait HadamardUnivariatePolyEvalKernel<F, EF> {
    fn hadamard_sum_as_poly_kernel() -> KernelPtr;
}

fn sum_as_poly_in_last_variable<F, EF>(
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
    let num_variables = poly.base.num_variables();
    let num_polys = poly.base.num_polynomials();
    assert_eq!(num_polys, 1);
    let scope = poly.backend();
    // let mut univariate_evals =
    //     Tensor::<EF, TaskScope>::with_sizes_in([2, 1 << (num_variables - 1)], scope.clone());

    const BLOCK_SIZE: usize = 1024;
    const STRIDE: usize = 32;

    let grid_dim = (1usize << (num_variables - 1)).div_ceil(BLOCK_SIZE).div_ceil(STRIDE);
    let mut univariate_evals = Tensor::<EF, TaskScope>::with_sizes_in([2, grid_dim], scope.clone());
    let num_tiles = BLOCK_SIZE.checked_div(32).unwrap_or(1);
    let shared_mem = num_tiles * std::mem::size_of::<EF>();
    let args = args!(
        univariate_evals.as_mut_ptr(),
        poly.base.as_ref().guts().as_ptr(),
        poly.ext.as_ref().guts().as_ptr(),
        num_variables - 1
    );
    unsafe {
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

    let univariate_evals = univariate_evals.sum(1);
    let host_evals = univariate_evals.into_buffer().copy_into_host_vec();

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

impl<'a, F> SumcheckPolyBackend<F, HadamardProduct<'a, F, F, TaskScope>> for TaskScope
where
    F: Field,
    TaskScope: MleFixLastVariableBackend<F, F>
        + HadamardUnivariatePolyEvalKernel<F, F>
        + ReduceSumBackend<F>,
{
    fn fix_last_variable(
        poly: HadamardProduct<'a, F, F, Self>,
        alpha: F,
    ) -> HadamardProduct<'static, F, F, TaskScope> {
        let base = poly.base.as_ref().fix_last_variable(alpha);
        let ext = poly.ext.as_ref().fix_last_variable(alpha);
        HadamardProduct { base: Cow::Owned(base), ext: Cow::Owned(ext) }
    }

    #[inline]
    fn sum_as_poly_in_last_variable(
        poly: &HadamardProduct<'a, F, F, Self>,
        claim: Option<F>,
    ) -> UnivariatePolynomial<F> {
        sum_as_poly_in_last_variable(poly, claim.unwrap())
    }
}

impl<F, EF> SumCheckPolyFirstRoundBackend<EF, HadamardProduct<'static, F, EF, TaskScope>>
    for TaskScope
where
    F: Field,
    EF: ExtensionField<F>,
    TaskScope: MleFixLastVariableBackend<F, EF>
        + MleFixLastVariableBackend<EF, EF>
        + HadamardUnivariatePolyEvalKernel<F, EF>
        + HadamardUnivariatePolyEvalKernel<EF, EF>
        + ReduceSumBackend<EF>,
{
    fn fix_t_variables(
        poly: HadamardProduct<'static, F, EF, TaskScope>,
        alpha: EF,
        _t: usize,
    ) -> impl slop_sumcheck::SumcheckPoly<EF> {
        let base = poly.base.as_ref().fix_last_variable(alpha);
        let ext = poly.ext.as_ref().fix_last_variable(alpha);
        HadamardProduct { base: Cow::Owned(base), ext: Cow::Owned(ext) }
    }

    fn sum_as_poly_in_last_t_variables(
        poly: &HadamardProduct<'static, F, EF, TaskScope>,
        claim: Option<EF>,
        t: usize,
    ) -> UnivariatePolynomial<EF> {
        assert_eq!(t, 1);
        sum_as_poly_in_last_variable(poly, claim.unwrap())
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
    use slop_baby_bear::{BabyBear, Challenger, DiffusionMatrixBabyBear, Perm};
    use slop_challenger::CanSample;
    use slop_multilinear::Mle;
    use slop_poseidon2::Poseidon2ExternalMatrixGeneral;
    use slop_sumcheck::{partially_verify_sumcheck_proof, reduce_sumcheck_to_evaluation};

    use super::*;

    #[tokio::test]
    async fn test_hadamard_product_sumcheck() {
        let mut rng = thread_rng();

        type F = BabyBear;
        type EF = BinomialExtensionField<BabyBear, 4>;

        let perm = Perm::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabyBear,
            &mut thread_rng(),
        );

        for num_variables in [19, 21, 24, 27, 29] {
            let base = Mle::<F>::rand(&mut rng, 1, num_variables);
            let ext = Mle::<EF>::rand(&mut rng, 1, num_variables);

            let perm = &perm;
            crate::task()
                .await
                .unwrap()
                .run(|t| async move {
                    let base_guts_device = t
                        .into_device(base.into_guts())
                        .await
                        .unwrap()
                        .reshape([1, 1 << num_variables]);
                    let ext_guts_device = t
                        .into_device(ext.into_guts())
                        .await
                        .unwrap()
                        .reshape([1, 1 << num_variables]);
                    let base_device = Mle::new(base_guts_device);
                    let ext_device = Mle::new(ext_guts_device);

                    let product_device = HadamardProduct::<F, EF, TaskScope> {
                        base: Cow::Owned(base_device),
                        ext: Cow::Owned(ext_device),
                    };

                    t.synchronize().await.unwrap();
                    let time = tokio::time::Instant::now();
                    let claim = product_device
                        .base
                        .as_ref()
                        .guts()
                        .dot(product_device.ext.as_ref().guts(), 1);
                    let claim = claim[[0]].copy_into_host(&t);
                    t.synchronize().await.unwrap();
                    println!("Time for getting claim for {num_variables}: {:?}", time.elapsed());

                    let mut challenger = Challenger::new(perm.clone());
                    let lambda: EF = challenger.sample();
                    t.synchronize().await.unwrap();
                    let time = tokio::time::Instant::now();
                    let (proof, _eval_claims) = reduce_sumcheck_to_evaluation::<F, EF, _>(
                        vec![product_device],
                        &mut challenger,
                        vec![claim],
                        1,
                        lambda,
                    );
                    t.synchronize().await.unwrap();
                    println!(
                        "Time for partial sumcheck proof for {num_variables}: {:?}",
                        time.elapsed()
                    );

                    // let point = &proof.point_and_eval.0;
                    // let [exp_eval_ext, exp_eval_base] = eval_claims.pop().unwrap().try_into().unwrap();

                    // let eval_ext = ext.eval_at(point).to_vec().pop().unwrap();
                    // let eval_base = base.eval_at(point).to_vec().pop().unwrap();

                    // assert_eq!(eval_ext, exp_eval_ext);
                    // assert_eq!(eval_base, exp_eval_base);

                    let mut challenger = Challenger::new(perm.clone());
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
