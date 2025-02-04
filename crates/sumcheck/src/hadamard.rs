use rayon::prelude::*;
use std::borrow::Cow;

use slop_algebra::{
    interpolate_univariate_polynomial, AbstractExtensionField, AbstractField, ExtensionField,
    Field, UnivariatePolynomial,
};
use slop_alloc::{Backend, CpuBackend, HasBackend};
use slop_multilinear::{Mle, MleBaseBackend};

use crate::{
    backend::{ComponentPolyEvalBackend, SumCheckPolyFirstRoundBackend, SumcheckPolyBackend},
    SumcheckPoly, SumcheckPolyBase, SumcheckPolyFirstRound,
};

pub struct HadamardProduct<'a, F: Clone, EF: Clone = F, A: Backend = CpuBackend> {
    pub base: Cow<'a, Mle<F, A>>,
    pub ext: Cow<'a, Mle<EF, A>>,
}

impl<'a, F, EF, A> HasBackend for HadamardProduct<'a, F, EF, A>
where
    A: Backend,
    F: AbstractField,
    EF: AbstractExtensionField<F>,
{
    type Backend = A;
    #[inline]
    fn backend(&self) -> &Self::Backend {
        self.base.backend()
    }
}

impl<'a, F, EF, A> SumcheckPolyBase for HadamardProduct<'a, F, EF, A>
where
    F: AbstractField,
    EF: AbstractExtensionField<F>,
    A: MleBaseBackend<F> + MleBaseBackend<EF>,
{
    #[inline]
    fn n_variables(&self) -> u32 {
        self.base.num_variables()
    }
}

impl<'a, F, EF> ComponentPolyEvalBackend<EF, HadamardProduct<'a, F, EF, CpuBackend>> for CpuBackend
where
    F: AbstractField,
    EF: AbstractExtensionField<F>,
{
    fn get_component_poly_evals(poly: &HadamardProduct<'a, F, EF, CpuBackend>) -> Vec<EF> {
        let base_eval: EF = (*poly.base.guts()[[0, 0]]).clone().into();
        let ext_eval: EF = (*poly.ext.guts()[[0, 0]]).clone();
        vec![ext_eval, base_eval]
    }
}

impl<F> SumcheckPolyBackend<F, HadamardProduct<'static, F>> for CpuBackend
where
    F: Field,
{
    fn fix_last_variable(
        poly: HadamardProduct<'static, F, F, CpuBackend>,
        alpha: F,
    ) -> HadamardProduct<'static, F, F, CpuBackend> {
        let base = poly.base.as_ref().fix_last_variable(alpha);
        let ext = poly.ext.as_ref().fix_last_variable(alpha);
        HadamardProduct { base: Cow::Owned(base), ext: Cow::Owned(ext) }
    }

    #[inline]
    fn sum_as_poly_in_last_variable(
        poly: &HadamardProduct<'static, F, F, CpuBackend>,
        claim: Option<F>,
    ) -> UnivariatePolynomial<F> {
        poly.sum_as_poly_in_last_t_variables(claim, 1)
    }
}

impl<'a, F, EF> SumCheckPolyFirstRoundBackend<EF, HadamardProduct<'a, F, EF, CpuBackend>>
    for CpuBackend
where
    F: Field,
    EF: ExtensionField<F>,
{
    fn fix_t_variables(
        poly: HadamardProduct<'a, F, EF, CpuBackend>,
        alpha: EF,
        t: usize,
    ) -> impl SumcheckPoly<EF> {
        assert_eq!(t, 1);
        let base = poly.base.as_ref().fix_last_variable(alpha);
        let ext = poly.ext.as_ref().fix_last_variable(alpha);
        HadamardProduct { base: Cow::Owned(base), ext: Cow::Owned(ext) }
    }

    fn sum_as_poly_in_last_t_variables(
        poly: &HadamardProduct<'a, F, EF, CpuBackend>,
        claim: Option<EF>,
        t: usize,
    ) -> UnivariatePolynomial<EF> {
        assert_eq!(t, 1);
        // The sumcheck polynomial is a multi-quadratic polynomial, so three evaluations are needed.
        let eval_0 = poly
            .ext
            .guts()
            .as_slice()
            .par_iter()
            .step_by(2)
            .zip(poly.base.guts().as_slice().par_iter().step_by(2))
            .map(|(x, y)| *x * *y)
            .sum();

        let eval_1 = claim.map(|x| x - eval_0).unwrap_or(
            poly.ext
                .guts()
                .as_slice()
                .par_iter()
                .skip(1)
                .step_by(2)
                .zip(poly.base.guts().as_slice().par_iter().skip(1).step_by(2))
                .map(|(x, y)| *x * *y)
                .sum(),
        );

        let eval_half: EF = poly
            .ext
            .guts()
            .as_slice()
            .par_iter()
            .step_by(2)
            .zip(poly.ext.guts().as_slice().par_iter().skip(1).step_by(2))
            .zip(poly.base.guts().as_slice().par_iter().step_by(2))
            .zip(poly.base.guts().as_slice().par_iter().skip(1).step_by(2))
            .map(|(((je_0, je_1), mle_0), mle_1)| (*je_0 + *je_1) * (*mle_0 + *mle_1))
            .sum();

        interpolate_univariate_polynomial(
            &[
                EF::from_canonical_u16(0),
                EF::from_canonical_u16(1),
                EF::from_canonical_u16(2).inverse(),
            ],
            &[eval_0, eval_1, eval_half * EF::from_canonical_u16(4).inverse()],
        )
    }
}

impl<F, EF, A> From<(Mle<F, A>, Mle<EF, A>)> for HadamardProduct<'static, F, EF, A>
where
    A: Backend,
    F: Field,
    EF: ExtensionField<F>,
{
    fn from(value: (Mle<F, A>, Mle<EF, A>)) -> Self {
        let (base, ext) = value;
        Self { base: Cow::Owned(base), ext: Cow::Owned(ext) }
    }
}

impl<'a, F, EF, A> From<(&'a Mle<F, A>, &'a Mle<EF, A>)> for HadamardProduct<'a, F, EF, A>
where
    A: Backend,
    F: AbstractField,
    EF: AbstractExtensionField<F>,
{
    fn from(value: (&'a Mle<F, A>, &'a Mle<EF, A>)) -> Self {
        let (base, ext) = value;
        Self { base: Cow::Borrowed(base), ext: Cow::Borrowed(ext) }
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;
    use slop_algebra::extension::BinomialExtensionField;
    use slop_baby_bear::{BabyBear, Challenger, DiffusionMatrixBabyBear, Perm};
    use slop_challenger::CanSample;
    use slop_poseidon2::Poseidon2ExternalMatrixGeneral;

    use crate::{partially_verify_sumcheck_proof, reduce_sumcheck_to_evaluation};

    use super::*;

    #[test]
    fn test_hadamard_product_sumcheck() {
        let mut rng = thread_rng();

        type F = BabyBear;
        type EF = BinomialExtensionField<BabyBear, 4>;

        let base = Mle::<F>::rand(&mut rng, 1, 10);
        let ext = Mle::<EF>::rand(&mut rng, 1, 10);

        let product = HadamardProduct::from((&base, &ext));

        let perm = Perm::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabyBear,
            &mut thread_rng(),
        );
        let mut challenger = Challenger::new(perm.clone());

        let claim: EF = product
            .ext
            .guts()
            .as_slice()
            .iter()
            .zip(product.base.guts().as_slice().iter())
            .map(|(x, y)| EF::from(*x) * EF::from(*y))
            .sum();

        let lambda: EF = challenger.sample();

        let (proof, mut eval_claims) = reduce_sumcheck_to_evaluation::<F, EF, _>(
            vec![product],
            &mut challenger,
            vec![claim],
            1,
            lambda,
        );

        let point = &proof.point_and_eval.0;
        let [exp_eval_ext, exp_eval_base] = eval_claims.pop().unwrap().try_into().unwrap();

        let eval_ext = ext.eval_at(point).to_vec().pop().unwrap();
        let eval_base = base.eval_at(point).to_vec().pop().unwrap();

        assert_eq!(eval_ext, exp_eval_ext);
        assert_eq!(eval_base, exp_eval_base);

        let mut challenger = Challenger::new(perm.clone());
        let _lambda: EF = challenger.sample();
        assert!(partially_verify_sumcheck_proof::<F, EF, _>(&proof, &mut challenger).is_ok());
        assert_eq!(proof.univariate_polys.len(), 10);
    }
}
