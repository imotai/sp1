use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use serde::{Deserialize, Serialize};
use slop_algebra::{Field, UnivariatePolynomial};
use slop_multilinear::{Mle, Point};

/// The fix_first_variable function applied to a sumcheck's first round's polynomial .
pub trait SumcheckPolyFirstRound<K>: SumcheckPolyBase<K> {
    fn fix_t_variables(self, alpha: K, t: usize) -> impl SumcheckPoly<K>;

    fn sum_as_poly_in_last_t_variables(
        &self,
        claim: Option<K>,
        t: usize,
    ) -> UnivariatePolynomial<K>;
}

/// The fix_first_variable function applied to a sumcheck's post first rounds' polynomial.
pub trait SumcheckPoly<K>: SumcheckPolyBase<K> {
    fn fix_last_variable(self, alpha: K) -> Self;

    fn sum_as_poly_in_last_variable(&self, claim: Option<K>) -> UnivariatePolynomial<K>;
}

impl<
        K: Field,
        S: Field
            + From<K>
            + Add<K, Output = S>
            + AddAssign<K>
            + Sub<K, Output = S>
            + SubAssign<K>
            + Mul<K, Output = S>
            + MulAssign<K>
            + Copy
            + Send
            + Sync
            + Default,
    > SumcheckPolyFirstRound<S> for Mle<K>
{
    fn fix_t_variables(self, alpha: S, t: usize) -> impl SumcheckPoly<S> {
        assert!(t == 1);
        self.sc_fix_last_variable(alpha)
    }

    fn sum_as_poly_in_last_t_variables(
        &self,
        _claim: Option<S>,
        t: usize,
    ) -> UnivariatePolynomial<S> {
        assert!(t == 1);
        self.sc_sum_as_poly_in_last_variable()
    }
}

impl<K: Field> SumcheckPoly<K> for Mle<K> {
    fn fix_last_variable(self, alpha: K) -> Self {
        self.sc_fix_last_variable(alpha)
    }

    fn sum_as_poly_in_last_variable(&self, _claim: Option<K>) -> UnivariatePolynomial<K> {
        self.sc_sum_as_poly_in_last_variable()
    }
}

impl<
        K: Field,
        S: Field
            + From<K>
            + Add<K, Output = S>
            + AddAssign<K>
            + Sub<K, Output = S>
            + SubAssign<K>
            + Mul<K, Output = S>
            + MulAssign<K>
            + Copy
            + Send
            + Sync
            + Default,
    > SumcheckPolyBase<S> for Mle<K>
{
    fn n_variables(&self) -> u32 {
        self.num_variables()
    }

    fn get_component_poly_evals(&self) -> Vec<S> {
        assert!(self.num_variables() == 0);
        vec![self.guts[0].into()]
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PartialSumcheckProof<K> {
    pub univariate_polys: Vec<UnivariatePolynomial<K>>,
    pub claimed_sum: K,
    pub point_and_eval: (Point<K>, K),
}

#[cfg(test)]
mod test {
    use slop_algebra::{extension::BinomialExtensionField, AbstractField, UnivariatePolynomial};
    use slop_baby_bear::BabyBear;
    use slop_multilinear::Mle;
    use slop_tensor::tensor;

    use crate::SumcheckPoly;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_mle_edge_case() {
        let mle = Mle::new(tensor![[F::two()]]);
        assert_eq!(
            <Mle::<F> as SumcheckPoly<F>>::sum_as_poly_in_last_variable(&mle, None),
            UnivariatePolynomial::new(vec![F::two()])
        );
        assert_eq!(mle.num_variables(), 0);
    }

    #[test]
    fn test_simple_mle_properties() {
        let mle = Mle::new(tensor![[EF::zero(), EF::one(), EF::zero(), EF::zero()]].into());
        assert_eq!(
            mle.sum_as_poly_in_last_variable(None),
            UnivariatePolynomial::new(vec![EF::zero(), EF::one()])
        );
        assert_eq!(mle.num_variables(), 2);
        assert_eq!(
            mle.fix_last_variable(EF::from_canonical_u16(0xBABE)),
            Mle::new(tensor![[EF::from_canonical_u32(0xBABE), EF::zero()]].into())
        );
    }
}
