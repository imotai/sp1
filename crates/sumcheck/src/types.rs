use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use slop_algebra::{Field, UnivariatePolynomial};
use slop_multilinear::{Mle, Point};

/// The basic functionality required of a struct for which a sumcheck proof can be generated.
pub trait SumcheckPolyBase<K> {
    fn sum_as_poly_in_first_variable(&self, claim: Option<K>) -> UnivariatePolynomial<K>;

    fn n_variables(&self) -> usize;

    fn get_component_poly_evals(&self) -> Vec<K>;
}

/// The fix_first_variable function applied to a sumcheck's first round's polynomial .
pub trait SumcheckPolyFirstRound<K>: SumcheckPolyBase<K> {
    fn fix_first_variable(self, alpha: K) -> impl SumcheckPoly<K>;
}

/// The fix_first_variable function applied to a sumcheck's post first rounds' polynomial.
pub trait SumcheckPoly<K>: SumcheckPolyBase<K> {
    fn fix_first_variable(self, alpha: K) -> Self;
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
    fn fix_first_variable(self, alpha: S) -> impl SumcheckPoly<S> {
        assert!(self.num_variables() > 0, "Cannot fix first variable of a 0-variate polynomial");
        let mut result: Vec<S> = Vec::with_capacity(self.guts.len() / 2);
        self.guts
            .par_iter()
            // All the evaluations with the first variable set to 0.
            .take(self.guts.len() / 2)
            // All the evaluations with the first variable set to 1.
            .zip(self.guts.par_iter().skip(self.guts.len() / 2))
            // Interpolate between the two evaluations, and evaluate at alpha.
            .map(|(x, y)| S::from(*x) + alpha * (*y - *x))
            .collect_into_vec(&mut result);
        Mle::new(result)
    }
}

impl<K: Field> SumcheckPoly<K> for Mle<K> {
    fn fix_first_variable(self, alpha: K) -> Self {
        assert!(self.num_variables() > 0, "Cannot fix first variable of a 0-variate polynomial");
        let mut result: Vec<K> = Vec::with_capacity(self.guts.len() / 2);
        self.guts
            .par_iter()
            // All the evaluations with the first variable set to 0.
            .take(self.guts.len() / 2)
            // All the evaluations with the first variable set to 1.
            .zip(self.guts.par_iter().skip(self.guts.len() / 2))
            // Interpolate between the two evaluations, and evaluate at alpha.
            .map(|(x, y)| *x + alpha * (*y - *x))
            .collect_into_vec(&mut result);
        Mle::new(result)
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
    fn sum_as_poly_in_first_variable(&self, _claim: Option<S>) -> UnivariatePolynomial<S> {
        // If the polynomial is 0-variate, the length of its guts is not divisible by 2, so we need
        // to handle this case separately.
        if self.num_variables() == 0 {
            return UnivariatePolynomial::new(vec![self.guts[0].into(), S::zero()]);
        }

        let chunks = self.guts.chunks_exact(self.guts.len() / 2);

        let [first_half_sum, second_half_sum]: [S; 2] = chunks
            .map(|chunk| (chunk.par_iter().map(|x| *x).sum::<K>()).into())
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        // In the formula for `fix_first_variable`,
        UnivariatePolynomial::new(vec![first_half_sum, second_half_sum - first_half_sum])
    }

    fn n_variables(&self) -> usize {
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
    use p3_baby_bear::BabyBear;
    use slop_algebra::{extension::BinomialExtensionField, AbstractField, UnivariatePolynomial};
    use slop_multilinear::Mle;

    use crate::{SumcheckPoly, SumcheckPolyBase};

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_mle_edge_case() {
        let mle = Mle::new(vec![F::two()]);
        assert_eq!(
            mle.sum_as_poly_in_first_variable(None),
            UnivariatePolynomial::new(vec![F::two()])
        );
        assert_eq!(mle.num_variables(), 0);
    }

    #[test]
    fn test_simple_mle_properties() {
        let mle = Mle::new(vec![EF::zero(), EF::one(), EF::zero(), EF::zero()]);
        assert_eq!(
            mle.sum_as_poly_in_first_variable(None),
            UnivariatePolynomial::new(vec![EF::one(), EF::neg_one()])
        );
        assert_eq!(mle.num_variables(), 2);
        assert_eq!(
            mle.fix_first_variable(EF::from_canonical_u16(0xBABE)),
            Mle::new(vec![EF::zero(), EF::one() - EF::from_canonical_u32(0xBABE)])
        );
    }
}
