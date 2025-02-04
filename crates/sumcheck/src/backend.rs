use slop_algebra::UnivariatePolynomial;
use slop_alloc::{Backend, HasBackend};

use crate::{ComponentPoly, SumcheckPoly, SumcheckPolyBase, SumcheckPolyFirstRound};

/// A trait to enable backend implementations of component polynomials.
///
/// An implementation of this trait for a type will imply a [crate::ComponentPoly] implementation
pub trait ComponentPolyEvalBackend<K, P>: Backend
where
    P: SumcheckPolyBase + HasBackend<Backend = Self>,
{
    fn get_component_poly_evals(poly: &P) -> Vec<K>;
}

impl<K, P> ComponentPoly<K> for P
where
    P: SumcheckPolyBase + HasBackend,
    P::Backend: ComponentPolyEvalBackend<K, P>,
{
    #[inline]
    fn get_component_poly_evals(&self) -> Vec<K> {
        P::Backend::get_component_poly_evals(self)
    }
}

/// A trait to enable backend implementations of sumcheck polynomials for the first round.
///
/// An implementation of this trait for a type will imply a [crate::SumcheckPolyFirstRound]
/// implementation for that type.
pub trait SumCheckPolyFirstRoundBackend<K, P>: Backend
where
    P: SumcheckPolyBase + HasBackend<Backend = Self>,
{
    fn fix_t_variables(poly: P, alpha: K, t: usize) -> impl SumcheckPoly<K>;

    fn sum_as_poly_in_last_t_variables(
        poly: &P,
        claim: Option<K>,
        t: usize,
    ) -> UnivariatePolynomial<K>;
}

impl<K, P> SumcheckPolyFirstRound<K> for P
where
    P: SumcheckPolyBase + ComponentPoly<K> + HasBackend,
    P::Backend: SumCheckPolyFirstRoundBackend<K, P>,
{
    #[inline]
    fn fix_t_variables(self, alpha: K, t: usize) -> impl SumcheckPoly<K> {
        P::Backend::fix_t_variables(self, alpha, t)
    }

    #[inline]
    fn sum_as_poly_in_last_t_variables(
        &self,
        claim: Option<K>,
        t: usize,
    ) -> UnivariatePolynomial<K> {
        P::Backend::sum_as_poly_in_last_t_variables(self, claim, t)
    }
}

/// A trait to enable backend implementations of sumcheck polynomials.
///
/// An implementation of this trait for a type will imply a [crate::SumcheckPoly] implementation
pub trait SumcheckPolyBackend<K, P>: Backend
where
    P: SumcheckPolyBase + ComponentPoly<K> + HasBackend<Backend = Self>,
{
    fn fix_last_variable(poly: P, alpha: K) -> P;

    fn sum_as_poly_in_last_variable(poly: &P, claim: Option<K>) -> UnivariatePolynomial<K>;
}

impl<K, P> SumcheckPoly<K> for P
where
    P: SumcheckPolyBase + ComponentPoly<K> + HasBackend,
    P::Backend: SumcheckPolyBackend<K, P>,
{
    #[inline]
    fn fix_last_variable(self, alpha: K) -> Self {
        P::Backend::fix_last_variable(self, alpha)
    }

    #[inline]
    fn sum_as_poly_in_last_variable(&self, claim: Option<K>) -> UnivariatePolynomial<K> {
        P::Backend::sum_as_poly_in_last_variable(self, claim)
    }
}
