use rayon::prelude::*;

use slop_algebra::{AbstractField, Field, UnivariatePolynomial};
use slop_alloc::CpuBackend;
use slop_multilinear::{Mle, MleBaseBackend};

use crate::{
    backend::{ComponentPolyEvalBackend, SumCheckPolyFirstRoundBackend, SumcheckPolyBackend},
    SumcheckPoly, SumcheckPolyBase,
};

impl<F, A> SumcheckPolyBase for Mle<F, A>
where
    F: AbstractField,
    A: MleBaseBackend<F>,
{
    #[inline]
    fn n_variables(&self) -> u32 {
        self.num_variables()
    }
}

impl<'a, F, A> SumcheckPolyBase for &'a Mle<F, A>
where
    F: AbstractField,
    A: MleBaseBackend<F>,
{
    #[inline]
    fn n_variables(&self) -> u32 {
        (*self).num_variables()
    }
}

impl<EF> ComponentPolyEvalBackend<EF, Mle<EF, CpuBackend>> for CpuBackend
where
    EF: AbstractField,
{
    fn get_component_poly_evals(poly: &Mle<EF, CpuBackend>) -> Vec<EF> {
        let eval: EF = (*poly.guts()[[0, 0]]).clone();
        vec![eval]
    }
}

impl<'a, EF> ComponentPolyEvalBackend<EF, &'a Mle<EF, CpuBackend>> for CpuBackend
where
    EF: AbstractField,
{
    fn get_component_poly_evals(poly: &&'a Mle<EF>) -> Vec<EF> {
        let eval: EF = (*poly.guts()[[0, 0]]).clone();
        vec![eval]
    }
}

impl<F> SumcheckPolyBackend<F, Mle<F, CpuBackend>> for CpuBackend
where
    F: Field,
{
    fn fix_last_variable(poly: Mle<F, CpuBackend>, alpha: F) -> Mle<F, CpuBackend> {
        assert!(poly.num_variables() > 0, "Cannot fix first variable of a 0-variate polynomial");
        let mut result = Vec::with_capacity((1 << poly.num_variables()) / 2);

        poly.guts()
            .as_buffer()
            .par_iter()
            .chunks(2)
            .map(|chunk| {
                let [x, y] = chunk.try_into().unwrap();
                alpha * (*y - *x) + (*x)
            })
            .collect_into_vec(&mut result);

        Mle::from(result)
    }

    fn sum_as_poly_in_last_variable(
        poly: &Mle<F, CpuBackend>,
        claim: Option<F>,
    ) -> UnivariatePolynomial<F> {
        // If the polynomial is 0-variate, the length of its guts is not divisible by 2, so we need
        // to handle this case separately.
        if poly.num_variables() == 0 {
            return UnivariatePolynomial::new(vec![*poly.guts()[[0, 0]], F::zero()]);
        }

        if let Some(claim) = claim {
            let even_sum = poly.guts().as_slice().par_iter().step_by(2).copied().sum();
            let odd_sum = claim - even_sum;

            // In the formula for `fix_first_variable`,
            UnivariatePolynomial::new(vec![even_sum, odd_sum - even_sum])
        } else {
            // let mut first_half_sum = F::zero();
            // let mut second_half_sum = F::zero();

            // poly.guts().as_slice().chunks(2).for_each(|chunk| {
            //     let [x, y] = chunk.try_into().unwrap();
            //     first_half_sum += x;
            //     second_half_sum += y;
            // });

            let [first_half_sum, second_half_sum] = poly
                .guts()
                .as_slice()
                .par_chunks_exact(2)
                .fold(
                    || [F::zero(); 2],
                    |mut acc, chunk| {
                        acc[0] += chunk[0];
                        acc[1] += chunk[1];
                        acc
                    },
                )
                .reduce(
                    || [F::zero(); 2],
                    |mut acc, arr| {
                        acc[0] += arr[0];
                        acc[1] += arr[1];
                        acc
                    },
                );

            // In the formula for `fix_first_variable`,
            UnivariatePolynomial::new(vec![first_half_sum, second_half_sum - first_half_sum])
        }
    }
}

impl<'a, F> SumCheckPolyFirstRoundBackend<F, &'a Mle<F, CpuBackend>> for CpuBackend
where
    F: Field,
{
    fn fix_t_variables(
        poly: &'a Mle<F, CpuBackend>,
        alpha: F,
        t: usize,
    ) -> impl crate::SumcheckPoly<F> {
        assert_eq!(t, 1);
        assert!(poly.num_variables() > 0, "Cannot fix first variable of a 0-variate polynomial");
        let mut result = Vec::with_capacity((1 << poly.num_variables()) / 2);

        poly.guts()
            .as_buffer()
            .par_iter()
            .chunks(2)
            .map(|chunk| {
                let [x, y] = chunk.try_into().unwrap();
                alpha * (*y - *x) + (*x)
            })
            .collect_into_vec(&mut result);

        Mle::from(result)
    }

    fn sum_as_poly_in_last_t_variables(
        poly: &&'a Mle<F, CpuBackend>,
        claim: Option<F>,
        t: usize,
    ) -> UnivariatePolynomial<F> {
        assert_eq!(t, 1);
        (**poly).sum_as_poly_in_last_variable(claim)
    }
}
