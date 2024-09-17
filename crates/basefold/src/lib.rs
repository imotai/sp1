mod prover;
mod types;
mod verifier;

pub use prover::*;
use spl_multi_pcs::Point;
pub use types::*;

use p3_fri::verifier::FriError;

use spl_algebra::Field;

use std::{
    fmt::Debug,
    ops::{Add, Mul},
};

#[derive(Debug, Clone)]
pub struct UnivariatePolynomial<K> {
    pub coefficients: Vec<K>,
}

/// Basic univariate polynomial operations.
impl<K: Field> UnivariatePolynomial<K> {
    pub fn new(mut coefficients: Vec<K>) -> Self {
        // Pop trailing zeros.
        while let Some(&x) = coefficients.last() {
            if x == K::zero() {
                coefficients.pop();
            } else {
                break;
            }
        }
        Self { coefficients }
    }

    pub fn mul_by_x(&self) -> Self {
        let mut result = Vec::with_capacity(self.coefficients.len() + 1);
        result.push(K::zero());
        result.extend(&self.coefficients[..]);
        Self::new(result)
    }

    pub fn zero() -> Self {
        Self { coefficients: vec![] }
    }

    pub fn one() -> Self {
        Self { coefficients: vec![K::one()] }
    }

    pub fn eval_at_point(&self, point: K) -> K {
        // Horner's method.
        self.coefficients.iter().rev().fold(K::zero(), |acc, x| acc * point + *x)
    }
}

/// Scalar multiplication for univariate polynomials.
impl<K: Field> Mul<K> for UnivariatePolynomial<K> {
    type Output = Self;

    fn mul(self, rhs: K) -> Self::Output {
        Self { coefficients: self.coefficients.into_iter().map(|x| x * rhs).collect() }
    }
}

/// Sum of two univariate polynomials.
impl<K: Field> Add for UnivariatePolynomial<K> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let mut new_coeffs = vec![K::zero(); self.coefficients.len().max(rhs.coefficients.len())];
        for (i, x) in new_coeffs.iter_mut().enumerate() {
            *x = *self.coefficients.get(i).unwrap_or(&K::zero())
                + *rhs.coefficients.get(i).unwrap_or(&K::zero());
        }
        UnivariatePolynomial::new(new_coeffs)
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use rand::Rng;
    use spl_algebra::AbstractField;
    use spl_multi_pcs::{full_lagrange_eval, partial_lagrange_eval, Mle};

    use crate::Point;

    type F = BabyBear;

    #[test]
    pub fn test_lagrange_eval() {
        for _ in 0..10 {
            let rng = &mut rand::thread_rng();
            let point_1: Point<F> = Point::new(vec![rng.gen(), rng.gen(), rng.gen()]);
            let point_2: Point<F> = Point::new(vec![rng.gen(), rng.gen(), rng.gen()]);
            assert_eq!(
                full_lagrange_eval(&point_1, &point_2),
                Mle::eval_at_point(&partial_lagrange_eval(&point_1).into(), &point_2)
            );
        }
    }

    #[test]
    fn test_eval_at_point() {
        // Choosing the entry corresponding to the bitstring 10.
        let mle = Mle::new(vec![F::zero(), F::zero(), F::one(), F::zero()]);

        let point = Point::new(vec![F::two(), F::one()]);

        assert_eq!(mle.eval_at_point(&point), F::zero());
    }
}
