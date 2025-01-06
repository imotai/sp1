use std::ops::Mul;

use itertools::Itertools;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_util::log2_strict_usize;
use slop_algebra::{ExtensionField, Field};

use crate::Point;

/// A struct wrapping Vec<K>.
///
/// The field `guts` is the vector of evaluations of a multilinear polynomial on the Boolean
/// hypercube.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Mle<K> {
    pub guts: Vec<K>,
}

impl<K> From<Vec<K>> for Mle<K> {
    fn from(value: Vec<K>) -> Self {
        Self { guts: value }
    }
}

impl<K: Field> Mle<K> {
    pub fn new(guts: Vec<K>) -> Self {
        Self { guts }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self { guts: Vec::with_capacity(capacity) }
    }

    /// The [`Mle`] obtained by fixing the last variable to 0. Since an [`Mle`] is encoded as a
    /// vector of evaluations, this is done by taking every even-indexed element of the vector.
    pub fn fix_last_to_zero(&self) -> Self {
        if self.num_variables() == 0 {
            self.clone()
        } else {
            Mle::new(self.guts.iter().step_by(2).copied().collect())
        }
    }

    /// The Mle obtained by fixing the last variable to 1. Since an [`Mle`] is encoded as a vector
    /// of evaluations, this is done by taking every odd-indexed element of the vector.
    pub fn fix_last_to_one(&self) -> Self {
        if self.num_variables() == 0 {
            self.clone()
        } else {
            Mle::new(self.guts.iter().skip(1).step_by(2).copied().collect())
        }
    }

    pub fn num_variables(&self) -> usize {
        log2_strict_usize(self.guts.len())
    }

    /// Splits a multilinear polynomial in n variables into 2^(n-idx) polynomials in idx variables.
    pub fn split_at(&self, idx: usize) -> Vec<Self> {
        // TODO: Remove clone.
        RowMajorMatrix::new(self.guts.clone(), 1 << (self.num_variables() - idx))
            .transpose()
            .values
            .chunks_exact(1 << idx)
            .map(|x| x.to_vec().into())
            .collect()
    }

    pub fn to_extension_field<EF: ExtensionField<K>>(self) -> Mle<EF> {
        Mle::new(self.guts.into_iter().map(EF::from_base).collect())
    }
}

impl<K: Field> Mle<K> {
    pub fn eval_at_point<EF: Field + Mul<K, Output = EF>>(&self, point: &Point<EF>) -> EF {
        self.guts.iter().zip(partial_lagrange_eval(point).iter()).map(|(x, y)| *y * *x).sum()
    }

    pub fn eval_batch_at_point<EF: Field + Mul<K, Output = EF>>(
        mles: &[&Mle<K>],
        point: &Point<EF>,
    ) -> Vec<EF> {
        let partial_lagrange = partial_lagrange_eval(point);
        mles.iter()
            .map(|mle| mle.guts.iter().zip(partial_lagrange.iter()).map(|(x, y)| *y * *x).sum())
            .collect()
    }

    pub fn eval_matrix_at_point<EK: Field + Mul<K, Output = EK>>(
        matrix: &RowMajorMatrix<K>,
        point: &Point<EK>,
    ) -> Vec<EK> {
        let partial_lagrange = partial_lagrange_eval(point);
        let mut evals = vec![EK::zero(); matrix.width()];
        matrix.rows().zip_eq(partial_lagrange.iter()).for_each(|(row, lagrange)| {
            row.zip_eq(evals.iter_mut()).for_each(|(x, y)| {
                *y += *lagrange * x;
            });
        });
        evals
    }

    pub fn random_linear_combination(&self, beta: K) -> Mle<K> {
        // Compute the random linear combination of the even and odd coefficients of `vals`. This is
        // used to reduce the two evaluation claims for new_point into a single evaluation claim.
        Mle::new(
            self.guts
                .iter()
                .step_by(2)
                .copied()
                .zip(self.guts.iter().skip(1).step_by(2).copied())
                .map(|(a, b)| a + beta * b)
                .collect::<Vec<_>>(),
        )
    }

    /// Compute the evaluation claims with the last variable fixed to 0, and the last variable
    /// fixed to 1 while fixing the remaining coordinates to their corresponding values in `point`.
    /// These are used to generate the messages sent to the verifier in a BaseFold proof.
    pub fn fixed_evaluations(&self, new_point: &Point<K>) -> [K; 2] {
        let evens = self.guts.iter().step_by(2).copied().collect::<Vec<_>>().into();
        let odds = self.guts.iter().skip(1).step_by(2).copied().collect::<Vec<_>>().into();
        let batch = vec![&evens, &odds];

        let batch_evals = Mle::eval_batch_at_point(&batch, new_point);
        [batch_evals[0], batch_evals[1]]
    }

    pub fn fixed_at_zero<EK: Field + Mul<K, Output = EK>>(&self, new_point: &Point<EK>) -> EK {
        // TODO: A smarter way to do this is pre-cache the partial_lagrange_evals that are implicit
        // in `eval_at_point` so we don't recompute it at every step of BaseFold.
        Mle::new(self.guts.iter().step_by(2).copied().collect()).eval_at_point(new_point)
    }
}

impl<K: Field> From<Mle<K>> for Vec<K> {
    fn from(value: Mle<K>) -> Self {
        value.guts
    }
}

impl<K> From<&[K]> for Mle<K>
where
    K: Copy,
{
    fn from(value: &[K]) -> Self {
        Mle { guts: value.to_vec() }
    }
}

impl<K: Field> Mul<K> for Mle<K> {
    type Output = Self;

    fn mul(self, rhs: K) -> Self::Output {
        Self { guts: self.guts.into_iter().map(|x| x * rhs).collect() }
    }
}

/// Consider the 2n-variate `Mle` f whose underlying Vec consists of the rows of the 2^n x 2^n
/// identity matrix concatenated into a Vec of length 2^(2n). Computes the n-variate `Mle` formed
/// from f by fixing the first n variables to the values in `point`.
///
/// The explicit formula for partial_lagrange_eval(n_variables, point) is as follows: let
/// `point.point = vec![x_1, x_2, ..., x_n]`. For y in {0,1}^n, let `y_i` be the ith bit of y. Then
/// partial_lagrange_eval(n, point) = Prod_i (x_i * y_i + (1-x_i) * (1-y_i)).
/// `partial_lagrange_eval(n,point)` has the nice property that to evaluate an n-variate `Mle` g at
/// a `point`, we simply take the dot product of partial_lagrange_eval(n, point) with g.
///  
/// Yet another perspective on `partial_lagrange_eval`. If we take the 2^{n_variables} `Mle`s
/// returned by letting `point` range over the Boolean hypercube, these `Mle`s form a basis for the
/// space of multilinear polynomials in `n_variables` variables.
///
/// The verifier should not have access to this function.
///
/// The implementation below has runtime O(2^n), which is faster than iterating over the Boolean
/// hypercube and evaluating the polynomial at each point (runtime: O(n2^n)).
pub fn partial_lagrange_eval<K: Field>(point: &Point<K>) -> Vec<K> {
    let one = K::one();
    let mut evals = Vec::with_capacity(1 << point.dimension());
    evals.push(one);

    // Build evals in n_variables rounds. In each round, we consider one more entry of `point`,
    // hence the zip.
    point.0.iter().for_each(|coordinate| {
        evals = evals
            .iter()
            // For each value in the previous round, multiply by (1-coordinate) and coordinate,
            // and collect all these values into a new vec.
            .flat_map(|val| {
                let prod = *val * *coordinate;
                [*val - prod, prod]
            })
            .collect();
    });
    evals
}

/// Evaluates the 2n-variate multilinear polynomial f(X,Y) = Prod_i (X_i * Y_i + (1-X_i) * (1-Y_i))
/// at a given pair (X,Y) of n-dimenional BabyBearExtensionField points.
///
///
/// This evaluation takes time linear in n to compute, so the verifier can easily compute it. Hence,
/// even though
/// ```full_lagrange_eval(point_1,
/// point_2)==partial_lagrange_eval(point_1).eval_at_point(point_2)```,
/// the RHS of the above equation runs in O(2^n) time, while the LHS runs in O(n).
/// The polynomial f(X,Y) is an important building block in zerocheck and other protocols which use
/// sumcheck.
pub fn full_lagrange_eval<F: Field>(point_1: &Point<F>, point_2: &Point<F>) -> F {
    assert_eq!(point_1.dimension(), point_2.dimension());

    //Iterate over all values in the n-variates X and Y.
    point_1
        .0
        .iter()
        .zip(point_2.0.iter())
        .map(|(x, y)| {
            // Multiply by (x_i * y_i + (1-x_i) * (1-y_i)).
            let prod = *x * *y;
            prod + prod + F::one() - *x - *y
        })
        .product()
}
