use std::{
    mem::ManuallyDrop,
    ops::{Add, Deref, DerefMut},
};

use rayon::prelude::*;

use derive_where::derive_where;
use rand::{distributions::Standard, prelude::Distribution, Rng};
use slop_algebra::{AbstractExtensionField, AbstractField, Field};
use slop_alloc::{Backend, Buffer, CpuBackend, HasBackend, GLOBAL_CPU_BACKEND};
use slop_matrix::Matrix;
use slop_tensor::Tensor;

use crate::{
    AddMleBackend, MleBaseBackend, MleEvaluationBackend, MleFixLastVariableBackend, MleFoldBackend,
    PartialLagrangeBackend, Point,
};

/// A bacth of multi-linear polynomials.
#[derive(Debug, Clone)]
#[derive_where(PartialEq, Eq; Tensor<T, A>)]
pub struct Mle<T, A: Backend = CpuBackend> {
    guts: Tensor<T, A>,
}

impl<F, A: Backend> HasBackend for Mle<F, A> {
    type Backend = A;

    #[inline]
    fn backend(&self) -> &Self::Backend {
        self.guts.backend()
    }
}

impl<F, A: Backend> Mle<F, A> {
    /// Creates a new MLE from a tensor in the correct shape.
    ///
    /// The tensor must be in the correct shape for the given backend.
    #[inline]
    pub const fn new(guts: Tensor<F, A>) -> Self {
        Self { guts }
    }

    #[inline]
    pub fn backend(&self) -> &A {
        self.guts.backend()
    }

    #[inline]
    pub fn into_guts(self) -> Tensor<F, A> {
        self.guts
    }

    /// Creates a new uninitialized MLE batch of the given size and number of variables.
    #[inline]
    pub fn uninit(num_polynomials: usize, num_non_zero_entries: usize, scope: &A) -> Self
    where
        F: AbstractField,
        A: MleBaseBackend<F>,
    {
        // The tensor is initialized in the correct shape by the backend.
        Self::new(scope.uninit_mle(num_polynomials, num_non_zero_entries))
    }

    #[inline]
    pub const fn guts(&self) -> &Tensor<F, A> {
        &self.guts
    }

    /// # Safety
    ///
    /// Changing the guts must preserve the layout that the MLE backend expects to have for a valid
    /// tensor to qualify as the guts of an MLE. For example, dimension matching the implementation
    /// of [Self::uninit].
    pub unsafe fn guts_mut(&mut self) -> &mut Tensor<F, A> {
        &mut self.guts
    }

    /// # Safety
    #[inline]
    pub unsafe fn assume_init(&mut self) {
        self.guts.assume_init();
    }

    /// Returns the number of polynomials in the batch.
    #[inline]
    pub fn num_polynomials(&self) -> usize
    where
        F: AbstractField,
        A: MleBaseBackend<F>,
    {
        A::num_polynomials(&self.guts)
    }

    /// Returns the number of variables in the polynomials.
    #[inline]
    pub fn num_variables(&self) -> u32
    where
        F: AbstractField,
        A: MleBaseBackend<F>,
    {
        A::num_variables(&self.guts)
    }

    /// Computes the partial lagrange polynomial eq(z, -) for a fixed z.
    #[inline]
    pub fn partial_lagrange(point: &Point<F, A>) -> Mle<F, A>
    where
        F: AbstractField,
        A: PartialLagrangeBackend<F>,
    {
        Mle::new(A::partial_lagrange(point))
    }

    /// Evaluates the MLE at a given point.
    #[inline]
    pub fn eval_at<EF: AbstractExtensionField<F>>(&self, point: &Point<EF, A>) -> MleEval<EF, A>
    where
        F: AbstractField,
        A: MleEvaluationBackend<F, EF>,
    {
        MleEval::new(A::eval_mle_at_point(&self.guts, point))
    }

    /// Compute the random linear combination of the even and odd coefficients of `vals`.
    ///
    /// This is used in the `Basefold` PCS.
    #[inline]
    pub fn fold(&self, beta: F) -> Mle<F, A>
    where
        F: AbstractField,
        A: MleFoldBackend<F>,
    {
        Mle::new(A::fold_mle(&self.guts, beta))
    }

    #[inline]
    pub fn fix_last_variable<EF>(&self, alpha: EF) -> Mle<EF, A>
    where
        F: AbstractField,
        EF: AbstractExtensionField<F>,
        A: MleFixLastVariableBackend<F, EF>,
    {
        Mle::new(A::mle_fix_last_variable(&self.guts, alpha))
    }
}

impl<T> Mle<T, CpuBackend> {
    pub fn rand<R: Rng>(rng: &mut R, num_polynomials: usize, num_variables: u32) -> Self
    where
        Standard: Distribution<T>,
    {
        Self::new(Tensor::rand(rng, [1 << num_variables, num_polynomials]))
    }

    /// Returns an iterator over the evaluations of the MLE on the Boolean hypercube.
    ///
    /// The iterator yields a slice for each index of the Boolean hypercube.
    pub fn hypercube_iter(&self) -> impl Iterator<Item = &[T]>
    where
        T: AbstractField,
    {
        let width = self.num_polynomials();
        let height = self.num_variables();
        (0..(1 << height)).map(move |i| &self.guts.as_slice()[i * width..(i + 1) * width])
    }

    /// Returns an iterator over the evaluations of the MLE on the Boolean hypercube.
    ///
    /// The iterator yields a slice for each index of the Boolean hypercube.
    pub fn hypercube_par_iter(&self) -> impl ParallelIterator<Item = &[T]>
    where
        T: AbstractField + Sync,
    {
        let width = self.num_polynomials();
        let height = self.num_variables();
        (0..(1 << height))
            .into_par_iter()
            .map(move |i| &self.guts.as_slice()[i * width..(i + 1) * width])
    }

    unsafe fn from_raw_parts(ptr: *mut T, num_polynomials: usize, len: usize) -> Self {
        let total_len = num_polynomials * len;
        let buffer = Buffer::from_raw_parts(ptr, total_len, total_len, GLOBAL_CPU_BACKEND);
        Self::new(Tensor::from(buffer).reshape([len, num_polynomials]))
    }

    /// legacy method to be compatible with the old API.
    pub fn eval_matrix_at_point<E>(
        matrix: &slop_matrix::dense::RowMajorMatrix<T>,
        point: &Point<E>,
    ) -> Vec<E>
    where
        T: AbstractField + Send + Sync,
        E: AbstractExtensionField<T> + Send + Sync,
    {
        let mle = unsafe {
            ManuallyDrop::new(Self::from_raw_parts(
                matrix.values.as_ptr() as *mut T,
                matrix.width,
                matrix.height(),
            ))
        };
        mle.eval_at(point).to_vec()
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
    pub fn full_lagrange_eval(point_1: &Point<T>, point_2: &Point<T>) -> T
    where
        T: AbstractField,
    {
        assert_eq!(point_1.dimension(), point_2.dimension());

        // Iterate over all values in the n-variates X and Y.
        point_1
            .iter()
            .zip(point_2.iter())
            .map(|(x, y)| {
                // Multiply by (x_i * y_i + (1-x_i) * (1-y_i)).
                let prod = x.clone() * y.clone();
                prod.clone() + prod + T::one() - x.clone() - y.clone()
            })
            .product()
    }

    pub fn fixed_at_zero<E>(&self, new_point: &Point<E>) -> E
    where
        T: AbstractField + Send + Sync + Copy,
        E: AbstractExtensionField<T> + Send + Sync + Copy,
    {
        // TODO: A smarter way to do this is pre-cache the partial_lagrange_evals that are implicit
        // in `eval_at_point` so we don't recompute it at every step of BaseFold.
        Mle::from(self.guts().as_slice().par_iter().step_by(2).copied().collect::<Vec<_>>())
            .eval_at(new_point)[0]
    }
}

impl<T> From<Vec<T>> for Mle<T, CpuBackend> {
    fn from(values: Vec<T>) -> Self {
        let len = values.len();
        let tensor = Tensor::from(values).reshape([len, 1]);
        Self::new(tensor)
    }
}

impl<T: Clone + Send + Sync> From<slop_matrix::dense::RowMajorMatrix<T>> for Mle<T, CpuBackend> {
    fn from(values: slop_matrix::dense::RowMajorMatrix<T>) -> Self {
        let num_polys = values.width;
        let num_vars = values.height().ilog2();
        assert_eq!(values.height(), 1 << num_vars);
        let tensor = Tensor::from(values.values).reshape([num_polys, 1]);
        Self::new(tensor)
    }
}

impl<T> FromIterator<T> for Mle<T, CpuBackend> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from(iter.into_iter().collect::<Vec<_>>())
    }
}

impl<'a, F: AbstractField, EF: AbstractExtensionField<F>, A: AddMleBackend<F, EF>>
    Add<&'a Mle<F, A>> for &'a Mle<EF, A>
{
    type Output = Mle<EF, A>;

    fn add(self, rhs: &'a Mle<F, A>) -> Self::Output {
        Mle::new(A::add_mle(&self.guts, &rhs.guts))
    }
}

impl<'a, F: AbstractField, EF: AbstractExtensionField<F>, A: AddMleBackend<F, EF>>
    Add<&'a Mle<F, A>> for Mle<EF, A>
{
    type Output = Mle<EF, A>;

    fn add(self, rhs: &'a Mle<F, A>) -> Self::Output {
        (&self) + rhs
    }
}

impl<F: AbstractField, EF: AbstractExtensionField<F>, A: AddMleBackend<F, EF>> Add<Mle<F, A>>
    for Mle<EF, A>
{
    type Output = Mle<EF, A>;

    fn add(self, rhs: Mle<F, A>) -> Self::Output {
        (&self) + &rhs
    }
}

/// The multilinear polynomial whose evaluation on the Boolean hypercube performs outputs 1 if the
/// Boolean hypercube point is the bit-string representation of a number greater than or equal to
/// `threshold`, and 0 otherwise.
pub fn partial_geq<F: Field>(threshold: usize, num_variables: usize) -> Vec<F> {
    assert!(threshold <= 1 << num_variables);

    (0..(1 << num_variables)).map(|x| if x >= threshold { F::one() } else { F::zero() }).collect()
}

/// A succinct way to compute the evaluation of `partial_geq` at `eval_point`. The threshold is passed
/// as a `Point` on the Boolean hypercube.
///
/// # Panics
/// If the dimensions of `threshold` and `eval_point` do not match.
/// If any of the entries in `threshold` are not boolean-valued.
pub fn full_geq<F: Field>(threshold: &Point<F>, eval_point: &Point<F>) -> F {
    assert_eq!(threshold.dimension(), eval_point.dimension());
    for bit in threshold.iter() {
        assert_eq!(*bit * (F::one() - *bit), F::zero());
    }
    threshold.iter().rev().zip(eval_point.iter().rev()).fold(F::one(), |acc, (x, y)| {
        ((F::one() - *x) * (F::one() - *y) + *x * *y) * acc + (F::one() - *x) * *y
    })
}

/// A bacth of multi-linear polynomial evaluations.
#[derive(Debug, Clone)]
pub struct MleEval<T, A: Backend = CpuBackend> {
    pub(crate) evaluations: Tensor<T, A>,
}

impl<T, A: Backend> MleEval<T, A> {
    /// Creates a new MLE evaluation from a tensor in the correct shape.
    #[inline]
    pub const fn new(evaluations: Tensor<T, A>) -> Self {
        Self { evaluations }
    }

    #[inline]
    pub fn evaluations(&self) -> &Tensor<T, A> {
        &self.evaluations
    }

    /// # Safety
    #[inline]
    pub unsafe fn evaluations_mut(&mut self) -> &mut Tensor<T, A> {
        &mut self.evaluations
    }

    #[inline]
    pub fn into_evaluations(self) -> Tensor<T, A> {
        self.evaluations
    }
}

impl<T: Clone> MleEval<T, CpuBackend> {
    pub fn to_vec(&self) -> Vec<T> {
        self.evaluations.as_buffer().to_vec()
    }
}

impl<T> Deref for MleEval<T, CpuBackend> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.evaluations.as_slice()
    }
}

impl<T> DerefMut for MleEval<T, CpuBackend> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.evaluations.as_mut_slice()
    }
}

#[cfg(test)]
mod tests {

    use slop_algebra::extension::BinomialExtensionField;
    use slop_alloc::Buffer;
    use slop_baby_bear::BabyBear;

    use super::*;

    use crate::{full_geq, partial_geq, Mle};

    #[test]
    fn test_mle_add() {
        let mut rng = rand::thread_rng();

        type F = BabyBear;

        let num_polynomials = 10;
        let num_variables = 11;

        let lhs = Mle::<F>::rand(&mut rng, num_polynomials, num_variables);
        let rhs = Mle::<F>::rand(&mut rng, num_polynomials, num_variables);

        let sum = &lhs + &rhs;

        assert_eq!(sum.num_polynomials(), num_polynomials);
        assert_eq!(sum.num_variables(), num_variables);

        let sum_vec = sum.guts.as_buffer().to_vec();
        let lhs_vec = lhs.guts.as_buffer().to_vec();
        let rhs_vec = rhs.guts.as_buffer().to_vec();

        for (sum, (lhs, rhs)) in sum_vec.iter().zip(lhs_vec.iter().zip(rhs_vec.iter())) {
            assert_eq!(*sum, *lhs + *rhs);
        }
    }

    #[test]
    fn test_mle_eval() {
        let mut rng = rand::thread_rng();

        type F = BabyBear;
        type EF = BinomialExtensionField<BabyBear, 4>;

        let num_variables = 11;
        let num_polynomials = 10;

        let mle = Mle::<F>::rand(&mut rng, num_polynomials, num_variables);

        // Test the correctness of values on the hypercube.
        for i in 0usize..(1 << num_variables) {
            // Get the big Endian bits of the index.
            let bits = (0..num_variables)
                .rev()
                .map(|j| (i >> j) & 1)
                .map(F::from_canonical_usize)
                .collect::<Vec<_>>();
            let point = Point::<F>::new(Buffer::from(bits));
            let value = mle.eval_at(&point).to_vec();
            for (j, v) in value.iter().enumerate() {
                assert_eq!(*mle.guts[[i, j]], *v);
            }
        }

        // Test the multi-linearity of evaluation.
        let point = Point::<EF>::rand(&mut rng, num_variables);

        let eval = mle.eval_at(&point);
        for i in 0..num_variables {
            let mut point_0 = point.clone();
            let mut point_1 = point.clone();
            let point_0_i_val: &mut EF = &mut point_0[i as usize];
            *point_0_i_val = EF::zero();
            let point_1_i_val: &mut EF = &mut point_1[i as usize];
            *point_1_i_val = EF::one();

            let eval_0 = mle.eval_at(&point_0);
            let eval_1 = mle.eval_at(&point_1);

            let z: EF = *point[i as usize];

            for ((eval_0, eval_1), eval) in
                eval_0.to_vec().iter().zip(eval_1.to_vec().iter()).zip(eval.to_vec().iter())
            {
                assert_eq!(*eval, *eval_0 * (EF::one() - z) + *eval_1 * z);
            }
        }

        // Test the linearity of evaluation.
        let rhs = Mle::<F>::rand(&mut rng, num_polynomials, num_variables);
        let point = Point::<EF>::rand(&mut rng, num_variables);

        let lhs_eval = mle.eval_at(&point);
        let rhs_eval = rhs.eval_at(&point);
        let sum_eval = (&mle + &rhs).eval_at(&point);

        let lhs_eval_values = lhs_eval.to_vec();
        let rhs_eval_values = rhs_eval.to_vec();
        let sum_eval_values = sum_eval.to_vec();

        for ((lhs, rhs), sum) in
            lhs_eval_values.iter().zip(rhs_eval_values.iter()).zip(sum_eval_values.iter())
        {
            assert_eq!(*lhs + *rhs, *sum);
        }
    }

    #[test]
    fn test_mle_fold() {
        let mut rng = rand::thread_rng();

        type EF = BinomialExtensionField<BabyBear, 4>;

        let mle = Mle::<EF>::rand(&mut rng, 1, 11);
        let point = Point::<EF>::rand(&mut rng, 10);

        let beta = rng.gen::<EF>();

        let fold = mle.fold(beta);

        let mut point_0 = point.to_vec();
        point_0.push(EF::zero());
        let point_0 = Point::<EF>::from(point_0);

        let mut point_1 = point.to_vec();
        point_1.push(EF::one());
        let point_1 = Point::<EF>::from(point_1);

        let eval_0 = *mle.eval_at(&point_0).evaluations()[[0]];
        let eval_1 = *mle.eval_at(&point_1).evaluations()[[0]];
        let fold_eval = *fold.eval_at(&point).evaluations()[[0]];

        assert_eq!(fold_eval, eval_0 + eval_1 * beta);
    }

    #[test]
    pub fn test_geq_polynomial() {
        let num_variables = 12;
        let mut rng = rand::thread_rng();

        type F = BabyBear;

        for threshold in 0..(1 << num_variables) {
            let eval_point =
                Point::<F>::from((0..num_variables).map(|_| rng.gen::<F>()).collect::<Vec<_>>());
            let geq_mle = Mle::from(partial_geq::<F>(threshold, num_variables));
            assert_eq!(
                geq_mle.eval_at(&eval_point).to_vec()[0],
                full_geq(&Point::from_usize(threshold, num_variables), &eval_point)
            );
        }
    }
}
