use crate::{
    backend::{MleBaseBackend, MleEvaluationBackend, PartialLargangeBackend},
    Point,
};
use csl_device::{cuda::TaskScope, mem::DeviceData, CudaSend, DeviceScope, Tensor};
use slop_algebra::{ExtensionField, Field};

/// A bacth of multi-linear polynomials.
#[derive(Debug, Clone, CudaSend)]
pub struct Mle<T: DeviceData, B: DeviceScope> {
    guts: Tensor<T, B>,
}

/// A bacth of multi-linear polynomials.
#[derive(Debug, Clone, CudaSend)]
pub struct MleEval<T: DeviceData, B: DeviceScope> {
    pub(crate) evalutions: Tensor<T, B>,
}

impl<T: DeviceData, B: DeviceScope> Mle<T, B> {
    /// Creates a new MLE from a tensor in the correct shape.
    #[inline]
    pub const fn new(guts: Tensor<T, B>) -> Self {
        Self { guts }
    }

    #[inline]
    pub fn scope(&self) -> &B {
        self.guts.scope()
    }

    #[inline]
    pub fn into_guts(self) -> Tensor<T, B> {
        self.guts
    }
}

impl<F: Field, B: DeviceScope> Mle<F, B> {
    /// Creates a new uninitialized MLE batch of the given size and number of variables.
    #[inline]
    pub fn uninit(num_polynomials: usize, num_variables: usize, scope: &B) -> Self
    where
        B: MleBaseBackend<F>,
    {
        Self::new(scope.uninit_mle(num_polynomials, num_variables))
    }

    pub const fn guts(&self) -> &Tensor<F, B> {
        &self.guts
    }

    /// # Safety
    ///
    /// Changing the guts must preserve the layout that the MLE backend expects to have for a valid
    /// tensor to qualify as the guts of an MLE. For example, dimension matching the implementation
    /// of [Self::uninit].
    pub unsafe fn guts_mut(&mut self) -> &mut Tensor<F, B> {
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
        B: MleBaseBackend<F>,
    {
        B::num_polynomials(&self.guts)
    }

    /// Returns the number of variables in the polynomials.
    #[inline]
    pub fn num_variables(&self) -> u32
    where
        B: MleBaseBackend<F>,
    {
        B::num_variables(&self.guts)
    }

    /// Computes the partial lagrange polynomial eq(z, -) for a fixed z.
    #[inline]
    pub fn partial_lagrange(point: &Point<F, B>) -> Mle<F, B>
    where
        B: PartialLargangeBackend<F>,
    {
        Mle::new(B::partial_lagrange(point.values()))
    }

    /// Evaluates the MLE at a given point.
    #[inline]
    pub fn eval_at<EF: ExtensionField<F>>(&self, point: &Point<EF, B>) -> MleEval<EF, B>
    where
        B: MleEvaluationBackend<F, EF>,
    {
        MleEval::new(B::eval_mle_at_point(&self.guts, point.values()))
    }
}

impl<T: DeviceData, B: DeviceScope> MleEval<T, B> {
    /// Creates a new MLE evaluation from a tensor in the correct shape.
    #[inline]
    pub const fn new(evalutions: Tensor<T, B>) -> Self {
        Self { evalutions }
    }

    #[inline]
    pub fn evalutions(&self) -> &Tensor<T, B> {
        &self.evalutions
    }

    /// # Safety
    #[inline]
    pub unsafe fn evalutions_mut(&mut self) -> &mut Tensor<T, B> {
        &mut self.evalutions
    }

    #[inline]
    pub fn into_evalutions(self) -> Tensor<T, B> {
        self.evalutions
    }
}
