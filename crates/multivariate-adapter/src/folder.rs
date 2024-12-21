use std::{
    marker::PhantomData,
    ops::{Add, Mul, MulAssign, Sub},
};

use p3_air::{AirBuilder, ExtensionBuilder};
use p3_matrix::{dense::RowMajorMatrixView, stack::VerticalPair};
use p3_uni_stark::{StarkGenericConfig, Val};

use slop_algebra::{AbstractField, ExtensionField, Field};
use slop_multilinear::Point;

use crate::MultivariateEvaluationAirBuilder;

type Challenge<SC> = <SC as StarkGenericConfig>::Challenge;

/// A folder for verifier constraints.
pub type VerifierConstraintFolder<'a, SC> =
    GenericVerifierConstraintFolder<'a, Val<SC>, Challenge<SC>, Challenge<SC>, Challenge<SC>>;

/// A folder for verifier constraints.
pub struct GenericVerifierConstraintFolder<'a, F, EF, Var, Expr> {
    /// The main trace.
    pub main: VerticalPair<RowMajorMatrixView<'a, Var>, RowMajorMatrixView<'a, Var>>,

    /// The adapter trace (eq column and cumulative sum).
    pub adapter: VerticalPair<RowMajorMatrixView<'a, Var>, RowMajorMatrixView<'a, Var>>,
    /// The selector for the first row.
    pub is_first_row: Var,
    /// The selector for the last row.
    pub is_last_row: Var,
    /// The selector for the transition.
    pub is_transition: Var,
    /// The constraint folding challenge.
    pub alpha: Var,

    /// The expected evaluation of the multilinear.
    pub expected_evals: &'a [Var],

    /// The random challenge used to batch the evaluations.
    pub batch_challenge: Var,

    /// The accumulator for the constraint folding.
    pub accumulator: Expr,
    /// The public values.
    pub evaluation_point: Point<Var>,
    /// The marker type.
    pub _marker: PhantomData<(F, EF)>,
}

impl<'a, F, EF, Var, Expr> AirBuilder for GenericVerifierConstraintFolder<'a, F, EF, Var, Expr>
where
    F: Field,
    EF: ExtensionField<F>,
    Expr: AbstractField
        + From<F>
        + Add<Var, Output = Expr>
        + Add<F, Output = Expr>
        + Sub<Var, Output = Expr>
        + Sub<F, Output = Expr>
        + Mul<Var, Output = Expr>
        + Mul<F, Output = Expr>
        + MulAssign<EF>,
    Var: Into<Expr>
        + Copy
        + Add<F, Output = Expr>
        + Add<Var, Output = Expr>
        + Add<Expr, Output = Expr>
        + Sub<F, Output = Expr>
        + Sub<Var, Output = Expr>
        + Sub<Expr, Output = Expr>
        + Mul<F, Output = Expr>
        + Mul<Var, Output = Expr>
        + Mul<Expr, Output = Expr>
        + Send
        + Sync,
{
    type F = F;
    type Expr = Expr;
    type Var = Var;
    type M = VerticalPair<RowMajorMatrixView<'a, Var>, RowMajorMatrixView<'a, Var>>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row.into()
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row.into()
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition.into()
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let x: Expr = x.into();

        // Horner's method for evaluating the folded constraint polynomial.
        self.accumulator *= self.alpha.into();
        self.accumulator += x;
    }
}

impl<'a, F, EF, Var, Expr> ExtensionBuilder
    for GenericVerifierConstraintFolder<'a, F, EF, Var, Expr>
where
    F: Field,
    EF: ExtensionField<F>,
    Expr: AbstractField<F = EF>
        + From<F>
        + Add<Var, Output = Expr>
        + Add<F, Output = Expr>
        + Sub<Var, Output = Expr>
        + Sub<F, Output = Expr>
        + Mul<Var, Output = Expr>
        + Mul<F, Output = Expr>
        + MulAssign<EF>,
    Var: Into<Expr>
        + Copy
        + Add<F, Output = Expr>
        + Add<Var, Output = Expr>
        + Add<Expr, Output = Expr>
        + Sub<F, Output = Expr>
        + Sub<Var, Output = Expr>
        + Sub<Expr, Output = Expr>
        + Mul<F, Output = Expr>
        + Mul<Var, Output = Expr>
        + Mul<Expr, Output = Expr>
        + Send
        + Sync,
{
    type EF = EF;
    type ExprEF = Expr;
    type VarEF = Var;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.assert_zero(x);
    }
}

impl<'a, F, EF, Var, Expr> MultivariateEvaluationAirBuilder
    for GenericVerifierConstraintFolder<'a, F, EF, Var, Expr>
where
    F: Field,
    EF: ExtensionField<F>,
    Expr: AbstractField<F = EF>
        + From<F>
        + Add<Var, Output = Expr>
        + Add<F, Output = Expr>
        + Sub<Var, Output = Expr>
        + Sub<F, Output = Expr>
        + Mul<Var, Output = Expr>
        + Mul<F, Output = Expr>
        + MulAssign<EF>,
    Var: Into<Expr>
        + Copy
        + Add<F, Output = Expr>
        + Add<Var, Output = Expr>
        + Add<Expr, Output = Expr>
        + Sub<F, Output = Expr>
        + Sub<Var, Output = Expr>
        + Sub<Expr, Output = Expr>
        + Mul<F, Output = Expr>
        + Mul<Var, Output = Expr>
        + Mul<Expr, Output = Expr>
        + Send
        + Sync,
{
    type MP = VerticalPair<RowMajorMatrixView<'a, Var>, RowMajorMatrixView<'a, Var>>;

    type Sum = Var;

    type RandomVar = Var;

    fn adapter(&self) -> Self::MP {
        self.adapter
    }

    fn expected_evals(&self) -> &[Self::Sum] {
        self.expected_evals
    }

    fn _evaluation_point(&self) -> Vec<Self::Sum> {
        self.evaluation_point.to_vec()
    }

    fn batch_randomness(&self) -> Self::RandomVar {
        self.batch_challenge
    }
}
