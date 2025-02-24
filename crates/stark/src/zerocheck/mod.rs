//! Zerocheck Sumcheck polynomial.

mod fix_last_variable;
mod sum_as_poly;
pub(crate) mod utils;

use fix_last_variable::fix_last_variable;
use sum_as_poly::sum_as_poly_in_last_variable;
use utils::ZeroCheckPolyVals;

use std::ops::{Add, Mul, Sub};

use p3_air::Air;
use p3_field::{AbstractExtensionField, ExtensionField};
use slop_algebra::{Field, UnivariatePolynomial};
use slop_alloc::{Backend, CpuBackend};
use slop_multilinear::{Mle, MleBaseBackend, Point};
use slop_sumcheck::{ComponentPoly, SumcheckPoly, SumcheckPolyBase, SumcheckPolyFirstRound};

use crate::{air::MachineAir, ConstraintSumcheckFolder};

/// Backend that to support `sum_as_poly_in_last_variable`.
pub trait SumAsPolyInLastVariableBackend<
    K: Field + From<F> + Add<F, Output = K> + Sub<F, Output = K> + Mul<F, Output = K>,
    F: Field,
    EF: ExtensionField<F> + From<K> + ExtensionField<F> + AbstractExtensionField<K>,
    A: for<'b> Air<ConstraintSumcheckFolder<'b, F, K, EF>> + MachineAir<F>,
    const IS_FIRST_ROUND: bool,
>: Backend
{
    /// Generate the univariate polynomial for a zerocheck poly.
    fn sum_as_poly_in_last_variable(
        partial_lagrange: &Mle<EF, Self>,
        preprocessed_values: Option<&Mle<K, Self>>,
        main_values: &Mle<K, Self>,
        num_non_padded_terms: usize,
        public_values: &[F],
        powers_of_alpha: &[EF],
        air: &A,
    ) -> (EF, EF, EF);
}

pub(crate) struct ZeroCheckPoly<
    'a,
    K: Field,
    F: Field,
    EF: ExtensionField<F>,
    A,
    B: Backend + MleBaseBackend<K> = CpuBackend,
> {
    /// The air that contains the constraint polynomial.
    pub air: &'a A,
    /// The public values.
    pub public_values: &'a Vec<F>,
    /// The random challenge point at which the polynomial is evaluated.
    pub zeta: Point<EF>,
    /// The preprocessed trace.
    pub preprocessed_columns: Option<ZeroCheckPolyVals<'a, K, B>>,
    /// The main trace.
    pub main_columns: ZeroCheckPolyVals<'a, K, B>,
    /// The adjustment factor from the constant part of the eq polynomial.
    pub eq_adjustment: EF,
    /// The geq polynomial value.  This will be 0 for all zerocheck polys that are at least one non-padded variable.
    pub geq_value: EF,
    /// The powers of alpha to combine all the constraints.
    pub powers_of_alpha: Vec<EF>,
    /// Num padded variables.  These padded variables are the first-most (e.g. the most significant) variables.
    pub num_padded_vars: usize,
    /// The padded row adjustment.
    pub padded_row_adjustment: EF,
}

impl<
        'a,
        K: Field,
        F: Field,
        EF: ExtensionField<F>,
        A: MachineAir<F>,
        B: Backend + MleBaseBackend<K>,
    > ZeroCheckPoly<'a, K, F, EF, A, B>
{
    /// Creates a new `ZeroCheckPoly`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        air: &'a A,
        zeta: Point<EF>,
        preprocessed_values: Option<ZeroCheckPolyVals<'a, K, B>>,
        main_values: ZeroCheckPolyVals<'a, K, B>,
        eq_adjustment: EF,
        geq_value: EF,
        public_values: &'a Vec<F>,
        powers_of_alpha: Vec<EF>,
        num_padded_vars: usize,
        padded_row_adjustment: EF,
    ) -> Self {
        // The zeta random point must have the same number of variables of the trace + num_padded_vars.
        let num_main_vars = main_values.mle_ref().num_variables() as usize;
        assert!(
            zeta.dimension() == num_main_vars + num_padded_vars,
            "point dimension must match main values height.  point dim: {:?}, main values height: {:?}",
            zeta.dimension(),
            num_main_vars + num_padded_vars
        );
        // The trace height for main and preprocessed must be the same.
        if let Some(preprocessed_values) = preprocessed_values.as_ref() {
            let num_preprocessed_vars: usize =
                preprocessed_values.mle_ref().num_variables().try_into().unwrap();
            assert!(num_preprocessed_vars == num_main_vars,);
        }

        Self {
            air,
            public_values,
            zeta,
            preprocessed_columns: preprocessed_values,
            main_columns: main_values,
            eq_adjustment,
            geq_value,
            powers_of_alpha,
            num_padded_vars,
            padded_row_adjustment,
        }
    }
}

impl<
        'a,
        K: Field + From<F> + Add<F, Output = K> + Sub<F, Output = K> + Mul<F, Output = K>,
        F: Field,
        EF: ExtensionField<F> + From<K> + ExtensionField<F> + AbstractExtensionField<K>,
        A: for<'b> Air<ConstraintSumcheckFolder<'b, F, K, EF>> + MachineAir<F>,
        B: Backend + MleBaseBackend<K>,
    > SumcheckPolyBase for ZeroCheckPoly<'a, K, F, EF, A, B>
{
    fn n_variables(&self) -> u32 {
        self.main_columns.mle_ref().num_variables() + self.num_padded_vars as u32
    }
}

impl<
        'a,
        K: Field + From<F> + Add<F, Output = K> + Sub<F, Output = K> + Mul<F, Output = K>,
        F: Field,
        EF: ExtensionField<F> + From<K> + ExtensionField<F> + AbstractExtensionField<K>,
        A: for<'b> Air<ConstraintSumcheckFolder<'b, F, K, EF>> + MachineAir<F>,
    > ComponentPoly<EF> for ZeroCheckPoly<'a, K, F, EF, A>
{
    async fn get_component_poly_evals(&self) -> Vec<EF> {
        assert!(self.n_variables() == 0);

        // First get the preprocessed values.
        let evals = if let Some(preprocessed_values) = self.preprocessed_columns.as_ref() {
            preprocessed_values.mle_ref().guts().as_slice()
        } else {
            &[]
        };

        // Add the main values.
        evals
            .iter()
            .chain(self.main_columns.mle_ref().guts().as_slice())
            .map(|x| (*x).into())
            .collect::<Vec<_>>()
    }
}

#[allow(clippy::mismatching_type_param_order)]
impl<
        'a,
        F: Field,
        EF: ExtensionField<F>,
        A: for<'b> Air<ConstraintSumcheckFolder<'b, F, F, EF>>
            + for<'b> Air<ConstraintSumcheckFolder<'b, F, EF, EF>>
            + MachineAir<F>,
    > SumcheckPolyFirstRound<EF> for ZeroCheckPoly<'a, F, F, EF, A>
{
    async fn fix_t_variables(self, alpha: EF, t: usize) -> impl SumcheckPoly<EF> {
        assert!(t == 1);
        fix_last_variable(self, alpha)
    }

    async fn sum_as_poly_in_last_t_variables(
        &self,
        claim: Option<EF>,
        t: usize,
    ) -> UnivariatePolynomial<EF> {
        assert!(t == 1);
        assert!(self.n_variables() > 0);
        sum_as_poly_in_last_variable::<F, F, EF, A, true>(self, claim)
    }
}

#[allow(clippy::mismatching_type_param_order)]
impl<
        'a,
        F: Field,
        EF: ExtensionField<F>,
        A: for<'b> Air<ConstraintSumcheckFolder<'b, F, F, EF>>
            + for<'b> Air<ConstraintSumcheckFolder<'b, F, EF, EF>>
            + MachineAir<F>,
    > SumcheckPoly<EF> for ZeroCheckPoly<'a, EF, F, EF, A>
{
    async fn fix_last_variable(self, alpha: EF) -> ZeroCheckPoly<'a, EF, F, EF, A> {
        fix_last_variable(self, alpha)
    }

    async fn sum_as_poly_in_last_variable(&self, claim: Option<EF>) -> UnivariatePolynomial<EF> {
        assert!(self.n_variables() > 0);
        sum_as_poly_in_last_variable::<EF, F, EF, A, false>(self, claim)
    }
}
