//! Zerocheck Sumcheck polynomial.

mod fix_last_variable;
mod sum_as_poly;

use fix_last_variable::fix_last_variable;

use std::{
    marker::PhantomData,
    ops::{Add, Mul, Sub},
    sync::Arc,
};

use p3_field::{AbstractExtensionField, ExtensionField};
use slop_algebra::{Field, UnivariatePolynomial};
use slop_alloc::{Backend, CpuBackend};
use slop_multilinear::{Mle, MleBaseBackend, Point};
use slop_sumcheck::{ComponentPoly, SumcheckPoly, SumcheckPolyBase, SumcheckPolyFirstRound};
pub use sum_as_poly::*;

/// Zerocheck sumcheck polynomial.
pub struct ZeroCheckPoly<
    K: Field,
    F: Field,
    EF: ExtensionField<F>,
    AirData,
    B: Backend + MleBaseBackend<K> = CpuBackend,
> {
    /// The data that contains the constraint polynomial.
    pub air_data: AirData,
    /// The random challenge point at which the polynomial is evaluated.
    pub zeta: Point<EF>,
    /// The preprocessed trace.
    pub preprocessed_columns: Option<Arc<Mle<K, B>>>,
    /// The main trace.
    pub main_columns: Arc<Mle<K, B>>,
    /// The adjustment factor from the constant part of the eq polynomial.
    pub eq_adjustment: EF,
    /// The geq polynomial value.  This will be 0 for all zerocheck polys that are at least one non-padded variable.
    pub geq_value: EF,
    /// Num padded variables.  These padded variables are the first-most (e.g. the most significant) variables.
    pub num_padded_vars: usize,
    /// The padded row adjustment.
    pub padded_row_adjustment: EF,
    _marker: PhantomData<F>,
}

impl<K: Field, F: Field, EF: ExtensionField<F>, AirData, B: Backend + MleBaseBackend<K>>
    ZeroCheckPoly<K, F, EF, AirData, B>
{
    /// Creates a new `ZeroCheckPoly`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        air_data: AirData,
        zeta: Point<EF>,
        preprocessed_values: Option<Arc<Mle<K, B>>>,
        main_values: Arc<Mle<K, B>>,
        eq_adjustment: EF,
        geq_value: EF,
        num_padded_vars: usize,
        padded_row_adjustment: EF,
    ) -> Self {
        // The zeta random point must have the same number of variables of the trace + num_padded_vars.
        let num_main_vars = main_values.num_variables() as usize;
        assert!(
            zeta.dimension() == num_main_vars + num_padded_vars,
            "point dimension must match main values height.  point dim: {:?}, main values height: {:?}",
            zeta.dimension(),
            num_main_vars + num_padded_vars
        );
        // The trace height for main and preprocessed must be the same.
        if let Some(preprocessed_values) = preprocessed_values.as_ref() {
            let num_preprocessed_vars: usize =
                preprocessed_values.num_variables().try_into().unwrap();
            assert!(num_preprocessed_vars == num_main_vars,);
        }

        Self {
            air_data,
            zeta,
            preprocessed_columns: preprocessed_values,
            main_columns: main_values,
            eq_adjustment,
            geq_value,
            num_padded_vars,
            padded_row_adjustment,
            _marker: PhantomData,
        }
    }
}

impl<
        K: Field + From<F> + Add<F, Output = K> + Sub<F, Output = K> + Mul<F, Output = K>,
        F: Field,
        EF: ExtensionField<F> + From<K> + ExtensionField<F> + AbstractExtensionField<K>,
        AirData,
        B: Backend + MleBaseBackend<K>,
    > SumcheckPolyBase for ZeroCheckPoly<K, F, EF, AirData, B>
{
    fn n_variables(&self) -> u32 {
        self.main_columns.num_variables() + self.num_padded_vars as u32
    }
}

impl<
        K: Field + From<F> + Add<F, Output = K> + Sub<F, Output = K> + Mul<F, Output = K>,
        F: Field,
        EF: ExtensionField<F> + From<K> + ExtensionField<F> + AbstractExtensionField<K>,
        AirData: Sync,
    > ComponentPoly<EF> for ZeroCheckPoly<K, F, EF, AirData>
{
    async fn get_component_poly_evals(&self) -> Vec<EF> {
        assert!(self.n_variables() == 0);

        let prep_columns = self.preprocessed_columns.as_ref();
        // First get the preprocessed values.
        let evals = if let Some(preprocessed_values) = prep_columns {
            preprocessed_values.guts().as_slice()
        } else {
            &[]
        };

        // Add the main values.
        evals
            .iter()
            .chain(self.main_columns.guts().as_slice())
            .map(|x| (*x).into())
            .collect::<Vec<_>>()
    }
}

#[allow(clippy::mismatching_type_param_order)]
impl<
        F: Field,
        EF: ExtensionField<F>,
        AirData: ZerocheckProver<F, F, EF> + ZerocheckProver<F, EF, EF>,
    > SumcheckPolyFirstRound<EF> for ZeroCheckPoly<F, F, EF, AirData>
{
    async fn fix_t_variables(self, alpha: EF, t: usize) -> impl SumcheckPoly<EF> {
        assert!(t == 1);
        fix_last_variable(self, alpha).await
    }

    async fn sum_as_poly_in_last_t_variables(
        &self,
        claim: Option<EF>,
        t: usize,
    ) -> UnivariatePolynomial<EF> {
        assert!(t == 1);
        assert!(self.n_variables() > 0);
        sum_as_poly_in_last_variable::<F, F, EF, AirData, true>(self, claim).await
    }
}

#[allow(clippy::mismatching_type_param_order)]
impl<F: Field, EF: ExtensionField<F>, AirData: ZerocheckProver<F, EF, EF>> SumcheckPoly<EF>
    for ZeroCheckPoly<EF, F, EF, AirData>
{
    async fn fix_last_variable(self, alpha: EF) -> ZeroCheckPoly<EF, F, EF, AirData> {
        fix_last_variable(self, alpha).await
    }

    async fn sum_as_poly_in_last_variable(&self, claim: Option<EF>) -> UnivariatePolynomial<EF> {
        assert!(self.n_variables() > 0);
        sum_as_poly_in_last_variable::<EF, F, EF, AirData, false>(self, claim).await
    }
}
