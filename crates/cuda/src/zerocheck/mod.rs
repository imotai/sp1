mod fix_last_variable;
mod sum_as_poly;

use fix_last_variable::zerocheck_fix_last_variable_device;
use futures::future::OptionFuture;
use slop_algebra::{ExtensionField, Field, UnivariatePolynomial};
use slop_alloc::{CanCopyIntoRef, CpuBackend, ToHost};
use slop_multilinear::{
    HostEvaluationBackend, Mle, MleBaseBackend, MleEvaluationBackend, MleFixLastVariableBackend,
    PartialLagrangeBackend, PointBackend,
};
use slop_sumcheck::{
    ComponentPolyEvalBackend, SumCheckPolyFirstRoundBackend, SumcheckPolyBackend, SumcheckPolyBase,
};
use slop_tensor::TransposeBackend;
use sp1_stark::prover::{ZeroCheckPoly, ZerocheckRoundProver};
use sum_as_poly::zerocheck_sum_as_poly_in_last_variable_device;

use crate::TaskScope;

impl<K, F, EF, AirData> ComponentPolyEvalBackend<ZeroCheckPoly<K, F, EF, AirData, TaskScope>, EF>
    for TaskScope
where
    K: Field,
    F: Field,
    EF: ExtensionField<F> + ExtensionField<K>,
    AirData: Sync,
    TaskScope: TransposeBackend<EF> + TransposeBackend<F> + TransposeBackend<K>,
{
    async fn get_component_poly_evals(
        poly: &ZeroCheckPoly<K, F, EF, AirData, TaskScope>,
    ) -> Vec<EF> {
        assert!(poly.num_variables() == 0);

        let prep_columns =
            OptionFuture::from(poly.preprocessed_columns.as_ref().map(|mle| mle.to_host()))
                .await
                .transpose()
                .unwrap();

        // First get the preprocessed values.
        let prep_evals = if let Some(preprocessed_values) = prep_columns.as_ref() {
            preprocessed_values.inner().as_ref().unwrap().guts().as_slice()
        } else {
            &[]
        };

        let main_evals = poly.main_columns.to_host().await.unwrap();
        let main_evals = main_evals
            .inner()
            .as_ref()
            .map(|mle| mle.guts().as_slice().to_vec())
            .unwrap_or(vec![K::zero(); poly.main_columns.num_polynomials()]);

        // Add the main values.
        prep_evals.iter().copied().chain(main_evals).map(Into::into).collect::<Vec<_>>()
    }
}

impl<F, EF, AirData> SumCheckPolyFirstRoundBackend<ZeroCheckPoly<F, F, EF, AirData, TaskScope>, EF>
    for TaskScope
where
    F: Field,
    EF: ExtensionField<F>,
    AirData: ZerocheckRoundProver<F, F, EF, TaskScope>
        + ZerocheckRoundProver<F, EF, EF, TaskScope>
        + Clone,
    TaskScope: MleFixLastVariableBackend<F, EF>
        + MleBaseBackend<EF>
        + HostEvaluationBackend<F, F>
        + HostEvaluationBackend<F, EF>
        + MleEvaluationBackend<F, EF>
        + PartialLagrangeBackend<EF>
        + MleBaseBackend<F>
        + PartialLagrangeBackend<EF>
        + HostEvaluationBackend<F, F>
        + CanCopyIntoRef<Mle<F, TaskScope>, CpuBackend, Output = Mle<F, CpuBackend>>
        + MleBaseBackend<EF>
        + HostEvaluationBackend<F, EF>
        + HostEvaluationBackend<F, EF>
        + MleEvaluationBackend<F, EF>
        + CanCopyIntoRef<Mle<F, TaskScope>, CpuBackend, Output = Mle<F, CpuBackend>>
        + PointBackend<EF>
        + PartialLagrangeBackend<EF>
        + HostEvaluationBackend<EF, EF>
        + CanCopyIntoRef<Mle<EF, TaskScope>, CpuBackend, Output = Mle<EF, CpuBackend>>
        + PointBackend<EF>
        + MleFixLastVariableBackend<EF, EF>
        + MleBaseBackend<EF>
        + MleEvaluationBackend<EF, EF>
        + HostEvaluationBackend<EF, EF>
        + TransposeBackend<EF>
        + TransposeBackend<F>,
{
    type NextRoundPoly = ZeroCheckPoly<EF, F, EF, AirData, TaskScope>;
    async fn fix_t_variables(
        poly: ZeroCheckPoly<F, F, EF, AirData, TaskScope>,
        alpha: EF,
        t: usize,
    ) -> Self::NextRoundPoly {
        assert!(t == 1);
        zerocheck_fix_last_variable_device(poly, alpha).await
    }

    async fn sum_as_poly_in_last_t_variables(
        poly: &ZeroCheckPoly<F, F, EF, AirData, TaskScope>,
        claim: Option<EF>,
        t: usize,
    ) -> UnivariatePolynomial<EF> {
        assert!(t == 1);
        assert!(poly.num_variables() > 0);
        zerocheck_sum_as_poly_in_last_variable_device::<F, F, EF, AirData, true>(poly, claim).await
    }
}

impl<F, EF, AirData> SumcheckPolyBackend<ZeroCheckPoly<EF, F, EF, AirData, TaskScope>, EF>
    for TaskScope
where
    F: Field,
    EF: ExtensionField<F>,
    AirData: ZerocheckRoundProver<F, F, EF, TaskScope>
        + ZerocheckRoundProver<F, EF, EF, TaskScope>
        + Clone,
    TaskScope: MleFixLastVariableBackend<EF, EF>
        + MleBaseBackend<EF>
        + HostEvaluationBackend<F, EF>
        + HostEvaluationBackend<F, EF>
        + MleEvaluationBackend<F, EF>
        + CanCopyIntoRef<Mle<F, TaskScope>, CpuBackend, Output = Mle<F, CpuBackend>>
        + PointBackend<EF>
        + PartialLagrangeBackend<EF>
        + HostEvaluationBackend<EF, EF>
        + CanCopyIntoRef<Mle<EF, TaskScope>, CpuBackend, Output = Mle<EF, CpuBackend>>
        + PointBackend<EF>
        + MleFixLastVariableBackend<EF, EF>
        + MleBaseBackend<EF>
        + MleEvaluationBackend<EF, EF>
        + HostEvaluationBackend<EF, EF>
        + TransposeBackend<EF>
        + TransposeBackend<F>,
{
    async fn fix_last_variable(
        poly: ZeroCheckPoly<EF, F, EF, AirData, TaskScope>,
        alpha: EF,
    ) -> ZeroCheckPoly<EF, F, EF, AirData, TaskScope> {
        zerocheck_fix_last_variable_device(poly, alpha).await
    }

    async fn sum_as_poly_in_last_variable(
        poly: &ZeroCheckPoly<EF, F, EF, AirData, TaskScope>,
        claim: Option<EF>,
    ) -> UnivariatePolynomial<EF> {
        assert!(poly.num_variables() > 0);
        zerocheck_sum_as_poly_in_last_variable_device::<EF, F, EF, AirData, false>(poly, claim)
            .await
    }
}
