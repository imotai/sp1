use std::{
    mem::ManuallyDrop,
    ops::{Add, Mul, Sub},
    sync::Arc,
};

use slop_algebra::{
    interpolate_univariate_polynomial, AbstractExtensionField, ExtensionField, Field,
    UnivariatePolynomial,
};
use slop_alloc::{CanCopyFromRef, CanCopyIntoRef, CpuBackend, HasBackend};
use slop_multilinear::{
    HostEvaluationBackend, Mle, MleBaseBackend, PartialLagrangeBackend, PointBackend,
};
use slop_sumcheck::SumcheckPolyBase;
use sp1_stark::prover::{ZeroCheckPoly, ZerocheckRoundProver};
use sp1_stark::MAX_CONSTRAINT_DEGREE;

use crate::TaskScope;

pub async fn zerocheck_sum_as_poly_in_last_variable_device<
    K: Field + From<F> + Add<F, Output = K> + Sub<F, Output = K> + Mul<F, Output = K>,
    F: Field,
    EF: ExtensionField<F> + ExtensionField<K> + ExtensionField<F> + AbstractExtensionField<K>,
    AirData: ZerocheckRoundProver<F, K, EF, TaskScope> + Clone,
    const IS_FIRST_ROUND: bool,
>(
    poly: &ZeroCheckPoly<K, F, EF, AirData, TaskScope>,
    claim: Option<EF>,
) -> UnivariatePolynomial<EF>
where
    TaskScope: MleBaseBackend<K>
        + PartialLagrangeBackend<EF>
        + HostEvaluationBackend<K, K>
        + CanCopyIntoRef<Mle<K, TaskScope>, CpuBackend, Output = Mle<K, CpuBackend>>
        + PointBackend<EF>,
{
    let num_real_entries = poly.main_columns.num_real_entries();
    if num_real_entries == 0 {
        // The +1 is from the zerocheck poly's eq.
        return UnivariatePolynomial::zero(MAX_CONSTRAINT_DEGREE + 1);
    }

    let claim = claim.expect("claim must be provided");

    let (rest_point, last) = poly.zeta.split_at(poly.zeta.dimension() - 1);
    let last = *last[0];

    // TODO:  Optimization of computing this once per zerocheck sumcheck.
    let scope = poly.main_columns.backend();

    let (tx, rx) = tokio::sync::oneshot::channel();
    let poly_main_columns = poly.main_columns.clone();
    let poly_preprocessed_columns = poly.preprocessed_columns.clone();
    let poly_air_data = poly.air_data.clone();
    let eq_adjustment = poly.eq_adjustment;
    let padded_row_adjustment = poly.padded_row_adjustment;
    let zeta = poly.zeta.clone();
    let virtual_geq = poly.virtual_geq;
    let num_variables = poly.num_variables();
    scope.spawn(move |s| async move {
        let rest_point = s.copy_to(&rest_point).await.unwrap();
        let partial_lagrange: Mle<EF, TaskScope> = Mle::partial_lagrange(&rest_point).await;
        let partial_lagrange = Arc::new(partial_lagrange);

        // Get an owned copy of the main and preprocessed columns in this new scope.
        let main_columns = unsafe { poly_main_columns.owned_unchecked_in(s.clone()) };
        let main_columns = ManuallyDrop::into_inner(main_columns);
        let preprocessed_columns = poly_preprocessed_columns.as_ref().map(|mle| unsafe {
            let mle = mle.owned_unchecked_in(s.clone());
            ManuallyDrop::into_inner(mle)
        });

        // For the first round, we know that at point 0 and 1, the zerocheck polynomial will evaluate to
        // 0. For all rounds, we can find a root of the zerocheck polynomial by finding a root of
        // the eq term in the last coord.
        // So for the first round, we need to find an additional 2 points (since the constraint
        // polynomial is degree 3). We calculate the eval at points 2 and 4 (since we don't need to
        // do any multiplications when interpolating the column evals).
        // For the other rounds, we need to find an additional 1 point since we don't know the zercheck
        // poly eval at point 0 and 1.
        // We calculate the eval at point 0 and then infer the eval at point 1 by the passed in claim.
        let mut xs = Vec::new();
        let mut ys = Vec::new();

        let (mut y_0, mut y_2, mut y_4) = poly_air_data
            .sum_as_poly_in_last_variable::<IS_FIRST_ROUND>(
                partial_lagrange.clone(),
                preprocessed_columns.clone(),
                main_columns.clone(),
            )
            .await;

        // Put the main and preprocessed columns back into ManuallyDrop.
        let _ = ManuallyDrop::new(main_columns);
        let _ = preprocessed_columns.map(ManuallyDrop::new);

        let threshold_half = num_real_entries.div_ceil(2) - 1;
        let msb_lagrange_eval: EF = eq_adjustment
            * if threshold_half < (1 << (num_variables - 1)) {
                partial_lagrange.guts().as_buffer()[threshold_half]
                    .copy_into_host(partial_lagrange.backend())
            } else {
                EF::zero()
            };

        let virtual_0 = virtual_geq.fix_last_variable(EF::zero()).eval_at_usize(threshold_half);
        let virtual_2 = virtual_geq.fix_last_variable(EF::two()).eval_at_usize(threshold_half);
        let virtual_4 = virtual_geq
            .fix_last_variable(EF::from_canonical_usize(4))
            .eval_at_usize(threshold_half);

        // Add the point 0 and it's eval to the xs and ys.
        xs.push(EF::zero());

        let eq_last_term_factor = EF::one() - last;
        y_0 *= eq_last_term_factor * eq_adjustment;
        y_0 -= padded_row_adjustment * virtual_0 * msb_lagrange_eval * eq_last_term_factor;
        ys.push(y_0);

        // Add the point 1 and it's eval to the xs and ys.
        xs.push(EF::one());

        let y_1 = claim - y_0;
        ys.push(y_1);

        // Add the point 2 and it's eval to the xs and ys.
        xs.push(EF::from_canonical_usize(2));
        let eq_last_term_factor = last * F::from_canonical_usize(3) - EF::one();
        y_2 *= eq_last_term_factor * eq_adjustment;
        y_2 -= padded_row_adjustment * virtual_2 * msb_lagrange_eval * eq_last_term_factor;
        ys.push(y_2);

        // Add the point 4 and it's eval to the xs and ys.
        xs.push(EF::from_canonical_usize(4));
        let eq_last_term_factor = last * F::from_canonical_usize(7) - F::from_canonical_usize(3);
        y_4 *= eq_last_term_factor * eq_adjustment;
        y_4 -= padded_row_adjustment * virtual_4 * msb_lagrange_eval * eq_last_term_factor;
        ys.push(y_4);

        // Add the eq_first_term_root point and it's eval to the xs and ys.
        let point_elements = zeta.to_vec();
        let point_first = point_elements.last().unwrap();
        let b_const = (EF::one() - *point_first) / (EF::one() - point_first.double());
        xs.push(b_const);
        ys.push(EF::zero());

        // let ys = ys.iter().map(|y| *y * eq_adjustment).collect::<Vec<_>>();

        tx.send(interpolate_univariate_polynomial(&xs, &ys)).unwrap();
    });

    rx.await.unwrap()
}
