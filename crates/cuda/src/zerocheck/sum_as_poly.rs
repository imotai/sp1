use std::{
    ops::{Add, Mul, Sub},
    sync::Arc,
};

use slop_algebra::{
    interpolate_univariate_polynomial, AbstractExtensionField, ExtensionField, Field,
    UnivariatePolynomial,
};
use slop_alloc::{CanCopyFromRef, CanCopyIntoRef, CpuBackend, HasBackend, ToHost};
use slop_multilinear::{
    HostEvaluationBackend, ManuallyDroppedPaddedMle, Mle, MleBaseBackend, Padding,
    PartialLagrangeBackend, PointBackend,
};
use sp1_stark::prover::{
    increment_y_values, interpolate_last_var_padded_values, ZeroCheckPoly, ZerocheckRoundProver,
};

use crate::TaskScope;

pub async fn zerocheck_sum_as_poly_in_last_variable_device<
    K: Field + From<F> + Add<F, Output = K> + Sub<F, Output = K> + Mul<F, Output = K>,
    F: Field,
    EF: ExtensionField<F> + From<K> + ExtensionField<F> + AbstractExtensionField<K>,
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
        return UnivariatePolynomial::zero();
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
    let geq_value = poly.geq_value;
    let padded_row_adjustment = poly.padded_row_adjustment;
    let zeta = poly.zeta.clone();
    scope.spawn(move |s| async move {
        let rest_point = s.copy_to(&rest_point).await.unwrap();
        let partial_lagrange: Mle<EF, TaskScope> = Mle::partial_lagrange(&rest_point).await;
        let partial_lagrange = Arc::new(partial_lagrange);

        // Get an owned copy of the main and preprocessed columns in this new scope.
        let main_columns = unsafe { poly_main_columns.owned_unchecked_in(s.clone()) };
        let main_columns = ManuallyDroppedPaddedMle::into_inner(main_columns);
        let preprocessed_columns = poly_preprocessed_columns.as_ref().map(|mle| unsafe {
            let mle = mle.owned_unchecked_in(s.clone());
            ManuallyDroppedPaddedMle::into_inner(mle)
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

        let mut y_0 = EF::zero();
        let mut y_2 = EF::zero();
        let mut y_4 = EF::zero();

        if num_real_entries > 1 {
            (y_0, y_2, y_4) = poly_air_data
                .sum_as_poly_in_last_variable::<IS_FIRST_ROUND>(
                    partial_lagrange,
                    preprocessed_columns.clone(),
                    main_columns.clone(),
                )
                .await;
        } else {
            // TODO: do not use the unsafe copy API.
            let eq_guts_0 = partial_lagrange.guts().as_buffer()[0].copy_into_host(&s);

            // Handle the case when the zerocheck polynomial is only padded variables.

            // Get the column values where the last variable is set to 0, 2, and 4.
            let (
                preprocessed_column_vals_0,
                preprocessed_column_vals_2,
                preprocessed_column_vals_4,
            ) = if let Some(preprocessed_values) = preprocessed_columns.as_ref() {
                let preprocessed_cols =
                    preprocessed_values.inner().as_ref().unwrap().as_ref().to_host().await.unwrap();
                interpolate_last_var_padded_values(&preprocessed_cols)
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };

            let main_cols = if let Some(main_cols) = main_columns.inner().as_ref() {
                main_cols.to_host().await.unwrap()
            } else {
                let padded_values = match main_columns.padding_values() {
                    Padding::Generic(pad_vals) => pad_vals.to_host().await.unwrap(),
                    Padding::Constant((val, _, _)) => {
                        vec![*val; main_columns.num_polynomials()].into()
                    }
                };
                let num_polys = main_columns.num_polynomials();
                let guts = padded_values.into_evaluations().reshape([1, num_polys]);
                Mle::new(guts)
            };
            let (main_column_vals_0, main_column_vals_2, main_column_vals_4) =
                interpolate_last_var_padded_values(&main_cols);

            // Evaluate the constraint polynomial at the points 0, 2, and 4, and
            // add the results to the y_0, y_2, and y_4 accumulators.
            increment_y_values::<K, F, EF, AirData::Air, IS_FIRST_ROUND>(
                poly_air_data.public_values(),
                poly_air_data.powers_of_alpha(),
                poly_air_data.air(),
                &mut y_0,
                &mut y_2,
                &mut y_4,
                &preprocessed_column_vals_0,
                &main_column_vals_0,
                &preprocessed_column_vals_2,
                &main_column_vals_2,
                &preprocessed_column_vals_4,
                &main_column_vals_4,
                eq_guts_0,
            );

            // Adjust the y_0 value by the padded_row_adjustment and the geq_value.
            y_0 -= geq_value * padded_row_adjustment * eq_guts_0;

            // Adjust the y_2 value by the padded_row_adjustment and the geq_value.
            let geq_last_term_value =
                (EF::one() - geq_value) * EF::from_canonical_usize(2) + geq_value;
            y_2 -= geq_last_term_value * padded_row_adjustment * eq_guts_0;

            // Adjust the y_4 value by the padded_row_adjustment and the geq_value.
            let geq_last_term_value =
                (EF::one() - geq_value) * EF::from_canonical_usize(4) + geq_value;
            y_4 -= geq_last_term_value * padded_row_adjustment * eq_guts_0;
        }

        // Put the main and preprocessed columns back into ManuallyDroppedPaddedMle.
        let _ = ManuallyDroppedPaddedMle::new(main_columns);
        let _ = preprocessed_columns.map(ManuallyDroppedPaddedMle::new);

        // Add the point 0 and it's eval to the xs and ys.
        xs.push(EF::zero());
        if IS_FIRST_ROUND {
            ys.push(EF::zero());
        } else {
            let eq_last_term_factor = EF::one() - last;
            y_0 *= eq_last_term_factor;
            ys.push(y_0);
        }

        // Add the point 1 and it's eval to the xs and ys.
        xs.push(EF::one());
        if IS_FIRST_ROUND {
            ys.push(EF::zero());
        } else {
            let y_1 = (claim / eq_adjustment) - y_0;
            ys.push(y_1);
        }

        // Add the point 2 and it's eval to the xs and ys.
        xs.push(EF::from_canonical_usize(2));
        let eq_last_term_factor = last * F::from_canonical_usize(3) - EF::one();
        y_2 *= eq_last_term_factor;
        ys.push(y_2);

        // Add the point 4 and it's eval to the xs and ys.
        xs.push(EF::from_canonical_usize(4));
        let eq_last_term_factor = last * F::from_canonical_usize(7) - F::from_canonical_usize(3);
        y_4 *= eq_last_term_factor;
        ys.push(y_4);

        // Add the eq_first_term_root point and it's eval to the xs and ys.
        let point_elements = zeta.to_vec();
        let point_first = point_elements.last().unwrap();
        let b_const = (EF::one() - *point_first) / (EF::one() - point_first.double());
        xs.push(b_const);
        ys.push(EF::zero());

        let ys = ys.iter().map(|y| *y * eq_adjustment).collect::<Vec<_>>();

        tx.send(interpolate_univariate_polynomial(&xs, &ys)).unwrap();
    });

    // let rest_point = backend.copy_to(&rest_point).await.unwrap();
    // let partial_lagrange: Mle<EF, TaskScope> = Mle::partial_lagrange(&rest_point).await;
    // let partial_lagrange = Arc::new(partial_lagrange);

    // // For the first round, we know that at point 0 and 1, the zerocheck polynomial will evaluate to
    // // 0. For all rounds, we can find a root of the zerocheck polynomial by finding a root of
    // // the eq term in the last coord.
    // // So for the first round, we need to find an additional 2 points (since the constraint
    // // polynomial is degree 3). We calculate the eval at points 2 and 4 (since we don't need to
    // // do any multiplications when interpolating the column evals).
    // // For the other rounds, we need to find an additional 1 point since we don't know the zercheck
    // // poly eval at point 0 and 1.
    // // We calculate the eval at point 0 and then infer the eval at point 1 by the passed in claim.
    // let mut xs = Vec::new();
    // let mut ys = Vec::new();

    // let mut y_0 = EF::zero();
    // let mut y_2 = EF::zero();
    // let mut y_4 = EF::zero();
    rx.await.unwrap()
}
