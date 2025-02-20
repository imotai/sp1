use std::ops::{Add, Mul, Sub};

use itertools::Itertools;
use p3_air::Air;
use p3_field::{AbstractExtensionField, ExtensionField};
use p3_matrix::dense::RowMajorMatrixView;
use slop_algebra::{interpolate_univariate_polynomial, Field, UnivariatePolynomial};
use slop_alloc::CpuBackend;
use slop_multilinear::Mle;

use crate::{air::MachineAir, ConstraintSumcheckFolder};

use super::{utils::ZeroCheckPolyVals, SumAsPolyInLastVariableBackend, ZeroCheckPoly};

/// This function will calculate the univariate polynomial where all variables other than the last are
/// summed on the boolean hypercube and the last variable is left as a free variable.
/// TODO:  Add flexibility to support degree 2 and degree 3 constraint polynomials.
#[allow(clippy::too_many_lines)]
pub(crate) fn sum_as_poly_in_last_variable<
    'a,
    K: Field + From<F> + Add<F, Output = K> + Sub<F, Output = K> + Mul<F, Output = K>,
    F: Field,
    EF: ExtensionField<F> + From<K> + ExtensionField<F> + AbstractExtensionField<K>,
    A: for<'b> Air<ConstraintSumcheckFolder<'b, F, K, EF>> + MachineAir<F>,
    const IS_FIRST_ROUND: bool,
>(
    poly: &ZeroCheckPoly<'a, K, F, EF, A>,
    claim: Option<EF>,
) -> UnivariatePolynomial<EF> {
    let claim = claim.expect("claim must be provided");

    let num_non_padded_vars = poly.main_columns.mle_ref().num_variables();

    let (rest_point, last) = poly.zeta.split_at(poly.zeta.dimension() - 1);
    let last = *last[0];

    // TODO:  Optimization of computing this once per zerocheck sumcheck.
    let partial_lagrange: Mle<EF> = Mle::partial_lagrange(&rest_point);

    // For the first round, we know that at point 0 and 1, the zerocheck polynomial will evaluate to 0.
    // For all rounds, we can find a root of the zerocheck polynomial by finding a root of the eq term
    // in the last coord.
    // So for the first round, we need to find an additional 2 points (since the constraint polynomial is degree 3).
    // We calculate the eval at points 2 and 4 (since we don't need to do any multiplications
    // when interpolating the column evals).
    // For the other rounds, we need to find an additional 1 point since we don't know the zercheck poly eval at
    // point 0 and 1.
    // We calculate the eval at point 0 and then infer the eval at point 1 by the passed in claim.
    let mut xs = Vec::new();
    let mut ys = Vec::new();

    let mut y_0 = EF::zero();
    let mut y_2 = EF::zero();
    let mut y_4 = EF::zero();

    if num_non_padded_vars > 0 {
        // Calculate the number of summed over terms that are non-padded.  The padded values will be zero,
        // since the adjusted constraint polynomial will be zero.
        let num_non_padded_terms = 2usize.pow(num_non_padded_vars - 1);

        (y_0, y_2, y_4) = <CpuBackend as SumAsPolyInLastVariableBackend<
            K,
            F,
            EF,
            A,
            IS_FIRST_ROUND,
        >>::sum_as_poly_in_last_variable(
            &partial_lagrange,
            poly.preprocessed_columns.as_ref().map(ZeroCheckPolyVals::mle_ref),
            poly.main_columns.mle_ref(),
            num_non_padded_terms,
            poly.public_values,
            &poly.powers_of_alpha,
            poly.air,
        );
    } else {
        let eq_guts = partial_lagrange.into_guts().into_buffer().into_vec();

        // Handle the case when the zerocheck polynomial is only padded variables.

        // Get the column values where the last variable is set to 0, 2, and 4.
        let (preprocessed_column_vals_0, preprocessed_column_vals_2, preprocessed_column_vals_4) =
            if let Some(preprocessed_values) = poly.preprocessed_columns.as_ref() {
                interpolate_last_var_padded_values(preprocessed_values.mle_ref())
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };

        let (main_column_vals_0, main_column_vals_2, main_column_vals_4) =
            interpolate_last_var_padded_values(poly.main_columns.mle_ref());

        // Evaluate the constraint polynomial at the points 0, 2, and 4, and
        // add the results to the y_0, y_2, and y_4 accumulators.
        increment_y_values::<K, F, EF, A, IS_FIRST_ROUND>(
            poly.public_values,
            &poly.powers_of_alpha,
            poly.air,
            &mut y_0,
            &mut y_2,
            &mut y_4,
            &preprocessed_column_vals_0,
            &main_column_vals_0,
            &preprocessed_column_vals_2,
            &main_column_vals_2,
            &preprocessed_column_vals_4,
            &main_column_vals_4,
            eq_guts[0],
        );

        // Adjust the y_0 value by the padded_row_adjustment and the geq_value.
        y_0 -= poly.geq_value * poly.padded_row_adjustment * eq_guts[0];

        // Adjust the y_2 value by the padded_row_adjustment and the geq_value.
        let geq_last_term_value =
            (EF::one() - poly.geq_value) * EF::from_canonical_usize(2) + poly.geq_value;
        y_2 -= geq_last_term_value * poly.padded_row_adjustment * eq_guts[0];

        // Adjust the y_4 value by the padded_row_adjustment and the geq_value.
        let geq_last_term_value =
            (EF::one() - poly.geq_value) * EF::from_canonical_usize(4) + poly.geq_value;
        y_4 -= geq_last_term_value * poly.padded_row_adjustment * eq_guts[0];
    }

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
        let y_1 = (claim / poly.eq_adjustment) - y_0;
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
    let point_elements = poly.zeta.to_vec();
    let point_first = point_elements.last().unwrap();
    let b_const = (EF::one() - *point_first) / (EF::one() - point_first.double());
    xs.push(b_const);
    ys.push(EF::zero());

    let ys = ys.iter().map(|y| *y * poly.eq_adjustment).collect::<Vec<_>>();

    interpolate_univariate_polynomial(&xs, &ys)
}

/// This function will calculate the column values where the last variable is set to 0, 2, and 4
/// and it's a non-padded variable.
fn interpolate_last_var_non_padded_values<K: Field, const IS_FIRST_ROUND: bool>(
    values: &Mle<K>,
    i: usize,
) -> (Vec<K>, Vec<K>, Vec<K>) {
    let num_variables = values.num_variables();
    assert!(2 * i + 1 < 2usize.pow(num_variables));
    let row_0 = values.guts().get(2 * i).unwrap().as_slice();
    let row_1 = values.guts().get(2 * i + 1).unwrap().as_slice();

    let mut vals_0 = Vec::with_capacity(values.num_polynomials());
    let mut vals_2 = Vec::with_capacity(values.num_polynomials());
    let mut vals_4 = Vec::with_capacity(values.num_polynomials());

    for (row_0_val, row_1_val) in row_0.iter().zip_eq(row_1.iter()) {
        let slope = *row_1_val - *row_0_val;
        let slope_times_2 = slope + slope;
        let slope_times_4 = slope_times_2 + slope_times_2;

        if !IS_FIRST_ROUND {
            vals_0.push(*row_0_val);
        }
        vals_2.push(slope_times_2 + *row_0_val);
        vals_4.push(slope_times_4 + *row_0_val);
    }

    (vals_0, vals_2, vals_4)
}

impl<
        K: Field + From<F> + Add<F, Output = K> + Sub<F, Output = K> + Mul<F, Output = K>,
        F: Field,
        EF: ExtensionField<F> + From<K> + ExtensionField<F> + AbstractExtensionField<K>,
        A: for<'b> Air<ConstraintSumcheckFolder<'b, F, K, EF>> + MachineAir<F>,
        const IS_FIRST_ROUND: bool,
    > SumAsPolyInLastVariableBackend<K, F, EF, A, IS_FIRST_ROUND> for CpuBackend
{
    fn sum_as_poly_in_last_variable(
        partial_lagrange: &Mle<EF>,
        preprocessed_values: Option<&Mle<K>>,
        main_values: &Mle<K>,
        num_non_padded_terms: usize,
        public_values: &[F],
        powers_of_alpha: &[EF],
        air: &A,
    ) -> (EF, EF, EF) {
        let mut y_0 = EF::zero();
        let mut y_2 = EF::zero();
        let mut y_4 = EF::zero();

        let eq_guts = partial_lagrange.guts().as_buffer().as_slice();

        // Handle the case when the zerocheck polynomial has non-padded variables.
        for (i, eq) in eq_guts.iter().enumerate().take(num_non_padded_terms) {
            // Get the column values where the last variable is set to 0, 2, and 4.
            let (
                preprocessed_column_vals_0,
                preprocessed_column_vals_2,
                preprocessed_column_vals_4,
            ) = if let Some(preprocessed_values) = preprocessed_values {
                interpolate_last_var_non_padded_values::<K, IS_FIRST_ROUND>(preprocessed_values, i)
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };

            let (main_column_vals_0, main_column_vals_2, main_column_vals_4) =
                interpolate_last_var_non_padded_values::<K, IS_FIRST_ROUND>(main_values, i);

            // Evaluate the constraint polynomial at the points 0, 2, and 4, and
            // add the results to the y_0, y_2, and y_4 accumulators.
            increment_y_values::<K, F, EF, A, IS_FIRST_ROUND>(
                public_values,
                powers_of_alpha,
                air,
                &mut y_0,
                &mut y_2,
                &mut y_4,
                &preprocessed_column_vals_0,
                &main_column_vals_0,
                &preprocessed_column_vals_2,
                &main_column_vals_2,
                &preprocessed_column_vals_4,
                &main_column_vals_4,
                *eq,
            );
        }

        (y_0, y_2, y_4)
    }
}

/// This function will calculate the column values where the last variable is set to 0, 2, and 4
/// and it's a padded variable.  The `row_0` values are taken from the values matrix (which should
/// have a height of 1).  The `row_1` values are all zero.
fn interpolate_last_var_padded_values<K: Field>(values: &Mle<K>) -> (Vec<K>, Vec<K>, Vec<K>) {
    let row_0 = values.guts().as_slice().iter();
    let vals_0 = row_0.clone().copied().collect::<Vec<_>>();
    let vals_2 = row_0.clone().map(|val| -(*val)).collect::<Vec<_>>();
    let vals_4 = row_0.clone().map(|val| -K::from_canonical_usize(3) * (*val)).collect::<Vec<_>>();

    (vals_0, vals_2, vals_4)
}

#[allow(clippy::too_many_arguments)]
fn increment_y_values<
    'a,
    K: Field + From<F> + Add<F, Output = K> + Sub<F, Output = K> + Mul<F, Output = K>,
    F: Field,
    EF: ExtensionField<F> + From<K> + ExtensionField<F> + AbstractExtensionField<K>,
    A: for<'b> Air<ConstraintSumcheckFolder<'b, F, K, EF>> + MachineAir<F>,
    const IS_FIRST_ROUND: bool,
>(
    public_values: &[F],
    powers_of_alpha: &[EF],
    air: &A,
    y_0: &mut EF,
    y_2: &mut EF,
    y_4: &mut EF,
    preprocessed_column_vals_0: &[K],
    main_column_vals_0: &[K],
    preprocessed_column_vals_2: &[K],
    main_column_vals_2: &[K],
    preprocessed_column_vals_4: &[K],
    main_column_vals_4: &[K],
    eq: EF,
) {
    // Add to the y_0 accumulator.
    if !IS_FIRST_ROUND {
        let mut folder = ConstraintSumcheckFolder {
            preprocessed: RowMajorMatrixView::new_row(preprocessed_column_vals_0),
            main: RowMajorMatrixView::new_row(main_column_vals_0),
            accumulator: EF::zero(),
            public_values,
            constraint_index: 0,
            powers_of_alpha,
        };
        air.eval(&mut folder);
        *y_0 += folder.accumulator * eq;
    }

    // Add to the y_2 accumulator.
    let mut folder = ConstraintSumcheckFolder {
        preprocessed: RowMajorMatrixView::new_row(preprocessed_column_vals_2),
        main: RowMajorMatrixView::new_row(main_column_vals_2),
        accumulator: EF::zero(),
        public_values,
        constraint_index: 0,
        powers_of_alpha,
    };
    air.eval(&mut folder);
    *y_2 += folder.accumulator * eq;

    // Add to the y_4 accumulator.
    let mut folder = ConstraintSumcheckFolder {
        preprocessed: RowMajorMatrixView::new_row(preprocessed_column_vals_4),
        main: RowMajorMatrixView::new_row(main_column_vals_4),
        accumulator: EF::zero(),
        public_values,
        constraint_index: 0,
        powers_of_alpha,
    };
    air.eval(&mut folder);
    *y_4 += folder.accumulator * eq;
}
