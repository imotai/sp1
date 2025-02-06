use std::ops::{Add, Mul};

use itertools::Itertools;
use p3_field::ExtensionField;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use slop_algebra::Field;

use crate::air::MachineAir;

use super::{ZeroCheckPoly, ZeroCheckPolyVals};

/// This function will set the last variable to `alpha`.
pub(crate) fn fix_last_variable<
    K: Field,
    F: Field,
    EF: ExtensionField<F> + Add<K, Output = EF> + Mul<K, Output = EF> + From<K>,
    A: MachineAir<F>,
>(
    poly: ZeroCheckPoly<'_, K, F, EF, A>,
    alpha: EF,
) -> ZeroCheckPoly<'_, EF, F, EF, A> {
    let (rest, last) = poly.zeta.split_at(poly.zeta.dimension() - 1);
    let last = *last[0];

    // When we are fixing the last variable, we can factor out one of the eq_terms, as it will be a constant.
    // That constant is equal to (alpha * last) + (1 - alpha) * (1 - last).
    let eq_adjustment =
        poly.eq_adjustment * ((alpha * last) + (EF::one() - alpha) * (EF::one() - last));

    // If the columns are constants, then we know that we are now processing padded variables.
    let is_in_padded_vars = poly.main_columns.matrix_ref().height() > 1;

    let (new_preprocessed_values, new_main_values, num_padded_vars, geq_value) =
        if is_in_padded_vars {
            let folded_preprocessed_columns =
                poly.preprocessed_columns.as_ref().map(|preprocessed_values| {
                    ZeroCheckPolyVals::Owned(fold_non_padded_values(
                        preprocessed_values.matrix_ref(),
                        alpha,
                    ))
                });

            let folded_main_columns = ZeroCheckPolyVals::Owned(fold_non_padded_values(
                poly.main_columns.matrix_ref(),
                alpha,
            ));

            (folded_preprocessed_columns, folded_main_columns, poly.num_padded_vars, EF::zero())
        } else {
            // Handling padded variables.
            let preprocessed_values =
                poly.preprocessed_columns.as_ref().map(|preprocessed_values| {
                    ZeroCheckPolyVals::Owned(fold_padded_values(
                        preprocessed_values.matrix_ref(),
                        alpha,
                    ))
                });

            let main_values =
                ZeroCheckPolyVals::Owned(fold_padded_values(poly.main_columns.matrix_ref(), alpha));

            // We can factor out one term of the geq polynomial.  We can think of the guts of the geq
            // polynomial as being 0 for all non padded values and 1 for all padded values.
            // When we are processing non padded values, the fix_as_last_variable will always be 0,
            // regardless of the value of alpha.
            // When we are processing padded variables, the new geq_value will be the interpolated value
            // at the point alpha where the value at 0 is the old geq_value and the value at 1 is 1.
            let geq_value = (EF::one() - poly.geq_value) * alpha + poly.geq_value;

            (preprocessed_values, main_values, poly.num_padded_vars - 1, geq_value)
        };

    let ret = ZeroCheckPoly::<EF, F, EF, _>::new(
        poly.air,
        rest,
        new_preprocessed_values,
        new_main_values,
        eq_adjustment,
        geq_value,
        poly.public_values,
        poly.powers_of_alpha,
        num_padded_vars,
        poly.padded_row_adjustment,
    );

    ret
}

/// This function will fold the preprocessed and main columns `ZerocheckPolys` that have non-padded variables.
#[inline]
fn fold_non_padded_values<
    K: Field,
    F: Field,
    EF: ExtensionField<F> + Add<K, Output = EF> + Mul<K, Output = EF> + From<K>,
>(
    rows: &RowMajorMatrix<K>,
    alpha: EF,
) -> RowMajorMatrix<EF> {
    let values = rows
        .rows()
        .chunks(2)
        .into_iter()
        .flat_map(|mut chunk| {
            let val_0 = chunk.next().expect("chunk must have 2 rows");
            let val_1 = chunk.next().expect("chunk must have 2 rows");

            val_0.zip_eq(val_1).map(|(val_0, val_1)| alpha * (val_1 - val_0) + val_0)
        })
        .collect::<Vec<_>>();

    RowMajorMatrix::new(values, rows.width())
}

/// This function will fold the preprocessed and main columns `ZerocheckPolys` that have only padded variables.
#[inline]
fn fold_padded_values<
    K: Field,
    F: Field,
    EF: ExtensionField<F> + Add<K, Output = EF> + Mul<K, Output = EF> + From<K>,
>(
    rows: &RowMajorMatrix<K>,
    alpha: EF,
) -> RowMajorMatrix<EF> {
    assert!(rows.height() == 1);

    let values = rows.row(0).map(|val| (EF::one() - alpha) * val).collect::<Vec<EF>>();
    RowMajorMatrix::new(values, rows.width())
}
