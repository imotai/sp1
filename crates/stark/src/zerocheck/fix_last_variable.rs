use std::{
    ops::{Add, Mul},
    sync::Arc,
};

use futures::future::OptionFuture;
use p3_field::ExtensionField;
use slop_algebra::Field;
use slop_multilinear::Mle;
use slop_tensor::Tensor;

use super::{sum_as_poly::ZerocheckProver, ZeroCheckPoly};

/// This function will set the last variable to `alpha`.
pub(crate) async fn fix_last_variable<
    K: Field,
    F: Field,
    EF: ExtensionField<F> + Add<K, Output = EF> + Mul<K, Output = EF> + From<K> + ExtensionField<K>,
    A: ZerocheckProver<F, K, EF>,
>(
    poly: ZeroCheckPoly<K, F, EF, A>,
    alpha: EF,
) -> ZeroCheckPoly<EF, F, EF, A> {
    let (rest, last) = poly.zeta.split_at(poly.zeta.dimension() - 1);
    let last = *last[0];

    // When we are fixing the last variable, we can factor out one of the eq_terms, as it will be a constant.
    // That constant is equal to (alpha * last) + (1 - alpha) * (1 - last).
    let eq_adjustment =
        poly.eq_adjustment * ((alpha * last) + (EF::one() - alpha) * (EF::one() - last));

    let has_non_padded_vars = poly.main_columns.num_variables() > 0;

    let (new_preprocessed_values, new_main_values, num_padded_vars, geq_value) =
        if has_non_padded_vars {
            let folded_preprocessed_columns =
                poly.preprocessed_columns.as_ref().map(|preprocessed_values| async move {
                    Arc::new(preprocessed_values.fix_last_variable(alpha).await)
                });
            let folded_preprocessed_columns = OptionFuture::from(folded_preprocessed_columns).await;

            let folded_main_columns = Arc::new(poly.main_columns.fix_last_variable(alpha).await);

            (folded_preprocessed_columns, folded_main_columns, poly.num_padded_vars, EF::zero())
        } else {
            // Handling padded variables.
            let preprocessed_values =
                poly.preprocessed_columns.as_ref().map(|preprocessed_values| {
                    Arc::new(fold_padded_values::<K, F, EF>(preprocessed_values, alpha))
                });

            let main_values =
                Arc::new(fold_padded_values::<K, F, EF>(poly.main_columns.as_ref(), alpha));

            // We can factor out one term of the geq polynomial.  We can think of the guts of the geq
            // polynomial as being 0 for all non padded values and 1 for all padded values.
            // When we are processing non padded values, the fix_as_last_variable will always be 0,
            // regardless of the value of alpha.
            // When we are processing padded variables, the new geq_value will be the interpolated value
            // at the point alpha where the value at 0 is the old geq_value and the value at 1 is 1.
            let geq_value = (EF::one() - poly.geq_value) * alpha + poly.geq_value;

            (preprocessed_values, main_values, poly.num_padded_vars - 1, geq_value)
        };

    ZeroCheckPoly::<EF, F, EF, _>::new(
        poly.air_data,
        rest,
        new_preprocessed_values,
        new_main_values,
        eq_adjustment,
        geq_value,
        num_padded_vars,
        poly.padded_row_adjustment,
    )
}

/// This function will fold the preprocessed and main columns `ZerocheckPolys` that have only padded variables.
#[inline]
fn fold_padded_values<
    K: Field,
    F: Field,
    EF: ExtensionField<F> + Add<K, Output = EF> + Mul<K, Output = EF> + From<K> + ExtensionField<K>,
>(
    rows: &Mle<K>,
    alpha: EF,
) -> Mle<EF> {
    assert!(rows.num_variables() == 0);

    let values =
        rows.guts().as_slice().iter().map(|val| (EF::one() - alpha) * *val).collect::<Vec<EF>>();
    let num_values = values.len();
    Mle::new(Tensor::from(values).reshape([1, num_values]))
}
