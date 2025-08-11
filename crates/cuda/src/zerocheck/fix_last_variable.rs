use std::{
    mem::ManuallyDrop,
    ops::{Add, Mul},
};

use futures::future::OptionFuture;
use slop_algebra::{ExtensionField, Field};
use slop_alloc::HasBackend;
use slop_multilinear::{
    HostEvaluationBackend, MleBaseBackend, MleEvaluationBackend, MleFixLastVariableBackend,
};
use sp1_hypercube::prover::{ZeroCheckPoly, ZerocheckRoundProver};

use crate::{sync::CudaSend, TaskScope};

/// This function will set the last variable to `alpha`.
pub async fn zerocheck_fix_last_variable_device<
    K: Field,
    F: Field,
    EF: ExtensionField<F> + Add<K, Output = EF> + Mul<K, Output = EF> + From<K> + ExtensionField<K>,
    A: ZerocheckRoundProver<F, K, EF, TaskScope>,
>(
    poly: ZeroCheckPoly<K, F, EF, A, TaskScope>,
    alpha: EF,
) -> ZeroCheckPoly<EF, F, EF, A, TaskScope>
where
    TaskScope: MleFixLastVariableBackend<K, EF>
        + MleBaseBackend<EF>
        + MleEvaluationBackend<K, EF>
        + HostEvaluationBackend<K, K>,
{
    let num_real_entries = poly.main_columns.num_real_entries();
    let main_columns = poly.main_columns.clone();
    let preprocessed_columns = poly.preprocessed_columns.clone();
    let (tx, rx) = tokio::sync::oneshot::channel();

    let scope = poly.main_columns.backend().clone();
    let original_scope = scope.clone();
    scope
        .run_in_place(move |s| async move {
            let t = original_scope;
            let main_columns = unsafe { main_columns.owned_unchecked_in(s.clone()) };
            let preprocessed_columns =
                unsafe { preprocessed_columns.map(|m| m.owned_unchecked_in(s.clone())) };
            let main_columns = ManuallyDrop::new(main_columns.fix_last_variable(alpha).await);
            let preprocessed_columns =
                OptionFuture::from(preprocessed_columns.as_ref().map(|mle| async {
                    let mle = ManuallyDrop::new(mle.fix_last_variable(alpha).await);
                    let mle = unsafe { mle.send_to_scope(&t) };
                    ManuallyDrop::into_inner(mle)
                }))
                .await;

            let main_columns = unsafe { main_columns.send_to_scope(&t) };
            let main_columns = ManuallyDrop::into_inner(main_columns);
            tx.send((preprocessed_columns, main_columns)).unwrap();
        })
        .await
        .unwrap();
    let (preprocessed_columns, main_columns) = rx.await.unwrap();

    if num_real_entries == 0 {
        // If the chip is pure padding, it's contribution to sumcheck is just zero, we don't need
        // to propagate any eq_adjustment or any other data relevant to the sumcheck.
        return ZeroCheckPoly::new(
            poly.air_data,
            poly.zeta,
            preprocessed_columns,
            main_columns,
            poly.eq_adjustment,
            poly.geq_value,
            poly.padded_row_adjustment,
            poly.virtual_geq.fix_last_variable(alpha),
        );
    }

    let (rest, last) = poly.zeta.split_at(poly.zeta.dimension() - 1);
    let last = *last[0];

    // When we are fixing the last variable, we can factor out one of the eq_terms, as it will be a
    // constant. That constant is equal to (alpha * last) + (1 - alpha) * (1 - last).
    let eq_adjustment =
        poly.eq_adjustment * ((alpha * last) + (EF::one() - alpha) * (EF::one() - last));

    let has_non_padded_vars = num_real_entries > 1;

    let geq_value = if has_non_padded_vars {
        EF::zero()
    } else {
        (EF::one() - poly.geq_value) * alpha + poly.geq_value
    };

    ZeroCheckPoly::new(
        poly.air_data,
        rest,
        preprocessed_columns,
        main_columns,
        eq_adjustment,
        geq_value,
        poly.padded_row_adjustment,
        poly.virtual_geq.fix_last_variable(alpha),
    )
}
