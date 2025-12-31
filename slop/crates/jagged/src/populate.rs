use std::sync::Arc;

use slop_commit::Rounds;
use tokio::sync::oneshot;

use slop_algebra::Field;
use slop_multilinear::Point;

use crate::{JaggedLittlePolynomialProverParams, LongMle};

pub(crate) async fn partial_jagged_multilinear<F: Field>(
    jagged_params: &JaggedLittlePolynomialProverParams,
    _row_data: Rounds<Arc<Vec<usize>>>,
    _column_data: Rounds<Arc<Vec<usize>>>,
    z_row: &Point<F>,
    z_col: &Point<F>,
    num_components: usize,
) -> LongMle<F> {
    let (tx, rx) = oneshot::channel();
    assert_eq!(num_components, 1, "only one component is supported for now");

    let z_row = z_row.clone();
    let z_col = z_col.clone();
    let jagged_params = jagged_params.clone();
    slop_futures::rayon::spawn(move || {
        let values = jagged_params.partial_jagged_little_polynomial_evaluation(&z_row, &z_col);
        let log_stacking_height = values.num_variables();
        let mle = LongMle::from_components(vec![values], log_stacking_height);
        tx.send(mle).unwrap();
    });

    rx.await.unwrap()
}
