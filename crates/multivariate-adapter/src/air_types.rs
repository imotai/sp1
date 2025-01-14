// use p3_air::ExtensionBuilder;
// use serde::{Deserialize, Serialize};

// use p3_matrix::{dense::RowMajorMatrixView, stack::VerticalPair, Matrix};

// #[derive(Debug, Clone, Serialize, Deserialize)]
// #[serde(bound(serialize = "T: Serialize"))]
// #[serde(bound(deserialize = "T: Deserialize<'de>"))]
// pub struct AirOpenedValues<T> {
//     pub local: Vec<T>,
//     pub next: Vec<T>,
// }

// impl<T: Send + Sync + Clone> AirOpenedValues<T> {
//     #[must_use]
//     pub fn view(&self) -> VerticalPair<RowMajorMatrixView<'_, T>, RowMajorMatrixView<'_, T>> {
//         let a = RowMajorMatrixView::new_row(&self.local);
//         let b = RowMajorMatrixView::new_row(&self.next);
//         VerticalPair::new(a, b)
//     }
// }

// #[derive(Debug, Clone, Serialize, Deserialize)]
// #[serde(bound(serialize = "T: Serialize"))]
// #[serde(bound(deserialize = "T: Deserialize<'de>"))]
// pub struct ChipOpenedValues<T> {
//     pub main: AirOpenedValues<T>,
//     pub adapter: AirOpenedValues<T>,
//     pub quotient: Vec<Vec<T>>,
//     pub log_degree: usize,
// }

// pub trait MultivariateEvaluationAirBuilder: ExtensionBuilder {
//     type MP: Matrix<Self::VarEF>;

//     type Sum: Into<Self::ExprEF> + Copy;

//     type RandomVar: Into<Self::ExprEF> + Copy;

//     /// The multivariate adapter columns (eq polynomial and cumulative sum).
//     fn adapter(&self) -> Self::MP;

//     /// The evaluation point of the multilinear polynomial. Unused for now but will get used when we
//     /// constrain the eq evaluation.
//     fn _evaluation_point(&self) -> Vec<Self::Sum>;

//     /// The expected evaluation of the multilinear. (Checked to be the last entry of the cumulative
//     /// sum).
//     fn expected_evals(&self) -> &[Self::Sum];

//     /// The random challenge used to batch the evaluations.
//     fn batch_randomness(&self) -> Self::RandomVar;
// }
