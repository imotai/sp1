use itertools::{izip, Itertools};
use serde::{Deserialize, Serialize};
use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use thiserror::Error;
use tracing::instrument;

use p3_air::{Air, AirBuilder, BaseAir, ExtensionBuilder};
use p3_commit::Pcs;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::{StarkGenericConfig, Val};

use slop_algebra::AbstractField;
use slop_multilinear::{partial_lagrange_eval, Point};

use crate::{air_types::ChipOpenedValues, verifier::AdapterAir, MultivariateEvaluationAirBuilder};

type Com<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Commitment;

pub type Dom<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Domain;

/// A univariate-to-multivariate adapter generic in a STARK configuration type.
#[derive(Default, Debug)]
pub struct MultivariateAdapterPCS<SC: StarkGenericConfig> {
    /// The STARK config.
    pub(crate) config: SC,

    /// The number of multivariates to be opened.
    pub(crate) batch_size: usize,
}

impl<SC: StarkGenericConfig> MultivariateAdapterPCS<SC> {
    /// Access the STARK config.
    pub fn config(&self) -> &SC {
        &self.config
    }

    /// Construct a new adapter PCS.
    pub fn new(config: SC, batch_size: usize) -> Self {
        Self { config, batch_size }
    }
}

/// The proof struct for the multivariate opening.
pub struct MultivariateAdapterProof<SC: StarkGenericConfig> {
    pub adapter_commit: Com<SC>,
    pub quotient_commit: Com<SC>,
    pub opening_proof: <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Proof,
    pub opened_values: Vec<ChipOpenedValues<SC::Challenge>>,
}

#[derive(Debug, Error)]
pub enum MultivariateAdapterError {
    #[error("Verification error")]
    Verification,

    #[error("Pcs error")]
    PcsError,

    #[error("Shape mismatch")]
    ShapeMismatch,
}

pub struct MultivariateAdapterAir {
    pub batch_size: usize,
}
pub const NUM_ADAPTER_COLS: usize = size_of::<MultivariateAdapterCols<u8>>();

#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct MultivariateAdapterCols<F> {
    /// The column of the evaluations of eq(i, eval_point) as i varies over the Boolean hypercube.
    pub lagrange_eval: F,

    /// A prefix-sum column to compute the inner product of the previous two columns.
    pub accum: F,
}

impl<F> BaseAir<F> for MultivariateAdapterAir {
    fn width(&self) -> usize {
        self.batch_size
    }
}

impl<F> AdapterAir<F> for MultivariateAdapterAir {
    fn adapter_width(&self) -> usize {
        NUM_ADAPTER_COLS
    }
}

impl<F> Borrow<MultivariateAdapterCols<F>> for [F] {
    fn borrow(&self) -> &MultivariateAdapterCols<F> {
        debug_assert_eq!(self.len(), NUM_ADAPTER_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<MultivariateAdapterCols<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<F> BorrowMut<MultivariateAdapterCols<F>> for [F] {
    fn borrow_mut(&mut self) -> &mut MultivariateAdapterCols<F> {
        debug_assert_eq!(self.len(), NUM_ADAPTER_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<MultivariateAdapterCols<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

impl<AB: AirBuilder + MultivariateEvaluationAirBuilder> Air<AB> for MultivariateAdapterAir {
    fn eval(&self, builder: &mut AB) {
        let adapter = builder.adapter();
        let main = builder.main();

        let (local_adapter, next_adapter) = (adapter.row_slice(0), adapter.row_slice(1));
        let local_adapter: &MultivariateAdapterCols<AB::VarEF> = (*local_adapter).borrow();
        let next_adapter: &MultivariateAdapterCols<AB::VarEF> = (*next_adapter).borrow();

        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let batch_challenge = builder.batch_randomness();

        let mut batch_challenge_powers = vec![AB::ExprEF::one()];

        for _ in 1..builder.main().width() {
            let new_batch_challenge_power =
                batch_challenge_powers.last().unwrap().clone() * batch_challenge.into();
            batch_challenge_powers.push(new_batch_challenge_power);
        }

        let batched_local = local
            .iter()
            .zip(batch_challenge_powers.iter())
            .map(|(x, batch_challenge_power)| batch_challenge_power.clone() * (*x).into())
            .sum::<AB::ExprEF>();

        let batched_next = next
            .iter()
            .zip(batch_challenge_powers.iter())
            .map(|(x, batch_challenge_power)| batch_challenge_power.clone() * (*x).into())
            .sum::<AB::ExprEF>();

        // Assert that the first row accumulator is equal to the product of the lagrange_eval and
        // the main trace element.
        builder
            .when_first_row()
            .assert_eq_ext(local_adapter.accum, local_adapter.lagrange_eval.into() * batched_local);

        // Assert that the accumulator is correctly computed.
        builder.when_transition().assert_eq_ext(
            local_adapter.accum.into() + next_adapter.lagrange_eval.into() * batched_next,
            next_adapter.accum,
        );

        let expected_evals = builder.expected_evals();

        let mut batch_challenge_power = AB::ExprEF::one();

        let mut total_expected_eval = AB::ExprEF::zero();

        for eval in expected_evals {
            total_expected_eval += batch_challenge_power.clone() * (*eval).into();
            batch_challenge_power *= batch_challenge.into();
        }

        // Assert that the last row of the accumulator and the claimed evaluation match.
        builder.when_last_row().assert_eq_ext(local_adapter.accum, total_expected_eval);

        // We also need to constrain the lagrange_evals to be correctly computed, but that requires
        // functionality which the current STARK/AIR API does not provide.
    }
}

#[instrument(name = "generate multivariate adapter trace", level = "debug", skip_all)]
pub fn generate_adapter_trace<SC: StarkGenericConfig>(
    data: &RowMajorMatrix<Val<SC>>,
    eval_point: &Point<SC::Challenge>,
    batch_randomness: SC::Challenge,
) -> RowMajorMatrix<SC::Challenge> {
    let mut trace = Vec::with_capacity(data.height() * NUM_ADAPTER_COLS);

    // The eq polynomial, with one set of variables fixed to `eval_point`.
    let lagrange = partial_lagrange_eval(eval_point);

    let mut batch_powers = Vec::with_capacity(data.width());

    let mut batch_randomness_power = SC::Challenge::one();

    for _ in 0..data.width() {
        batch_powers.push(batch_randomness_power);
        batch_randomness_power *= batch_randomness;
    }

    let batched_data: Vec<SC::Challenge> = data
        .rows()
        .map(|row| row.zip(batch_powers.iter()).map(|(x, batch_power)| *batch_power * x).sum())
        .collect();

    // Compute the cumulative sum of the coordinate-wise product of eq polynomial and the Mle data.
    let accum = batched_data
        .iter()
        .zip(lagrange.iter())
        .scan(<SC::Challenge as AbstractField>::zero(), |acc, (x, y)| {
            *acc += *x * *y;
            Some(*acc)
        })
        .collect_vec();

    for (lagrange_eval, accum) in izip!(lagrange.iter(), accum) {
        let mut row = [SC::Challenge::zero(); NUM_ADAPTER_COLS];

        let cols: &mut MultivariateAdapterCols<_> = row.as_mut_slice().borrow_mut();

        cols.lagrange_eval = *lagrange_eval;
        cols.accum = accum;

        trace.extend(row);
    }

    RowMajorMatrix::new(trace, NUM_ADAPTER_COLS)
}
