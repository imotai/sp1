use itertools::izip;
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator},
    slice::ParallelSlice,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

use slop_algebra::{ExtensionField, Field, Powers};
use slop_alloc::{Backend, Buffer, CanCopyFromRef, CpuBackend};
use slop_challenger::FieldChallenger;
use slop_jagged::JaggedProverComponents;
use slop_matrix::dense::RowMajorMatrix;
use slop_multilinear::{
    HostEvaluationBackend, Mle, MleBaseBackend, MleEval, MleEvaluationBackend,
    MleFixLastVariableBackend, PaddedMle, Padding, Point, PointBackend,
};
use slop_sumcheck::{
    ComponentPolyEvalBackend, PartialSumcheckProof, SumCheckPolyFirstRoundBackend, SumcheckError,
    SumcheckPolyBackend,
};
use slop_tensor::AddAssignBackend;

use crate::Interaction;

use super::LogupGkrPoly;

/// A backend capable of implementing `SumcheckPoly` for `LogupGkrPoly`.
pub trait GkrBackend<F: Field, EF: ExtensionField<F>>:
    Backend
    + MleFixLastVariableBackend<F, EF>
    + MleFixLastVariableBackend<EF, EF>
    + MleFixLastVariableBackend<F, F>
    + MleEvaluationBackend<F, EF>
    + MleEvaluationBackend<EF, EF>
    + MleEvaluationBackend<F, F>
    + HostEvaluationBackend<F, EF>
    + HostEvaluationBackend<F, F>
    + HostEvaluationBackend<EF, EF>
    + PointBackend<EF>
    + MleBaseBackend<F>
    + MleBaseBackend<EF>
    + ComponentPolyEvalBackend<LogupGkrPoly<EF, EF, Self>, EF>
    + ComponentPolyEvalBackend<LogupGkrPoly<F, EF, Self>, EF>
    + SumcheckPolyBackend<LogupGkrPoly<EF, EF, Self>, EF>
    + SumCheckPolyFirstRoundBackend<
        LogupGkrPoly<F, EF, Self>,
        EF,
        NextRoundPoly = LogupGkrPoly<EF, EF, Self>,
    > + SumCheckPolyFirstRoundBackend<
        LogupGkrPoly<EF, EF, Self>,
        EF,
        NextRoundPoly = LogupGkrPoly<EF, EF, Self>,
    > + PointBackend<F>
    + PointBackend<EF>
    + CanCopyFromRef<Buffer<EF>, CpuBackend, Output = Buffer<EF, Self>>
    + AddAssignBackend<EF>
{
}

impl<F: Field, EF: ExtensionField<F>, B> GkrBackend<F, EF> for B where
    B: Backend
        + MleFixLastVariableBackend<F, EF>
        + MleFixLastVariableBackend<EF, EF>
        + MleFixLastVariableBackend<F, F>
        + MleEvaluationBackend<F, EF>
        + MleEvaluationBackend<EF, EF>
        + MleEvaluationBackend<F, F>
        + HostEvaluationBackend<F, EF>
        + HostEvaluationBackend<F, F>
        + HostEvaluationBackend<EF, EF>
        + PointBackend<EF>
        + MleBaseBackend<F>
        + MleBaseBackend<EF>
        + ComponentPolyEvalBackend<LogupGkrPoly<EF, EF, Self>, EF>
        + ComponentPolyEvalBackend<LogupGkrPoly<F, EF, Self>, EF>
        + SumcheckPolyBackend<LogupGkrPoly<EF, EF, Self>, EF>
        + SumCheckPolyFirstRoundBackend<
            LogupGkrPoly<F, EF, Self>,
            EF,
            NextRoundPoly = LogupGkrPoly<EF, EF, Self>,
        > + SumCheckPolyFirstRoundBackend<
            LogupGkrPoly<EF, EF, Self>,
            EF,
            NextRoundPoly = LogupGkrPoly<EF, EF, Self>,
        > + PointBackend<F>
        + PointBackend<EF>
        + CanCopyFromRef<Buffer<EF>, CpuBackend, Output = Buffer<EF, Self>>
        + AddAssignBackend<EF>
{
}

/// The GKR prover trait.
pub trait GkrProver: Clone + Send + Sync + 'static + std::fmt::Debug {
    /// The base field of the zkVM.
    type F: Field;
    /// The "cryptographically secure" extension field.
    type EF: ExtensionField<Self::F>;
    /// The challenger type.
    type Challenger: FieldChallenger<Self::F> + 'static + Send + Sync + Clone;
    /// The backend type, which provides the necessary operations for the GKR prover.
    type B: GkrBackend<Self::F, Self::EF>;

    /// The first layer of the GKR circuit.
    fn first_circuit_layer_execution(
        numerator_input_mle: &PaddedMle<Self::F, Self::B>,
        denom_input_mle: &PaddedMle<Self::EF, Self::B>,
    ) -> impl futures::Future<Output = (PaddedMle<Self::EF, Self::B>, PaddedMle<Self::EF, Self::B>)> + Send;

    /// The subsequent layers of the GKR circuit.
    fn circuit_layer_execution(
        current_numerator_mle: &PaddedMle<Self::EF, Self::B>,
        current_denom_mle: &PaddedMle<Self::EF, Self::B>,
    ) -> impl futures::Future<Output = (PaddedMle<Self::EF, Self::B>, PaddedMle<Self::EF, Self::B>)> + Send;

    /// Generate the input MLEs for the GKR circuit. This involves computing the numerator and
    /// denominator MLEs for each interaction in the chip.
    fn generate_gkr_input_mles(
        preprocessed: Option<&PaddedMle<Self::F, Self::B>>,
        main: &PaddedMle<Self::F, Self::B>,
        sends: &[Interaction<Self::F>],
        receives: &[Interaction<Self::F>],
        alpha: Self::EF,
        betas: &Powers<Self::EF>,
        log_max_row_height: usize,
    ) -> impl futures::Future<Output = (PaddedMle<Self::F, Self::B>, PaddedMle<Self::EF, Self::B>)> + Send;
}

/// Run the GKR circuit to generate the MLEs for each layer.
pub async fn generate_mles<SC: GkrProver>(
    numerator_input_mle: PaddedMle<SC::F, SC::B>,
    denom_input_mle: PaddedMle<SC::EF, SC::B>,
) -> GkrMles<SC> {
    // First handle the input MLEs.
    let (next_numerator_mle, next_denom_mle) =
        SC::first_circuit_layer_execution(&numerator_input_mle, &denom_input_mle).await;

    let mut numerator_mles = vec![next_numerator_mle];
    let mut denom_mles = vec![next_denom_mle];

    let mut current_numerator_mle =
        numerator_mles.last().expect("numerator_mle_s must be non-empty");
    let mut current_denom_mle = denom_mles.last().expect("denom_mle_s must be non-empty");

    while current_denom_mle.num_variables() > 0 {
        let (next_numerator_mle, next_denom_mle) =
            SC::circuit_layer_execution(current_numerator_mle, current_denom_mle).await;

        numerator_mles.push(next_numerator_mle);
        denom_mles.push(next_denom_mle);

        current_numerator_mle = numerator_mles.last().expect("numerator_mle_s must be non-empty");
        current_denom_mle = denom_mles.last().expect("denom_mle_s must be non-empty");
    }

    // Reverse the MLEs so that the first round is the first element in the vec.
    numerator_mles.reverse();
    denom_mles.reverse();

    GkrMles {
        input_numerators: numerator_input_mle,
        input_denoms: denom_input_mle,
        numerator_mles,
        denom_mles,
    }
}

/// Execute one layer of the GKR circuit.
///
/// This is essentially the grade-school algorithm for adding fractions:
/// `a/b + c/d = (ad + bc) / bd`: for adjacent `(p_0, p_1)` in `numerator_input_mle` and adjacent
/// `(q_0, q_1)` in `denom_input_mle`, compute  `(new_p, new_q) = (q_1 * p_0 + q_0 * p_1, q_0 * q_1)`.
pub fn circuit_layer<F: Field, EF: ExtensionField<F>>(
    numerator_input_mle: &PaddedMle<F>,
    denom_input_mle: &PaddedMle<EF>,
) -> (PaddedMle<EF>, PaddedMle<EF>) {
    assert!(numerator_input_mle.num_polynomials() == denom_input_mle.num_polynomials());
    let num_interactions = numerator_input_mle.num_polynomials();
    let (numerator_guts, denom_guts): (Option<Arc<Mle<_>>>, Option<Arc<Mle<_>>>) =
        match numerator_input_mle.inner() {
            Some(numerator_input_mle_inner) => {
                let (numerator_guts, denom_guts) = numerator_input_mle_inner
                    .guts()
                    .as_buffer()
                    .par_chunks(2 * num_interactions)
                    .zip(
                        denom_input_mle
                            .inner()
                            .as_ref()
                            .unwrap()
                            .guts()
                            .as_buffer()
                            .par_chunks(2 * num_interactions),
                    )
                    .flat_map_iter(|(numerator_chunk, denom_chunk)| {
                        let (numerator_chunk_0, numerator_chunk_1, denom_chunk_0, denom_chunk_1) =
                            if numerator_chunk.len() == 2 * num_interactions {
                                (
                                    numerator_chunk
                                        .par_iter()
                                        .copied()
                                        .take(num_interactions)
                                        .collect::<Vec<_>>(),
                                    numerator_chunk
                                        .par_iter()
                                        .copied()
                                        .skip(num_interactions)
                                        .collect::<Vec<_>>(),
                                    denom_chunk
                                        .par_iter()
                                        .copied()
                                        .take(num_interactions)
                                        .collect::<Vec<_>>(),
                                    denom_chunk
                                        .par_iter()
                                        .copied()
                                        .skip(num_interactions)
                                        .collect::<Vec<_>>(),
                                )
                            } else {
                                (
                                    numerator_chunk
                                        .iter()
                                        .copied()
                                        .take(num_interactions)
                                        .collect::<Vec<_>>(),
                                    vec![F::zero(); num_interactions],
                                    denom_chunk
                                        .par_iter()
                                        .copied()
                                        .take(num_interactions)
                                        .collect::<Vec<_>>(),
                                    vec![EF::one(); num_interactions],
                                )
                            };
                        izip!(numerator_chunk_0, numerator_chunk_1, denom_chunk_0, denom_chunk_1)
                            .map(|(n0, n1, d0, d1)| (d1 * n0 + d0 * n1, d0 * d1))
                    })
                    .unzip();
                (
                    Some(Arc::new(RowMajorMatrix::new(numerator_guts, num_interactions).into())),
                    Some(Arc::new(RowMajorMatrix::new(denom_guts, num_interactions).into())),
                )
            }
            None => (None, None),
        };

    (
        PaddedMle::new(
            numerator_guts,
            numerator_input_mle.num_variables() - 1,
            Padding::Constant((EF::zero(), num_interactions, CpuBackend)),
        ),
        PaddedMle::new(
            denom_guts,
            denom_input_mle.num_variables() - 1,
            Padding::Constant((EF::one(), num_interactions, CpuBackend)),
        ),
    )
}

impl<PcsComponents: JaggedProverComponents<A = CpuBackend> + std::fmt::Debug> GkrProver
    for PcsComponents
{
    type F = PcsComponents::F;
    type EF = PcsComponents::EF;
    type Challenger = PcsComponents::Challenger;
    type B = CpuBackend;

    fn first_circuit_layer_execution(
        numerator_input_mle: &PaddedMle<Self::F, Self::B>,
        denom_input_mle: &PaddedMle<Self::EF, Self::B>,
    ) -> impl futures::Future<Output = (PaddedMle<Self::EF, Self::B>, PaddedMle<Self::EF, Self::B>)> + Send
    {
        std::future::ready(circuit_layer(numerator_input_mle, denom_input_mle))
    }

    fn circuit_layer_execution(
        current_numerator_mle: &PaddedMle<Self::EF, Self::B>,
        current_denom_mle: &PaddedMle<Self::EF, Self::B>,
    ) -> impl futures::Future<Output = (PaddedMle<Self::EF, Self::B>, PaddedMle<Self::EF, Self::B>)> + Send
    {
        std::future::ready(circuit_layer(current_numerator_mle, current_denom_mle))
    }

    async fn generate_gkr_input_mles(
        preprocessed: Option<&PaddedMle<Self::F, Self::B>>,
        main: &PaddedMle<Self::F, Self::B>,
        sends: &[Interaction<Self::F>],
        receives: &[Interaction<Self::F>],
        alpha: Self::EF,
        betas: &Powers<Self::EF>,
        log_max_row_height: usize,
    ) -> (PaddedMle<Self::F, Self::B>, PaddedMle<Self::EF, Self::B>) {
        super::generate_gkr_input_mles(
            preprocessed,
            main,
            sends
                .iter()
                .map(|i| (i, true))
                .chain(receives.iter().map(|i| (i, false)))
                .collect::<Vec<_>>()
                .as_slice(),
            alpha,
            betas,
            log_max_row_height,
        )
    }
}

/// The MLEs for each layer of the GKR circuit.
pub struct GkrMles<SC: GkrProver> {
    pub(crate) input_numerators: PaddedMle<SC::F, SC::B>,
    pub(crate) input_denoms: PaddedMle<SC::EF, SC::B>,
    pub(crate) numerator_mles: Vec<PaddedMle<SC::EF, SC::B>>,
    pub(crate) denom_mles: Vec<PaddedMle<SC::EF, SC::B>>,
}

/// The GKR messages that the prover will send to the verifier.  It will send one message per round.
#[derive(Serialize, Deserialize, Clone)]
pub(crate) struct ProverMessage<EF> {
    /// Set to the evaluation of NumeratorMLE(numerator's sumcheck point || 0).
    pub(crate) numerator_0: Vec<EF>,
    /// Set to the evaluation of NumeratorMLE(numerator's sumcheck point || 1).
    pub(crate) numerator_1: Vec<EF>,
    /// Set to the evaluation of DenomMLE(denominator's sumcheck point || 0).
    pub(crate) denom_0: Vec<EF>,
    /// Set to the evaluation of DenomMLE(denominator's sumcheck point || 1).
    pub(crate) denom_1: Vec<EF>,
}

/// The proof for `LogUp` GKR.
#[derive(Clone, Serialize, Deserialize)]
pub struct LogupGkrProof<EF> {
    // The prover messages for each round.
    pub(crate) prover_messages: Vec<ProverMessage<EF>>,
    // The sumcheck proofs.  Note that the i'th entry is for round i+1.  There is no proof for the first round.
    pub(crate) sc_proofs: Vec<PartialSumcheckProof<EF>>,
    // The numerator claim.
    pub(crate) numerator_claims: Vec<EF>,
    // The denominator claim.
    pub(crate) denom_claims: Vec<EF>,

    // The data to pass to the multivariate PCS, namely the per-interaction multilinear evaluation
    // claims.
    pub(crate) main_log_height: usize,
    pub(crate) column_openings: (MleEval<EF>, Option<MleEval<EF>>),
}

#[derive(Serialize, Deserialize, Clone)]
/// The proof for `LogUp` GKR without computing the column openings of the main traces.
pub struct GkrProofWithoutOpenings<EF> {
    pub(crate) prover_messages: Vec<ProverMessage<EF>>,
    pub(crate) sc_proofs: Vec<PartialSumcheckProof<EF>>,
    pub(crate) numerator_claims: Vec<EF>,
    pub(crate) denom_claims: Vec<EF>,
    pub(crate) challenge: Point<EF>,
}

/// An error type for `LogUp` GKR.
#[derive(Debug, Error)]
pub enum LogupGkrVerificationError {
    /// The final claim is not consistent with the first prover message.
    #[error("final ")]
    InconsistentFinalClaim,
    /// The sumcheck claim is not consistent with the calculated one from the prover messages.
    #[error("inconsistent sumcheck claim")]
    InconsistentSumcheckClaim,
    /// Inconsistency between the calculated evaluation and the sumcheck evaluation.
    #[error("inconsistent evaluation")]
    InconsistentEvaluation,
    /// Inconsistency between the individual evaluations per interaction, and the final eval claim
    /// from the last round of GKR.
    #[error("inconsistent individual evaluations")]
    InconsistentIndividualEvals,
    /// Error when verifying sumcheck proof.
    #[error("sumcheck error: {0}")]
    SumcheckError(#[from] SumcheckError),
}
