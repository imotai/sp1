use itertools::izip;
use rayon::{
    iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
    slice::ParallelSlice,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

use slop_algebra::{ExtensionField, Field, Powers};
use slop_alloc::{Backend, Buffer, CanCopyFrom, CanCopyFromRef, CanCopyInto, CpuBackend};
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
use slop_tensor::{AddAssignBackend, ReduceSumBackend, Tensor};

use crate::Interaction;

use super::{GkrMle, LogupGkrPoly};

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
    + CanCopyFrom<Tensor<EF>, CpuBackend, Output = Tensor<EF, Self>>
    + CanCopyInto<Tensor<EF, Self>, CpuBackend, Output = Tensor<EF>>
    + ReduceSumBackend<EF>
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
        + CanCopyFrom<Tensor<EF>, CpuBackend, Output = Tensor<EF, Self>>
        + CanCopyInto<Tensor<EF, Self>, CpuBackend, Output = Tensor<EF>>
        + ReduceSumBackend<EF>
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
        input_mles: &GkrMle<Self::F, Self::EF, Self::B>,
    ) -> impl futures::Future<Output = GkrMle<Self::EF, Self::EF, Self::B>> + Send;

    /// The subsequent layers of the GKR circuit.
    fn circuit_layer_execution(
        current_mle: &GkrMle<Self::EF, Self::EF, Self::B>,
    ) -> impl futures::Future<Output = GkrMle<Self::EF, Self::EF, Self::B>> + Send;

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
    ) -> impl futures::Future<Output = GkrMle<Self::F, Self::EF, Self::B>> + Send;
}

/// Run the GKR circuit to generate the MLEs for each layer.
pub async fn generate_mles<SC: GkrProver>(input_mles: GkrMle<SC::F, SC::EF, SC::B>) -> GkrMles<SC> {
    // First handle the input MLEs.
    let next_mle = SC::first_circuit_layer_execution(&input_mles).await;
    // println!("NExt mle num vars: {}", next_mle.numerator_0.num_variables());

    let mut mles = vec![next_mle];

    let mut current_mle = mles.last().expect("mles must be non-empty");

    while current_mle.numerator_0.num_variables() > 0 {
        let next_mle = SC::circuit_layer_execution(current_mle).await;

        mles.push(next_mle);

        current_mle = mles.last().expect("mles must be non-empty");
    }

    // Reverse the MLEs so that the first round is the first element in the vec.

    GkrMles { mles }
}

/// Execute one layer of the GKR circuit.
///
/// This is essentially the grade-school algorithm for adding fractions:
/// `a/b + c/d = (ad + bc) / bd`: for adjacent `(p_0, p_1)` in `numerator_input_mle` and adjacent
/// `(q_0, q_1)` in `denom_input_mle`, compute  `(new_p, new_q) = (q_1 * p_0 + q_0 * p_1, q_0 * q_1)`.
#[allow(clippy::too_many_lines)]
pub fn circuit_layer<F: Field, EF: ExtensionField<F>>(
    input_mle: &GkrMle<F, EF, CpuBackend>,
) -> GkrMle<EF, EF, CpuBackend> {
    let num_interactions = input_mle.numerator_0.num_polynomials();

    #[allow(clippy::type_complexity)]
    let (numerator_0_guts, denom_0_guts, numerator_1_guts, denom_1_guts): (
        Option<Arc<Mle<_>>>,
        Option<Arc<Mle<_>>>,
        Option<Arc<Mle<_>>>,
        Option<Arc<Mle<_>>>,
    ) = match input_mle.numerator_0.inner() {
        Some(numerator_input_mle_0_inner) => {
            let numer_denom_guts = (
                numerator_input_mle_0_inner.guts().as_buffer().par_chunks(num_interactions),
                input_mle
                    .numerator_1
                    .inner()
                    .as_ref()
                    .unwrap()
                    .guts()
                    .as_buffer()
                    .as_slice()
                    .par_chunks(num_interactions),
                input_mle
                    .denom_0
                    .inner()
                    .as_ref()
                    .unwrap()
                    .guts()
                    .as_buffer()
                    .par_chunks(num_interactions),
                input_mle
                    .denom_1
                    .inner()
                    .as_ref()
                    .unwrap()
                    .guts()
                    .as_buffer()
                    .as_slice()
                    .par_chunks(num_interactions),
            )
                .into_par_iter()
                .flat_map_iter(
                    |(numerator_0_chunk, numerator_1_chunk, denom_0_chunk, denom_1_chunk)| {
                        izip!(numerator_0_chunk, numerator_1_chunk, denom_0_chunk, denom_1_chunk)
                            .map(|(n0, n1, d0, d1)| (*d1 * *n0 + *d0 * *n1, *d0 * *d1))
                    },
                )
                .collect::<Vec<_>>();

            let (numerator_0_guts, denom_0_guts): (Vec<EF>, Vec<EF>) = numer_denom_guts
                .par_chunks(num_interactions)
                .step_by(2)
                .flat_map(|chunk| chunk)
                .copied()
                .unzip();
            let (numerator_1_guts, denom_1_guts): (Vec<_>, Vec<_>) = numer_denom_guts
                .par_chunks(num_interactions)
                .skip(1)
                .step_by(2)
                .flat_map(|chunk| chunk)
                .copied()
                .unzip();

            let (numerator_1_guts, denom_1_guts) = if numerator_1_guts.is_empty() {
                assert!(denom_1_guts.is_empty());
                assert!(numerator_0_guts.len() == num_interactions);
                (vec![EF::zero(); num_interactions], vec![EF::one(); num_interactions])
            } else {
                (numerator_1_guts, denom_1_guts)
            };

            (
                Some(Arc::new(RowMajorMatrix::new(numerator_0_guts, num_interactions).into())),
                Some(Arc::new(RowMajorMatrix::new(denom_0_guts, num_interactions).into())),
                Some(Arc::new(RowMajorMatrix::new(numerator_1_guts, num_interactions).into())),
                Some(Arc::new(RowMajorMatrix::new(denom_1_guts, num_interactions).into())),
            )
        }
        None => (None, None, None, None),
    };

    // println!("Numerator_0_num_variables");

    let numerator_0 = PaddedMle::new(
        numerator_0_guts,
        input_mle.numerator_0.num_variables() - 1,
        Padding::Constant((EF::zero(), num_interactions, CpuBackend)),
    );
    let denom_0 = PaddedMle::new(
        denom_0_guts,
        input_mle.denom_0.num_variables() - 1,
        Padding::Constant((EF::one(), num_interactions, CpuBackend)),
    );
    let numerator_1 = PaddedMle::new(
        numerator_1_guts,
        input_mle.numerator_1.num_variables() - 1,
        Padding::Constant((EF::zero(), num_interactions, CpuBackend)),
    );
    let denom_1 = PaddedMle::new(
        denom_1_guts,
        input_mle.denom_1.num_variables() - 1,
        Padding::Constant((EF::one(), num_interactions, CpuBackend)),
    );
    GkrMle { numerator_0, numerator_1, denom_0, denom_1 }
}

impl<PcsComponents: JaggedProverComponents<A = CpuBackend> + std::fmt::Debug> GkrProver
    for PcsComponents
{
    type F = PcsComponents::F;
    type EF = PcsComponents::EF;
    type Challenger = PcsComponents::Challenger;
    type B = CpuBackend;

    fn first_circuit_layer_execution(
        input_mles: &GkrMle<Self::F, Self::EF, Self::B>,
    ) -> impl futures::Future<Output = GkrMle<Self::EF, Self::EF, Self::B>> + Send {
        std::future::ready(circuit_layer(input_mles))
    }

    fn circuit_layer_execution(
        current_mle: &GkrMle<Self::EF, Self::EF, Self::B>,
    ) -> impl futures::Future<Output = GkrMle<Self::EF, Self::EF, Self::B>> + Send {
        std::future::ready(circuit_layer(current_mle))
    }

    async fn generate_gkr_input_mles(
        preprocessed: Option<&PaddedMle<Self::F, Self::B>>,
        main: &PaddedMle<Self::F, Self::B>,
        sends: &[Interaction<Self::F>],
        receives: &[Interaction<Self::F>],
        alpha: Self::EF,
        betas: &Powers<Self::EF>,
        log_max_row_height: usize,
    ) -> GkrMle<Self::F, Self::EF> {
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
    pub(crate) mles: Vec<GkrMle<SC::EF, SC::EF, SC::B>>,
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
    InconsistentEvaluation(usize),
    /// Inconsistency between the individual evaluations per interaction, and the final eval claim
    /// from the last round of GKR.
    #[error("inconsistent individual evaluations")]
    InconsistentIndividualEvals,
    /// Error when verifying sumcheck proof.
    #[error("sumcheck error: {0}")]
    SumcheckError(#[from] SumcheckError),
}
