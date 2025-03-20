use crate::Interaction;
use itertools::{izip, Itertools};
use serde::{Deserialize, Serialize};
use slop_algebra::AbstractField;
use slop_algebra::{ExtensionField, Field};
use slop_alloc::{Buffer, CanCopyFrom, CpuBackend, HasBackend, ToHost};
use slop_challenger::FieldChallenger;
use slop_multilinear::{
    HostEvaluationBackend, MleBaseBackend, MleEval, MleEvaluationBackend,
    MleFixLastVariableBackend, PaddedMle, Padding, Point,
};
use slop_sumcheck::{
    reduce_sumcheck_to_evaluation, ComponentPolyEvalBackend, PartialSumcheckProof,
    SumCheckPolyFirstRoundBackend,
};

use super::{
    generate_mles, GkrMle, GkrMles, GkrProofWithoutOpenings, GkrProver, LogupGkrPoly,
    LogupGkrProof, ProverMessage,
};

#[derive(Serialize, Deserialize, Clone)]
pub enum Either<L, R> {
    Left(L),
    Right(R),
}

pub struct LogUpProverData<SC: GkrProver> {
    pub proof: LogupGkrProof<SC::EF>,
    pub final_point: Point<SC::EF>,
    pub challenger: SC::Challenger,
}

/// Generates the GKR proof for the interactions logup permutation argument.  Specifically, it will
/// generate a proof of the numerator and denominator of the argument's cumulative sum.
pub(crate) async fn generate_gkr_logup_proof_and_data<SC: GkrProver>(
    sends: &[Interaction<SC::F>],
    receives: &[Interaction<SC::F>],
    preprocessed: Option<&PaddedMle<SC::F, SC::B>>,
    main: &PaddedMle<SC::F, SC::B>,
    random_elements: &[SC::EF],
    mut challenger: SC::Challenger,
    log_max_row_height: usize,
) -> LogUpProverData<SC>
where
    SC::EF: Clone,
    SC::B: CanCopyFrom<Buffer<SC::EF>, CpuBackend, Output = Buffer<SC::EF, SC::B>>,
{
    // Generate the RLC elements to uniquely identify each interaction.
    let alpha = random_elements[0];

    // Generate the RLC elements to uniquely identify each item in the looked up tuple.
    let betas = random_elements[1].powers();

    let input_mles = SC::generate_gkr_input_mles(
        preprocessed,
        main,
        sends,
        receives,
        alpha,
        &betas.clone(),
        log_max_row_height,
    )
    .await;

    let GkrProofWithoutOpenings {
        prover_messages,
        sc_proofs,
        numerator_claims: perm_numerator_claims,
        denom_claims: perm_denom_claims,
        challenge: last_challenge,
    } = generate_gkr_proof::<SC>(
        input_mles,
        &mut challenger,
        sends,
        receives,
        preprocessed,
        main,
        random_elements,
        log_max_row_height,
    )
    .await;

    let k = main.num_variables();

    let column_openings = generate_column_openings::<SC>(preprocessed, main, &last_challenge).await;

    LogUpProverData {
        proof: LogupGkrProof {
            main_log_height: k as usize,
            prover_messages,
            sc_proofs,
            numerator_claims: perm_numerator_claims,
            denom_claims: perm_denom_claims,
            column_openings,
        },
        final_point: last_challenge,
        challenger,
    }
}

async fn run_gkr_round<SC: GkrProver, NumeratorType>(
    round_num: usize,
    mles: GkrMle<NumeratorType, SC::EF, SC::B>,
    numerator_claims: &[SC::EF],
    denom_claims: &[SC::EF],
    challenge: &Point<SC::EF>,
    challenger: &mut SC::Challenger,
) -> (ProverMessage<SC::EF>, Option<PartialSumcheckProof<SC::EF>>)
where
    SC::EF: ExtensionField<NumeratorType>,
    NumeratorType: Field,

    SC::B: MleFixLastVariableBackend<NumeratorType, NumeratorType>
        + MleEvaluationBackend<NumeratorType, NumeratorType>
        + HostEvaluationBackend<NumeratorType, NumeratorType>
        + MleBaseBackend<NumeratorType>
+ SumCheckPolyFirstRoundBackend<LogupGkrPoly<NumeratorType, SC::EF, SC::B>, SC::EF>
        + ComponentPolyEvalBackend<LogupGkrPoly<NumeratorType, SC::EF, SC::B>, SC::EF>
        +   CanCopyFrom<Buffer<SC::EF>, CpuBackend, Output = Buffer<SC::EF, SC::B>>,

    <SC::B as SumCheckPolyFirstRoundBackend<LogupGkrPoly<NumeratorType, SC::EF, SC::B>, SC::EF>>::NextRoundPoly: Send+Sync,
{
    // For the first round, we don't need to generate a sumcheck proof, since the the verifier will
    // do the sum itself (it's only summing two elements for each claim).

    let logup_gkr_mle_or_prover_message =
        reduce_gkr_to_sumcheck::<SC, NumeratorType>(round_num, mles, challenge, challenger).await;

    match logup_gkr_mle_or_prover_message {
        Either::Left(prover_message) => (prover_message, None),
        Either::Right(logup_gkr_mles) => {
            let lambda = logup_gkr_mles.lambda;
            let sumcheck_claim = match logup_gkr_mles.numerator_0.inner() {
                Some(_) => numerator_claims
                    .iter()
                    .zip_eq(denom_claims.iter())
                    .map(|(numerator_claim, denom_claim)| *numerator_claim + lambda * *denom_claim)
                    .zip_eq(logup_gkr_mles.batching_randomness_powers.iter().copied())
                    .map(|(claim, challenge)| claim * challenge)
                    .sum::<SC::EF>(),
                None => logup_gkr_mles.batching_randomness_powers_sum * lambda,
            };
            let (sc_proof, last_poly) =
                reduce_sumcheck_to_evaluation::<SC::F, SC::EF, SC::Challenger>(
                    vec![logup_gkr_mles],
                    challenger,
                    vec![sumcheck_claim],
                    1,
                    SC::EF::one(),
                )
                .await;

            assert!(last_poly.len() == 1);

            let last_poly = &last_poly[0];

            let numerator_0: Vec<_> = last_poly.iter().copied().step_by(4).collect();
            let numerator_1: Vec<_> = last_poly.iter().copied().skip(1).step_by(4).collect();
            let denom_0: Vec<_> = last_poly.iter().copied().skip(2).step_by(4).collect();
            let denom_1: Vec<_> = last_poly.iter().copied().skip(3).step_by(4).collect();

            for (numerator_0_elem, numerator_1_elem, denom_0_elem, denom_1_elem) in
                izip!(numerator_0.iter(), numerator_1.iter(), denom_0.iter(), denom_1.iter())
            {
                challenger.observe_ext_element(*numerator_0_elem);
                challenger.observe_ext_element(*numerator_1_elem);
                challenger.observe_ext_element(*denom_0_elem);
                challenger.observe_ext_element(*denom_1_elem);
            }
            (ProverMessage { numerator_0, numerator_1, denom_0, denom_1 }, Some(sc_proof))
        }
    }
}
pub type MessageOrLogupPoly<NumeratorType, SC> = Either<
    ProverMessage<<SC as GkrProver>::EF>,
    LogupGkrPoly<NumeratorType, <SC as GkrProver>::EF, <SC as GkrProver>::B>,
>;

async fn mle_to_host<
    F: Field,
    EF: ExtensionField<F>,
    B: MleEvaluationBackend<F, F> + HostEvaluationBackend<F, F> + MleBaseBackend<F>,
>(
    mle: &PaddedMle<F, B>,
) -> Vec<EF> {
    match mle.inner() {
        Some(inner) => (*inner)
            .eval_at(&Point::new(Buffer::with_capacity_in(
                0,
                mle.padding_values().backend().clone(),
            )))
            .await
            .to_host()
            .await
            .unwrap()
            .evaluations()
            .as_slice()
            .iter()
            .copied()
            .map(EF::from_base)
            .collect(),
        None => match mle.padding_values() {
            Padding::Generic(padding_values) => padding_values
                .to_host()
                .await
                .unwrap()
                .evaluations()
                .as_slice()
                .iter()
                .copied()
                .map(EF::from_base)
                .collect(),
            Padding::Constant((value, num_polynomials, _)) => {
                vec![EF::from_base(*value); *num_polynomials]
            }
        },
    }
}

/// Reduce the circuit evaluation claim to a sumcheck proof. Returns an `Either` type because the
/// in the first round, the prover will simply send the evaluations.
async fn reduce_gkr_to_sumcheck<SC: GkrProver, NumeratorType>(
    round_num: usize,
    mles: GkrMle<NumeratorType, SC::EF, SC::B>,
    challenge: &Point<SC::EF>,
    challenger: &mut SC::Challenger,
) -> MessageOrLogupPoly<NumeratorType, SC>
where
    NumeratorType: Field,
    SC::EF: ExtensionField<NumeratorType>,
    SC::B: MleFixLastVariableBackend<NumeratorType, NumeratorType>
        + MleEvaluationBackend<NumeratorType, NumeratorType>
        + HostEvaluationBackend<NumeratorType, NumeratorType>
        + MleBaseBackend<NumeratorType>
        + CanCopyFrom<Buffer<SC::EF>, CpuBackend, Output = Buffer<SC::EF, SC::B>>,
{
    // TODO: Traitify this.
    let GkrMle {
        numerator_0: numerator_mles_fixed_0,
        numerator_1: numerator_mles_fixed_1,
        denom_0: denom_mles_fixed_0,
        denom_1: denom_mles_fixed_1,
    } = mles;

    // For the first round, we don't need to generate a sumcheck proof, since the the verifier will
    // do the sum itself (it's only summing two elements for each claim).
    if round_num == 0 {
        assert!(numerator_mles_fixed_0.num_variables() == 0);
        assert!(numerator_mles_fixed_1.num_variables() == 0);
        assert!(denom_mles_fixed_0.num_variables() == 0);
        assert!(denom_mles_fixed_1.num_variables() == 0);

        assert!(numerator_mles_fixed_0.num_real_entries() <= 1);
        assert!(numerator_mles_fixed_1.num_real_entries() <= 1);
        assert!(denom_mles_fixed_0.num_real_entries() <= 1);
        assert!(denom_mles_fixed_1.num_real_entries() <= 1);

        assert!(numerator_mles_fixed_0.num_real_entries() <= 1);
        assert!(numerator_mles_fixed_1.num_real_entries() <= 1);
        assert!(denom_mles_fixed_0.num_real_entries() <= 1);
        assert!(denom_mles_fixed_1.num_real_entries() <= 1);

        let numerator_0 =
            mle_to_host::<NumeratorType, SC::EF, SC::B>(&numerator_mles_fixed_0).await;
        let numerator_1 = mle_to_host(&numerator_mles_fixed_1).await;
        let denom_0 = mle_to_host(&denom_mles_fixed_0).await;
        let denom_1 = mle_to_host(&denom_mles_fixed_1).await;

        for (numerator_0_elem, numerator_1_elem, denom_0_elem, denom_1_elem) in
            izip!(numerator_0.iter(), numerator_1.iter(), denom_0.iter(), denom_1.iter())
        {
            challenger.observe_ext_element(*numerator_0_elem);
            challenger.observe_ext_element(*numerator_1_elem);
            challenger.observe_ext_element(*denom_0_elem);
            challenger.observe_ext_element(*denom_1_elem);
        }

        Either::Left(ProverMessage { numerator_0, numerator_1, denom_0, denom_1 })
    } else {
        // Get the sumcheck challenges.  It will reduce the summation of the MLE polynomial to an evaluation of a single random point.
        let lambda: SC::EF = challenger.sample_ext_element();
        let batch_randomness = challenger.sample_ext_element();

        let logup_gkr_mles = LogupGkrPoly::<NumeratorType, SC::EF, SC::B>::new(
            challenge.clone(),
            (numerator_mles_fixed_0, numerator_mles_fixed_1),
            (denom_mles_fixed_0, denom_mles_fixed_1),
            lambda,
            SC::EF::one(),
            batch_randomness,
        )
        .await;

        Either::Right(logup_gkr_mles)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn generate_gkr_proof<SC: GkrProver>(
    input_mles: GkrMle<SC::F, SC::EF, SC::B>,
    challenger: &mut SC::Challenger,
    sends: &[Interaction<SC::F>],
    receives: &[Interaction<SC::F>],
    preprocessed: Option<&PaddedMle<SC::F, SC::B>>,
    main: &PaddedMle<SC::F, SC::B>,
    random_elements: &[SC::EF],
    log_max_row_height: usize,
) -> GkrProofWithoutOpenings<SC::EF>
where
    SC::B: CanCopyFrom<Buffer<SC::EF>, CpuBackend, Output = Buffer<SC::EF, SC::B>>,
{
    let mut gkr_mles: GkrMles<SC> = generate_mles(input_mles).await;

    let num_rounds = gkr_mles.mles.len();

    let final_mle = gkr_mles.mles.last().unwrap();

    assert!(final_mle.numerator_0.num_variables() == 0);
    assert!(final_mle.numerator_1.num_variables() == 0);
    assert!(final_mle.denom_0.num_variables() == 0);
    assert!(final_mle.denom_1.num_variables() == 0);

    let final_numerator_0_claims =
        final_mle.numerator_0.eval_at::<SC::EF>(&vec![].into()).await.to_host().await.unwrap();

    let final_numerator_1_claims =
        final_mle.numerator_1.eval_at::<SC::EF>(&vec![].into()).await.to_host().await.unwrap();
    let final_denominator_0_claims =
        final_mle.denom_0.eval_at::<SC::EF>(&vec![].into()).await.to_host().await.unwrap();
    let final_denominator_1_claims =
        final_mle.denom_1.eval_at::<SC::EF>(&vec![].into()).await.to_host().await.unwrap();

    let (final_numerator_claims, final_denom_claims): (Vec<SC::EF>, Vec<SC::EF>) = izip!(
        final_numerator_0_claims.evaluations().as_slice().iter().copied(),
        final_numerator_1_claims.evaluations().as_slice().iter().copied(),
        final_denominator_0_claims.evaluations().as_slice().iter().copied(),
        final_denominator_1_claims.evaluations().as_slice().iter().copied(),
    )
    .map(|(num_0, num_1, denom_0, denom_1)| (num_0 * denom_1 + num_1 * denom_0, denom_0 * denom_1))
    .unzip();

    for (numerator_claim, denom_claim) in
        final_numerator_claims.iter().zip(final_denom_claims.iter())
    {
        challenger.observe_ext_element(*numerator_claim);
        challenger.observe_ext_element(*denom_claim);
    }

    let mut challenge: Point<SC::EF> = vec![].into();
    let mut prover_messages = Vec::new();
    let mut sc_proofs = Vec::new();

    let mut numerator_claims = final_numerator_claims.clone();
    let mut denom_claims = final_denom_claims.clone();

    for round_num in 0..num_rounds {
        let mle = gkr_mles.mles.pop().unwrap();

        let (prover_message, sc_proof) = run_gkr_round::<SC, SC::EF>(
            round_num,
            mle,
            &numerator_claims,
            &denom_claims,
            &challenge,
            challenger,
        )
        .await;

        if let Some(sc_proof) = sc_proof {
            sc_proofs.push(sc_proof.clone());
            challenge = sc_proof.point_and_eval.0;
        }

        let gkr_round_challenge: SC::EF = challenger.sample_ext_element();

        // Do the 2-to-1 trick.  Calculate a random linear combination of the next round's 0 and 1 points.
        numerator_claims = prover_message
            .numerator_0
            .iter()
            .zip(prover_message.numerator_1.iter())
            .map(|(numerator_0, numerator_1)| {
                *numerator_0 + gkr_round_challenge * (*numerator_1 - *numerator_0)
            })
            .collect();

        denom_claims = prover_message
            .denom_0
            .iter()
            .zip(prover_message.denom_1.iter())
            .map(|(denom_0, denom_1)| *denom_0 + gkr_round_challenge * (*denom_1 - *denom_0))
            .collect();

        prover_messages.push(prover_message);

        challenge.add_dimension_back(gkr_round_challenge);
    }

    let input_mles = SC::generate_gkr_input_mles(
        preprocessed,
        main,
        sends,
        receives,
        random_elements[0],
        &random_elements[1].powers(),
        log_max_row_height,
    )
    .await;

    let (prover_message, maybe_sc_proof) = run_gkr_round::<SC, SC::F>(
        num_rounds,
        input_mles,
        &numerator_claims,
        &denom_claims,
        &challenge,
        challenger,
    )
    .await;

    prover_messages.push(prover_message);
    if let Some(sc_proof) = maybe_sc_proof {
        sc_proofs.push(sc_proof.clone());
        challenge = sc_proof.point_and_eval.0;
    }

    let gkr_round_challenge: SC::EF = challenger.sample_ext_element();
    challenge.add_dimension_back(gkr_round_challenge);

    GkrProofWithoutOpenings {
        prover_messages,
        sc_proofs,
        numerator_claims: final_numerator_claims,
        denom_claims: final_denom_claims,
        challenge,
    }
}

async fn generate_column_openings<SC: GkrProver>(
    preprocessed: Option<&PaddedMle<SC::F, SC::B>>,
    main: &PaddedMle<SC::F, SC::B>,
    challenge: &Point<SC::EF>,
) -> (MleEval<SC::EF>, Option<MleEval<SC::EF>>) {
    let main_evals = main.eval_at(challenge).await;

    let prep_evals = match preprocessed {
        Some(prep) => Some(prep.eval_at(challenge).await),
        None => None,
    };

    (
        main_evals.to_host().await.unwrap(),
        match prep_evals {
            Some(prep_evals) => Some(prep_evals.to_host().await.unwrap()),
            None => None,
        },
    )
}
