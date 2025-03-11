use crate::MachineConfig;
use itertools::izip;
use slop_algebra::AbstractField;
use slop_challenger::FieldChallenger;
use slop_multilinear::{full_geq, Mle, Point};
use slop_sumcheck::{partially_verify_sumcheck_proof, PartialSumcheckProof};

use crate::Interaction;

use super::{LogupGkrProof, LogupGkrVerificationError, ProverMessage};

pub struct GkrPointAndEvals<EF> {
    pub point: Point<EF>,
    pub numerator_evals: Vec<EF>,
    pub denom_evals: Vec<EF>,
}

pub(crate) fn verify_gkr_rounds<SC: MachineConfig>(
    prover_messages: &[ProverMessage<SC::EF>],
    sc_proofs: &[PartialSumcheckProof<SC::EF>],
    numerator_claims: &[SC::EF],
    denom_claims: &[SC::EF],
    challenger: &mut SC::Challenger,
) -> Result<GkrPointAndEvals<SC::EF>, LogupGkrVerificationError> {
    let num_rounds = prover_messages.len();

    let mut round_challenge: Point<_> = Vec::new().into();

    let mut numerator_claims = numerator_claims.to_vec();
    let mut denom_claims = denom_claims.to_vec();

    for round_num in 0..num_rounds {
        // For the first round, we don't have a sumcheck proof, since the verifier can check the claimed
        // value directly from the next round's 0 and 1 points.
        if round_num == 0 {
            let expected_final_numerator_claims = izip!(
                prover_messages[round_num].numerator_0.iter().copied(),
                prover_messages[round_num].numerator_1.iter().copied(),
                prover_messages[round_num].denom_0.iter().copied(),
                prover_messages[round_num].denom_1.iter().copied()
            )
            .map(|(n0, n1, d0, d1)| n0 * d1 + n1 * d0)
            .collect::<Vec<_>>();
            let expected_final_denom_claims = prover_messages[round_num]
                .denom_0
                .iter()
                .zip(&prover_messages[round_num].denom_1)
                .map(|(d0, d1)| *d0 * *d1)
                .collect::<Vec<_>>();

            if (expected_final_numerator_claims != numerator_claims)
                || (expected_final_denom_claims != denom_claims)
            {
                return Err(LogupGkrVerificationError::InconsistentFinalClaim);
            }
        } else {
            // Reduce the sumcheck to an evaluation of the sumcheck'ed polynomial at a random point.

            let sc_proof = &sc_proofs[round_num - 1];

            let lambda: SC::EF = challenger.sample_ext_element();
            let batch_challenge: SC::EF = challenger.sample_ext_element();

            if numerator_claims
                .iter()
                .zip(denom_claims.iter())
                .rev()
                .fold(SC::EF::zero(), |acc, x| acc * batch_challenge + (lambda * *x.1 + *x.0))
                != sc_proof.claimed_sum
            {
                return Err(LogupGkrVerificationError::InconsistentSumcheckClaim);
            }

            if let Err(sc_error) = partially_verify_sumcheck_proof(sc_proof, challenger) {
                return Err(LogupGkrVerificationError::SumcheckError(sc_error));
            }

            let expected_eval = izip!(
                prover_messages[round_num].numerator_0.iter().copied(),
                prover_messages[round_num].numerator_1.iter().copied(),
                prover_messages[round_num].denom_0.iter().copied(),
                prover_messages[round_num].denom_1.iter().copied()
            )
            .map(|(n0, n1, d0, d1)| {
                let expected_numerator_eval =
                    Mle::full_lagrange_eval(&round_challenge, &sc_proof.point_and_eval.0)
                        * (n1 * d0 + n0 * d1);

                let expected_denom_eval =
                    Mle::full_lagrange_eval(&round_challenge, &sc_proof.point_and_eval.0) * d0 * d1;

                (expected_numerator_eval, expected_denom_eval)
            })
            .rev()
            .fold(SC::EF::zero(), |acc, x| acc * batch_challenge + (lambda * x.1 + x.0));

            if expected_eval != sc_proof.point_and_eval.1 {
                return Err(LogupGkrVerificationError::InconsistentEvaluation);
            }
            round_challenge = sc_proof.point_and_eval.0.clone();
        }

        let gkr_round_challenge: SC::EF = challenger.sample_ext_element();

        // Do the 2-to-1 trick.  Calculate a random linear combination of the next round's 0 and 1 points.
        numerator_claims = prover_messages[round_num]
            .numerator_0
            .iter()
            .zip(&prover_messages[round_num].numerator_1)
            .map(|(&n0, &n1)| n0 + gkr_round_challenge * (n1 - n0))
            .collect::<Vec<_>>();

        denom_claims = prover_messages[round_num]
            .denom_0
            .iter()
            .zip(&prover_messages[round_num].denom_1)
            .map(|(&d0, &d1)| d0 + gkr_round_challenge * (d1 - d0))
            .collect::<Vec<_>>();

        round_challenge.add_dimension_back(gkr_round_challenge);
    }

    Ok(GkrPointAndEvals {
        point: round_challenge,
        numerator_evals: numerator_claims,
        denom_evals: denom_claims,
    })
}

// Verifies the GKR proof for the interactions logup permutation argument.
pub(crate) fn verify_permutation_gkr_proof<SC: MachineConfig>(
    proof: &LogupGkrProof<SC::EF>,
    challenger: &mut SC::Challenger,
    sends: &[Interaction<SC::F>],
    receives: &[Interaction<SC::F>],
    permutation_challenges: (SC::EF, SC::EF),
    log_degree: Option<u32>,
    max_log_row_count: usize,
) -> Result<Point<SC::EF>, LogupGkrVerificationError> {
    let (alpha, beta) = permutation_challenges;
    let LogupGkrProof {
        prover_messages,
        sc_proofs,
        numerator_claims,
        denom_claims,
        column_openings,
        ..
    } = proof;

    let GkrPointAndEvals {
        point: eval_point,
        numerator_evals: numerator_claims,
        denom_evals: denom_claims,
    } = verify_gkr_rounds::<SC>(
        prover_messages,
        sc_proofs.as_slice(),
        numerator_claims,
        denom_claims,
        challenger,
    )?;

    let geq_val = match log_degree {
        None => SC::EF::one(),
        Some(log_d) if log_d < max_log_row_count as u32 => {
            // TODO: This will be available from the jagged parameters.
            // Create the threshold point. This should be the big-endian bit representation
            // of 2^openings.log_degree.
            let mut threshold_point_vals = vec![SC::EF::zero(); max_log_row_count];
            threshold_point_vals[max_log_row_count - (log_d as usize) - 1] = SC::EF::one();
            let threshold_point = Point::new(threshold_point_vals.into());
            full_geq(&threshold_point, &eval_point)
        }
        _ => SC::EF::zero(),
    };

    let interactions = &sends
        .iter()
        .map(|int| (int, true))
        .chain(receives.iter().map(|int| (int, false)))
        .collect::<Vec<_>>();

    let (mut numerator_guts, mut denom_guts): (Vec<_>, Vec<_>) = interactions
        .iter()
        .map(|(interaction, is_send)| {
            let (mult_value, fingerprint_value) = interaction.values_from_pair_cols::<SC::EF>(
                (&column_openings.0, &column_openings.1),
                alpha,
                &beta.powers(),
            );

            (if *is_send { mult_value } else { -mult_value }, fingerprint_value)
        })
        .unzip();

    let (numerator_dummy_evals, denom_dummy_evals): (Vec<_>, Vec<_>) = interactions
        .iter()
        .map(|(interaction, is_send)| {
            let (mult_value, fingerprint_value) = interaction.values_from_pair_cols::<SC::EF>(
                (
                    &column_openings.0.to_vec().iter().map(|_| SC::EF::zero()).collect(),
                    &column_openings
                        .1
                        .as_ref()
                        .map(|mle| mle.to_vec().iter().map(|_| SC::EF::zero()).collect()),
                ),
                alpha,
                &beta.powers(),
            );

            (if *is_send { mult_value } else { -mult_value }, fingerprint_value)
        })
        .unzip();

    numerator_guts
        .iter_mut()
        .zip(numerator_dummy_evals.iter())
        .for_each(|(gut, eval)| *gut -= *eval * geq_val);

    denom_guts
        .iter_mut()
        .zip(denom_dummy_evals.iter())
        .for_each(|(gut, eval)| *gut += (SC::EF::one() - *eval) * geq_val);

    if numerator_claims != numerator_guts || denom_claims != denom_guts {
        return Err(LogupGkrVerificationError::InconsistentIndividualEvals);
    }

    Ok(eval_point)
}
