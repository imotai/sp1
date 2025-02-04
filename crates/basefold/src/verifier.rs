use std::iter::repeat;

use itertools::izip;
use slop_algebra::{ExtensionField, TwoAdicField};
use slop_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use slop_commit::{ExtensionMmcs, Mmcs};
use slop_fri::{
    verifier::{FriChallenges, FriError},
    FriConfig, QueryProof,
};
use slop_matrix::Dimensions;
use slop_multilinear::{MultilinearPcsBatchVerifier, Point};
use slop_utils::reverse_bits_len;

use crate::{BaseFoldError, BaseFoldPcs, BaseFoldProof};

type DataFromProof<K> = (K, K, usize);

/// The FRI verifier for a single query. We modify this from Plonky3 to be compatible with opening
/// only a single vector.
pub(crate) fn verify_query<F, M>(
    config: &FriConfig<M>,
    commit_phase_commits: &[M::Commitment],
    mut index: usize,
    proof: &QueryProof<F, M>,
    betas: &[F],
    data_from_proof: DataFromProof<F>,
) -> Result<(), FriError<M::Error>>
where
    F: TwoAdicField,
    M: Mmcs<F>,
{
    let (final_poly, reduced_opening, log_max_height) = data_from_proof;

    let mut folded_eval = reduced_opening;
    let mut x = F::two_adic_generator(log_max_height)
        .exp_u64(reverse_bits_len(index, log_max_height) as u64);

    for (log_folded_height, commit, step, &beta) in izip!(
        (config.log_blowup..log_max_height + 1).rev(),
        commit_phase_commits,
        &proof.commit_phase_openings,
        betas,
    ) {
        let index_sibling = index ^ 1;
        let index_pair = index >> 1;

        let mut evals = vec![folded_eval; 2];
        evals[index_sibling % 2] = step.sibling_value;

        let dims = &[Dimensions { width: 2, height: 1 << log_folded_height }];
        config
            .mmcs
            .verify_batch(commit, dims, index_pair, &[evals.clone()], &step.opening_proof)
            .map_err(FriError::CommitPhaseMmcsError)?;

        let mut xs = [x; 2];
        xs[index_sibling % 2] *= F::two_adic_generator(1);
        // interpolate and evaluate at beta
        folded_eval = evals[0] + (beta - xs[0]) * (evals[1] - evals[0]) / (xs[1] - xs[0]);

        index = index_pair;
        x = x.square();
    }

    debug_assert!(index < config.blowup(), "index was {}", index);
    debug_assert_eq!(x.exp_power_of_2(config.log_blowup), F::one());

    if folded_eval != final_poly {
        return Err(FriError::FinalPolyMismatch);
    }

    Ok(())
}

impl<K: TwoAdicField, EK: TwoAdicField + ExtensionField<K>, InnerMmcs: Mmcs<K>, Challenger>
    MultilinearPcsBatchVerifier for BaseFoldPcs<K, EK, InnerMmcs, Challenger>
where
    Challenger: GrindingChallenger
        + FieldChallenger<K>
        + CanObserve<<ExtensionMmcs<K, EK, InnerMmcs> as Mmcs<EK>>::Commitment>,
    <ExtensionMmcs<K, EK, InnerMmcs> as Mmcs<EK>>::Commitment: Eq,
{
    type Proof = BaseFoldProof<
        K,
        EK,
        InnerMmcs::Commitment,
        Challenger::Witness,
        InnerMmcs::Proof,
        InnerMmcs,
    >;
    type Commitment = <ExtensionMmcs<K, EK, InnerMmcs> as Mmcs<EK>>::Commitment;
    type Error = BaseFoldError<<ExtensionMmcs<K, EK, InnerMmcs> as Mmcs<EK>>::Error>;
    type F = K;
    type EF = EK;
    type Challenger = Challenger;
    type FinalizeCommit = ();

    /// Verify a BaseFold proof of a claim: commitment D represents a batch of matrices whose
    /// columns encode multilinear polynomials `g_i` whose joint evaluations at `point` are
    /// `expected_evals`.
    ///
    /// As in Plonky3, the BaseFold protocol is run on a random linear combination of the committed
    /// multilinear polynomials. The verifier check consistency of the FRI query proofs with the
    /// committed-to values of the matrices.
    ///
    /// The verifier proceeds in rounds. First, it checks that the claimed evaluation of `g` at
    /// `point` is consistent with the proof's claims about `g(X_0, ..., X_{d-2}, 0)` and
    /// `g(X_0, ..., X_{d-2}, 1)`. Then, it takes a random linear combination of those
    /// evaluations to produce a claim about a multilinear in one fewer variable. It iterates
    /// these two checks d times in total to produce a claim about a 0-variate polynomial.
    ///
    /// Then, the verifier verifies the FRI queries sent by the prover.
    ///
    /// Finally, the verifier checks that the prover's claim about the final polynomial in FRI is
    /// consistent with the prover's implied claim about the 0-variate multilinear.
    fn verify_trusted_evaluations(
        &self,
        mut point: Point<EK>,
        eval_claims: &[&[&[EK]]],
        commitments: &[Self::Commitment],
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        let batching_challenge = challenger.sample_ext_element::<EK>();

        let eval_claim: EK = eval_claims
            .iter()
            .flat_map(|eval_set| eval_set.iter().flat_map(|evals| evals.iter()))
            .zip(batching_challenge.powers())
            .map(|(eval, batch_power)| *eval * batch_power)
            .sum();

        // Assert correctness of shape.
        if proof.commitments.len() != proof.univariate_messages.len()
            || proof.query_phase_proofs.len() != self.fri_config.num_queries
        {
            return Err(BaseFoldError::IncorrectShape);
        }

        // The prover messages correspond to fixing the last coordinate first, so we reverse the
        // underlying point for the verification.
        point.reverse();

        // Sample the challenges used for FRI folding and BaseFold random linear combinations.
        let betas = proof
            .commitments
            .iter()
            .zip(proof.univariate_messages.iter())
            .map(|(commitment, poly)| {
                poly.iter().copied().for_each(|x| challenger.observe_ext_element(x));
                challenger.observe(commitment.clone());
                challenger.sample_ext_element::<EK>()
            })
            .collect::<Vec<_>>();

        // Check the consistency of the first univariate message with the claimed evaluation. The
        // first_poly is supposed to be `vals(X_0, X_1, ..., X_{d-1}, 0), vals(X_0, X_1, ...,
        // X_{d-1}, 1)`. Given this, the claimed evaluation should be `(1 - X_d) *
        // first_poly[0] + X_d * first_poly[1]`.
        let first_poly = proof.univariate_messages[0];
        if eval_claim != (EK::one() - *point[0]) * first_poly[0] + *point[0] * first_poly[1] {
            return Err(BaseFoldError::Sumcheck);
        };

        // Fold the two messages into a single evaluation claim for the next round, using the
        // sampled randomness.
        let mut expected_eval = first_poly[0] + betas[0] * first_poly[1];

        // Check round-by-round consistency between the successive sumcheck univariate messages.
        for (i, (poly, beta)) in
            proof.univariate_messages[1..].iter().zip(betas[1..].iter()).enumerate()
        {
            // The check is similar to the one for `first_poly`.
            let i = i + 1;
            if expected_eval != (EK::one() - *point[i]) * poly[0] + *point[i] * poly[1] {
                return Err(BaseFoldError::Sumcheck);
            }

            // Fold the two pieces of the message.
            expected_eval = poly[0] + *beta * poly[1];
        }

        challenger.observe_ext_element(proof.final_poly);

        // Check proof of work (grinding to find a number that hashes to have
        // `self.config.proof_of_work_bits` zeroes at the beginning).
        if !challenger.check_witness(self.fri_config.proof_of_work_bits, proof.pow_witness) {
            return Err(BaseFoldError::Pow);
        }

        let log_len = proof.commitments.len();

        // Sample query indices for the FRI query IOPP part of BaseFold. This part is very similar
        // to the corresponding part in the Plonky3 verifier.
        let query_indices = (0..self.fri_config.num_queries)
            .map(|_| challenger.sample_bits(log_len + self.fri_config.log_blowup))
            .collect::<Vec<_>>();

        let challenges = FriChallenges { query_indices, betas };

        izip!(
            proof.query_openings.iter(),
            challenges.query_indices.clone(),
            proof.query_phase_proofs.iter(),
        )
        .map(|(query_opening_proof, index, query_phase_proof)| {
            let mut batch_challenge_power = 0;
            let mut batch_eval: EK = EK::zero();
            let result = query_opening_proof.iter().zip(commitments.iter()).try_for_each(
                |((openings, opening_proof), commitment)| -> Result<(), Self::Error> {
                    // Verify the openings of the individual columns.
                    self.inner_mmcs
                        .verify_batch(
                            commitment,
                            &repeat(Dimensions {
                                width: 0,
                                height: 1
                                    << (self.fri_config.log_blowup
                                        + proof.univariate_messages.len()),
                            })
                            .take(openings.len())
                            .collect::<Vec<_>>(),
                            index,
                            &openings.clone(),
                            opening_proof,
                        )
                        .map_err(BaseFoldError::Mmcs)?;

                    // Fold the openings into an index opening value for the single FRI polynomial.
                    let batch_openings = openings.iter().flat_map(|set| set.iter());
                    let num_columns = batch_openings.clone().count();
                    batch_eval += batch_openings
                        .zip(batching_challenge.powers().skip(batch_challenge_power))
                        .map(|(opening, batch_power)| batch_power * *opening)
                        .sum::<EK>();

                    batch_challenge_power += num_columns;

                    Ok(())
                },
            );

            verify_query(
                &self.fri_config,
                &proof.commitments,
                index,
                &query_phase_proof.1,
                &challenges.betas,
                (proof.final_poly, query_phase_proof.0, log_len + self.fri_config.log_blowup),
            )
            .map_err(BaseFoldError::Fri)?;

            // Check consistency of the batch computation with the FRI query proof opening.
            if batch_eval == query_phase_proof.0 {
                result
            } else {
                Err(BaseFoldError::Batching)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

        // The final consistency check between the FRI messages and the partial evaluation messages.
        if proof.final_poly
            != proof.univariate_messages.last().unwrap()[0]
                + *challenges.betas.last().unwrap() * proof.univariate_messages.last().unwrap()[1]
        {
            Err(BaseFoldError::Sumcheck)
        } else {
            Ok(())
        }
    }

    fn incorporate_finalize_data(
        &self,
        _data: Self::FinalizeCommit,
        _challenger: &mut Self::Challenger,
    ) {
        // No-op.
    }
}
