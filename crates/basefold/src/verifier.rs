use itertools::{izip, Itertools};
use p3_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use p3_commit::Mmcs;
use p3_fri::{
    verifier::{FriChallenges, FriError},
    FriConfig, QueryProof,
};
use p3_matrix::Dimensions;
use p3_util::reverse_bits_len;
use spl_algebra::TwoAdicField;
use spl_multi_pcs::{MultilinearPcs, Point};

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

impl<K: TwoAdicField, M: Mmcs<K>, Challenger> MultilinearPcs<K, Challenger>
    for BaseFoldPcs<K, M, Challenger>
where
    Challenger: GrindingChallenger + FieldChallenger<K> + CanObserve<M::Commitment>,
    M::Commitment: Eq,
{
    type Proof = BaseFoldProof<K, M, Challenger::Witness>;
    type Commitment = M::Commitment;
    type Error = BaseFoldError<M::Error>;

    /// Verify a BaseFold proof of a claim: commitment D represents a multilinear polynomial `g` whose
    /// evaluation at `point` is `proof.eval`.
    ///
    /// The verifier proceeds in rounds. First, it checks that the claimed evaluation of `g` at `point`
    /// is consistent with the proof's claims about `g(X_0, ..., X_{d-2}, 0)` and `g(X_0, ..., X_{d-2}, 1)`.
    /// Then, it takes a random linear combination of those evaluations to produce a claim about a
    /// multilinear in one fewer variable. It iterates these two checks d times in total to produce
    /// a claim about a 0-variate polynomial.
    ///
    /// Then, the verifier verifies the FRI queries sent by the prover.
    ///
    /// Finally, the verifier checks that the prover's claim about the final polynomial in FRI is
    /// consistent with the prover's implied claim about the 0-variate multilinear.
    fn verify(
        &self,
        mut point: Point<K>,
        eval_claims: Vec<K>,
        commitment: Self::Commitment,
        proof: &Self::Proof,
        challenger: &mut Challenger,
    ) -> Result<(), Self::Error> {
        if commitment != proof.commitments[0] {
            return Err(BaseFoldError::IncorrectShape);
        }

        // We don't support batching for BaseFold yet.
        assert!(eval_claims.len() == 1);
        let eval_claim = eval_claims[0];

        // Assert correctness of shape.
        if proof.commitments.len() != proof.univariate_messages.len()
            || proof.query_phase_proofs.len() != self.config.fri_config.num_queries
        {
            return Err(BaseFoldError::IncorrectShape);
        }

        // The prover messages correspond to fixing the last coordinate first, so we reverse the
        // underlying point for the verification.
        point.0.reverse();

        // Sample the challenges used for FRI folding and BaseFold random linear combinations.
        let betas = proof
            .commitments
            .iter()
            .zip(proof.univariate_messages.iter())
            .map(|(commitment, poly)| {
                challenger.observe_slice(poly);
                challenger.observe(commitment.clone());
                challenger.sample_ext_element::<K>()
            })
            .collect_vec();

        // Check the consistency of the first univariate message with the claimed evaluation. The
        // first_poly is supposed to be `vals(X_0, X_1, ..., X_{d-1}, 0), vals(X_0, X_1, ..., X_{d-1}, 1)`.
        // Given this, the claimed evaluation should be `(1 - X_d) * first_poly[0] + X_d * first_poly[1]`.
        let first_poly = proof.univariate_messages[0];
        if eval_claim != (K::one() - point.0[0]) * first_poly[0] + point.0[0] * first_poly[1] {
            return Err(BaseFoldError::Sumcheck);
        };

        // Fold the two messages into a single evaluation claim for the next round, using the sampled
        // randomness.
        let mut expected_eval = first_poly[0] + betas[0] * first_poly[1];

        // Check round-by-round consistency between the successive sumcheck univariate messages.
        for (i, (poly, beta)) in
            proof.univariate_messages[1..].iter().zip(betas[1..].iter()).enumerate()
        {
            // The check is similar to the one for `first_poly`.
            let i = i + 1;
            if expected_eval != (K::one() - point.0[i]) * poly[0] + point.0[i] * poly[1] {
                return Err(BaseFoldError::Sumcheck);
            }

            // Fold the two pieces of the message.
            expected_eval = poly[0] + *beta * poly[1];
        }

        // Check proof of work (grinding to find a number that hashes to have
        // `self.config.proof_of_work_bits` zeroes at the beginning).
        if !challenger.check_witness(self.config.fri_config.proof_of_work_bits, proof.pow_witness) {
            return Err(BaseFoldError::Pow);
        }

        let log_len = proof.commitments.len();

        // Sample query indices for the FRI query IOPP part of BaseFold. This part is very similar to
        // the corresponding part in the Plonky3 verifier.
        let query_indices = (0..self.config.fri_config.num_queries)
            .map(|_| challenger.sample_bits(log_len + self.config.fri_config.log_blowup))
            .collect_vec();

        let challenges = FriChallenges { query_indices, betas };

        // Verify the FRI queries.
        challenges
            .query_indices
            .iter()
            .zip(proof.query_phase_proofs.iter())
            .map(|(idx, (opening, query_proof))| {
                verify_query(
                    &self.config.fri_config,
                    &proof.commitments,
                    *idx,
                    query_proof,
                    &challenges.betas,
                    (proof.final_poly, *opening, log_len + self.config.fri_config.log_blowup),
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(BaseFoldError::Fri)?;

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
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_fri::FriConfig;
    use p3_merkle_tree::{self, FieldMerkleTreeMmcs};
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use rand::Rng;
    use spl_algebra::Field;
    use spl_multi_pcs::{Mle, MultilinearPcs};

    use crate::{BaseFoldPcs, BaseFoldProver, Point};

    use crate::BaseFoldConfig;

    type F = BabyBear;

    type Perm = Poseidon2<F, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs =
        FieldMerkleTreeMmcs<<F as Field>::Packing, <F as Field>::Packing, MyHash, MyCompress, 8>;
    type Challenger = DuplexChallenger<F, Perm, 16, 8>;

    #[test]
    fn test_prover() {
        let mut rng = rand::thread_rng();

        (1..13).for_each(|i| {
            println!("Testing an instance with {} variables.", i);
            let num_variables = i;

            let vals = (0..(1 << num_variables)).map(|_| rng.gen::<F>()).collect_vec();
            let perm = Perm::new_from_rng_128(
                Poseidon2ExternalMatrixGeneral,
                DiffusionMatrixBabyBear,
                &mut rng,
            );
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm.clone());
            let mmcs = ValMmcs::new(hash, compress);
            let config = BaseFoldConfig {
                fri_config: FriConfig {
                    log_blowup: 1,
                    num_queries: 10,
                    proof_of_work_bits: 8,
                    mmcs,
                },
            };

            let pcs = BaseFoldPcs::<F, ValMmcs, Challenger>::new(config);

            let new_eval_point = Point::new((0..num_variables).map(|_| rng.gen::<F>()).collect());

            let expected_eval = Mle::new(vals.clone()).eval_at_point(&new_eval_point);

            let prover = BaseFoldProver::new(pcs);

            let (commit, data) = prover.commit(vals.clone());

            let proof = prover.prove_evaluation(
                data,
                new_eval_point.clone(),
                expected_eval,
                &mut Challenger::new(perm.clone()),
            );

            prover
                .pcs
                .verify(
                    new_eval_point,
                    vec![expected_eval],
                    commit,
                    &proof,
                    &mut Challenger::new(perm.clone()),
                )
                .unwrap();
        });
    }
}
