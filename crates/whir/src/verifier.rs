use serde::Serialize;
use slop_algebra::{AbstractField, UnivariatePolynomial};
use slop_basefold::BasefoldConfig;
use slop_challenger::{CanObserve, CanSampleBits, FieldChallenger, GrindingChallenger};
use slop_commit::{TensorCs, TensorCsOpening};
use slop_multilinear::{Mle, Point};
use slop_utils::reverse_bits_len;
use std::{iter, marker::PhantomData};
use thiserror::Error;

use crate::config::WhirProofShape;

pub struct Verifier<'a, C>
where
    C: BasefoldConfig,
{
    merkle_verifier: C::Tcs,
    _marker: PhantomData<&'a C>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ParsedCommitment<C>
where
    C: BasefoldConfig,
{
    pub commitment: C::Commitment,
    pub ood_points: Vec<Point<C::EF>>,
    pub ood_answers: Vec<C::EF>,
}

#[derive(Serialize)]
pub struct SumcheckPoly<F>(pub [F; 3]);

impl<F> SumcheckPoly<F>
where
    F: AbstractField,
{
    /// Equivalent to `eval_one_plus_eval_zero` for the `UnivariatePolynomial` struct.
    pub fn sum_over_hypercube(&self) -> F {
        let [c0, c1, c2] = self.0.clone();
        c0.double() + c1 + c2
    }

    // Equivalent to `eval_at_point` for the `UnivariatePolynomial` struct.
    pub fn evaluate_at_point(&self, point: F) -> F {
        let [c0, c1, c2] = self.0.clone();
        c0 + c1 * point.clone() + c2 * point.square()
    }
}

pub type ProofOfWork<C> = <<C as BasefoldConfig>::Challenger as GrindingChallenger>::Witness;
pub type ProverMessage<C> = (SumcheckPoly<<C as BasefoldConfig>::EF>, ProofOfWork<C>);

#[derive(Serialize)]
pub struct WhirProof<C>
where
    C: BasefoldConfig,
{
    // First sumcheck
    pub initial_sumcheck_polynomials: Vec<(SumcheckPoly<C::EF>, ProofOfWork<C>)>,

    // For internal rounds
    pub commitments: Vec<ParsedCommitment<C>>,
    pub merkle_proofs: Vec<TensorCsOpening<C::Tcs>>,
    pub query_proof_of_works: Vec<ProofOfWork<C>>,
    pub sumcheck_polynomials: Vec<Vec<ProverMessage<C>>>,

    // Final round
    pub final_polynomial: Vec<C::EF>,
    pub final_merkle_proof: TensorCsOpening<C::Tcs>,
    pub final_sumcheck_polynomials: Vec<ProverMessage<C>>,
    pub final_pow: ProofOfWork<C>,
    pub _config: PhantomData<C>,
}

#[derive(Debug, Error)]
pub enum WhirProofError {
    #[error("invalid number of OOD samples: expected {0}, got {1}")]
    InvalidNumberOfOODSamples(usize, usize),
    #[error("sumcheck error")]
    SumcheckError(#[from] SumcheckError),
    #[error("invalid proof of work")]
    PowError,
    #[error("invalid OOD evaluation")]
    InvalidOOD,
    #[error("invalid Merkle authentication")]
    InvalidMerkleAuthentication,
    #[error("invalid degree of final polynomial: expected {0}, got {1}")]
    InvalidDegreeFinalPolynomial(usize, usize),
    #[error("final query mismatch")]
    FinalQueryMismatch,
}

#[derive(Debug, Error)]
pub enum SumcheckError {
    #[error("expected {0} sumcheck polynomials, got {1}")]
    InvalidNumberOfSumcheckPoly(usize, usize),
    #[error("invalid sum")]
    InvalidSum,
    #[error("invalid proof of work")]
    PowError,
}

pub fn map_to_pow<F: AbstractField>(mut elem: F, len: usize) -> Point<F> {
    assert!(len > 0);
    let mut res = Vec::with_capacity(len);
    for _ in 0..len {
        res.push(elem.clone());
        elem = elem.square();
    }
    res.reverse();
    res.into()
}

impl<'a, C> Verifier<'a, C>
where
    C: BasefoldConfig,
{
    pub const fn new(merkle_verifier: C::Tcs) -> Self {
        Self { _marker: PhantomData, merkle_verifier }
    }

    pub fn observe_commitment(
        &self,
        commitment: &ParsedCommitment<C>,
        challenger: &mut C::Challenger,
        config: &WhirProofShape<C::F>,
    ) -> Result<(), WhirProofError> {
        challenger.observe(commitment.commitment.clone());
        let ood_points: Vec<Point<C::EF>> = (0..config.starting_ood_samples)
            .map(|_| {
                (0..config.num_variables)
                    .map(|_| challenger.sample_ext_element())
                    .collect::<Vec<C::EF>>()
                    .into()
            })
            .collect();

        if ood_points != commitment.ood_points {
            return Err(WhirProofError::InvalidOOD);
        }

        challenger.observe_ext_element_slice(&commitment.ood_answers);

        Ok(())
    }

    /// The claim is that < f, v > = claim.
    /// WHIR reduces it to a claim that v(point) = claim'
    pub fn verify(
        &self,
        commitment: &ParsedCommitment<C>,
        claim: C::EF,
        proof: &WhirProof<C>,
        challenger: &mut C::Challenger,
        config: &WhirProofShape<C::F>,
    ) -> Result<(Point<C::EF>, C::EF), WhirProofError> {
        let n_rounds = config.round_parameters.len();

        // Check that the number of OOD answers in the proof matches the expected value.
        if commitment.ood_answers.len() != config.starting_ood_samples {
            return Err(WhirProofError::InvalidNumberOfOODSamples(
                config.starting_ood_samples,
                commitment.ood_answers.len(),
            ));
        }

        // Batch the initial claim with the OOD claims of the commitment
        let claim_batching_randomness: C::EF = challenger.sample_ext_element();
        let claimed_sum: C::EF = claim_batching_randomness
            .powers()
            .zip(iter::once(&claim).chain(&commitment.ood_answers))
            .map(|(r, &v)| v * r)
            .sum();

        // Initialize the collection of points at which we will need to compute the monomial basis
        // polynomial evaluations.
        let mut final_evaluation_points = vec![commitment.ood_points.clone()];

        // Check the initial sumcheck.
        let (mut folding_randomness, mut claimed_sum) = self
            .verify_sumcheck(
                &proof.initial_sumcheck_polynomials,
                claimed_sum,
                config.starting_folding_factor,
                &config.starting_folding_pow_bits,
                challenger,
            )
            .map_err(WhirProofError::SumcheckError)?;

        // This contains all the sumcheck randomnesses (these are the alphas)
        let mut concatenated_folding_randomness = folding_randomness.clone();

        // This contains all the batching randomness for sumcheck (these are the epsilons) for
        // batching in- and out-of-domain claims from round to round.
        let mut all_claim_batching_randomness = vec![claim_batching_randomness];

        // This is relative to the previous commitment (i.e. prev_commitment has a domain size of this size)
        let mut domain_size =
            config.num_variables - config.starting_folding_factor + config.starting_log_inv_rate;
        let mut generator = config.domain_generator;
        let mut prev_commitment = commitment;

        let mut prev_folding_factor = config.starting_folding_factor;
        let mut num_variables = config.num_variables - config.starting_folding_factor;

        for round_index in 0..n_rounds {
            let round_params = &config.round_parameters[round_index];
            let new_commitment = &proof.commitments[round_index];
            if new_commitment.ood_answers.len() != round_params.ood_samples {
                return Err(WhirProofError::InvalidNumberOfOODSamples(
                    round_params.ood_samples,
                    new_commitment.ood_answers.len(),
                ));
            }

            // Observe the commitment
            challenger.observe(new_commitment.commitment.clone());

            // Squeeze the ood points
            let ood_points: Vec<Point<C::EF>> = (0..round_params.ood_samples)
                .map(|_| {
                    (0..num_variables)
                        .map(|_| challenger.sample_ext_element())
                        .collect::<Vec<C::EF>>()
                        .into()
                })
                .collect();

            if ood_points != new_commitment.ood_points {
                return Err(WhirProofError::InvalidOOD);
            }

            // Absorb the OOD answers
            challenger.observe_ext_element_slice(&new_commitment.ood_answers);

            // Squeeze the STIR queries
            let id_query_indices = (0..round_params.num_queries)
                .map(|_| challenger.sample_bits(domain_size))
                .collect::<Vec<_>>();
            let id_query_values: Vec<C::F> = id_query_indices
                .iter()
                .map(|val| reverse_bits_len(*val, domain_size))
                .map(|pos| generator.exp_u64(pos as u64))
                .collect();
            let claim_batching_randomness: C::EF = challenger.sample_ext_element();

            if !challenger.check_witness(
                round_params.queries_pow_bits.ceil() as usize,
                proof.query_proof_of_works[round_index],
            ) {
                return Err(WhirProofError::PowError);
            }

            let merkle_proof = &proof.merkle_proofs[round_index];
            self.merkle_verifier
                .verify_tensor_openings(
                    &prev_commitment.commitment,
                    &id_query_indices,
                    merkle_proof,
                    domain_size,
                )
                .map_err(|_| WhirProofError::InvalidMerkleAuthentication)?;

            // Chunk the Merkle openings into chunks of size `1<<prev_folding_factor`
            // so that the verifier can induce in-domain evaluation claims about the next codeword.
            // Except in the first round, the opened values in the Merkle proof are secretly extension
            // field elements, so we have to reinterpret them as such. (The Merkle tree API commits
            // to and opens only base-field values.)
            let merkle_read_values: Vec<Mle<C::EF>> = if round_index != 0 {
                merkle_proof
                    .values
                    .clone()
                    .into_buffer()
                    .into_extension::<C::EF>()
                    .to_vec()
                    .chunks_exact(1 << prev_folding_factor)
                    .map(|v| Mle::new(v.to_vec().into()))
                    .collect()
            } else {
                merkle_proof
                    .values
                    .clone()
                    .into_buffer()
                    .to_vec()
                    .into_iter()
                    .map(C::EF::from)
                    .collect::<Vec<_>>()
                    .chunks_exact(1 << prev_folding_factor)
                    .map(|v| Mle::new(v.to_vec().into()))
                    .collect()
            };

            // Compute the STIR values by reading the merkle values and folding across the column.
            let stir_values: Vec<C::EF> = merkle_read_values
                .iter()
                .map(|coeffs| coeffs.blocking_eval_at(&folding_randomness.clone().into())[0])
                .collect();

            // Update the claimed sum using the STIR values and the OOD answers.
            claimed_sum = claim_batching_randomness
                .powers()
                .zip(
                    iter::once(&claimed_sum).chain(&new_commitment.ood_answers).chain(&stir_values),
                )
                .map(|(r, &v)| r * v)
                .sum();

            (folding_randomness, claimed_sum) = self
                .verify_sumcheck(
                    &proof.sumcheck_polynomials[round_index],
                    claimed_sum,
                    round_params.folding_factor,
                    &round_params.pow_bits,
                    challenger,
                )
                .map_err(WhirProofError::SumcheckError)?;

            // Prepend the folding randomness from the sumcheck into the combined folding randomness.
            concatenated_folding_randomness =
                [folding_randomness.clone(), concatenated_folding_randomness].concat();

            all_claim_batching_randomness.push(claim_batching_randomness);

            // Add both the in-domain and out-of-domain claims to the set of final evaluation points.
            final_evaluation_points.push(
                [
                    ood_points.clone(),
                    id_query_values
                        .into_iter()
                        .map(|point| map_to_pow(point, num_variables).to_extension())
                        .collect(),
                ]
                .concat(),
            );

            domain_size = round_params.evaluation_domain_log_size;
            prev_commitment = new_commitment;
            prev_folding_factor = round_params.folding_factor;
            generator = generator.square();
            num_variables -= round_params.folding_factor;
        }

        // Now, we want to verify the final evaluations
        challenger.observe_ext_element_slice(&proof.final_polynomial);
        if proof.final_polynomial.len() > 1 << config.final_poly_log_degree {
            return Err(WhirProofError::InvalidDegreeFinalPolynomial(
                1 << config.final_poly_log_degree,
                proof.final_polynomial.len(),
            ));
        }
        let final_poly = proof.final_polynomial.clone();
        let final_poly_uv = UnivariatePolynomial::new(final_poly.clone());

        let final_id_indices = (0..config.final_queries)
            .map(|_| challenger.sample_bits(domain_size))
            .collect::<Vec<_>>();
        let final_id_values: Vec<C::F> = final_id_indices
            .iter()
            .map(|val| reverse_bits_len(*val, domain_size))
            .map(|pos| generator.exp_u64(pos as u64))
            .collect();

        self.merkle_verifier
            .verify_tensor_openings(
                &prev_commitment.commitment,
                &final_id_indices,
                &proof.final_merkle_proof,
                domain_size,
            )
            .map_err(|_| WhirProofError::InvalidMerkleAuthentication)?;

        let final_merkle_read_values: Vec<Mle<C::EF>> = proof
            .final_merkle_proof
            .values
            .clone()
            .into_buffer()
            .into_extension::<C::EF>()
            .to_vec()
            .chunks_exact(1 << prev_folding_factor)
            .map(|v| Mle::new(v.to_vec().into()))
            .collect();

        // Compute the STIR values by reading the merkle values and folding across the column
        let final_stir_values: Vec<C::EF> = final_merkle_read_values
            .iter()
            .map(|coeffs| coeffs.blocking_eval_at(&folding_randomness.clone().into())[0])
            .collect();

        if final_stir_values
            != final_id_values
                .into_iter()
                .map(|val| final_poly_uv.eval_at_point(val.into()))
                .collect::<Vec<_>>()
        {
            return Err(WhirProofError::FinalQueryMismatch);
        }

        if !challenger.check_witness(config.final_pow_bits.ceil() as usize, proof.final_pow) {
            return Err(WhirProofError::PowError);
        }

        (folding_randomness, claimed_sum) = self
            .verify_sumcheck(
                &proof.final_sumcheck_polynomials,
                claimed_sum,
                config.final_poly_log_degree,
                &config.final_folding_pow_bits,
                challenger,
            )
            .map_err(WhirProofError::SumcheckError)?;

        concatenated_folding_randomness =
            [folding_randomness.clone(), concatenated_folding_randomness].concat();

        let f = Mle::new(proof.final_polynomial.clone().into())
            .blocking_eval_at(&Point::from(folding_randomness))[0];

        let mut summand = C::EF::zero();
        for (i, eval_points) in final_evaluation_points.into_iter().enumerate() {
            let combination_randomness = all_claim_batching_randomness[i];
            let len = eval_points[0].len();
            let eval_randomness: Point<C::EF> =
                concatenated_folding_randomness[..len].to_vec().into();

            let sum_modification = combination_randomness
                .powers()
                .skip(1)
                .zip(eval_points)
                .map(|(r, point)| r * { Mle::full_monomial_basis_eq(&point, &eval_randomness) })
                .sum::<C::EF>();

            summand += sum_modification;
        }

        // This is the claimed value of the query vector. It is trusted and assumed to be easily
        // computable by the verifier.
        let claimed_value = claimed_sum / f - summand;

        Ok((concatenated_folding_randomness.into(), claimed_value))
    }

    // Verifies the sumcheck polynomial, returning the new claim value
    fn verify_sumcheck(
        &self,
        sumcheck_polynomials: &[(SumcheckPoly<C::EF>, ProofOfWork<C>)],
        mut claimed_sum: C::EF,
        rounds: usize,
        pow_bits: &[f64],
        challenger: &mut C::Challenger,
    ) -> Result<(Vec<C::EF>, C::EF), SumcheckError> {
        if sumcheck_polynomials.len() != rounds {
            return Err(SumcheckError::InvalidNumberOfSumcheckPoly(
                rounds,
                sumcheck_polynomials.len(),
            ));
        }
        let mut randomness = Vec::with_capacity(rounds);
        for i in 0..rounds {
            let (sumcheck_poly, pow_witness) = &sumcheck_polynomials[i];
            challenger.observe_ext_element_slice(&sumcheck_poly.0);
            if sumcheck_poly.sum_over_hypercube() != claimed_sum {
                return Err(SumcheckError::InvalidSum);
            }

            let folding_randomness_single: C::EF = challenger.sample_ext_element();
            randomness.push(folding_randomness_single);

            if !challenger.check_witness(pow_bits[i].ceil() as usize, *pow_witness) {
                return Err(SumcheckError::PowError);
            }

            claimed_sum = sumcheck_poly.evaluate_at_point(folding_randomness_single);
        }

        randomness.reverse();
        Ok((randomness, claimed_sum))
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use slop_algebra::AbstractField;
    use slop_multilinear::monomial_basis_evals_blocking;

    use crate::verifier::map_to_pow;

    type F = slop_koala_bear::KoalaBear;
    #[test]
    fn test_monomial_basis_evals_and_map_to_pow() {
        let mut rng = rand::thread_rng();
        let x = rng.gen::<F>();
        let point = map_to_pow(x, 12);
        let select = monomial_basis_evals_blocking(&point);
        let select_vec = select.as_slice().to_vec();

        for (i, elem) in select_vec.iter().enumerate() {
            assert_eq!(*elem, x.exp_u64(i as u64));
        }
    }
}
