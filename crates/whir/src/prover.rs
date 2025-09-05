use std::{iter, sync::Arc};

use crate::{
    config::WhirProofShape,
    verifier::{map_to_pow, ParsedCommitment, ProofOfWork, SumcheckPoly, WhirProof},
    RoundConfig,
};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use slop_algebra::{AbstractField, ExtensionField, Field, TwoAdicField};
use slop_alloc::CpuBackend;
use slop_basefold::BasefoldConfig;
use slop_challenger::{CanObserve, CanSampleBits, FieldChallenger, GrindingChallenger, IopCtx};
use slop_commit::Message;
use slop_dft::Dft;
use slop_merkle_tree::{ComputeTcsOpenings, MerkleTreeOpening, TensorCsProver};
use slop_multilinear::{monomial_basis_evals_blocking, Mle, Point};
use slop_tensor::Tensor;
use slop_utils::reverse_bits_len;

fn batch_dft<D, F, EF>(dft: &D, data: Tensor<EF>, log_blowup: usize) -> Tensor<EF>
where
    F: Field,
    EF: ExtensionField<F>,
    D: Dft<F>,
{
    assert_eq!(data.sizes().len(), 2, "Expected a 2D tensor");

    let base_tensor = data.flatten_to_base();
    let base_tensor =
        dft.dft(&base_tensor, log_blowup, slop_dft::DftOrdering::BitReversed, 0).unwrap();
    base_tensor.into_extension()
}

pub struct Prover<GC, MerkleProver, D, C>
where
    GC: IopCtx,
    C: BasefoldConfig<GC>,
    MerkleProver: TensorCsProver<GC, CpuBackend, MerkleConfig = C::Tcs>
        + ComputeTcsOpenings<GC, CpuBackend, MerkleConfig = C::Tcs>,
{
    dft: D,
    merkle_prover: MerkleProver,
    _marker: std::marker::PhantomData<(C, GC)>,
}

pub struct WitnessData<GC, MerkleProver, BC: BasefoldConfig<GC>>
where
    GC: IopCtx,
    MerkleProver: TensorCsProver<GC, CpuBackend> + ComputeTcsOpenings<GC, CpuBackend>,
{
    parsed_commitment: ParsedCommitment<GC, BC>,
    polynomial: Mle<GC::F>,
    committed_data: Tensor<GC::F>,
    commitment_data: MerkleProver::ProverData,
}

impl<GC, C, MerkleProver, D> Prover<GC, MerkleProver, D, C>
where
    GC: IopCtx,
    D: Dft<GC::F>,
    C: BasefoldConfig<GC>,
    MerkleProver: TensorCsProver<GC, CpuBackend, MerkleConfig = C::Tcs>
        + ComputeTcsOpenings<GC, CpuBackend, MerkleConfig = C::Tcs>,
{
    pub async fn new(dft: D, merkle_prover: MerkleProver) -> Self {
        Self { dft, merkle_prover, _marker: std::marker::PhantomData }
    }

    pub async fn commit(
        &self,
        polynomial: Mle<GC::F>,
        challenger: &mut GC::Challenger,
        config: &WhirProofShape<GC::F>,
    ) -> (ParsedCommitment<GC, C>, WitnessData<GC, MerkleProver, C>) {
        let num_variables = polynomial.num_variables() as usize;
        let inner_evals = polynomial.guts().clone().reshape([
            (1 << num_variables) / (1 << config.starting_folding_factor),
            1 << config.starting_folding_factor,
        ]);

        let encoding = batch_dft(&self.dft, inner_evals, config.starting_log_inv_rate);

        let (commitment, prover_data) =
            self.merkle_prover.commit_tensors(encoding.clone().into()).await.unwrap();

        challenger.observe(commitment);
        let ood_points: Vec<Point<GC::EF>> = (0..config.starting_ood_samples)
            .map(|_| {
                (0..num_variables)
                    .map(|_| challenger.sample_ext_element())
                    .collect::<Vec<GC::EF>>()
                    .into()
            })
            .collect();

        let ood_answers: Vec<GC::EF> = ood_points
            .iter()
            .map(|point| polynomial.blocking_monomial_basis_eval_at(point)[0])
            .collect();

        challenger.observe_ext_element_slice(&ood_answers);

        let parsed_commitment =
            ParsedCommitment { commitment, ood_points, ood_answers, _marker: Default::default() };

        (
            parsed_commitment.clone(),
            WitnessData {
                parsed_commitment,
                polynomial,
                committed_data: encoding,
                commitment_data: prover_data,
            },
        )
    }

    pub async fn prove(
        &self,
        query_vector: Mle<GC::EF>,
        witness_data: WitnessData<GC, MerkleProver, C>,
        challenger: &mut GC::Challenger,
        config: &WhirProofShape<GC::F>,
    ) -> WhirProof<GC, C> {
        let n_rounds = config.round_parameters.len();

        // Compute <f, v>
        let claim: GC::EF = witness_data
            .polynomial
            .hypercube_iter()
            .zip(query_vector.hypercube_iter())
            .map(|(a, b)| b[0] * a[0])
            .sum();

        let claim_batching_randomness: GC::EF = challenger.sample_ext_element();
        let claimed_sum: GC::EF = claim_batching_randomness
            .powers()
            .zip(iter::once(&claim).chain(&witness_data.parsed_commitment.ood_answers))
            .map(|(r, &v)| r * v)
            .sum();

        let mut sumcheck_prover = SumcheckProver::<GC, GC::F>::new(
            witness_data.polynomial.clone(),
            query_vector,
            witness_data.parsed_commitment.ood_points.clone(),
            claim_batching_randomness,
        )
        .await;

        let (initial_sumcheck_polynomials, mut folding_randomness, mut claimed_sum) =
            sumcheck_prover
                .compute_sumcheck_polynomials(
                    claimed_sum,
                    config.starting_folding_factor,
                    &config.starting_folding_pow_bits,
                    challenger,
                )
                .await;

        let mut generator = config.domain_generator;
        let mut parsed_commitments = Vec::with_capacity(n_rounds);
        let mut merkle_proofs = Vec::with_capacity(n_rounds);
        let mut query_proof_of_works = Vec::with_capacity(n_rounds);
        let mut sumcheck_polynomials = Vec::with_capacity(n_rounds);

        let mut prev_domain_log_size = config.starting_domain_log_size;
        let mut prev_folding_factor = config.starting_folding_factor;
        let (mut prev_prover_data, mut prev_committed_data) =
            (witness_data.commitment_data, Arc::new(witness_data.committed_data));

        for round_index in 0..n_rounds {
            let round_params = &config.round_parameters[round_index];

            let num_variables = match &sumcheck_prover.f_vec {
                KOrEfMle::K(mle) => mle.num_variables() as usize,
                KOrEfMle::EF(mle) => mle.num_variables() as usize,
            };
            let inner_evals = match &sumcheck_prover.f_vec {
                KOrEfMle::K(_) => unreachable!("Should be of type EF after first sumcheck"),
                KOrEfMle::EF(mle) => mle.guts().clone().reshape([
                    (1 << num_variables) / (1 << round_params.folding_factor),
                    1 << round_params.folding_factor,
                ]),
            };

            let encoding =
                batch_dft::<_, GC::F, GC::EF>(&self.dft, inner_evals, round_params.log_inv_rate);

            let encoding_base = Arc::new(encoding.flatten_to_base());

            let (commitment, prover_data) = self
                .merkle_prover
                .commit_tensors(Message::<Tensor<GC::F>>::from(vec![encoding_base.clone()]))
                .await
                .unwrap();

            // Observe the commitment
            challenger.observe(commitment);

            // Squeeze the ood points
            let ood_points: Vec<Point<GC::EF>> = (0..round_params.ood_samples)
                .map(|_| {
                    (0..num_variables)
                        .map(|_| challenger.sample_ext_element())
                        .collect::<Vec<GC::EF>>()
                        .into()
                })
                .collect();

            let f_vec = match sumcheck_prover.f_vec {
                KOrEfMle::K(_) => unreachable!("Should be of type EF after first sumcheck"),
                KOrEfMle::EF(ref mle) => mle,
            };

            let ood_answers: Vec<GC::EF> = ood_points
                .iter()
                .map(|point| f_vec.blocking_monomial_basis_eval_at(point)[0])
                .collect();

            challenger.observe_ext_element_slice(&ood_answers);

            parsed_commitments.push(ParsedCommitment::<GC, C> {
                commitment,
                ood_points: ood_points.clone(),
                ood_answers: ood_answers.clone(),
                _marker: Default::default(),
            });

            let id_query_indices = (0..round_params.num_queries)
                .map(|_| challenger.sample_bits(prev_domain_log_size))
                .collect::<Vec<_>>();
            let id_query_values: Vec<GC::F> = id_query_indices
                .iter()
                .map(|val| reverse_bits_len(*val, prev_domain_log_size))
                .map(|pos| generator.exp_u64(pos as u64))
                .collect();

            let claim_batching_randomness: GC::EF = challenger.sample_ext_element();

            query_proof_of_works
                .push(challenger.grind(round_params.queries_pow_bits.ceil() as usize));

            let merkle_openings = self
                .merkle_prover
                .compute_openings_at_indices(
                    Message::<Tensor<GC::F>>::from(vec![prev_committed_data]),
                    &id_query_indices,
                )
                .await;

            let merkle_proof = self
                .merkle_prover
                .prove_openings_at_indices(prev_prover_data, &id_query_indices)
                .await
                .unwrap();
            let merkle_proof = MerkleTreeOpening { values: merkle_openings, proof: merkle_proof };
            let merkle_read_values: Vec<Mle<GC::EF>> = if round_index != 0 {
                merkle_proof
                    .values
                    .clone()
                    .into_buffer()
                    .into_extension::<GC::EF>()
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
                    .map(GC::EF::from)
                    .collect::<Vec<_>>()
                    .chunks_exact(1 << prev_folding_factor)
                    .map(|v| Mle::new(v.to_vec().into()))
                    .collect()
            };
            merkle_proofs.push(merkle_proof);

            let stir_values: Vec<GC::EF> = merkle_read_values
                .iter()
                .map(|coeffs| coeffs.blocking_eval_at(&folding_randomness.clone().into())[0])
                .collect();

            // Update the claimed sum
            claimed_sum = claim_batching_randomness
                .powers()
                .zip(iter::once(&claimed_sum).chain(&ood_answers).chain(&stir_values))
                .map(|(r, &v)| v * r)
                .sum();

            let new_eq_polys = [
                ood_points.clone(),
                id_query_values
                    .into_iter()
                    .map(|point| map_to_pow(point, num_variables).to_extension())
                    .collect(),
            ]
            .concat();
            sumcheck_prover.add_equality_polynomials(new_eq_polys, claim_batching_randomness).await;

            let (round_sumcheck_polynomials, round_folding_randomness, round_claimed_sum) =
                sumcheck_prover
                    .compute_sumcheck_polynomials(
                        claimed_sum,
                        round_params.folding_factor,
                        &round_params.pow_bits,
                        challenger,
                    )
                    .await;
            folding_randomness = round_folding_randomness;
            claimed_sum = round_claimed_sum;

            sumcheck_polynomials.push(round_sumcheck_polynomials);

            // Update
            generator = generator.square();
            prev_folding_factor = round_params.folding_factor;
            prev_domain_log_size = round_params.evaluation_domain_log_size;
            (prev_prover_data, prev_committed_data) = (prover_data, encoding_base);
        }

        let f_vec = match &sumcheck_prover.f_vec {
            KOrEfMle::K(_) => unreachable!("Should be of type EF after first sumcheck"),
            KOrEfMle::EF(mle) => mle,
        };

        let final_polynomial = f_vec.guts().clone().into_buffer().to_vec();
        challenger.observe_ext_element_slice(&final_polynomial);

        let final_id_indices = (0..config.final_queries)
            .map(|_| challenger.sample_bits(prev_domain_log_size))
            .collect::<Vec<_>>();

        let final_pow = challenger.grind(config.final_pow_bits.ceil() as usize);

        let final_merkle_openings = self
            .merkle_prover
            .compute_openings_at_indices(
                Message::<Tensor<GC::F>>::from(vec![prev_committed_data]),
                &final_id_indices,
            )
            .await;
        let final_merkle_proof = self
            .merkle_prover
            .prove_openings_at_indices(prev_prover_data, &final_id_indices)
            .await
            .unwrap();
        let final_merkle_proof =
            MerkleTreeOpening { values: final_merkle_openings, proof: final_merkle_proof };

        let (final_sumcheck_polynomials, _, _) = sumcheck_prover
            .compute_sumcheck_polynomials(
                claimed_sum,
                config.final_poly_log_degree,
                &config.final_folding_pow_bits,
                challenger,
            )
            .await;

        WhirProof {
            initial_sumcheck_polynomials,
            commitments: parsed_commitments,
            merkle_proofs,
            query_proof_of_works,
            sumcheck_polynomials,
            final_polynomial,
            final_merkle_proof,
            final_sumcheck_polynomials,
            final_pow,
            _config: Default::default(),
        }
    }
}

enum KOrEfMle<K, EF> {
    K(Mle<K>),
    EF(Mle<EF>),
}

impl<K, EF> KOrEfMle<K, EF>
where
    K: Field,
    EF: ExtensionField<K>,
{
    pub fn inner_prod(&self, other: Mle<EF>) -> (EF, EF) {
        match self {
            KOrEfMle::K(mle) => mle
                .guts()
                .as_slice()
                .par_iter()
                .zip_eq(other.guts().as_slice().par_iter())
                .map(|(m, z)| (*m, *z))
                .chunks(2)
                .map(|chunk| {
                    let (e0, e1) = (chunk[0], chunk[1]);
                    let f0 = e0.0;
                    let f1 = e1.0;
                    let v0 = e0.1;
                    let v1 = e1.1;

                    (v0 * f0, (v1 - v0) * (f1 - f0))
                })
                .reduce(
                    || (EF::zero(), EF::zero()),
                    |(acc0, acc1), (v0, v1)| (acc0 + v0, acc1 + v1),
                ),
            KOrEfMle::EF(mle) => mle
                .guts()
                .as_slice()
                .par_iter()
                .zip_eq(other.guts().as_slice().par_iter())
                .map(|(m, z)| (*m, *z))
                .chunks(2)
                .map(|chunk| {
                    let (e0, e1) = (chunk[0], chunk[1]);
                    let f0 = e0.0;
                    let f1 = e1.0;
                    let v0 = e0.1;
                    let v1 = e1.1;

                    (v0 * f0, (v1 - v0) * (f1 - f0))
                })
                .reduce(
                    || (EF::zero(), EF::zero()),
                    |(acc0, acc1), (v0, v1)| (acc0 + v0, acc1 + v1),
                ),
        }
    }
    pub async fn fix_last_variable(&self, value: EF) -> Self {
        match self {
            KOrEfMle::K(mle) => KOrEfMle::EF(mle.fix_last_variable(value).await),
            KOrEfMle::EF(mle) => KOrEfMle::EF(mle.fix_last_variable(value).await),
        }
    }
}

pub struct SumcheckProver<GC, K>
where
    GC: IopCtx,
    K: Field,
    GC::EF: ExtensionField<K>,
{
    f_vec: KOrEfMle<K, GC::EF>,
    eq_vec: Mle<GC::EF>,
}

impl<GC, K> SumcheckProver<GC, K>
where
    GC: IopCtx,
    K: Field,
    GC::EF: ExtensionField<K>,
{
    async fn new(
        f_vec: Mle<K>,
        query_vector: Mle<GC::EF>,
        eq_points: Vec<Point<GC::EF>>,
        combination_randomness: GC::EF,
    ) -> Self {
        // assert!(!eq_points.is_empty());
        let mut acc = combination_randomness;
        let mut eq_vec = query_vector.into_guts().into_buffer().to_vec();
        for mle in eq_points.iter().map(monomial_basis_evals_blocking) {
            Mle::new(mle)
                .hypercube_iter()
                .enumerate()
                .for_each(|(i, val)| eq_vec[i] += acc * val[0]);
            acc *= combination_randomness;
        }

        SumcheckProver { f_vec: KOrEfMle::K(f_vec), eq_vec: eq_vec.into() }
    }

    async fn add_equality_polynomials(
        &mut self,
        eq_points: Vec<Point<GC::EF>>,
        combination_randomness: GC::EF,
    ) {
        let mut eq_vec = self.eq_vec.guts().clone().into_buffer().to_vec();
        let mut acc = combination_randomness;
        for mle in eq_points.iter().map(monomial_basis_evals_blocking) {
            Mle::new(mle)
                .hypercube_iter()
                .enumerate()
                .for_each(|(i, val)| eq_vec[i] += acc * val[0]);
            acc *= combination_randomness;
        }
        self.eq_vec = eq_vec.into();
    }

    async fn compute_sumcheck_polynomials(
        &mut self,
        mut claimed_sum: GC::EF,
        num_rounds: usize,
        pow_bits: &[f64],
        challenger: &mut GC::Challenger,
    ) -> (Vec<(SumcheckPoly<GC::EF>, ProofOfWork<GC>)>, Vec<GC::EF>, GC::EF) {
        let mut res = Vec::with_capacity(num_rounds);
        let mut folding_randomness = Vec::with_capacity(num_rounds);

        for round_pow_bits in &pow_bits[..num_rounds] {
            // Constant and quadratic term
            let (c0, c2) = self.f_vec.inner_prod(self.eq_vec.clone());
            let c1 = claimed_sum - c0.double() - c2;

            let sumcheck_poly = SumcheckPoly([c0, c1, c2]);

            challenger.observe_ext_element_slice(&sumcheck_poly.0);
            let folding_randomness_single: GC::EF = challenger.sample_ext_element();
            let pow = challenger.grind(round_pow_bits.ceil() as usize);
            claimed_sum = sumcheck_poly.evaluate_at_point(folding_randomness_single);
            res.push((sumcheck_poly, pow));
            folding_randomness.push(folding_randomness_single);

            self.f_vec = self.f_vec.fix_last_variable(folding_randomness_single).await;
            self.eq_vec = self.eq_vec.fix_last_variable(folding_randomness_single).await;
        }
        folding_randomness.reverse();
        (res, folding_randomness, claimed_sum)
    }
}

pub fn big_beautiful_whir_config<F: TwoAdicField>() -> WhirProofShape<F> {
    let folding_factor = 4;
    WhirProofShape::<F> {
        num_variables: 28,
        domain_generator: F::two_adic_generator(21),
        starting_ood_samples: 2,
        starting_log_inv_rate: 1,
        starting_folding_factor: 8,
        starting_domain_log_size: 21,
        starting_folding_pow_bits: vec![0.; 8],
        round_parameters: vec![
            RoundConfig {
                folding_factor,
                evaluation_domain_log_size: 20,
                queries_pow_bits: 16.0,
                pow_bits: vec![0.0; folding_factor],
                num_queries: 84,
                ood_samples: 2,
                log_inv_rate: 4,
            },
            RoundConfig {
                folding_factor,
                evaluation_domain_log_size: 19,
                queries_pow_bits: 16.0,
                pow_bits: vec![0.0; folding_factor],
                num_queries: 21,
                ood_samples: 2,
                log_inv_rate: 7,
            },
            RoundConfig {
                folding_factor,
                evaluation_domain_log_size: 18,
                queries_pow_bits: 16.0,
                pow_bits: vec![0.0; folding_factor],
                num_queries: 12,
                ood_samples: 2,
                log_inv_rate: 10,
            },
        ],
        final_poly_log_degree: 8,
        final_queries: 9,
        final_pow_bits: 16.0,
        final_folding_pow_bits: vec![0.0; 8],
    }
}

pub fn default_whir_config<F: TwoAdicField>() -> WhirProofShape<F> {
    let folding_factor = 4;
    WhirProofShape::<F> {
        num_variables: 16,
        domain_generator: F::two_adic_generator(13),
        starting_ood_samples: 1,
        starting_log_inv_rate: 1,
        starting_folding_factor: folding_factor,
        starting_domain_log_size: 13,
        starting_folding_pow_bits: vec![10.0; folding_factor],
        round_parameters: vec![
            RoundConfig {
                folding_factor,
                evaluation_domain_log_size: 12,
                queries_pow_bits: 10.0,
                pow_bits: vec![10.0; folding_factor],
                num_queries: 90,
                ood_samples: 1,
                log_inv_rate: 4,
            },
            RoundConfig {
                folding_factor,
                evaluation_domain_log_size: 11,
                queries_pow_bits: 10.0,
                pow_bits: vec![10.0; folding_factor],
                num_queries: 15,
                ood_samples: 1,
                log_inv_rate: 7,
            },
        ],
        final_poly_log_degree: 4,
        final_queries: 10,
        final_pow_bits: 10.0,
        final_folding_pow_bits: vec![10.0; 8],
    }
}

#[cfg(test)]
mod tests {
    use rand::{distributions::Standard, prelude::Distribution, Rng, SeedableRng};
    use slop_algebra::{extension::BinomialExtensionField, TwoAdicField, UnivariatePolynomial};
    use slop_baby_bear::BabyBear;
    use slop_basefold::{
        DefaultBasefoldConfig, Poseidon2BabyBear16BasefoldConfig,
        Poseidon2KoalaBear16BasefoldConfig,
    };
    use slop_dft::p3::Radix2DitParallel;
    use slop_koala_bear::{KoalaBear, KoalaBearDegree4Duplex};
    use slop_matrix::{bitrev::BitReversableMatrix, dense::RowMajorMatrix, Matrix};
    use slop_merkle_tree::{
        DefaultMerkleTreeConfig, FieldMerkleTreeProver, MerkleTreeTcs, Poseidon2BabyBear16Prover,
        Poseidon2KoalaBear16Prover,
    };
    use slop_utils::setup_logger;

    use super::*;
    use crate::{
        config::{RoundConfig, WhirProofShape},
        verifier::Verifier,
    };

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;

    fn big_beautiful_whir_config<F: TwoAdicField>() -> WhirProofShape<F> {
        let folding_factor = 4;
        WhirProofShape::<F> {
            num_variables: 28,
            domain_generator: F::two_adic_generator(21),
            starting_ood_samples: 2,
            starting_log_inv_rate: 1,
            starting_folding_factor: 8,
            starting_domain_log_size: 21,
            starting_folding_pow_bits: vec![0.; 8],
            round_parameters: vec![
                RoundConfig {
                    folding_factor,
                    evaluation_domain_log_size: 20,
                    queries_pow_bits: 16.0,
                    pow_bits: vec![0.0; folding_factor],
                    num_queries: 84,
                    ood_samples: 2,
                    log_inv_rate: 4,
                },
                RoundConfig {
                    folding_factor,
                    evaluation_domain_log_size: 19,
                    queries_pow_bits: 16.0,
                    pow_bits: vec![0.0; folding_factor],
                    num_queries: 21,
                    ood_samples: 2,
                    log_inv_rate: 7,
                },
                RoundConfig {
                    folding_factor,
                    evaluation_domain_log_size: 18,
                    queries_pow_bits: 16.0,
                    pow_bits: vec![0.0; folding_factor],
                    num_queries: 12,
                    ood_samples: 2,
                    log_inv_rate: 10,
                },
            ],
            final_poly_log_degree: 8,
            final_queries: 9,
            final_pow_bits: 16.0,
            final_folding_pow_bits: vec![0.0; 8],
        }
    }

    #[tokio::test]
    async fn whir_folding() {
        const FOLDING_FACTOR: usize = 4;
        let blowup_factor = 2;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let dft = Radix2DitParallel;

        let polynomial: Mle<F> = Mle::rand(&mut rng, 1, 16);

        let num_variables = polynomial.num_variables() as usize;
        let inner_evals = polynomial
            .guts()
            .clone()
            .reshape([(1 << num_variables) / (1 << FOLDING_FACTOR), 1 << FOLDING_FACTOR]);

        let encoding = batch_dft::<_, F, F>(&dft, inner_evals, blowup_factor);

        let [r1, r2, r3, r4]: [EF; FOLDING_FACTOR] = rng.gen();

        let folded_poly = polynomial
            .fix_last_variable(r1)
            .await
            .fix_last_variable(r2)
            .await
            .fix_last_variable(r3)
            .await
            .fix_last_variable(r4)
            .await;

        let encoding_of_fold =
            batch_dft::<_, F, EF>(&dft, folded_poly.guts().clone(), blowup_factor);

        let encoding_of_fold_vec = encoding_of_fold.into_buffer().to_vec();

        let columns: Vec<_> = encoding
            .clone()
            .into_buffer()
            .to_vec()
            .chunks_exact(1 << FOLDING_FACTOR)
            .map(|v| Mle::new(v.to_vec().into()))
            .collect();

        assert_eq!(columns.len(), 1 << (num_variables + blowup_factor - FOLDING_FACTOR));

        let uv_coeff = folded_poly.guts().clone().into_buffer().to_vec();
        let mle_evals = folded_poly.clone();
        let uv = UnivariatePolynomial::new(uv_coeff);

        let gen = EF::two_adic_generator(num_variables - FOLDING_FACTOR + blowup_factor);
        let powers: Vec<_> =
            gen.powers().take(1 << (num_variables + blowup_factor - FOLDING_FACTOR)).collect();
        let bit_reversed_powers =
            RowMajorMatrix::new(powers, 1).bit_reverse_rows().to_row_major_matrix().values;

        for ((col, enc), val) in
            columns.into_iter().zip(encoding_of_fold_vec).zip(bit_reversed_powers)
        {
            // We fixed `r1` as last variable first, so it should be the last coordinate of the point.
            // This assertion tests that the encoding of the folded polynomial matches the folding
            // onf the encoded polynomial.
            assert_eq!(enc, col.blocking_eval_at(&vec![r4, r3, r2, r1].into())[0]);

            // This assertion checks that the encoding of the folded polynomial is the bit-reversed
            // RS-encoding of the univariate polynomial whose coefficients are the same as the elements
            // of the folded polynomial (we always represent multilinears in the evaluation basis).
            assert_eq!(enc, uv.eval_at_point(val));
            let num_variables = mle_evals.num_variables() as usize;
            let point = (0..num_variables)
                .map(|i| val.exp_power_of_2(num_variables - 1 - i))
                .collect::<Point<_>>();

            // This assertion checks the compatibility between the multilinear representation of the
            // folded polynomial and its encoding: namely if we form the point (val^{2^{num_variables-1}}, ..., val^2, val)
            // and evaluate `mle_evals` in the monomial basis representation, that should be the same
            // thing as computing the DFT value at the current location.
            assert_eq!(enc, mle_evals.blocking_monomial_basis_eval_at(&point)[0]);
        }
    }

    type C = Poseidon2KoalaBear16BasefoldConfig;
    type GC = KoalaBearDegree4Duplex;

    #[tokio::test]
    async fn whir_test_sumcheck() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let verifier = C::default_verifier(1);
        let mut challenger_prover = C::default_challenger(&verifier);

        let num_variables: usize = 4;
        let polynomial: Mle<F> = Mle::rand(&mut rng, 1, num_variables as u32);
        let query_vector: Mle<EF> = Mle::rand(&mut rng, 1, num_variables as u32);

        let mut sumcheck_prover = SumcheckProver::<GC, KoalaBear>::new(
            polynomial.clone(),
            query_vector.clone(),
            vec![vec![EF::zero(); num_variables].into(); 1],
            EF::zero(),
        )
        .await;

        let claim: EF = polynomial
            .hypercube_iter()
            .zip(query_vector.hypercube_iter())
            .map(|(a, b)| b[0] * a[0])
            .sum();

        let (_, folding_randmness, claimed_sum) = sumcheck_prover
            .compute_sumcheck_polynomials(
                claim,
                num_variables,
                &vec![0.; num_variables],
                &mut challenger_prover,
            )
            .await;

        assert_eq!(
            query_vector.blocking_eval_at(&folding_randmness.clone().into())[0]
                * polynomial.blocking_eval_at(&folding_randmness.clone().into())[0],
            claimed_sum
        );
    }

    #[tokio::test]
    async fn whir_test_sumcheck_with_eq_modification() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let mut challenger_prover = C::default_challenger(&C::default_verifier(1));

        let num_variables: usize = 8;
        let polynomial: Mle<F> = Mle::rand(&mut rng, 1, num_variables as u32);
        let query_vector: Mle<EF> = Mle::rand(&mut rng, 1, num_variables as u32);

        let z_initial: Point<EF> = (0..num_variables).map(|_| rng.gen()).collect();

        let mut sumcheck_prover = SumcheckProver::<GC, F>::new(
            polynomial.clone(),
            query_vector.clone(),
            vec![z_initial.clone()],
            EF::one(),
        )
        .await;

        let claim: EF = polynomial
            .hypercube_iter()
            .zip(query_vector.hypercube_iter())
            .map(|(a, b)| b[0] * a[0])
            .sum::<EF>()
            + polynomial.blocking_monomial_basis_eval_at(&z_initial)[0];

        let (_, folding_randomness, claimed_sum) = sumcheck_prover
            .compute_sumcheck_polynomials(
                claim,
                num_variables / 2,
                &vec![0.; num_variables / 2],
                &mut challenger_prover,
            )
            .await;

        let z_1: Point<EF> = (0..4).map(|_| rng.gen()).collect();
        let combination_randomness: EF = rng.gen();

        sumcheck_prover.add_equality_polynomials(vec![z_1.clone()], combination_randomness).await;

        let f_vec = match &sumcheck_prover.f_vec {
            KOrEfMle::EF(f_vec) => f_vec,
            KOrEfMle::K(_) => panic!(),
        };
        let f_eval = f_vec.blocking_monomial_basis_eval_at(&z_1);

        let (_, folding_randomness_2, claimed_sum) = sumcheck_prover
            .compute_sumcheck_polynomials(
                claimed_sum + combination_randomness * f_eval[0],
                2,
                &[0.; 2],
                &mut challenger_prover,
            )
            .await;

        let z_2: Point<EF> = (0..2).map(|_| rng.gen()).collect();
        let combination_randomness_2: EF = rng.gen();

        sumcheck_prover.add_equality_polynomials(vec![z_2.clone()], combination_randomness_2).await;

        let f_vec = match &sumcheck_prover.f_vec {
            KOrEfMle::EF(f_vec) => f_vec,
            KOrEfMle::K(_) => panic!(),
        };
        let f_eval = f_vec.blocking_monomial_basis_eval_at(&z_2);

        let (_, folding_randomness_3, claimed_sum) = sumcheck_prover
            .compute_sumcheck_polynomials(
                claimed_sum + combination_randomness_2 * f_eval[0],
                2,
                &[0.; 2],
                &mut challenger_prover,
            )
            .await;

        let full_concatenated: Point<EF> = folding_randomness_3
            .iter()
            .copied()
            .chain(folding_randomness_2.iter().copied())
            .chain(folding_randomness.iter().copied())
            .collect();
        let partial_concatenated: Point<EF> = folding_randomness_3
            .iter()
            .copied()
            .chain(folding_randomness_2.iter().copied())
            .collect();
        assert_eq!(
            claimed_sum,
            (query_vector.blocking_eval_at(&full_concatenated).to_vec()[0]
                + Mle::full_monomial_basis_eq(&z_initial, &full_concatenated))
                * polynomial.blocking_eval_at(&full_concatenated).to_vec()[0]
                + combination_randomness
                    * polynomial.blocking_eval_at(&full_concatenated).to_vec()[0]
                    * Mle::full_monomial_basis_eq(&z_1, &partial_concatenated)
                + combination_randomness_2
                    * polynomial.blocking_eval_at(&full_concatenated).to_vec()[0]
                    * Mle::full_monomial_basis_eq(&z_2, &folding_randomness_3.into())
        );
    }

    async fn whir_test_generic<
        GC: IopCtx<F: TwoAdicField, EF: TwoAdicField + ExtensionField<GC::F>>,
        C: DefaultBasefoldConfig<GC, Tcs: DefaultMerkleTreeConfig<GC>>,
        MerkleProver: TensorCsProver<GC, CpuBackend, MerkleConfig = C::Tcs>
            + ComputeTcsOpenings<GC, CpuBackend, MerkleConfig = C::Tcs>,
    >(
        config: WhirProofShape<GC::F>,
        merkle_prover: MerkleProver,
    ) where
        Standard: Distribution<GC::F> + Distribution<GC::EF>,
    {
        setup_logger();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let mut challenger_prover = C::default_challenger(&C::default_verifier(1));
        let mut challenger_verifier = C::default_challenger(&C::default_verifier(1));

        let prover = Prover::<_, _, _, C>::new(Radix2DitParallel, merkle_prover).await;
        let merkle_verifier = MerkleTreeTcs::default();
        let polynomial: Mle<GC::F> = Mle::rand(&mut rng, 1, config.num_variables as u32);
        let query_vector: Mle<GC::EF> =
            Mle::<GC::EF>::rand(&mut rng, 1, config.num_variables as u32);

        let claim: GC::EF = polynomial
            .hypercube_iter()
            .zip(query_vector.hypercube_iter())
            .map(|(a, b)| b[0] * a[0])
            .sum();

        let (commitment, prover_data) =
            prover.commit(polynomial, &mut challenger_prover, &config).await;
        let now = std::time::Instant::now();
        let proof =
            prover.prove(query_vector.clone(), prover_data, &mut challenger_prover, &config).await;

        let elapsed = now.elapsed();
        tracing::debug!("Proof generation took: {:?}", elapsed);

        let proof_bytes = bincode::serialize(&proof).unwrap();
        tracing::debug!("Proof size: {} bytes", proof_bytes.len());

        let verifier = Verifier::new(merkle_verifier);
        verifier.observe_commitment(&commitment, &mut challenger_verifier, &config).unwrap();
        let (point, value) =
            verifier.verify(&commitment, claim, &proof, &mut challenger_verifier, &config).unwrap();

        assert_eq!(point.dimension(), config.num_variables);

        assert_eq!(query_vector.blocking_eval_at(&point)[0], value)
    }

    #[tokio::test]
    async fn whir_test_e2e_koala_bear() {
        let config = default_whir_config::<KoalaBear>();
        let merkle_prover: Poseidon2KoalaBear16Prover = FieldMerkleTreeProver::default();
        whir_test_generic::<_, Poseidon2KoalaBear16BasefoldConfig, _>(config, merkle_prover).await;
    }

    #[tokio::test]
    #[ignore = "test used for benchmarking"]
    async fn whir_test_realistic_koala_bear() {
        let config = big_beautiful_whir_config::<KoalaBear>();
        let merkle_prover: Poseidon2KoalaBear16Prover = FieldMerkleTreeProver::default();
        whir_test_generic::<_, Poseidon2KoalaBear16BasefoldConfig, _>(config, merkle_prover).await;
    }

    #[tokio::test]
    async fn whir_test_e2e_baby_bear() {
        let config = default_whir_config::<BabyBear>();
        let merkle_prover: Poseidon2BabyBear16Prover = FieldMerkleTreeProver::default();
        whir_test_generic::<_, Poseidon2BabyBear16BasefoldConfig, _>(config, merkle_prover).await;
    }

    #[tokio::test]
    #[ignore = "test used for benchmarking"]
    async fn whir_test_realistic_baby_bear() {
        let config = big_beautiful_whir_config::<BabyBear>();
        let merkle_prover: Poseidon2BabyBear16Prover = FieldMerkleTreeProver::default();
        whir_test_generic::<_, Poseidon2BabyBear16BasefoldConfig, _>(config, merkle_prover).await;
    }
}
