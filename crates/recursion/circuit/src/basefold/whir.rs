#![cfg(test)]
use std::{iter, marker::PhantomData};

use crate::{
    basefold::{
        tcs::{RecursiveTcs, RecursiveTensorCsOpening},
        RecursiveBasefoldConfig,
    },
    challenger::{CanObserveVariable, CanSampleBitsVariable, FieldChallengerVariable},
    sumcheck::{evaluate_mle_ext, evaluate_mle_ext_batch},
    symbolic::IntoSymbolic,
    witness::Witnessable,
    AsRecursive, CircuitConfig,
};
use slop_algebra::{
    extension::BinomialExtensionField, AbstractField, ExtensionField, Field, UnivariatePolynomial,
};
use slop_basefold::BasefoldConfig;
use slop_challenger::GrindingChallenger;
use slop_commit::TensorCsOpening;
use slop_multilinear::{Mle, Point};
use slop_whir::{map_to_pow, ParsedCommitment, SumcheckPoly, WhirProof, WhirProofShape};
use sp1_primitives::SP1Field;
use sp1_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Ext, ExtensionOperand, Felt, SymbolicExt},
};

#[derive(Clone)]
pub struct RecursiveWhirVerifier<C: RecursiveBasefoldConfig> {
    _marker: PhantomData<C>,
}

impl<C: CircuitConfig> IntoSymbolic<C> for SumcheckPoly<Ext<C::F, C::EF>> {
    type Output = SumcheckPoly<SymbolicExt<C::F, C::EF>>;

    fn as_symbolic(&self) -> Self::Output {
        SumcheckPoly(self.0.map(SymbolicExt::from))
    }
}

#[derive(Debug, Clone)]
pub struct RecursiveParsedCommitment<C>
where
    C: RecursiveBasefoldConfig,
{
    pub(crate) commitment: C::Commitment,
    pub(crate) ood_points: Vec<Point<Ext<C::F, C::EF>>>,
    pub(crate) ood_answers: Vec<Ext<C::F, C::EF>>,
}

pub type RecursiveProverMessage<C> = (
    SumcheckPoly<Ext<<C as RecursiveBasefoldConfig>::F, <C as RecursiveBasefoldConfig>::EF>>,
    Felt<<C as RecursiveBasefoldConfig>::F>,
);

type ProverMessage<C> = (
    SumcheckPoly<Ext<<C as RecursiveBasefoldConfig>::F, <C as RecursiveBasefoldConfig>::EF>>,
    Felt<<C as RecursiveBasefoldConfig>::F>,
);

type PointAndEval<F> = (Point<F>, F);
pub struct RecursiveWhirProof<C>
where
    C: RecursiveBasefoldConfig,
    C::Challenger: CanObserveVariable<C::Circuit, C::Commitment>,
    C::Commitment: Copy,
{
    // First sumcheck
    pub initial_sumcheck_polynomials: Vec<ProverMessage<C>>,

    // For internal rounds
    pub commitments: Vec<RecursiveParsedCommitment<C>>,
    pub merkle_proofs: Vec<RecursiveTensorCsOpening<C::Tcs>>,
    pub query_proof_of_works: Vec<Felt<C::F>>,
    pub sumcheck_polynomials: Vec<Vec<RecursiveProverMessage<C>>>,

    // Final round
    pub final_polynomial: Vec<Ext<C::F, C::EF>>,
    pub final_merkle_proof: RecursiveTensorCsOpening<C::Tcs>,
    pub final_sumcheck_polynomials: Vec<RecursiveProverMessage<C>>,
    pub final_pow: Felt<C::F>,
    pub _config: PhantomData<C>,
}

impl<C: RecursiveBasefoldConfig<F = SP1Field, EF = BinomialExtensionField<SP1Field, 4>>>
    RecursiveWhirVerifier<C>
where
    C::Challenger:
        FieldChallengerVariable<C::Circuit, C::Bit> + CanObserveVariable<C::Circuit, C::Commitment>,
    C::F: Field,
    C::EF: ExtensionField<C::F>,
    C::Commitment: Copy,
    C::Bit: Clone,
{
    pub(crate) fn observe_commitment(
        &self,
        builder: &mut Builder<C::Circuit>,
        commitment: &RecursiveParsedCommitment<C>,
        challenger: &mut C::Challenger,
        config: &WhirProofShape<C::F>,
    ) {
        challenger.observe(builder, commitment.commitment);
        let ood_points: Vec<Point<Ext<C::F, C::EF>>> = (0..config.starting_ood_samples)
            .map(|_| {
                (0..config.num_variables)
                    .map(|_| challenger.sample_ext(builder))
                    .collect::<Vec<Ext<C::F, C::EF>>>()
                    .into()
            })
            .collect();

        for (ood_point, commitment_ood_point) in ood_points.iter().zip(commitment.ood_points.iter())
        {
            for (a, b) in ood_point.iter().zip(commitment_ood_point.iter()) {
                builder.assert_ext_eq(*a, *b);
            }
        }

        for ood_answer in commitment.ood_answers.iter() {
            challenger.observe_ext_element(builder, *ood_answer);
        }
    }

    pub(crate) fn verify_whir(
        &self,
        builder: &mut Builder<C::Circuit>,
        commitment: &RecursiveParsedCommitment<C>,
        claim: Ext<C::F, C::EF>,
        proof: &RecursiveWhirProof<C>,
        challenger: &mut C::Challenger,
        config: &WhirProofShape<C::F>,
    ) -> PointAndEval<Ext<C::F, C::EF>> {
        let n_rounds = config.round_parameters.len();

        // Batch the initial claim with the OOD claims of the commitment
        let claim_batching_randomness: Ext<C::F, C::EF> = challenger.sample_ext(builder);
        let claimed_sum: Ext<C::F, C::EF> = builder.eval(
            IntoSymbolic::<C::Circuit>::as_symbolic(&claim_batching_randomness)
                .powers()
                .zip(iter::once(&claim).chain(&commitment.ood_answers))
                .map(|(r, &v)| v * r)
                .sum::<SymbolicExt<_, _>>(),
        );

        // Initialize the collection of points at which we will need to compute the monomial basis
        // polynomial evaluations.
        let mut final_evaluation_points = vec![commitment.ood_points.clone()];

        // Check the initial sumcheck.
        let (mut folding_randomness, mut claimed_sum) = self.verify_whir_sumcheck(
            builder,
            &proof.initial_sumcheck_polynomials,
            claimed_sum,
            config.starting_folding_factor,
            &config.starting_folding_pow_bits,
            challenger,
        );

        // This contains all the sumcheck randomnesses (these are the alphas)
        let mut concatenated_folding_randomness = folding_randomness.clone();

        // This contains all the batching randomness for sumcheck (these are the epsilons) for
        // batching in- and out-of-domain claims from round to round.
        let mut all_claim_batching_randomness = vec![claim_batching_randomness];

        // This is relative to the previous commitment (i.e. prev_commitment has a domain size of
        // this size)
        let mut domain_size =
            config.num_variables - config.starting_folding_factor + config.starting_log_inv_rate;
        let mut generator: Felt<C::F> = builder.constant(config.domain_generator);
        let mut prev_commitment = commitment;

        let mut prev_folding_factor = config.starting_folding_factor;
        let mut num_variables = config.num_variables - config.starting_folding_factor;

        for round_index in 0..n_rounds {
            let round_params = &config.round_parameters[round_index];
            let new_commitment = &proof.commitments[round_index];

            // Observe the commitment
            challenger.observe(builder, new_commitment.commitment);

            // Squeeze the ood points
            let ood_points: Vec<Point<Ext<C::F, C::EF>>> = (0..round_params.ood_samples)
                .map(|_| {
                    (0..num_variables)
                        .map(|_| challenger.sample_ext(builder))
                        .collect::<Vec<Ext<C::F, C::EF>>>()
                        .into()
                })
                .collect();

            for (ood_point, commitment_ood_point) in
                ood_points.iter().zip(&new_commitment.ood_points)
            {
                for (ood_elem, commitment_ood_elem) in
                    ood_point.iter().zip(commitment_ood_point.iter())
                {
                    builder.assert_ext_eq(*ood_elem, *commitment_ood_elem);
                }
            }

            // Absorb the OOD answers
            for ood_answer in &new_commitment.ood_answers {
                challenger.observe_ext_element(builder, *ood_answer);
            }

            // Squeeze the STIR queries
            let id_query_indices = (0..round_params.num_queries)
                .map(|_| challenger.sample_bits(builder, domain_size))
                .collect::<Vec<_>>();
            let id_query_values: Vec<Felt<C::F>> = id_query_indices
                .iter()
                .map(|val| {
                    <C::Circuit as CircuitConfig>::exp_reverse_bits(builder, generator, val.clone())
                })
                .collect();
            let claim_batching_randomness: Ext<C::F, C::EF> = challenger.sample_ext(builder);

            challenger.check_witness(
                builder,
                round_params.queries_pow_bits.ceil() as usize,
                proof.query_proof_of_works[round_index],
            );

            let merkle_proof = &proof.merkle_proofs[round_index];
            <C::Tcs as RecursiveTcs>::verify_tensor_openings(
                builder,
                &prev_commitment.commitment,
                &id_query_indices,
                merkle_proof,
            );

            // Chunk the Merkle openings into chunks of size `1<<prev_folding_factor`
            // so that the verifier can induce in-domain evaluation claims about the next codeword.
            // Except in the first round, the opened values in the Merkle proof are secretly
            // extension field elements, so we have to reinterpret them as such. (The
            // Merkle tree API commits to and opens only base-field values.)
            let merkle_read_values: Vec<Mle<Ext<C::F, C::EF>>> = if round_index != 0 {
                merkle_proof
                    .values
                    .clone()
                    .into_buffer()
                    .into_vec()
                    .chunks_exact(sp1_recursion_executor::D)
                    .map(|felt_chunk| {
                        <C::Circuit as CircuitConfig>::felt2ext(
                            builder,
                            felt_chunk.try_into().unwrap(),
                        )
                    })
                    .collect::<Vec<_>>()
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
                    .map(|f| {
                        let e: SymbolicExt<C::F, C::EF> = f.into();
                        builder.eval(e)
                    })
                    .collect::<Vec<_>>()
                    .chunks_exact(1 << prev_folding_factor)
                    .map(|v| Mle::new(v.to_vec().into()))
                    .collect()
            };
            // Compute the STIR values by reading the merkle values and folding across the column.
            let stir_values: Vec<Ext<C::F, C::EF>> =
                evaluate_mle_ext_batch(builder, merkle_read_values, folding_randomness.clone())
                    .iter()
                    .map(|eval| eval[0])
                    .collect();

            if round_index == 0 {
                builder.cycle_tracker_v2_enter("first round stir values");
            }
            if round_index == 0 {
                builder.cycle_tracker_v2_exit();
            }

            // Update the claimed sum using the STIR values and the OOD answers.
            claimed_sum = builder.eval(
                IntoSymbolic::<C::Circuit>::as_symbolic(&claim_batching_randomness)
                    .powers()
                    .zip(
                        iter::once(&claimed_sum)
                            .chain(&new_commitment.ood_answers)
                            .chain(&stir_values),
                    )
                    .map(|(r, &v)| r * v)
                    .sum::<SymbolicExt<C::F, C::EF>>(),
            );

            (folding_randomness, claimed_sum) = self.verify_whir_sumcheck(
                builder,
                &proof.sumcheck_polynomials[round_index],
                claimed_sum,
                round_params.folding_factor,
                &round_params.pow_bits,
                challenger,
            );

            // Prepend the folding randomness from the sumcheck into the combined folding
            // randomness.
            concatenated_folding_randomness = folding_randomness
                .iter()
                .cloned()
                .chain(concatenated_folding_randomness.iter().cloned())
                .collect();

            all_claim_batching_randomness.push(claim_batching_randomness);

            // Add both the in-domain and out-of-domain claims to the set of final evaluation
            // points.
            final_evaluation_points.push(
                [
                    ood_points.clone(),
                    id_query_values
                        .into_iter()
                        .map(|point| {
                            map_to_pow(
                                IntoSymbolic::<C::Circuit>::as_symbolic(&point),
                                num_variables,
                            )
                            .iter()
                            .cloned()
                            .map(|el| {
                                let ext = el.to_operand().symbolic();
                                builder.eval(ext)
                            })
                            .collect()
                        })
                        .collect(),
                ]
                .concat(),
            );

            domain_size = round_params.evaluation_domain_log_size;
            prev_commitment = new_commitment;
            prev_folding_factor = round_params.folding_factor;
            generator = builder.eval(IntoSymbolic::<C::Circuit>::as_symbolic(&generator).square());
            num_variables -= round_params.folding_factor;
        }

        // Now, we want to verify the final evaluations
        challenger.observe_ext_element_slice(builder, &proof.final_polynomial);

        let final_poly = proof.final_polynomial.clone();
        let final_poly_uv =
            UnivariatePolynomial::new(IntoSymbolic::<C::Circuit>::as_symbolic(&final_poly));

        let final_id_indices = (0..config.final_queries)
            .map(|_| challenger.sample_bits(builder, domain_size))
            .collect::<Vec<_>>();
        let final_id_values: Vec<Felt<C::F>> = final_id_indices
            .iter()
            .map(|val| {
                <C::Circuit as CircuitConfig>::exp_reverse_bits(builder, generator, val.clone())
            })
            .collect();

        <C::Tcs as RecursiveTcs>::verify_tensor_openings(
            builder,
            &prev_commitment.commitment,
            &final_id_indices,
            &proof.final_merkle_proof,
        );

        let final_merkle_read_values: Vec<Mle<Ext<C::F, C::EF>>> = proof
            .final_merkle_proof
            .values
            .clone()
            .into_buffer()
            .into_vec()
            .chunks_exact(sp1_recursion_executor::D)
            .map(|felt_slice| {
                <C::Circuit as CircuitConfig>::felt2ext(builder, felt_slice.try_into().unwrap())
            })
            .collect::<Vec<_>>()
            .chunks_exact(1 << prev_folding_factor)
            .map(|v| Mle::new(v.to_vec().into()))
            .collect();

        let final_stir_values: Vec<Ext<_, _>> =
            evaluate_mle_ext_batch(builder, final_merkle_read_values, folding_randomness.clone())
                .iter()
                .map(|eval| eval[0])
                .collect();

        for (final_stir_val, final_id_val) in final_stir_values.iter().zip(final_id_values.iter()) {
            builder.assert_ext_eq(
                *final_stir_val,
                final_poly_uv.eval_at_point((*final_id_val).into()),
            );
        }

        challenger.check_witness(builder, config.final_pow_bits.ceil() as usize, proof.final_pow);

        (folding_randomness, claimed_sum) = self.verify_whir_sumcheck(
            builder,
            &proof.final_sumcheck_polynomials,
            claimed_sum,
            config.final_poly_log_degree,
            &config.final_folding_pow_bits,
            challenger,
        );

        concatenated_folding_randomness = folding_randomness
            .iter()
            .cloned()
            .chain(concatenated_folding_randomness.iter().cloned())
            .collect();

        let f: Ext<_, _> = evaluate_mle_ext(
            builder,
            proof.final_polynomial.clone().into(),
            folding_randomness.clone(),
        )[0];

        builder.cycle_tracker_v2_enter("compute summand");
        let mut summand = SymbolicExt::<C::F, C::EF>::zero();
        for (i, eval_points) in final_evaluation_points.into_iter().enumerate() {
            let combination_randomness = all_claim_batching_randomness[i];
            let len = eval_points[0].len();
            let eval_randomness: Point<Ext<C::F, C::EF>> =
                concatenated_folding_randomness.split_at(len).0;

            let sum_modification = IntoSymbolic::<C::Circuit>::as_symbolic(&combination_randomness)
                .powers()
                .skip(1)
                .zip(eval_points)
                .map(|(r, point)| {
                    r * Mle::<SymbolicExt<C::F, C::EF>>::full_monomial_basis_eq(
                        &IntoSymbolic::<C::Circuit>::as_symbolic(&point),
                        &IntoSymbolic::<C::Circuit>::as_symbolic(&eval_randomness),
                    )
                })
                .sum::<SymbolicExt<C::F, C::EF>>();

            summand += sum_modification;
        }

        let summand: Ext<_, _> = builder.eval(summand);

        builder.cycle_tracker_v2_exit();

        // This is the claimed value of the query vector. It is trusted and assumed to be easily
        // computable by the verifier.
        let claimed_value = claimed_sum / f - summand;

        let claimed_value = builder.eval(claimed_value);
        (concatenated_folding_randomness, claimed_value)
    }

    pub(crate) fn verify_whir_sumcheck(
        &self,
        builder: &mut Builder<C::Circuit>,
        sumcheck_polynomials: &[ProverMessage<C>],
        mut claimed_sum: Ext<C::F, C::EF>,
        rounds: usize,
        pow_bits: &[f64],
        challenger: &mut C::Challenger,
    ) -> PointAndEval<Ext<C::F, C::EF>> {
        let mut randomness = Vec::with_capacity(rounds);
        for i in 0..rounds {
            let (sumcheck_poly, pow_witness) = &sumcheck_polynomials[i];
            for elem in sumcheck_poly.0.iter() {
                challenger.observe_ext_element(builder, *elem);
            }

            let sum = IntoSymbolic::<C::Circuit>::as_symbolic(sumcheck_poly).sum_over_hypercube();

            builder.assert_ext_eq(claimed_sum, sum);

            let folding_randomness_single: Ext<C::F, C::EF> = challenger.sample_ext(builder);
            randomness.push(folding_randomness_single);

            challenger.check_witness(builder, pow_bits[i].ceil() as usize, *pow_witness);
            claimed_sum = builder.eval(
                IntoSymbolic::<C::Circuit>::as_symbolic(sumcheck_poly).evaluate_at_point(
                    IntoSymbolic::<C::Circuit>::as_symbolic(&folding_randomness_single),
                ),
            );
        }

        randomness.reverse();
        (randomness.into(), claimed_sum)
    }
}

impl<C: CircuitConfig, BC: BasefoldConfig<F = C::F, EF = C::EF> + AsRecursive<C>> Witnessable<C>
    for ParsedCommitment<BC>
where
    BC::Recursive: RecursiveBasefoldConfig<F = C::F, EF = C::EF, Circuit = C>,
    BC::Commitment:
        Witnessable<C, WitnessVariable = <BC::Recursive as RecursiveBasefoldConfig>::Commitment>,
    C::F: Witnessable<C, WitnessVariable = Felt<C::F>>,
    C::EF: Witnessable<C, WitnessVariable = Ext<C::F, C::EF>>,
{
    type WitnessVariable = RecursiveParsedCommitment<BC::Recursive>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let commitment_variable = self.commitment.read(builder);
        let ood_point_variable = self.ood_points.iter().map(|point| point.read(builder)).collect();
        let ood_answer_variable =
            self.ood_answers.iter().map(|answer| answer.read(builder)).collect();
        RecursiveParsedCommitment {
            commitment: commitment_variable,
            ood_points: ood_point_variable,
            ood_answers: ood_answer_variable,
        }
    }

    fn write(&self, witness: &mut impl crate::witness::WitnessWriter<C>) {
        self.commitment.write(witness);
        for point in &self.ood_points {
            point.write(witness);
        }
        for answer in &self.ood_answers {
            answer.write(witness);
        }
    }
}

impl<C: CircuitConfig> Witnessable<C> for SumcheckPoly<C::EF>
where
    C::EF: Witnessable<C, WitnessVariable = Ext<C::F, C::EF>>,
{
    type WitnessVariable = SumcheckPoly<Ext<C::F, C::EF>>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let coeffs = std::array::from_fn(|i| self.0[i].read(builder));
        SumcheckPoly(coeffs)
    }

    fn write(&self, witness: &mut impl crate::witness::WitnessWriter<C>) {
        for coeff in &self.0 {
            coeff.write(witness);
        }
    }
}

impl<C: CircuitConfig, BC: BasefoldConfig<F = C::F, EF = C::EF> + AsRecursive<C>> Witnessable<C>
    for WhirProof<BC>
where
    BC::Recursive: RecursiveBasefoldConfig<F = C::F, EF = C::EF, Circuit = C>,
    BC::Commitment:
        Witnessable<C, WitnessVariable = <BC::Recursive as RecursiveBasefoldConfig>::Commitment>,
    C::F: Witnessable<C, WitnessVariable = Felt<C::F>>,
    C::EF: Witnessable<C, WitnessVariable = Ext<C::F, C::EF>>,
    <BC::Challenger as GrindingChallenger>::Witness: Witnessable<C, WitnessVariable = Felt<C::F>>,
    <BC::Recursive as RecursiveBasefoldConfig>::Challenger:
        CanObserveVariable<C, <BC::Recursive as RecursiveBasefoldConfig>::Commitment>,
    <BC::Recursive as RecursiveBasefoldConfig>::Commitment: Copy,
    TensorCsOpening<BC::Tcs>: Witnessable<
        C,
        WitnessVariable = RecursiveTensorCsOpening<<BC::Recursive as RecursiveBasefoldConfig>::Tcs>,
    >,
{
    type WitnessVariable = RecursiveWhirProof<BC::Recursive>;

    fn read(&self, builder: &mut Builder<C>) -> Self::WitnessVariable {
        let initial_sumcheck_polynomials = self
            .initial_sumcheck_polynomials
            .iter()
            .map(|(poly, pow)| (poly.read(builder), pow.read(builder)))
            .collect();
        let commitments = self.commitments.iter().map(|comm| comm.read(builder)).collect();
        let merkle_proofs = self.merkle_proofs.iter().map(|proof| proof.read(builder)).collect();
        let query_proof_of_works =
            self.query_proof_of_works.iter().map(|pow| pow.read(builder)).collect();
        let sumcheck_polynomials = self
            .sumcheck_polynomials
            .iter()
            .map(|round| {
                round.iter().map(|(poly, pow)| (poly.read(builder), pow.read(builder))).collect()
            })
            .collect();
        let final_polynomial = self.final_polynomial.read(builder);
        let final_merkle_proof = self.final_merkle_proof.read(builder);
        let final_sumcheck_polynomials = self
            .final_sumcheck_polynomials
            .iter()
            .map(|(poly, pow)| (poly.read(builder), pow.read(builder)))
            .collect();
        let final_pow = self.final_pow.read(builder);
        RecursiveWhirProof {
            initial_sumcheck_polynomials,
            commitments,
            merkle_proofs,
            query_proof_of_works,
            sumcheck_polynomials,
            final_polynomial,
            final_merkle_proof,
            final_sumcheck_polynomials,
            final_pow,
            _config: PhantomData,
        }
    }

    fn write(&self, witness: &mut impl crate::witness::WitnessWriter<C>) {
        for (poly, pow) in &self.initial_sumcheck_polynomials {
            poly.write(witness);
            pow.write(witness);
        }
        for comm in &self.commitments {
            comm.write(witness);
        }
        for proof in &self.merkle_proofs {
            proof.write(witness);
        }
        for pow in &self.query_proof_of_works {
            pow.write(witness);
        }
        for round in &self.sumcheck_polynomials {
            for (poly, pow) in round {
                poly.write(witness);
                pow.write(witness);
            }
        }
        self.final_polynomial.write(witness);
        self.final_merkle_proof.write(witness);
        for (poly, pow) in &self.final_sumcheck_polynomials {
            poly.write(witness);
            pow.write(witness);
        }
        self.final_pow.write(witness);
    }
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use slop_dft::p3::Radix2DitParallel;
    use slop_merkle_tree::{FieldMerkleTreeProver, MerkleTreeTcs, Poseidon2KoalaBear16Prover};
    use slop_whir::{default_whir_config, Prover, Verifier};
    use sp1_core_machine::utils::setup_logger;
    use sp1_hypercube::{
        prover::CpuProverBuilder, MachineProof, MachineVerifier, SP1BasefoldConfig,
        SP1CoreJaggedConfig, ShardVerifier,
    };
    use sp1_recursion_compiler::circuit::AsmConfig;
    use sp1_recursion_machine::RecursionAir;
    use std::{collections::VecDeque, marker::PhantomData, sync::Arc};

    use slop_algebra::extension::BinomialExtensionField;
    use sp1_primitives::SP1DiffusionMatrix;

    use crate::{
        basefold::RecursiveBasefoldConfigImpl, challenger::DuplexChallengerVariable,
        witness::Witnessable,
    };

    use super::*;

    use slop_basefold::{BasefoldConfig, DefaultBasefoldConfig};

    use slop_multilinear::Mle;
    use sp1_hypercube::inner_perm;
    use sp1_recursion_compiler::circuit::{AsmBuilder, AsmCompiler};
    use sp1_recursion_executor::Runtime;

    use sp1_primitives::SP1Field;
    type F = SP1Field;
    type EF = BinomialExtensionField<SP1Field, 4>;

    #[tokio::test]
    async fn test_whir() {
        setup_logger();
        let config = default_whir_config();
        type C = SP1BasefoldConfig;

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let mut challenger_prover = C::default_challenger(&C::default_verifier(1));
        let mut challenger_verifier = C::default_challenger(&C::default_verifier(1));

        let merkle_prover: Poseidon2KoalaBear16Prover = FieldMerkleTreeProver::default();

        let prover = Prover::<_, _, C>::new(Radix2DitParallel, merkle_prover).await;
        let merkle_verifier = MerkleTreeTcs::default();
        let verifier = Verifier::<C>::new(merkle_verifier);
        let polynomial: Mle<<C as BasefoldConfig>::F> =
            Mle::rand(&mut rng, 1, config.num_variables as u32);
        let query_vector: Mle<EF> = Mle::<EF>::rand(&mut rng, 1, config.num_variables as u32);

        let claim: EF = polynomial
            .hypercube_iter()
            .zip(query_vector.hypercube_iter())
            .map(|(a, b)| b[0] * a[0])
            .sum();

        tracing::debug!("claimed: {:?}", claim);

        let (commitment, prover_data) =
            prover.commit(polynomial, &mut challenger_prover, &config).await;

        let proof =
            prover.prove(query_vector.clone(), prover_data, &mut challenger_prover, &config).await;

        verifier.observe_commitment(&commitment, &mut challenger_verifier, &config).unwrap();

        let (point, value) =
            verifier.verify(&commitment, claim, &proof, &mut challenger_verifier, &config).unwrap();

        let mut builder = AsmBuilder::<F, EF>::default();
        let mut witness_stream = Vec::new();
        let mut challenger_variable = DuplexChallengerVariable::new(&mut builder);

        Witnessable::<AsmConfig<F, EF>>::write(&commitment, &mut witness_stream);
        let commitment = commitment.read(&mut builder);

        let recursive_verifier = RecursiveWhirVerifier::<
            RecursiveBasefoldConfigImpl<AsmConfig<F, EF>, SP1CoreJaggedConfig>,
        > {
            _marker: PhantomData,
        };

        recursive_verifier.observe_commitment(
            &mut builder,
            &commitment,
            &mut challenger_variable,
            &config,
        );

        Witnessable::<AsmConfig<F, EF>>::write(&point, &mut witness_stream);
        let point = point.read(&mut builder);

        Witnessable::<AsmConfig<F, EF>>::write(&value, &mut witness_stream);
        let value = value.read(&mut builder);

        Witnessable::<AsmConfig<F, EF>>::write(&proof, &mut witness_stream);
        let proof = proof.read(&mut builder);

        Witnessable::<AsmConfig<F, EF>>::write(&claim, &mut witness_stream);
        let eval_claim = claim.read(&mut builder);

        let (point_var, claim_var) = recursive_verifier.verify_whir(
            &mut builder,
            &commitment,
            eval_claim,
            &proof,
            &mut challenger_variable,
            &config,
        );

        for (coord, coord_var) in point_var.iter().zip(point.iter()) {
            builder.assert_ext_eq(*coord, *coord_var);
        }

        builder.assert_ext_eq(claim_var, value);

        let mut buf = VecDeque::<u8>::new();
        let block = builder.into_root_block();
        let mut compiler = AsmCompiler::default();
        let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
        let mut runtime = Runtime::<F, EF, SP1DiffusionMatrix>::new(program.clone(), inner_perm());
        runtime.witness_stream = witness_stream.into();
        runtime.debug_stdout = Box::new(&mut buf);
        runtime.run().unwrap();

        type A = RecursionAir<SP1Field, 3, 2>;
        let machine = A::compress_machine();
        let log_blowup = 1;
        let log_stacking_height = 22;
        let max_log_row_count = 21;
        let verifier = ShardVerifier::from_basefold_parameters(
            log_blowup,
            log_stacking_height,
            max_log_row_count,
            machine,
        );
        let prover = CpuProverBuilder::simple(verifier.clone()).build();

        let (pk, vk) = prover.setup(program, None).await;

        let records = vec![runtime.record.clone()];

        let pk = unsafe { pk.into_inner() };
        let mut shard_proofs = Vec::with_capacity(records.len());
        for record in records {
            let proof = prover.prove_shard(pk.clone(), record).await;
            shard_proofs.push(proof);
        }

        assert!(shard_proofs.len() == 1);

        let proof = MachineProof { shard_proofs };

        let machine_verifier = MachineVerifier::new(verifier);
        tracing::debug_span!("verify the proof")
            .in_scope(|| machine_verifier.verify(&vk, &proof))
            .unwrap();
    }
}
