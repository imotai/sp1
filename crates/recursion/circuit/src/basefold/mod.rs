use crate::{
    challenger::{CanObserveVariable, CanSampleBitsVariable, FieldChallengerVariable},
    CircuitConfig, SP1FieldConfigVariable,
};
use itertools::Itertools;
use slop_algebra::{
    extension::{BinomialExtensionField, BinomiallyExtendable, HasTwoAdicBionmialExtension},
    AbstractField, PrimeField31, TwoAdicField,
};
use slop_basefold::FriConfig;
use slop_multilinear::{Evaluations, Point};
use sp1_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, DslIr, Ext, Felt, SymbolicExt},
};

use sp1_primitives::SP1Field;
use sp1_recursion_executor::D;
use std::{iter::once, marker::PhantomData};
use tcs::{RecursiveMerkleTreeTcs, RecursiveTcs, RecursiveTensorCsOpening};
pub mod merkle_tree;
pub mod stacked;
pub mod tcs;
mod whir;
pub mod witness;
use crate::AsRecursive;
use slop_basefold::Poseidon2Bn254FrBasefoldConfig;
use sp1_hypercube::{SP1BasefoldConfig, SP1CoreJaggedConfig, SP1OuterConfig};

pub trait RecursiveBasefoldConfig: Sized {
    type F: Copy;
    type EF: Copy;
    type Commitment;
    type Circuit: CircuitConfig<F = Self::F, EF = Self::EF, Bit = Self::Bit>;
    type Bit;
    type Tcs: RecursiveTcs<
        Data = Felt<Self::F>,
        Commitment = Self::Commitment,
        Circuit = Self::Circuit,
        Bit = Self::Bit,
    >;
    type Challenger: CanObserveVariable<Self::Circuit, Felt<Self::F>>;
}

pub struct RecursiveBasefoldConfigImpl<C, SC>(PhantomData<(C, SC)>);

impl<C: CircuitConfig> AsRecursive<C> for SP1BasefoldConfig {
    type Recursive = RecursiveBasefoldConfigImpl<C, SP1CoreJaggedConfig>;
}

impl<C: CircuitConfig> AsRecursive<C> for Poseidon2Bn254FrBasefoldConfig<C::F>
where
    C::F: PrimeField31 + TwoAdicField + BinomiallyExtendable<4> + HasTwoAdicBionmialExtension<4>,
{
    type Recursive = RecursiveBasefoldConfigImpl<C, SP1OuterConfig>;
}

impl<
        C: CircuitConfig<F = SP1Field, EF = BinomialExtensionField<SP1Field, 4>>,
        SC: SP1FieldConfigVariable<C>,
    > RecursiveBasefoldConfig for RecursiveBasefoldConfigImpl<C, SC>
{
    type F = SP1Field;
    type EF = BinomialExtensionField<SP1Field, 4>;
    type Commitment = SC::DigestVariable;
    type Circuit = C;
    type Bit = C::Bit;
    type Tcs = RecursiveMerkleTreeTcs<C, SC>;
    type Challenger = SC::FriChallengerVariable;
}

pub struct RecursiveBasefoldProof<B: RecursiveBasefoldConfig> {
    /// The univariate polynomials that are used in the sumcheck part of the BaseFold protocol.
    pub univariate_messages: Vec<[Ext<B::F, B::EF>; 2]>,
    /// The FRI parts of the proof.
    /// The commitments to the folded polynomials produced in the commit phase.
    pub fri_commitments: Vec<<B::Tcs as RecursiveTcs>::Commitment>,
    /// The query openings for the individual multilinear polynmomials.
    /// The vector is indexed by the batch number.
    pub component_polynomials_query_openings: Vec<RecursiveTensorCsOpening<B::Tcs>>,
    /// The query openings and the FRI query proofs for the FRI query phase.
    pub query_phase_openings: Vec<RecursiveTensorCsOpening<B::Tcs>>,
    /// The prover performs FRI until we reach a polynomial of degree 0, and return the constant
    /// value of this polynomial.
    pub final_poly: Ext<B::F, B::EF>,
    /// Proof-of-work witness.
    pub pow_witness: Felt<B::F>,
}

pub struct RecursiveBasefoldVerifier<B: RecursiveBasefoldConfig> {
    pub fri_config: FriConfig<B::F>,
    pub tcs: B::Tcs,
}

pub trait RecursiveMultilinearPcsVerifier: Sized {
    type F: Copy;
    type EF: Copy;
    type Commitment;
    type Proof;
    type Circuit: CircuitConfig<F = Self::F, EF = Self::EF, Bit = Self::Bit>;
    type Bit;
    type Challenger: CanObserveVariable<Self::Circuit, Felt<Self::F>>;

    fn verify_trusted_evaluations(
        &self,
        builder: &mut Builder<Self::Circuit>,
        commitments: &[Self::Commitment],
        point: Point<Ext<Self::F, Self::EF>>,
        evaluation_claims: &[Evaluations<Ext<Self::F, Self::EF>>],
        proof: &Self::Proof,
        challenger: &mut Self::Challenger,
    );

    fn verify_untrusted_evaluations(
        &self,
        builder: &mut Builder<Self::Circuit>,
        commitments: &[Self::Commitment],
        point: Point<Ext<Self::F, Self::EF>>,
        evaluation_claims: &[Evaluations<Ext<Self::F, Self::EF>>],
        proof: &Self::Proof,
        challenger: &mut Self::Challenger,
    ) {
        for round in evaluation_claims.iter() {
            for round_evaluations in round.iter() {
                for evaluations in round_evaluations.iter() {
                    for evaluation in evaluations.iter() {
                        let evaluation_felts = Self::Circuit::ext2felt(builder, *evaluation);
                        evaluation_felts.iter().for_each(|felt| challenger.observe(builder, *felt));
                    }
                }
            }
        }
        self.verify_trusted_evaluations(
            builder,
            commitments,
            point,
            evaluation_claims,
            proof,
            challenger,
        )
    }
}

impl<
        C: CircuitConfig<F = SP1Field, EF = BinomialExtensionField<SP1Field, 4>>,
        SC: SP1FieldConfigVariable<C>,
    > RecursiveMultilinearPcsVerifier
    for RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>
{
    type F = SP1Field;
    type EF = BinomialExtensionField<SP1Field, 4>;
    type Commitment = SC::DigestVariable;
    type Proof = RecursiveBasefoldProof<RecursiveBasefoldConfigImpl<C, SC>>;
    type Circuit = C;
    type Bit = C::Bit;
    type Challenger = SC::FriChallengerVariable;

    fn verify_trusted_evaluations(
        &self,
        builder: &mut Builder<Self::Circuit>,
        commitments: &[Self::Commitment],
        point: Point<Ext<Self::F, Self::EF>>,
        evaluation_claims: &[Evaluations<Ext<Self::F, Self::EF>>],
        proof: &Self::Proof,
        challenger: &mut Self::Challenger,
    ) {
        self.verify_mle_evaluations(
            builder,
            commitments,
            point,
            evaluation_claims,
            proof,
            challenger,
        )
    }
}

impl<
        C: CircuitConfig<F = SP1Field, EF = BinomialExtensionField<SP1Field, 4>>,
        SC: SP1FieldConfigVariable<C>,
    > RecursiveBasefoldVerifier<RecursiveBasefoldConfigImpl<C, SC>>
{
    fn verify_mle_evaluations(
        &self,
        builder: &mut Builder<C>,
        commitments: &[SC::DigestVariable],
        mut point: Point<Ext<C::F, C::EF>>,
        evaluation_claims: &[Evaluations<Ext<C::F, C::EF>>],
        proof: &RecursiveBasefoldProof<RecursiveBasefoldConfigImpl<C, SC>>,
        challenger: &mut SC::FriChallengerVariable,
    ) {
        // Sample the challenge used to batch all the different polynomials.
        let batching_challenge = SymbolicExt::<C::F, C::EF>::from(challenger.sample_ext(builder));

        builder.cycle_tracker_v2_enter("compute eval_claim");
        // Compute the batched evaluation claim.
        let eval_claim = evaluation_claims
            .iter()
            .flat_map(|batch_claims| batch_claims.iter().flat_map(|eval| eval.iter()))
            .flatten()
            .zip(batching_challenge.powers())
            .map(|(eval, batch_power)| *eval * batch_power)
            .sum::<SymbolicExt<C::F, C::EF>>();
        builder.cycle_tracker_v2_exit();

        // Assert correctness of shape.
        assert_eq!(
            proof.fri_commitments.len(),
            proof.univariate_messages.len(),
            "Sumcheck FRI Length Mismatch"
        );

        // The prover messages correspond to fixing the last coordinate first, so we reverse the
        // underlying point for the verification.
        point.reverse();

        // Sample the challenges used for FRI folding and BaseFold random linear combinations.
        let betas = proof
            .fri_commitments
            .iter()
            .zip(proof.univariate_messages.iter())
            .map(|(commitment, poly)| {
                poly.iter().copied().for_each(|x| {
                    let x_felts = C::ext2felt(builder, x);
                    x_felts.iter().for_each(|felt| challenger.observe(builder, *felt));
                });
                challenger.observe(builder, *commitment);
                challenger.sample_ext(builder)
            })
            .collect::<Vec<_>>();

        // Check the consistency of the first univariate message with the claimed evaluation. The
        // first_poly is supposed to be `vals(X_0, X_1, ..., X_{d-1}, 0), vals(X_0, X_1, ...,
        // X_{d-1}, 1)`. Given this, the claimed evaluation should be `(1 - X_d) *
        // first_poly[0] + X_d * first_poly[1]`.
        let first_poly = proof.univariate_messages[0];
        let one: Ext<C::F, C::EF> = builder.constant(C::EF::one());

        builder.assert_ext_eq(
            eval_claim,
            (one - *point[0]) * first_poly[0] + *point[0] * first_poly[1],
        );

        // Fold the two messages into a single evaluation claim for the next round, using the
        // sampled randomness.
        let mut expected_eval = first_poly[0] + betas[0] * first_poly[1];

        // Check round-by-round consistency between the successive sumcheck univariate messages.
        for (i, (poly, beta)) in
            proof.univariate_messages[1..].iter().zip(betas[1..].iter()).enumerate()
        {
            // The check is similar to the one for `first_poly`.
            let i = i + 1;
            builder.assert_ext_eq(expected_eval, (one - *point[i]) * poly[0] + *point[i] * poly[1]);

            // Fold the two pieces of the message.
            expected_eval = poly[0] + *beta * poly[1];
        }

        let final_poly_felts = C::ext2felt(builder, proof.final_poly);
        final_poly_felts.iter().for_each(|felt| {
            challenger.observe(builder, *felt);
        });

        // Check proof of work (grinding to find a number that hashes to have
        // `self.config.proof_of_work_bits` zeroes at the beginning).
        challenger.check_witness(builder, self.fri_config.proof_of_work_bits, proof.pow_witness);

        let log_len = proof.fri_commitments.len();

        builder.cycle_tracker_v2_enter("sample query_indices");
        // Sample query indices for the FRI query IOPP part of BaseFold. This part is very similar
        // to the corresponding part in the univariate FRI verifier.
        let query_indices = (0..self.fri_config.num_queries)
            .map(|_| challenger.sample_bits(builder, log_len + self.fri_config.log_blowup()))
            .collect::<Vec<_>>();
        builder.cycle_tracker_v2_exit();

        builder.cycle_tracker_v2_enter("compute batch_evals");

        // Compute the batch evaluations from the openings of the component polynomials.
        let zero = SymbolicExt::<C::F, C::EF>::zero();
        let mut batch_evals = vec![zero; query_indices.len()];
        let mut batch_challenge_power = SymbolicExt::from(one);
        for opening in proof.component_polynomials_query_openings.iter() {
            let values = &opening.values;
            for (batch_eval, values) in batch_evals.iter_mut().zip_eq(values.split()) {
                let beta_powers = batching_challenge.shifted_powers(batch_challenge_power);
                for (value, beta_power) in values.as_slice().iter().zip(beta_powers) {
                    *batch_eval += beta_power * *value;
                }
            }
            let count = values.get(0).unwrap().as_slice().len();
            batch_challenge_power =
                batching_challenge.shifted_powers(batch_challenge_power).nth(count).unwrap();
        }
        let batch_evals: Vec<Ext<C::F, C::EF>> =
            batch_evals.into_iter().map(|x| builder.eval(x)).collect_vec();
        builder.cycle_tracker_v2_exit();

        builder.cycle_tracker_v2_enter("verify_tensor_openings");

        // Verify the proof of the claimed values.
        for (commit, opening) in
            commitments.iter().zip_eq(proof.component_polynomials_query_openings.iter())
        {
            RecursiveMerkleTreeTcs::<C, SC>::verify_tensor_openings(
                builder,
                commit,
                &query_indices,
                opening,
            );
        }
        builder.cycle_tracker_v2_exit();

        builder.cycle_tracker_v2_enter("verify_queries");
        // Check that the query openings are consistent as FRI messages.
        self.verify_queries(
            builder,
            &proof.fri_commitments,
            &query_indices,
            proof.final_poly,
            batch_evals,
            &proof.query_phase_openings,
            &betas,
        );
        builder.cycle_tracker_v2_exit();

        // The final consistency check between the FRI messages and the partial evaluation messages.
        builder.assert_ext_eq(
            proof.final_poly,
            proof.univariate_messages.last().unwrap()[0]
                + *betas.last().unwrap() * proof.univariate_messages.last().unwrap()[1],
        );
    }

    /// The FRI verifier for a single query. We modify this from Plonky3 to be compatible with
    /// opening only a single vector.
    #[allow(clippy::too_many_arguments)]
    fn verify_queries(
        &self,
        builder: &mut Builder<C>,
        commitments: &[SC::DigestVariable],
        indices: &[Vec<C::Bit>],
        final_poly: Ext<C::F, C::EF>,
        reduced_openings: Vec<Ext<C::F, C::EF>>,
        query_openings: &[RecursiveTensorCsOpening<RecursiveMerkleTreeTcs<C, SC>>],
        betas: &[Ext<C::F, C::EF>],
    ) {
        let log_max_height = commitments.len() + self.fri_config.log_blowup();

        let mut folded_evals = reduced_openings;
        builder.cycle_tracker_v2_enter("compute exp reverse bits");
        let mut xis: Vec<Felt<C::F>> = indices
            .iter()
            .map(|index| {
                let two_adic_generator: Felt<C::F> =
                    builder.constant(C::F::two_adic_generator(log_max_height));
                C::exp_reverse_bits(builder, two_adic_generator, index.to_vec())
            })
            .collect::<Vec<_>>();
        builder.cycle_tracker_v2_exit();

        let mut indices = indices.to_vec();

        // Loop over the FRI queries.
        for ((commitment, query_opening), beta) in
            commitments.iter().zip_eq(query_openings.iter()).zip_eq(betas)
        {
            let openings = &query_opening.values;

            for (((index, folded_eval), opening), x) in indices
                .iter_mut()
                .zip_eq(folded_evals.iter_mut())
                .zip_eq(openings.split())
                .zip_eq(xis.iter_mut())
            {
                let index_sibling_complement = index[0];
                let index_pair = &index[1..];

                builder.reduce_e(*folded_eval);

                let evals: [Ext<C::F, C::EF>; 2] = opening
                    .as_slice()
                    .chunks_exact(D)
                    .map(|slice| {
                        let reconstructed_ext: Ext<C::F, C::EF> =
                            C::felt2ext(builder, slice.try_into().unwrap());
                        reconstructed_ext
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();

                let eval_ordered = C::select_chain_ef(
                    builder,
                    index_sibling_complement,
                    once(evals[0]),
                    once(evals[1]),
                );

                // Check that the folded evaluation is consistent with the FRI query proof opening.
                builder.assert_ext_eq(eval_ordered[0], *folded_eval);

                let xs_new = builder.eval((*x) * C::F::two_adic_generator(1));
                let xs =
                    C::select_chain_f(builder, index_sibling_complement, once(*x), once(xs_new));

                // interpolate and evaluate at beta
                let temp_1: Felt<_> = builder.uninit();
                builder.push_op(DslIr::SubF(temp_1, xs[1], xs[0]));

                // let temp_2 = evals_ext[1] - evals_ext[0];
                let temp_2: Ext<_, _> = builder.uninit();
                builder.push_op(DslIr::SubE(temp_2, evals[1], evals[0]));

                // let temp_3 = temp_2 / temp_1;
                let temp_3: Ext<_, _> = builder.uninit();
                builder.push_op(DslIr::DivEF(temp_3, temp_2, temp_1));

                // let temp_4 = beta - xs[0];
                let temp_4: Ext<_, _> = builder.uninit();
                builder.push_op(DslIr::SubEF(temp_4, *beta, xs[0]));

                // let temp_5 = temp_4 * temp_3;
                let temp_5: Ext<_, _> = builder.uninit();
                builder.push_op(DslIr::MulE(temp_5, temp_4, temp_3));

                // let temp6 = evals_ext[0] + temp_5;
                let temp_6: Ext<_, _> = builder.uninit();
                builder.push_op(DslIr::AddE(temp_6, evals[0], temp_5));
                *folded_eval = temp_6;

                // let temp_7 = x * x;
                let temp_7: Felt<_> = builder.uninit();
                builder.push_op(DslIr::MulF(temp_7, *x, *x));
                *x = temp_7;

                *index = index_pair.to_vec();
            }
            // Check that the opening is consistent with the commitment.
            RecursiveMerkleTreeTcs::<C, SC>::verify_tensor_openings(
                builder,
                commitment,
                &indices,
                query_opening,
            );
        }

        for folded_eval in folded_evals {
            builder.assert_ext_eq(folded_eval, final_poly);
        }
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;
    use slop_commit::Message;
    use sp1_recursion_compiler::circuit::AsmConfig;
    use std::sync::Arc;

    use slop_algebra::extension::BinomialExtensionField;
    use sp1_primitives::SP1DiffusionMatrix;

    use crate::{challenger::DuplexChallengerVariable, witness::Witnessable};

    use super::*;
    use futures::prelude::*;

    use slop_basefold::BasefoldVerifier;
    use slop_basefold_prover::BasefoldProver;
    use slop_challenger::CanObserve;

    use slop_commit::Rounds;

    use slop_multilinear::Mle;
    use sp1_hypercube::inner_perm;
    use sp1_primitives::SP1Field;
    use sp1_recursion_compiler::circuit::{AsmBuilder, AsmCompiler};
    use sp1_recursion_executor::Runtime;

    type F = SP1Field;
    type EF = BinomialExtensionField<SP1Field, 4>;

    #[tokio::test]
    async fn test_basefold_proof() {
        type C = SP1BasefoldConfig;
        type Prover = BasefoldProver<sp1_hypercube::prover::SP1BasefoldCpuProverComponents>;

        let num_variables = 16;
        let round_widths = [vec![16, 10, 14], vec![20, 78, 34], vec![10, 10]];
        let log_blowup = 1;

        let mut rng = thread_rng();
        let round_mles = round_widths
            .iter()
            .map(|widths| {
                widths
                    .iter()
                    .map(|&w| Mle::<SP1Field>::rand(&mut rng, w, num_variables))
                    .collect::<Message<_>>()
            })
            .collect::<Rounds<_>>();

        let verifier = BasefoldVerifier::<C>::new(log_blowup);
        let recursive_verifier = RecursiveBasefoldVerifier::<
            RecursiveBasefoldConfigImpl<AsmConfig<F, EF>, SP1CoreJaggedConfig>,
        > {
            fri_config: verifier.fri_config,
            tcs: RecursiveMerkleTreeTcs::<AsmConfig<F, EF>, SP1CoreJaggedConfig>(PhantomData),
        };

        let prover = Prover::new(&verifier);

        let mut challenger = verifier.challenger();
        let mut commitments = vec![];
        let mut prover_data = Rounds::new();
        let mut eval_claims = Rounds::new();
        let point = Point::<EF>::rand(&mut rng, num_variables);
        for mles in round_mles.iter() {
            let (commitment, data) = prover.commit_mles(mles.clone()).await.unwrap();
            challenger.observe(commitment);
            commitments.push(commitment);
            prover_data.push(data);
            let evaluations = stream::iter(mles.iter())
                .then(|mle| mle.eval_at(&point))
                .collect::<Evaluations<_>>()
                .await;
            eval_claims.push(evaluations);
        }

        let proof = prover
            .prove_trusted_mle_evaluations(
                point.clone(),
                round_mles,
                eval_claims.clone(),
                prover_data,
                &mut challenger,
            )
            .await
            .unwrap();

        let mut builder = AsmBuilder::<F, EF>::default();
        let mut witness_stream = Vec::new();
        let mut challenger_variable = DuplexChallengerVariable::new(&mut builder);

        for commitment in commitments.iter() {
            challenger.observe(*commitment);
        }

        Witnessable::<AsmConfig<F, EF>>::write(&commitments, &mut witness_stream);
        let commitments = commitments.read(&mut builder);

        for commitment in commitments.iter() {
            challenger_variable.observe(&mut builder, *commitment);
        }

        Witnessable::<AsmConfig<F, EF>>::write(&point, &mut witness_stream);
        let point = point.read(&mut builder);

        Witnessable::<AsmConfig<F, EF>>::write(&eval_claims, &mut witness_stream);
        let eval_claims = eval_claims.read(&mut builder);

        Witnessable::<AsmConfig<F, EF>>::write(&proof, &mut witness_stream);
        let proof = proof.read(&mut builder);

        RecursiveBasefoldVerifier::<
            RecursiveBasefoldConfigImpl<AsmConfig<F, EF>, SP1CoreJaggedConfig>,
        >::verify_mle_evaluations(
            &recursive_verifier,
            &mut builder,
            &commitments,
            point,
            &eval_claims,
            &proof,
            &mut challenger_variable,
        );
        let block = builder.into_root_block();
        let mut compiler = AsmCompiler::default();
        let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
        let mut runtime = Runtime::<F, EF, SP1DiffusionMatrix>::new(program.clone(), inner_perm());
        runtime.witness_stream = witness_stream.into();
        runtime.run().unwrap();
    }

    #[tokio::test]
    async fn test_invalid_basefold_proof() {
        type C = SP1BasefoldConfig;
        type Prover = BasefoldProver<sp1_hypercube::prover::SP1BasefoldCpuProverComponents>;

        let num_variables = 16;
        let round_widths = [vec![16, 10, 14], vec![20, 78, 34], vec![10, 10]];
        let log_blowup = 1;

        let mut rng = thread_rng();
        let round_mles = round_widths
            .iter()
            .map(|widths| {
                widths
                    .iter()
                    .map(|&w| Mle::<SP1Field>::rand(&mut rng, w, num_variables))
                    .collect::<Message<_>>()
            })
            .collect::<Rounds<_>>();

        let verifier = BasefoldVerifier::<C>::new(log_blowup);
        let recursive_verifier = RecursiveBasefoldVerifier::<
            RecursiveBasefoldConfigImpl<AsmConfig<F, EF>, SP1CoreJaggedConfig>,
        > {
            fri_config: verifier.fri_config,
            tcs: RecursiveMerkleTreeTcs::<AsmConfig<F, EF>, SP1CoreJaggedConfig>(PhantomData),
        };

        let prover = Prover::new(&verifier);

        let mut challenger = verifier.challenger();
        let mut commitments = vec![];
        let mut prover_data = Rounds::new();
        let mut eval_claims = Rounds::new();
        let point = Point::<EF>::rand(&mut rng, num_variables);
        for mles in round_mles.iter() {
            let (commitment, data) = prover.commit_mles(mles.clone()).await.unwrap();
            challenger.observe(commitment);
            commitments.push(commitment);
            prover_data.push(data);
            let evaluations = stream::iter(mles.iter())
                .then(|mle| mle.eval_at(&point))
                .collect::<Evaluations<_>>()
                .await;
            eval_claims.push(evaluations);
        }

        let proof = prover
            .prove_trusted_mle_evaluations(
                point.clone(),
                round_mles,
                eval_claims.clone(),
                prover_data,
                &mut challenger,
            )
            .await
            .unwrap();

        // Make a new point that is different from the original point.
        let point = Point::<EF>::rand(&mut rng, num_variables);

        let mut builder = AsmBuilder::<F, EF>::default();
        let mut witness_stream = Vec::new();
        let mut challenger_variable = DuplexChallengerVariable::new(&mut builder);

        for commitment in commitments.iter() {
            challenger.observe(*commitment);
        }

        Witnessable::<AsmConfig<F, EF>>::write(&commitments, &mut witness_stream);
        let commitments = commitments.read(&mut builder);

        for commitment in commitments.iter() {
            challenger_variable.observe(&mut builder, *commitment);
        }

        Witnessable::<AsmConfig<F, EF>>::write(&point, &mut witness_stream);
        let point = point.read(&mut builder);

        Witnessable::<AsmConfig<F, EF>>::write(&eval_claims, &mut witness_stream);
        let eval_claims = eval_claims.read(&mut builder);

        Witnessable::<AsmConfig<F, EF>>::write(&proof, &mut witness_stream);
        let proof = proof.read(&mut builder);

        RecursiveBasefoldVerifier::<
            RecursiveBasefoldConfigImpl<AsmConfig<F, EF>, SP1CoreJaggedConfig>,
        >::verify_mle_evaluations(
            &recursive_verifier,
            &mut builder,
            &commitments,
            point,
            &eval_claims,
            &proof,
            &mut challenger_variable,
        );
        let block = builder.into_root_block();
        let mut compiler = AsmCompiler::default();
        let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
        let mut runtime = Runtime::<F, EF, SP1DiffusionMatrix>::new(program.clone(), inner_perm());
        runtime.witness_stream = witness_stream.into();
        runtime.run().expect_err("invalid proof should not be verified");
    }
}
