use slop_algebra::extension::BinomialExtensionField;
use slop_baby_bear::BabyBear;
use slop_commit::Rounds;
use slop_multilinear::{Evaluations, Mle, Point};
use sp1_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{Builder, Ext, SymbolicExt},
};

use crate::primitives::sumcheck::evaluate_mle_ext;

use super::RecursiveMultilinearPcsVerifier;

pub struct RecursiveStackedPcsVerifier<P> {
    pub recursive_pcs_verifier: P,
    pub log_stacking_height: u32,
}

pub struct RecursiveStackedPcsProof<PcsProof, F, EF> {
    pub pcs_proof: PcsProof,
    pub batch_evaluations: Rounds<Evaluations<Ext<F, EF>>>,
}

impl<
        P: RecursiveMultilinearPcsVerifier<F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>>,
    > RecursiveStackedPcsVerifier<P>
{
    pub const fn new(recursive_pcs_verifier: P, log_stacking_height: u32) -> Self {
        Self { recursive_pcs_verifier, log_stacking_height }
    }

    pub fn verify_trusted_evaluation(
        &self,
        builder: &mut Builder<P::Circuit>,
        commitments: &[P::Commitment],
        point: &Point<Ext<P::F, P::EF>>,
        proof: &RecursiveStackedPcsProof<P::Proof, P::F, P::EF>,
        evaluation_claim: SymbolicExt<P::F, P::EF>,
        challenger: &mut P::Challenger,
    ) {
        let (batch_point, stack_point) =
            point.split_at(point.dimension() - self.log_stacking_height as usize);
        let batch_evaluations =
            proof.batch_evaluations.iter().flatten().flatten().cloned().collect::<Mle<_>>();

        builder.cycle_tracker_v2_enter("rizz - evaluate_mle_ext");
        let expected_evaluation = evaluate_mle_ext(builder, batch_evaluations, batch_point)[0];
        builder.assert_ext_eq(evaluation_claim, expected_evaluation);
        builder.cycle_tracker_v2_exit();

        builder.cycle_tracker_v2_enter("rizz - verify_untrusted_evaluations");
        self.recursive_pcs_verifier.verify_untrusted_evaluations(
            builder,
            commitments,
            stack_point,
            &proof.batch_evaluations,
            &proof.pcs_proof,
            challenger,
        );
        builder.cycle_tracker_v2_exit();
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;
    use slop_commit::Message;
    use sp1_core_machine::utils::setup_logger;
    use sp1_recursion_compiler::circuit::AsmConfig;
    use sp1_stark::BabyBearPoseidon2;
    use std::collections::VecDeque;
    use std::marker::PhantomData;
    use std::sync::Arc;

    use slop_algebra::extension::BinomialExtensionField;
    use slop_baby_bear::{BabyBear, DiffusionMatrixBabyBear};

    use crate::basefold::tcs::RecursiveMerkleTreeTcs;
    use crate::basefold::RecursiveBasefoldConfigImpl;
    use crate::witness::Witnessable;
    use crate::{basefold::RecursiveBasefoldVerifier, challenger::DuplexChallengerVariable};

    use super::*;

    use p3_challenger::CanObserve;
    use slop_basefold::{BasefoldVerifier, Poseidon2BabyBear16BasefoldConfig};
    use slop_basefold_prover::{BasefoldProver, Poseidon2BabyBear16BasefoldCpuProverComponents};

    use slop_commit::Rounds;

    use crate::challenger::CanObserveVariable;
    use slop_merkle_tree::my_bb_16_perm;
    use slop_multilinear::Mle;
    use slop_stacked::{FixedRateInterleave, StackedPcsProver, StackedPcsVerifier};
    use sp1_recursion_compiler::circuit::{AsmBuilder, AsmCompiler};
    use sp1_recursion_executor::Runtime;

    type F = BabyBear;

    async fn test_round_widths_and_log_heights(
        round_widths_and_log_heights: &[Vec<(usize, u32)>],
        log_stacking_height: u32,
        batch_size: usize,
    ) {
        type C = Poseidon2BabyBear16BasefoldConfig;
        type Prover = BasefoldProver<Poseidon2BabyBear16BasefoldCpuProverComponents>;
        type EF = BinomialExtensionField<BabyBear, 4>;
        let total_data_length = round_widths_and_log_heights
            .iter()
            .map(|dims| dims.iter().map(|&(w, log_h)| w << log_h).sum::<usize>())
            .sum::<usize>();
        let total_number_of_variables = total_data_length.next_power_of_two().ilog2();
        assert_eq!(1 << total_number_of_variables, total_data_length);

        let log_blowup = 1;

        let mut rng = thread_rng();
        let round_mles = round_widths_and_log_heights
            .iter()
            .map(|dims| {
                dims.iter()
                    .map(|&(w, log_h)| Mle::<BabyBear>::rand(&mut rng, w, log_h))
                    .collect::<Message<_>>()
            })
            .collect::<Rounds<_>>();

        let pcs_verifier = BasefoldVerifier::<C>::new(log_blowup);
        let pcs_prover = Prover::new(&pcs_verifier);
        let stacker = FixedRateInterleave::new(batch_size);

        let verifier = StackedPcsVerifier::new(pcs_verifier, log_stacking_height);
        let prover = StackedPcsProver::new(pcs_prover, stacker, log_stacking_height);

        let mut challenger = verifier.pcs_verifier.challenger();
        let mut commitments = vec![];
        let mut prover_data = Rounds::new();
        let mut batch_evaluations = Rounds::new();
        let point = Point::<EF>::rand(&mut rng, total_number_of_variables);

        let (batch_point, stack_point) =
            point.split_at(point.dimension() - log_stacking_height as usize);
        for mles in round_mles.iter() {
            let (commitment, data) = prover.commit_multilinears(mles.clone()).await.unwrap();
            challenger.observe(commitment);
            commitments.push(commitment);
            let evaluations = prover.round_batch_evaluations(&stack_point, &data).await;
            prover_data.push(data);
            batch_evaluations.push(evaluations);
        }

        // Interpolate the batch evaluations as a multilinear polynomial.
        let batch_evaluations_mle =
            batch_evaluations.iter().flatten().flatten().cloned().collect::<Mle<_>>();
        // Verify that the climed evaluations matched the interpolated evaluations.
        let eval_claim = batch_evaluations_mle.eval_at(&batch_point).await[0];

        let proof = prover
            .prove_trusted_evaluation(
                point.clone(),
                eval_claim,
                prover_data,
                batch_evaluations,
                &mut challenger,
            )
            .await
            .unwrap();

        let mut builder = AsmBuilder::<F, EF>::default();
        let mut witness_stream = Vec::new();
        let mut challenger_variable = DuplexChallengerVariable::new(&mut builder);

        Witnessable::<AsmConfig<F, EF>>::write(&commitments, &mut witness_stream);
        let commitments = commitments.read(&mut builder);

        for commitment in commitments.iter() {
            challenger_variable.observe(&mut builder, *commitment);
        }

        Witnessable::<AsmConfig<F, EF>>::write(&point, &mut witness_stream);
        let point = point.read(&mut builder);

        Witnessable::<AsmConfig<F, EF>>::write(&proof, &mut witness_stream);
        let proof = proof.read(&mut builder);

        Witnessable::<AsmConfig<F, EF>>::write(&eval_claim, &mut witness_stream);
        let eval_claim = eval_claim.read(&mut builder);

        let verifier = BasefoldVerifier::<C>::new(log_blowup);
        let recursive_verifier = RecursiveBasefoldVerifier::<
            RecursiveBasefoldConfigImpl<AsmConfig<F, EF>, BabyBearPoseidon2>,
        > {
            fri_config: verifier.fri_config,
            tcs: RecursiveMerkleTreeTcs::<AsmConfig<F, EF>, BabyBearPoseidon2>(PhantomData),
        };
        let recursive_verifier =
            RecursiveStackedPcsVerifier::new(recursive_verifier, log_stacking_height);

        recursive_verifier.verify_trusted_evaluation(
            &mut builder,
            &commitments,
            &point,
            &proof,
            eval_claim.into(),
            &mut challenger_variable,
        );

        let mut buf = VecDeque::<u8>::new();
        let block = builder.into_root_block();
        let mut compiler = AsmCompiler::default();
        let program = Arc::new(compiler.compile_inner(block).validate().unwrap());
        let mut runtime =
            Runtime::<F, EF, DiffusionMatrixBabyBear>::new(program.clone(), my_bb_16_perm());
        runtime.witness_stream = witness_stream.into();
        runtime.debug_stdout = Box::new(&mut buf);
        runtime.run().unwrap();
    }

    #[tokio::test]
    async fn test_stacked_pcs_proof() {
        setup_logger();
        let round_widths_and_log_heights: Vec<(usize, u32)> =
            vec![(1 << 10, 10), (1 << 4, 11), (496, 11)];
        test_round_widths_and_log_heights(&[round_widths_and_log_heights], 10, 10).await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_stacked_pcs_proof_core_shard() {
        setup_logger();
        let round_widths_and_log_heights = [vec![
            (30, 21),
            (44, 21),
            (45, 21),
            (18, 20),
            (400, 18),
            (25, 20),
            (100, 20),
            (40, 19),
            (22, 19),
        ]];
        test_round_widths_and_log_heights(&round_widths_and_log_heights, 21, 1).await;
        test_round_widths_and_log_heights(&round_widths_and_log_heights, 21, 5).await;
    }

    #[tokio::test]
    #[ignore]
    async fn test_stacked_pcs_proof_precompile_shard() {
        setup_logger();
        let round_widths_and_log_heights = [vec![(4000, 16), (400, 19), (20, 20), (21, 21)]];
        test_round_widths_and_log_heights(&round_widths_and_log_heights, 21, 1).await;
        test_round_widths_and_log_heights(&round_widths_and_log_heights, 21, 5).await;
    }
}
