use serde::{Deserialize, Serialize};
use slop_algebra::{extension::BinomialExtensionField, TwoAdicField};
use slop_alloc::CpuBackend;
use slop_baby_bear::{baby_bear_poseidon2::BabyBearDegree4Duplex, BabyBear};
use slop_basefold::{BasefoldVerifier, FriConfig};
use slop_basefold_prover::BasefoldProver;
use slop_bn254::BNGC;
use slop_challenger::IopCtx;
use slop_koala_bear::{KoalaBear, KoalaBearDegree4Duplex};
use slop_merkle_tree::{
    BnProver, ComputeTcsOpenings, Poseidon2BabyBear16Prover, Poseidon2KoalaBear16Prover,
};
use slop_multilinear::MultilinearPcsProver;
use slop_stacked::{StackedPcsProver, StackedPcsVerifier};
use std::marker::PhantomData;

use crate::{
    DefaultJaggedProver, JaggedAssistProver, JaggedAssistSumAsPolyCPUImpl,
    JaggedEvalSumcheckProver, JaggedPcsVerifier, JaggedProver,
};

pub type BabyBearPoseidon2 = StackedPcsVerifier<BabyBearDegree4Duplex>;

pub type KoalaBearPoseidon2 = StackedPcsVerifier<KoalaBearDegree4Duplex>;

pub type Bn254JaggedConfig<F, EF> = StackedPcsVerifier<BNGC<F, EF>>;
pub type SP1OuterConfig = Bn254JaggedConfig<KoalaBear, BinomialExtensionField<KoalaBear, 4>>;

pub type Poseidon2BabyBearJaggedCpuProverComponents = JaggedBasefoldProverComponents<
    Poseidon2BabyBear16Prover,
    JaggedEvalSumcheckProver<
        BabyBear,
        JaggedAssistSumAsPolyCPUImpl<
            BabyBear,
            BinomialExtensionField<BabyBear, 4>,
            <BabyBearDegree4Duplex as IopCtx>::Challenger,
        >,
        CpuBackend,
        <BabyBearDegree4Duplex as IopCtx>::Challenger,
    >,
    BabyBearDegree4Duplex,
>;

pub type Poseidon2KoalaBearJaggedCpuProverComponents = JaggedBasefoldProverComponents<
    Poseidon2KoalaBear16Prover,
    JaggedEvalSumcheckProver<
        KoalaBear,
        JaggedAssistSumAsPolyCPUImpl<
            KoalaBear,
            BinomialExtensionField<KoalaBear, 4>,
            <KoalaBearDegree4Duplex as IopCtx>::Challenger,
        >,
        CpuBackend,
        <KoalaBearDegree4Duplex as IopCtx>::Challenger,
    >,
    KoalaBearDegree4Duplex,
>;

pub type Poseidon2Bn254JaggedCpuProverComponents<F, EF> = JaggedBasefoldProverComponents<
    BnProver<F, EF>,
    JaggedEvalSumcheckProver<
        F,
        JaggedAssistSumAsPolyCPUImpl<
            F,
            BinomialExtensionField<F, 4>,
            <BNGC<F, EF> as IopCtx>::Challenger,
        >,
        CpuBackend,
        <BNGC<F, EF> as IopCtx>::Challenger,
    >,
    BNGC<F, EF>,
>;

#[derive(Serialize, Deserialize, PartialEq, Eq, Clone)]
pub struct JaggedBasefoldProverComponents<B, E, GC>(B, E, PhantomData<GC>);

impl<GC> JaggedPcsVerifier<GC, StackedPcsVerifier<GC>>
where
    GC: IopCtx<F: TwoAdicField, EF: TwoAdicField>,
{
    pub fn new_from_basefold_params(
        fri_config: FriConfig<GC::F>,
        log_stacking_height: u32,
        max_log_row_count: usize,
        expected_number_of_commits: usize,
    ) -> Self {
        let basefold_verifer = BasefoldVerifier::<GC>::new(fri_config, expected_number_of_commits);
        let stacked_pcs_verifier = StackedPcsVerifier::new(basefold_verifer, log_stacking_height);
        Self::new(stacked_pcs_verifier, max_log_row_count)
    }
}

impl<GC, Bpc> JaggedProver<GC, StackedPcsProver<Bpc, GC>>
where
    Bpc: ComputeTcsOpenings<GC, CpuBackend>,
    GC: IopCtx<F: TwoAdicField, EF: TwoAdicField>,
{
    pub fn from_basefold_components(
        verifier: &JaggedPcsVerifier<GC, StackedPcsVerifier<GC>>,
        interleave_batch_size: usize,
    ) -> Self {
        let pcs_prover = BasefoldProver::new(&verifier.pcs_verifier.basefold_verifier);
        let stacked_pcs_prover = StackedPcsProver::new(
            pcs_prover,
            verifier.pcs_verifier.log_stacking_height,
            interleave_batch_size,
        );

        Self::new(
            verifier.max_log_row_count,
            stacked_pcs_prover,
            JaggedAssistProver::<GC>::default(),
        )
    }
}

const DEFAULT_INTERLEAVE_BATCH_SIZE: usize = 32;

impl<Bpc, GC> DefaultJaggedProver<GC> for StackedPcsProver<Bpc, GC>
where
    GC: IopCtx<F: TwoAdicField, EF: TwoAdicField>,
    Bpc: ComputeTcsOpenings<GC, CpuBackend>,
{
    fn prover_from_verifier(
        verifier: &JaggedPcsVerifier<GC, <Self as MultilinearPcsProver<GC>>::Verifier>,
    ) -> JaggedProver<GC, Self> {
        JaggedProver::from_basefold_components(verifier, DEFAULT_INTERLEAVE_BATCH_SIZE)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rand::{thread_rng, Rng};
    use slop_challenger::CanObserve;
    use slop_commit::Rounds;
    use slop_multilinear::{Evaluations, Mle, MleEval, PaddedMle, Point};
    use slop_stacked::ToMle;

    use super::*;

    #[tokio::test]
    async fn test_baby_bear_jagged_basefold() {
        test_jagged_basefold::<
            BabyBearDegree4Duplex,
            StackedPcsProver<Poseidon2BabyBear16Prover, _>,
        >()
        .await;
    }

    #[tokio::test]
    async fn test_koala_bear_jagged_basefold() {
        test_jagged_basefold::<
            KoalaBearDegree4Duplex,
            StackedPcsProver<Poseidon2KoalaBear16Prover, _>,
        >()
        .await;
    }

    #[tokio::test]
    async fn test_bn254_jagged_basefold() {
        test_jagged_basefold::<
            BNGC<BabyBear, BinomialExtensionField<BabyBear, 4>>,
            StackedPcsProver<BnProver<BabyBear, BinomialExtensionField<BabyBear, 4>>, _>,
        >()
        .await;
    }

    #[tokio::test]
    async fn test_bn254_jagged_kb_basefold() {
        test_jagged_basefold::<
            BNGC<KoalaBear, BinomialExtensionField<KoalaBear, 4>>,
            StackedPcsProver<BnProver<KoalaBear, BinomialExtensionField<KoalaBear, 4>>, _>,
        >()
        .await;
    }

    // #[tokio::test]
    async fn test_jagged_basefold<
        GC: IopCtx<F: TwoAdicField, EF: TwoAdicField>,
        PcsProver: MultilinearPcsProver<GC, Verifier = StackedPcsVerifier<GC>> + DefaultJaggedProver<GC>,
    >()
    where
        rand::distributions::Standard: rand::distributions::Distribution<GC::F>,
        rand::distributions::Standard: rand::distributions::Distribution<GC::EF>,
        PcsProver::ProverData: ToMle<GC::F, CpuBackend>,
    {
        let row_counts_rounds = vec![vec![1 << 10, 0, 1 << 10], vec![1 << 8]];
        let column_counts_rounds = vec![vec![128, 45, 32], vec![512]];
        let num_rounds = row_counts_rounds.len();

        let log_stacking_height = 11;
        let max_log_row_count = 10;

        let row_counts = row_counts_rounds.into_iter().collect::<Rounds<Vec<usize>>>();
        let column_counts = column_counts_rounds.into_iter().collect::<Rounds<Vec<usize>>>();

        assert!(row_counts.len() == column_counts.len());

        let mut rng = thread_rng();

        let round_mles = row_counts
            .iter()
            .zip(column_counts.iter())
            .map(|(row_counts, col_counts)| {
                row_counts
                    .iter()
                    .zip(col_counts.iter())
                    .map(|(num_rows, num_cols)| {
                        if *num_rows == 0 {
                            PaddedMle::zeros(*num_cols, max_log_row_count)
                        } else {
                            let mle = Mle::<GC::F>::rand(&mut rng, *num_cols, num_rows.ilog(2));
                            PaddedMle::padded_with_zeros(Arc::new(mle), max_log_row_count)
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Rounds<_>>();

        let jagged_verifier =
            JaggedPcsVerifier::<GC, StackedPcsVerifier<GC>>::new_from_basefold_params(
                FriConfig::default_fri_config(),
                log_stacking_height,
                max_log_row_count as usize,
                num_rounds,
            );

        let jagged_prover = JaggedProver::<GC, PcsProver>::from_verifier(&jagged_verifier);

        let eval_point = (0..max_log_row_count).map(|_| rng.gen::<GC::EF>()).collect::<Point<_>>();

        // Begin the commit rounds
        let mut challenger = jagged_verifier.challenger();

        let mut prover_data = Rounds::new();
        let mut commitments = Rounds::new();
        for round in round_mles.iter() {
            let (commit, data) =
                jagged_prover.commit_multilinears(round.clone()).await.ok().unwrap();
            challenger.observe(commit);
            prover_data.push(data);
            commitments.push(commit);
        }

        let mut evaluation_claims = Rounds::new();
        for round in round_mles.iter() {
            let mut evals = Evaluations::default();
            for mle in round.iter() {
                let eval = mle.eval_at(&eval_point).await;
                evals.push(eval);
            }
            evaluation_claims.push(evals);
        }

        let proof = jagged_prover
            .prove_trusted_evaluations(
                eval_point.clone(),
                evaluation_claims.clone(),
                prover_data,
                &mut challenger,
            )
            .await
            .ok()
            .unwrap();

        let mut challenger = jagged_verifier.challenger();
        for commitment in commitments.iter() {
            challenger.observe(*commitment);
        }

        let evaluation_claims = evaluation_claims
            .iter()
            .map(|round| {
                round.iter().flat_map(|evals| evals.iter().cloned()).collect::<MleEval<_>>()
            })
            .collect::<Vec<_>>();

        jagged_verifier
            .verify_trusted_evaluations(
                &commitments,
                eval_point,
                &evaluation_claims,
                &proof,
                &mut challenger,
            )
            .unwrap();
    }
}
