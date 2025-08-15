use serde::{Deserialize, Serialize};
use slop_algebra::extension::BinomialExtensionField;
use slop_alloc::CpuBackend;
use slop_baby_bear::BabyBear;
use slop_basefold::{
    BasefoldConfig, BasefoldProof, BasefoldVerifier, DefaultBasefoldConfig,
    Poseidon2BabyBear16BasefoldConfig, Poseidon2Bn254FrBasefoldConfig,
    Poseidon2KoalaBear16BasefoldConfig,
};
use slop_basefold_prover::{
    BasefoldProver, BasefoldProverComponents, DefaultBasefoldProver,
    Poseidon2BabyBear16BasefoldCpuProverComponents, Poseidon2Bn254BasefoldCpuProverComponents,
    Poseidon2KoalaBear16BasefoldCpuProverComponents,
};
use slop_challenger::{CanObserve, FieldChallenger};
use slop_commit::TensorCs;
use slop_koala_bear::KoalaBear;
use slop_stacked::{FixedRateInterleave, StackedPcsProver, StackedPcsVerifier};
use std::fmt::Debug;

use crate::{
    CpuJaggedMleGenerator, DefaultJaggedProver, HadamardJaggedSumcheckProver,
    JaggedAssistSumAsPolyCPUImpl, JaggedBackend, JaggedConfig, JaggedEvalConfig, JaggedEvalProver,
    JaggedEvalSumcheckConfig, JaggedEvalSumcheckProver, JaggedPcsVerifier, JaggedProver,
    JaggedProverComponents, JaggedSumcheckProver, TrivialJaggedEvalConfig,
};

pub type BabyBearPoseidon2 =
    JaggedBasefoldConfig<Poseidon2BabyBear16BasefoldConfig, JaggedEvalSumcheckConfig<BabyBear>>;

pub type KoalaBearPoseidon2 =
    JaggedBasefoldConfig<Poseidon2KoalaBear16BasefoldConfig, JaggedEvalSumcheckConfig<KoalaBear>>;

pub type Bn254JaggedConfig<F> =
    JaggedBasefoldConfig<Poseidon2Bn254FrBasefoldConfig<F>, JaggedEvalSumcheckConfig<F>>;

pub type SP1OuterConfig = Bn254JaggedConfig<KoalaBear>;

pub type BabyBearPoseidon2TrivialEval =
    JaggedBasefoldConfig<Poseidon2BabyBear16BasefoldConfig, TrivialJaggedEvalConfig>;

pub type Poseidon2BabyBearJaggedCpuProverComponents = JaggedBasefoldProverComponents<
    Poseidon2BabyBear16BasefoldCpuProverComponents,
    HadamardJaggedSumcheckProver<CpuJaggedMleGenerator>,
    JaggedEvalSumcheckProver<
        BabyBear,
        JaggedAssistSumAsPolyCPUImpl<
            BabyBear,
            BinomialExtensionField<BabyBear, 4>,
            <BabyBearPoseidon2 as JaggedConfig>::Challenger,
        >,
        CpuBackend,
        <BabyBearPoseidon2 as JaggedConfig>::Challenger,
    >,
>;

pub type Poseidon2KoalaBearJaggedCpuProverComponents = JaggedBasefoldProverComponents<
    Poseidon2KoalaBear16BasefoldCpuProverComponents,
    HadamardJaggedSumcheckProver<CpuJaggedMleGenerator>,
    JaggedEvalSumcheckProver<
        KoalaBear,
        JaggedAssistSumAsPolyCPUImpl<
            KoalaBear,
            BinomialExtensionField<KoalaBear, 4>,
            <KoalaBearPoseidon2 as JaggedConfig>::Challenger,
        >,
        CpuBackend,
        <KoalaBearPoseidon2 as JaggedConfig>::Challenger,
    >,
>;

pub type Poseidon2Bn254JaggedCpuProverComponents<F> = JaggedBasefoldProverComponents<
    Poseidon2Bn254BasefoldCpuProverComponents<F>,
    HadamardJaggedSumcheckProver<CpuJaggedMleGenerator>,
    JaggedEvalSumcheckProver<
        F,
        JaggedAssistSumAsPolyCPUImpl<
            F,
            BinomialExtensionField<F, 4>,
            <Bn254JaggedConfig<F> as JaggedConfig>::Challenger,
        >,
        CpuBackend,
        <Bn254JaggedConfig<F> as JaggedConfig>::Challenger,
    >,
>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JaggedBasefoldConfig<BC, E>(BC, E);

impl<BC, E> JaggedConfig for JaggedBasefoldConfig<BC, E>
where
    BC: BasefoldConfig,
    E: JaggedEvalConfig<BC::F, BC::EF, BC::Challenger> + Clone,
    BC::Commitment: Debug,
{
    type F = BC::F;
    type EF = BC::EF;
    type Commitment = BC::Commitment;
    type BatchPcsProof = BasefoldProof<BC>;
    type Challenger = BC::Challenger;
    type BatchPcsVerifier = BasefoldVerifier<BC>;
    type JaggedEvaluator = E;
}

#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JaggedBasefoldProverComponents<B, J, E>(B, J, E);

impl<Bpc, JaggedProver, EvalProver> JaggedProverComponents
    for JaggedBasefoldProverComponents<Bpc, JaggedProver, EvalProver>
where
    Bpc: BasefoldProverComponents,
    Bpc::A: JaggedBackend<Bpc::F, Bpc::EF>,
    Bpc::Challenger:
        CanObserve<<Bpc::Config as BasefoldConfig>::Commitment> + FieldChallenger<Bpc::F>,
    JaggedProver: JaggedSumcheckProver<Bpc::F, Bpc::EF, Bpc::A>,
    EvalProver: JaggedEvalProver<Bpc::F, Bpc::EF, Bpc::Challenger, A = Bpc::A>
        + 'static
        + Send
        + Sync
        + Clone,
    <<Bpc as BasefoldProverComponents>::Tcs as TensorCs>::Commitment: std::fmt::Debug,
{
    type F = Bpc::F;
    type EF = Bpc::EF;
    type A = Bpc::A;
    type Challenger = Bpc::Challenger;
    type Commitment = <Bpc::Config as BasefoldConfig>::Commitment;
    type BatchPcsProof = BasefoldProof<Bpc::Config>;
    type Config = JaggedBasefoldConfig<Bpc::Config, EvalProver::EvalConfig>;

    type JaggedSumcheckProver = JaggedProver;
    type BatchPcsProver = BasefoldProver<Bpc>;
    type Stacker = FixedRateInterleave<Bpc::F, Bpc::A>;
    type JaggedEvalProver = EvalProver;
}

impl<BC, E> JaggedPcsVerifier<JaggedBasefoldConfig<BC, E>>
where
    BC: DefaultBasefoldConfig,
    BC::Commitment: Debug,
    E: JaggedEvalConfig<BC::F, BC::EF, BC::Challenger> + Default,
{
    pub fn new(log_blowup: usize, log_stacking_height: u32, max_log_row_count: usize) -> Self {
        let basefold_verifer = BasefoldVerifier::<BC>::new(log_blowup);
        let stacked_pcs_verifier = StackedPcsVerifier::new(basefold_verifer, log_stacking_height);
        Self { stacked_pcs_verifier, max_log_row_count, jagged_evaluator: E::default() }
    }
}

impl<Bpc, JP, E> JaggedProver<JaggedBasefoldProverComponents<Bpc, JP, E>>
where
    Bpc: BasefoldProverComponents + DefaultBasefoldProver,
    Bpc::Config: DefaultBasefoldConfig,
    <Bpc::Config as BasefoldConfig>::Commitment: Debug,
    Bpc::A: JaggedBackend<Bpc::F, Bpc::EF>,
    Bpc::Challenger: CanObserve<<Bpc::Config as BasefoldConfig>::Commitment>,
    JP: JaggedSumcheckProver<Bpc::F, Bpc::EF, Bpc::A>,
    E: JaggedEvalProver<Bpc::F, Bpc::EF, Bpc::Challenger, A = Bpc::A> + Default,
{
    pub fn from_basefold_components(
        verifier: &JaggedPcsVerifier<JaggedBasefoldConfig<Bpc::Config, E::EvalConfig>>,
        jagged_generator: JP,
        interleave_batch_size: usize,
    ) -> Self {
        let pcs_prover = BasefoldProver::new(&verifier.stacked_pcs_verifier.pcs_verifier);
        let stacker = FixedRateInterleave::new(interleave_batch_size);
        let stacked_pcs_prover = StackedPcsProver::new(
            pcs_prover,
            stacker,
            verifier.stacked_pcs_verifier.log_stacking_height,
        );

        Self::new(verifier.max_log_row_count, stacked_pcs_prover, jagged_generator, E::default())
    }
}

const DEFAULT_INTERLEAVE_BATCH_SIZE: usize = 32;

impl<Bpc, JP, E> DefaultJaggedProver for JaggedBasefoldProverComponents<Bpc, JP, E>
where
    Bpc: BasefoldProverComponents + DefaultBasefoldProver,
    Bpc::Config: DefaultBasefoldConfig,
    <Bpc::Config as BasefoldConfig>::Commitment: Debug,
    Bpc::A: JaggedBackend<Bpc::F, Bpc::EF>,
    Bpc::Challenger: CanObserve<<Bpc::Config as BasefoldConfig>::Commitment>,
    JP: JaggedSumcheckProver<Bpc::F, Bpc::EF, Bpc::A> + Default,
    E: JaggedEvalProver<Bpc::F, Bpc::EF, Bpc::Challenger, A = Bpc::A> + Default,
{
    fn prover_from_verifier(verifier: &JaggedPcsVerifier<Self::Config>) -> JaggedProver<Self> {
        JaggedProver::from_basefold_components(
            verifier,
            JP::default(),
            DEFAULT_INTERLEAVE_BATCH_SIZE,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rand::{thread_rng, Rng};
    use slop_commit::Rounds;
    use slop_multilinear::{Evaluations, Mle, PaddedMle, Point};

    use crate::MachineJaggedPcsVerifier;

    use super::*;

    //     type JC = BabyBearPoseidon2;
    // type Prover = JaggedProver<Poseidon2BabyBearJaggedCpuProverComponents>;
    // type F = <JC as JaggedConfig>::F;
    // type EF = <JC as JaggedConfig>::EF;

    #[tokio::test]
    async fn test_baby_bear_jagged_basefold() {
        test_jagged_basefold::<
            Poseidon2BabyBear16BasefoldConfig,
            Poseidon2BabyBearJaggedCpuProverComponents,
        >()
        .await;
    }

    #[tokio::test]
    async fn test_koala_bear_jagged_basefold() {
        test_jagged_basefold::<
            Poseidon2KoalaBear16BasefoldConfig,
            Poseidon2KoalaBearJaggedCpuProverComponents,
        >()
        .await;
    }

    #[tokio::test]
    async fn test_bn254_jagged_basefold() {
        test_jagged_basefold::<
            Poseidon2Bn254FrBasefoldConfig<BabyBear>,
            Poseidon2Bn254JaggedCpuProverComponents<BabyBear>,
        >()
        .await;
    }

    #[tokio::test]
    async fn test_bn254_jagged_kb_basefold() {
        test_jagged_basefold::<
            Poseidon2Bn254FrBasefoldConfig<KoalaBear>,
            Poseidon2Bn254JaggedCpuProverComponents<KoalaBear>,
        >()
        .await;
    }

    type E<F> = JaggedEvalSumcheckConfig<F>;
    type JC<B> = JaggedBasefoldConfig<B, E<<B as BasefoldConfig>::F>>;

    // #[tokio::test]
    async fn test_jagged_basefold<
        BC: BasefoldConfig<Commitment: Debug> + DefaultBasefoldConfig,
        Prover: JaggedProverComponents<
                Config = JaggedBasefoldConfig<BC, E<BC::F>>,
                F = BC::F,
                EF = BC::EF,
                Challenger = <JC<BC> as JaggedConfig>::Challenger,
                A = CpuBackend,
                Commitment = <JC<BC> as JaggedConfig>::Commitment,
            > + DefaultJaggedProver,
    >()
    where
        rand::distributions::Standard: rand::distributions::Distribution<<BC as BasefoldConfig>::F>,
        rand::distributions::Standard:
            rand::distributions::Distribution<<BC as BasefoldConfig>::EF>,
        <JC<BC> as JaggedConfig>::Commitment: Copy,
    {
        let row_counts_rounds = vec![vec![1 << 10, 0, 1 << 10], vec![1 << 8]];
        let column_counts_rounds = vec![vec![128, 45, 32], vec![512]];

        let log_blowup = 1;
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
                            let mle = Mle::<BC::F>::rand(&mut rng, *num_cols, num_rows.ilog(2));
                            PaddedMle::padded_with_zeros(Arc::new(mle), max_log_row_count)
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Rounds<_>>();

        let jagged_verifier = JaggedPcsVerifier::<JaggedBasefoldConfig<BC, E<BC::F>>>::new(
            log_blowup,
            log_stacking_height,
            max_log_row_count as usize,
        );

        let jagged_prover = JaggedProver::<Prover>::from_verifier(&jagged_verifier);

        let machine_verifier = MachineJaggedPcsVerifier::new(
            &jagged_verifier,
            vec![column_counts[0].clone(), column_counts[1].clone()],
        );

        let eval_point = (0..max_log_row_count).map(|_| rng.gen::<BC::EF>()).collect::<Point<_>>();

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

        machine_verifier
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
