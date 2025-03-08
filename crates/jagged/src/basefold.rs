use serde::{Deserialize, Serialize};
use slop_basefold::{
    BasefoldConfig, BasefoldProof, BasefoldVerifier, DefaultBasefoldConfig,
    Poseidon2BabyBear16BasefoldConfig,
};
use slop_basefold_prover::{
    BasefoldProver, BasefoldProverComponents, DefaultBasefoldProver,
    Poseidon2BabyBear16BasefoldCpuProverComponents,
};
use slop_challenger::CanObserve;
use slop_stacked::{FixedRateInterleave, StackedPcsProver, StackedPcsVerifier};

use crate::{
    CpuJaggedMleGenerator, DfeaultJaggedProver, HadamardJaggedSumcheckProver, JaggedBackend,
    JaggedConfig, JaggedPcsVerifier, JaggedProver, JaggedProverComponents, JaggedSumcheckProver,
};

pub type BabyBearPoseidon2 = JaggedBasefoldConfig<Poseidon2BabyBear16BasefoldConfig>;

pub type Poseidon2BabyBearJaggedCpuProverComponents = JaggedBasefoldProverComponents<
    Poseidon2BabyBear16BasefoldCpuProverComponents,
    HadamardJaggedSumcheckProver<CpuJaggedMleGenerator>,
>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JaggedBasefoldConfig<BC>(BC);

impl<BC> JaggedConfig for JaggedBasefoldConfig<BC>
where
    BC: BasefoldConfig,
{
    type F = BC::F;
    type EF = BC::EF;
    type Commitment = BC::Commitment;
    type BatchPcsProof = BasefoldProof<BC>;
    type Challenger = BC::Challenger;
    type BatchPcsVerifier = BasefoldVerifier<BC>;
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JaggedBasefoldProverComponents<B, J>(B, J);

impl<Bpc, JaggedProver> JaggedProverComponents for JaggedBasefoldProverComponents<Bpc, JaggedProver>
where
    Bpc: BasefoldProverComponents,
    Bpc::A: JaggedBackend<Bpc::F, Bpc::EF>,
    Bpc::Challenger: CanObserve<<Bpc::Config as BasefoldConfig>::Commitment>,
    JaggedProver: JaggedSumcheckProver<Bpc::F, Bpc::EF, Bpc::A>,
{
    type F = Bpc::F;
    type EF = Bpc::EF;
    type A = Bpc::A;
    type Challenger = Bpc::Challenger;
    type Commitment = <Bpc::Config as BasefoldConfig>::Commitment;
    type BatchPcsProof = BasefoldProof<Bpc::Config>;
    type Config = JaggedBasefoldConfig<Bpc::Config>;

    type JaggedSumcheckProver = JaggedProver;
    type BatchPcsProver = BasefoldProver<Bpc>;
    type Stacker = FixedRateInterleave<Bpc::F, Bpc::A>;
}

impl<BC> JaggedPcsVerifier<JaggedBasefoldConfig<BC>>
where
    BC: DefaultBasefoldConfig,
{
    pub fn new(log_blowup: usize, log_stacking_height: u32, max_log_row_count: usize) -> Self {
        let basefold_verifer = BasefoldVerifier::<BC>::new(log_blowup);
        let stacked_pcs_verifier = StackedPcsVerifier::new(basefold_verifer, log_stacking_height);
        Self { stacked_pcs_verifier, max_log_row_count }
    }

    pub fn challenger(&self) -> BC::Challenger {
        self.stacked_pcs_verifier.pcs_verifier.challenger()
    }
}

impl<Bpc, JP> JaggedProver<JaggedBasefoldProverComponents<Bpc, JP>>
where
    Bpc: BasefoldProverComponents + DefaultBasefoldProver,
    Bpc::Config: DefaultBasefoldConfig,
    Bpc::A: JaggedBackend<Bpc::F, Bpc::EF>,
    Bpc::Challenger: CanObserve<<Bpc::Config as BasefoldConfig>::Commitment>,
    JP: JaggedSumcheckProver<Bpc::F, Bpc::EF, Bpc::A>,
{
    pub fn from_basefold_components(
        verifier: &JaggedPcsVerifier<JaggedBasefoldConfig<Bpc::Config>>,
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

        Self::new(verifier.max_log_row_count, stacked_pcs_prover, jagged_generator)
    }
}

const DEFAULT_INTERLEAVE_BATCH_SIZE: usize = 128;

impl<Bpc, JP> DfeaultJaggedProver for JaggedBasefoldProverComponents<Bpc, JP>
where
    Bpc: BasefoldProverComponents + DefaultBasefoldProver,
    Bpc::Config: DefaultBasefoldConfig,
    Bpc::A: JaggedBackend<Bpc::F, Bpc::EF>,
    Bpc::Challenger: CanObserve<<Bpc::Config as BasefoldConfig>::Commitment>,
    JP: JaggedSumcheckProver<Bpc::F, Bpc::EF, Bpc::A> + Default,
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

    #[tokio::test]
    async fn test_jagged_basefold() {
        let row_counts_rounds = vec![vec![1 << 10, 0, 1 << 10], vec![1 << 8]];
        let column_counts_rounds = vec![vec![128, 45, 32], vec![512]];

        let log_blowup = 1;
        let log_stacking_height = 10;
        let max_log_row_count = 10;

        type JC = BabyBearPoseidon2;
        type Prover = JaggedProver<Poseidon2BabyBearJaggedCpuProverComponents>;
        type F = <JC as JaggedConfig>::F;
        type EF = <JC as JaggedConfig>::EF;

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
                            let mle = Mle::<F>::rand(&mut rng, *num_cols, num_rows.ilog(2));
                            PaddedMle::padded_with_zeros(Arc::new(mle), max_log_row_count)
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Rounds<_>>();

        let jagged_verifier = JaggedPcsVerifier::<JC>::new(
            log_blowup,
            log_stacking_height,
            max_log_row_count as usize,
        );

        let jagged_prover = Prover::from_verifier(&jagged_verifier);

        let machine_verifier = MachineJaggedPcsVerifier::new(
            &jagged_verifier,
            vec![column_counts[0].clone(), column_counts[1].clone()],
        );

        let eval_point = (0..max_log_row_count).map(|_| rng.gen::<EF>()).collect::<Point<_>>();

        // Begin the commit rounds
        let mut challenger = jagged_verifier.challenger();

        let mut prover_data = Rounds::new();
        let mut commitments = Rounds::new();
        for round in round_mles.iter() {
            let (commit, data) =
                jagged_prover.commit_multilinears(round.clone()).await.ok().unwrap();
            challenger.observe(commit);
            let data_bytes = bincode::serialize(&data).unwrap();
            let data = bincode::deserialize(&data_bytes).unwrap();
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
