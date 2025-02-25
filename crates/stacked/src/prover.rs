use derive_where::derive_where;
use futures::prelude::*;
use serde::{Deserialize, Serialize};
use slop_alloc::{HasBackend, ToHost};
use slop_commit::{Message, Rounds};
use slop_multilinear::{Evaluations, Mle, MultilinearPcsProver, Point};
use std::fmt::Debug;
use thiserror::Error;

use crate::{InterleaveMultilinears, StackedPcsProof};

#[derive(Debug, Clone)]
pub struct StackedPcsProver<P, S> {
    pcs_prover: P,
    stacker: S,
    pub log_stacking_height: u32,
}

#[derive(Serialize, Deserialize)]
#[derive_where(Debug, Clone; P::ProverData: Debug + Clone)]
#[serde(bound(
    serialize = "Mle<P::F, P::A>: Serialize, P::ProverData: Serialize",
    deserialize = "Mle<P::F, P::A>: Deserialize<'de>, P::ProverData: Deserialize<'de>"
))]
pub struct StackedPcsProverData<P: MultilinearPcsProver> {
    pcs_batch_data: P::ProverData,
    pub interleaved_mles: Message<Mle<P::F, P::A>>,
}

#[derive(Error, Debug)]
pub enum StackedPcsProverError<E> {
    PcsProverError(E),
}

impl<P, S> StackedPcsProver<P, S>
where
    P: MultilinearPcsProver,
    S: InterleaveMultilinears<P::F, P::A>,
{
    pub const fn new(pcs_prover: P, stacker: S, log_stacking_height: u32) -> Self {
        Self { pcs_prover, stacker, log_stacking_height }
    }

    pub async fn round_batch_evaluations(
        &self,
        stacked_point: &Point<P::EF, P::A>,
        prover_data: &StackedPcsProverData<P>,
    ) -> Evaluations<P::EF, P::A> {
        stream::iter(prover_data.interleaved_mles.iter())
            .then(|mle| mle.eval_at(stacked_point))
            .collect::<Evaluations<_, _>>()
            .await
    }

    pub async fn commit_multilinears(
        &self,
        multilinears: Message<Mle<P::F, P::A>>,
    ) -> Result<(P::Commitment, StackedPcsProverData<P>), StackedPcsProverError<P::ProverError>>
    {
        let interleaved_mles =
            self.stacker.interleave_multilinears(multilinears, self.log_stacking_height).await;
        let (commit, pcs_batch_data) = self
            .pcs_prover
            .commit_multilinears(interleaved_mles.clone())
            .await
            .map_err(StackedPcsProverError::PcsProverError)?;
        let prover_data = StackedPcsProverData { pcs_batch_data, interleaved_mles };
        Ok((commit, prover_data))
    }

    pub async fn prove_trusted_evaluation(
        &self,
        eval_point: Point<P::EF>,
        _evaluation_claim: P::EF,
        prover_data: Rounds<StackedPcsProverData<P>>,
        batch_evaluations: Rounds<Evaluations<P::EF, P::A>>,
        challenger: &mut P::Challenger,
    ) -> Result<StackedPcsProof<P::Proof, P::EF>, StackedPcsProverError<P::ProverError>> {
        let mut host_batch_evaluations = Rounds::new();
        for round_evals in batch_evaluations.iter() {
            let mut host_round_evals = vec![];
            for eval in round_evals.iter() {
                let host_eval = eval.to_host().await.unwrap();
                host_round_evals.push(host_eval);
            }
            let host_round_evals = Evaluations::new(host_round_evals);
            host_batch_evaluations.push(host_round_evals);
        }
        let (pcs_prover_data, mle_rounds): (Rounds<_>, Rounds<_>) = prover_data
            .into_iter()
            .map(|data| (data.pcs_batch_data, data.interleaved_mles))
            .unzip();

        let (_, stack_point) =
            eval_point.split_at(eval_point.dimension() - self.log_stacking_height as usize);

        let pcs_proof = self
            .pcs_prover
            .prove_untrusted_evaluations(
                stack_point,
                mle_rounds,
                batch_evaluations,
                pcs_prover_data,
                challenger,
            )
            .await
            .map_err(StackedPcsProverError::PcsProverError)?;

        Ok(StackedPcsProof { pcs_proof, batch_evaluations: host_batch_evaluations })
    }
}

impl<P: MultilinearPcsProver> HasBackend for StackedPcsProverData<P> {
    type Backend = P::A;

    fn backend(&self) -> &Self::Backend {
        self.interleaved_mles[0].backend()
    }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;
    use slop_algebra::extension::BinomialExtensionField;
    use slop_baby_bear::BabyBear;
    use slop_basefold::{BasefoldVerifier, Poseidon2BabyBear16BasefoldConfig};
    use slop_basefold_prover::{BasefoldProver, Poseidon2BabyBear16BasefoldCpuProverComponents};
    use slop_challenger::CanObserve;

    use crate::{FixedRateInterleave, StackedPcsVerifier};

    use super::*;

    #[tokio::test]
    async fn test_stacked_prover_with_fixed_rate_interleave() {
        let log_stacking_height = 10;
        let batch_size = 10;

        type C = Poseidon2BabyBear16BasefoldConfig;
        type Prover = BasefoldProver<Poseidon2BabyBear16BasefoldCpuProverComponents>;
        type EF = BinomialExtensionField<BabyBear, 4>;

        let round_widths_and_log_heights = [vec![(1 << 10, 10), (1 << 4, 11), (496, 11)]];

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

        let mut challenger = verifier.pcs_verifier.challenger();
        for commitment in commitments.iter() {
            challenger.observe(*commitment);
        }
        verifier
            .verify_trusted_evaluation(&commitments, &point, &proof, eval_claim, &mut challenger)
            .unwrap();
    }
}
