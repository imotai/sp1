use std::marker::PhantomData;

use csl_cuda::TaskScope;
use csl_dft::SpparkDftBabyBear;
use csl_merkle_tree::Poseidon2BabyBear16CudaProver;
use slop_algebra::extension::BinomialExtensionField;
use slop_baby_bear::BabyBear;
use slop_basefold::{BasefoldVerifier, Poseidon2BabyBear16BasefoldConfig};
use slop_basefold_prover::{BasefoldProver, BasefoldProverComponents, DefaultBasefoldProver};
use slop_merkle_tree::{MerkleTreeTcs, Poseidon2BabyBearConfig};

use crate::{BasefoldCudaConfig, CudaDftEncoder, FriCudaProver, GrindingPowCudaProver};

#[derive(Debug, Clone, Copy, Default, PartialEq, PartialOrd, Eq, Ord)]
pub struct Poseidon2BabyBear16BasefoldCudaProverComponents;

impl BasefoldProverComponents for Poseidon2BabyBear16BasefoldCudaProverComponents {
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
    type A = TaskScope;
    type Tcs = MerkleTreeTcs<Poseidon2BabyBearConfig>;
    type Challenger = <Poseidon2BabyBear16BasefoldConfig as BasefoldCudaConfig>::DeviceChallenger;
    type Config = Poseidon2BabyBear16BasefoldConfig;
    type Encoder = CudaDftEncoder<BabyBear, SpparkDftBabyBear>;
    type FriProver = FriCudaProver<Self::Encoder, Self::TcsProver>;
    type TcsProver = Poseidon2BabyBear16CudaProver;
    type PowProver = GrindingPowCudaProver;
}

impl DefaultBasefoldProver for Poseidon2BabyBear16BasefoldCudaProverComponents {
    fn default_prover(verifier: &BasefoldVerifier<Self::Config>) -> BasefoldProver<Self> {
        let dft = SpparkDftBabyBear::default();
        let encoder = CudaDftEncoder { config: verifier.fri_config, dft };
        let fri_prover = FriCudaProver::<_, _>(PhantomData);
        let tcs_prover = Poseidon2BabyBear16CudaProver::default();
        let pow_prover = GrindingPowCudaProver;
        BasefoldProver { encoder, fri_prover, tcs_prover, pow_prover }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use futures::prelude::*;
    use rand::thread_rng;
    use slop_alloc::{IntoHost, ToHost};
    use slop_baby_bear::BabyBear;
    use slop_basefold::{BasefoldVerifier, Poseidon2BabyBear16BasefoldConfig};
    use slop_challenger::CanObserve;
    use slop_commit::{Message, Rounds};
    use slop_multilinear::{Evaluations, Mle, MultilinearPcsProver, MultilinearPcsVerifier, Point};
    use slop_stacked::{FixedRateInterleave, StackedPcsProver, StackedPcsVerifier};

    use super::*;

    #[tokio::test]
    async fn test_basefold_prover_backend() {
        type C = Poseidon2BabyBear16BasefoldConfig;
        type Prover = BasefoldProver<Poseidon2BabyBear16BasefoldCudaProverComponents>;
        type EF = BinomialExtensionField<BabyBear, 4>;

        let round_widths = [vec![16, 10, 14], vec![20, 78, 34], vec![10, 10]];
        let log_blowup = 1;

        let mut rng = thread_rng();

        for num_variables in [10, 11, 16, 20, 21, 22] {
            let round_mles_host = round_widths
                .iter()
                .map(|widths| {
                    widths
                        .iter()
                        .map(|&w| Mle::<BabyBear>::rand(&mut rng, w, num_variables))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();

            let verifier = BasefoldVerifier::<C>::new(log_blowup);
            let prover = Prover::new(&verifier);

            let point = Point::<EF>::rand(&mut rng, num_variables);

            let mut challenger = verifier.challenger();

            let point_ref = point.clone();
            let (commitments, proof, eval_claims) = csl_cuda::spawn(move |t| async move {
                let mut round_mles = vec![];
                for round in round_mles_host {
                    let mut mles = vec![];
                    for mle in round {
                        let mle = t.into_device(mle).await.unwrap();
                        mles.push(mle.clone());
                    }
                    let mles = Message::<Mle<BabyBear, TaskScope>>::from(mles);
                    round_mles.push(mles);
                }
                let rounds = Rounds { rounds: round_mles };

                let mut commitments = vec![];
                let mut prover_data = Rounds::new();
                let mut eval_claims = Rounds::new();
                let d_point = point_ref.copy_into(&t);
                let mut commit_time = Duration::ZERO;
                for mles in rounds.iter() {
                    t.synchronize().await.unwrap();
                    let time = std::time::Instant::now();
                    let (commitment, data) =
                        prover.commit_multilinears(mles.clone()).await.unwrap();
                    t.synchronize().await.unwrap();
                    commit_time += time.elapsed();
                    challenger.observe(commitment);
                    commitments.push(commitment);
                    prover_data.push(data);
                    let evaluations = stream::iter(mles.iter())
                        .then(|mle| mle.eval_at(&d_point))
                        .collect::<Evaluations<_, _>>()
                        .await;
                    eval_claims.push(evaluations);
                }
                t.synchronize().await.unwrap();
                println!("commit time for {} variables: {:?}", num_variables, commit_time);

                t.synchronize().await.unwrap();
                let time = std::time::Instant::now();
                let proof = prover
                    .prove_trusted_evaluations(
                        point_ref.clone(),
                        rounds,
                        eval_claims.clone(),
                        prover_data,
                        &mut challenger,
                    )
                    .await
                    .unwrap();

                t.synchronize().await.unwrap();
                println!("proof time for {} variables: {:?}", num_variables, time.elapsed());

                let mut eval_claims_host = vec![];
                for eval_claim in eval_claims {
                    let eval_claim = eval_claim.into_host().await.unwrap();
                    eval_claims_host.push(eval_claim.clone());
                }
                (commitments, proof, eval_claims_host)
            })
            .await
            .unwrap();

            let mut challenger = verifier.challenger();
            for commitment in commitments.iter() {
                challenger.observe(*commitment);
            }
            verifier
                .verify_trusted_evaluations(
                    &commitments,
                    point,
                    &eval_claims,
                    &proof,
                    &mut challenger,
                )
                .unwrap();
        }
    }

    #[tokio::test]
    async fn test_stacked_prover_with_fixed_rate_interleave() {
        type C = Poseidon2BabyBear16BasefoldConfig;
        type Prover = BasefoldProver<Poseidon2BabyBear16BasefoldCudaProverComponents>;
        type EF = BinomialExtensionField<BabyBear, 4>;

        for (log_stacking_height, batch_size) in
            [(10usize, 10usize), (16, 128), (20, 128), (21, 128)]
        {
            let mut round_widths_and_log_heights = [vec![
                (200, log_stacking_height),
                (16, log_stacking_height - 1),
                (100, log_stacking_height - 1),
                (1, log_stacking_height + 1),
            ]];

            let total_data_length = round_widths_and_log_heights
                .iter()
                .map(|dims| dims.iter().map(|&(w, log_h)| w << log_h).sum::<usize>())
                .sum::<usize>();
            let total_number_of_variables = total_data_length.next_power_of_two().ilog2();
            println!("total_number_of_variables: {}", total_number_of_variables);

            let last_log_height = log_stacking_height;
            let last_batch_size = ((1 << total_number_of_variables) - total_data_length)
                .checked_div(1 << last_log_height)
                .unwrap();
            println!("last_batch_size: {}", last_batch_size);
            if last_batch_size > 0 {
                round_widths_and_log_heights
                    .last_mut()
                    .unwrap()
                    .push((last_batch_size, last_log_height));
            }

            let log_blowup = 1;

            let mut rng = thread_rng();
            let round_mles_host = round_widths_and_log_heights
                .iter()
                .map(|dims| {
                    dims.iter()
                        .map(|&(w, log_h)| Mle::<BabyBear>::rand(&mut rng, w, log_h as u32))
                        .collect::<Vec<_>>()
                })
                .collect::<Rounds<_>>();

            let pcs_verifier = BasefoldVerifier::<C>::new(log_blowup);
            let pcs_prover = Prover::new(&pcs_verifier);
            let stacker = FixedRateInterleave::new(batch_size);

            let verifier = StackedPcsVerifier::new(pcs_verifier, log_stacking_height as u32);
            let prover = StackedPcsProver::new(pcs_prover, stacker, log_stacking_height as u32);

            let mut challenger = verifier.pcs_verifier.challenger();
            let mut commitments = vec![];
            let mut prover_data = Rounds::new();
            let mut batch_evaluations = Rounds::new();
            let point = Point::<EF>::rand(&mut rng, total_number_of_variables);

            csl_cuda::spawn(move |t| async move {
                let (batch_point, stack_point) =
                    point.split_at(point.dimension() - log_stacking_height);
                let mut round_mles = vec![];
                for round in round_mles_host {
                    let mut mles = vec![];
                    for mle in round {
                        let mle = t.into_device(mle).await.unwrap();
                        mles.push(mle.clone());
                    }
                    let mles = Message::<Mle<BabyBear, TaskScope>>::from(mles);
                    round_mles.push(mles);
                }
                let round_mles = Rounds { rounds: round_mles };

                let stack_point_device = t.to_device(&stack_point).await.unwrap();

                let mut commit_time = Duration::ZERO;
                for mles in round_mles.iter() {
                    t.synchronize().await.unwrap();
                    let time = std::time::Instant::now();
                    let (commitment, data) =
                        prover.commit_multilinears(mles.clone()).await.unwrap();
                    t.synchronize().await.unwrap();
                    commit_time += time.elapsed();
                    challenger.observe(commitment);
                    commitments.push(commitment);
                    let evaluations =
                        prover.round_batch_evaluations(&stack_point_device, &data).await;
                    prover_data.push(data);
                    batch_evaluations.push(evaluations);
                }
                println!(
                    "commit time for {} variables: {:?}",
                    total_number_of_variables, commit_time
                );

                let mut host_batch_evaluations = Rounds::new();
                for round_evals in batch_evaluations.iter() {
                    let mut host_round_evals = vec![];
                    for eval in round_evals.iter() {
                        host_round_evals.push(eval.to_host().await.unwrap());
                    }
                    let host_round_evals = Evaluations::new(host_round_evals);
                    host_batch_evaluations.push(host_round_evals);
                }

                // Interpolate the batch evaluations as a multilinear polynomial.
                let batch_evaluations_mle =
                    host_batch_evaluations.iter().flatten().flatten().cloned().collect::<Mle<_>>();
                // Verify that the climed evaluations matched the interpolated evaluations.
                let eval_claim = batch_evaluations_mle.eval_at(&batch_point).await[0];

                t.synchronize().await.unwrap();
                let time = std::time::Instant::now();
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
                t.synchronize().await.unwrap();
                println!(
                    "proof time for {} variables: {:?}",
                    total_number_of_variables,
                    time.elapsed()
                );

                let mut challenger = verifier.pcs_verifier.challenger();
                for commitment in commitments.iter() {
                    challenger.observe(*commitment);
                }
                verifier
                    .verify_trusted_evaluation(
                        &commitments,
                        &point,
                        &proof,
                        eval_claim,
                        &mut challenger,
                    )
                    .unwrap();
            })
            .await
            .unwrap();
        }
    }
}
