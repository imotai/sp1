use csl_basefold::Poseidon2BabyBear16BasefoldCudaProverComponents;
use slop_jagged::JaggedBasefoldProverComponents;

use crate::VirtualJaggedSumcheckProver;

pub type Poseidon2BabyBearJaggedCudaProverComponents = JaggedBasefoldProverComponents<
    Poseidon2BabyBear16BasefoldCudaProverComponents,
    VirtualJaggedSumcheckProver,
>;

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use futures::prelude::*;
    use rand::{thread_rng, Rng};
    use serial_test::serial;
    use slop_alloc::IntoHost;
    use slop_challenger::CanObserve;
    use slop_commit::Rounds;
    use slop_multilinear::{Evaluations, Mle, PaddedMle, Point};

    use slop_jagged::{
        BabyBearPoseidon2, JaggedConfig, JaggedPcsVerifier, JaggedProver, MachineJaggedPcsVerifier,
    };

    use crate::Poseidon2BabyBearJaggedCudaProverComponents;

    #[tokio::test]
    #[serial]
    async fn test_jagged_basefold() {
        let log_blowup = 1;

        type JC = BabyBearPoseidon2;
        type Prover = JaggedProver<Poseidon2BabyBearJaggedCudaProverComponents>;
        type F = <JC as JaggedConfig>::F;
        type EF = <JC as JaggedConfig>::EF;

        let mut rng = thread_rng();

        for (log_stacking_height, max_log_row_count) in
            [(10, 10), (11, 11), (16, 16), (20, 20), (21, 21)]
        {
            let row_counts_rounds = vec![
                vec![1 << (max_log_row_count - 2), 1 << max_log_row_count],
                vec![
                    1 << (max_log_row_count - 6),
                    1 << (max_log_row_count),
                    1 << (max_log_row_count - 1),
                ],
            ];
            let column_counts_rounds = vec![vec![128, 32], vec![512, 128, 100]];

            let row_counts = row_counts_rounds.into_iter().collect::<Rounds<Vec<usize>>>();
            let column_counts = column_counts_rounds.into_iter().collect::<Rounds<Vec<usize>>>();

            let total_number_of_variables = row_counts
                .iter()
                .zip(column_counts.iter())
                .map(|(row_counts, column_counts)| {
                    row_counts
                        .iter()
                        .zip(column_counts.iter())
                        .map(|(row_count, column_count)| row_count * column_count)
                        .sum::<usize>()
                })
                .sum::<usize>()
                .next_power_of_two()
                .ilog2();

            let lde_area = ((1 << (total_number_of_variables + log_blowup))
                * std::mem::size_of::<F>()) as f64
                / 1e9;
            println!(
                "lde_area for total_number_of_variables: {:?}, lde_area: {:2}",
                total_number_of_variables, lde_area
            );

            assert!(row_counts.len() == column_counts.len());
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
                log_blowup as usize,
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

            let eval_point_host = eval_point.clone();
            let (commitments, evaluation_claims, proof) = csl_cuda::task()
                .await
                .unwrap()
                .run(|t| async move {
                    let mut prover_data = Rounds::new();
                    let mut commitments = Rounds::new();

                    let rounds = stream::iter(round_mles.into_iter())
                        .then(|round| {
                            stream::iter(round.into_iter())
                                .then(|mle| async { t.into_device(mle).await.unwrap() })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Rounds<_>>()
                        .await;

                    let mut commit_time = Duration::ZERO;
                    for round in rounds.iter() {
                        t.synchronize().await.unwrap();
                        let start = tokio::time::Instant::now();
                        let (commit, data) =
                            jagged_prover.commit_multilinears(round.clone()).await.ok().unwrap();
                        commit_time += start.elapsed();
                        challenger.observe(commit);
                        prover_data.push(data);
                        commitments.push(commit);
                    }
                    println!(
                        "commit_time for total_number_of_variables: {:?}, commit_time: {:?}",
                        total_number_of_variables, commit_time
                    );

                    let evaluation_claims = stream::iter(rounds.iter())
                        .then(|round| {
                            stream::iter(round.iter())
                                .then(|mle| mle.eval_at(&eval_point))
                                .collect::<Evaluations<_, _>>()
                        })
                        .collect::<Rounds<_>>()
                        .await;

                    t.synchronize().await.unwrap();
                    let start = tokio::time::Instant::now();
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
                    let proof_time = start.elapsed();
                    println!(
                        "proof_time for total_number_of_variables: {:?}, proof_time: {:?}",
                        total_number_of_variables, proof_time
                    );
                    let evaluation_claims = stream::iter(evaluation_claims.into_iter())
                        .then(|e| async move { e.into_host().await.unwrap() })
                        .collect::<Rounds<_>>()
                        .await;
                    (commitments, evaluation_claims, proof)
                })
                .await
                .await
                .unwrap();

            let mut challenger = jagged_verifier.challenger();
            for commitment in commitments.iter() {
                challenger.observe(*commitment);
            }
            machine_verifier
                .verify_trusted_evaluations(
                    &commitments,
                    eval_point_host,
                    &evaluation_claims,
                    &proof,
                    &mut challenger,
                )
                .unwrap();
        }
    }
}
