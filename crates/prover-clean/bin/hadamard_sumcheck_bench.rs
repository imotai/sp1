use csl_cuda::run_in_place;
use csl_prover_clean::config::{Ext, Felt};
use csl_prover_clean::hadamard_sumcheck::simple_hadamard_sumcheck;
use itertools::Itertools;

use rand::SeedableRng;
use slop_basefold::BasefoldVerifier;
use slop_basefold::Poseidon2BabyBear16BasefoldConfig;
use slop_challenger::CanSample;
use slop_multilinear::Mle;

const NUM_ITERATIONS: usize = 2;

/// Compares our simple hadamard sumcheck implementation with the slop implementation, which is more complicated and supports batching.
#[tokio::main]
async fn main() {
    // Simple base-ext bench.
    for num_variables in [27] {
        // Run hadamard sumcheck

        run_in_place(|t| async move {
            let mut rng = rand::rngs::StdRng::seed_from_u64(0);

            let verifier = BasefoldVerifier::<Poseidon2BabyBear16BasefoldConfig>::new(1);

            let mut challenger = verifier.challenger();

            let _yuwen_lambda: Ext = challenger.sample();
            let base = Mle::<Felt>::rand(&mut rng, 1, num_variables);
            let ext = Mle::<Ext>::rand(&mut rng, 1, num_variables);

            let claim = ext
                .guts()
                .as_slice()
                .iter()
                .zip_eq(base.guts().as_slice().iter())
                .map(|(e_i, b_i)| *e_i * *b_i)
                .sum::<Ext>();

            let verifier = BasefoldVerifier::<Poseidon2BabyBear16BasefoldConfig>::new(1);
            let mut challenger = verifier.challenger();

            let _yuwen_lambda: Ext = challenger.sample();

            let base_warmup = t.into_device(base.clone()).await.unwrap();
            let ext_warmup = t.into_device(ext.clone()).await.unwrap();
            let challenger_warmup = challenger.clone();
            t.synchronize().await.unwrap();

            // Time the simple implementation
            let (_proof, _claims) =
                simple_hadamard_sumcheck(base_warmup, ext_warmup, challenger_warmup, claim).await;
            t.synchronize().await.unwrap();

            let mut simple_durations = Vec::with_capacity(NUM_ITERATIONS);
            for _ in 0..NUM_ITERATIONS {
                let base_simple_bench = t.into_device(base.clone()).await.unwrap();
                let ext_simple_bench = t.into_device(ext.clone()).await.unwrap();
                let challenger_simple_bench = challenger.clone();
                t.synchronize().await.unwrap();
                let now = std::time::Instant::now();
                let (_proof, _claims) = simple_hadamard_sumcheck(
                    base_simple_bench,
                    ext_simple_bench,
                    challenger_simple_bench,
                    claim,
                )
                .await;
                t.synchronize().await.unwrap();
                simple_durations.push(now.elapsed());
            }

            let avg_simple_duration = simple_durations.iter().sum::<std::time::Duration>()
                / NUM_ITERATIONS.try_into().unwrap();

            let max_simple_duration = simple_durations.iter().max().unwrap();

            let min_simple_duration = simple_durations.iter().min().unwrap();
            println!("\n🚀 Performance Comparison:");
            println!("Simple implementation: {avg_simple_duration:?}");
            println!("Simple implementation max: {max_simple_duration:?}");
            println!("Simple implementation min: {min_simple_duration:?}");
        })
        .await;
    }

    // Gkr-style bench.
    let sizes = (4u32..29).rev().collect_vec();
    run_in_place(|t| async move {
        println!("\n🚀 Gkr-style bench");
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        // Allocate ext1 and ext2 MLE's
        let mut all_ext1 = Vec::with_capacity(sizes.len());
        let mut all_ext2 = Vec::with_capacity(sizes.len());
        let mut all_claims = Vec::with_capacity(sizes.len());
        for size in sizes.iter() {
            let ext1 = Mle::<Ext>::rand(&mut rng, 1, *size);
            let ext2 = Mle::<Ext>::rand(&mut rng, 1, *size);
            let claim = ext1
                .guts()
                .as_slice()
                .iter()
                .zip_eq(ext2.guts().as_slice().iter())
                .map(|(e_i, b_i)| *e_i * *b_i)
                .sum::<Ext>();
            let ext1_device = t.into_device(ext1).await.unwrap();
            let ext2_device = t.into_device(ext2).await.unwrap();
            all_ext1.push(ext1_device);
            all_ext2.push(ext2_device);
            all_claims.push(claim);
        }

        t.synchronize().await.unwrap();
        let verifier = BasefoldVerifier::<Poseidon2BabyBear16BasefoldConfig>::new(1);
        let mut challenger = verifier.challenger();

        // Warm up on the first element of all_ext1 and all_ext2
        let _yuwen_lambda: Ext = challenger.sample();
        let base = all_ext1[0].clone();
        let ext = all_ext2[0].clone();
        let claim = all_claims[0];
        t.synchronize().await.unwrap();
        let now = std::time::Instant::now();
        let (_proof, _claims) =
            simple_hadamard_sumcheck(base, ext, challenger.clone(), claim).await;
        t.synchronize().await.unwrap();
        let warmup_duration = now.elapsed();

        println!("warmup duration: {warmup_duration:?}");

        let now = std::time::Instant::now();
        let mut durations = Vec::with_capacity(sizes.len());
        for ((ext1, ext2), claim) in
            all_ext1.into_iter().zip(all_ext2.into_iter()).zip(all_claims.into_iter())
        {
            let (_proof, _claims) =
                simple_hadamard_sumcheck(ext1, ext2, challenger.clone(), claim).await;
            // t.synchronize().await.unwrap();
            durations.push(now.elapsed());
        }
        t.synchronize().await.unwrap();
        let total_duration = now.elapsed();
        println!("total duration: {total_duration:?}");

        println!("first duration: {:?}", durations[0]);
    })
    .await;
}
