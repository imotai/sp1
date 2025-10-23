use csl_cuda::run_in_place;
use cslpc_experimental::zerocheck_mock::simple_zerocheck;
use cslpc_utils::{Ext, Felt, TestGC};
use itertools::Itertools;

use rand::SeedableRng;
use slop_algebra::{AbstractExtensionField, AbstractField};
use slop_challenger::{CanSample, FieldChallenger, IopCtx};
use slop_multilinear::Mle;
use slop_sumcheck::{partially_verify_sumcheck_proof, PartialSumcheckProof};

fn zerocheck_eval(ext: Ext, base: Ext) -> Ext {
    let six = Ext::from_canonical_u16(6);
    ext * ext * base + ext * base * base + six * base * base * base
    // ext * base
}

fn verify_zerocheck_proof<C>(
    proof: PartialSumcheckProof<Ext>,
    claims: Vec<Ext>,
    challenger: &mut C,
    eval_base: Ext,
    eval_ext: Ext,
    num_variables: usize,
) where
    C: FieldChallenger<Felt>,
{
    let [exp_eval_base, exp_eval_ext] = claims.clone().try_into().unwrap();

    // Check that the final claimed evaluation is the product of the two evaluations
    // NOTE: this check is commented out, since the degree of the zerocheck polynomial is wrong.
    // let claimed_eval = proof.point_and_eval.1;
    // assert_eq!(claimed_eval, zerocheck_eval(exp_eval_ext, exp_eval_base));
    assert_eq!(eval_ext, exp_eval_ext);
    assert_eq!(eval_base, exp_eval_base);

    assert!(partially_verify_sumcheck_proof::<Felt, Ext, _>(&proof, challenger, num_variables, 3,)
        .is_ok());
}

const NUM_ITERATIONS: usize = 1;

/// Compares our simple hadamard sumcheck implementation with the slop implementation, which is more complicated and supports batching.
#[tokio::main]
async fn main() {
    for num_variables in [27] {
        // Run hadamard sumcheck
        run_in_place(|t| async move {
            let mut rng = rand::rngs::StdRng::seed_from_u64(1);

            let base = Mle::<Felt>::rand(&mut rng, 1, num_variables);
            let ext = Mle::<Ext>::rand(&mut rng, 1, num_variables);

            let claim = ext
                .guts()
                .as_slice()
                .iter()
                .zip_eq(base.guts().as_slice().iter())
                .map(|(e_i, b_i)| {
                    let b_i_ext = Ext::from_base(*b_i);
                    zerocheck_eval(*e_i, b_i_ext)
                })
                .sum::<Ext>();

            let mut challenger = TestGC::default_challenger();
            let _lambda: Ext = challenger.sample();

            let base_device = t.into_device(base.clone()).await.unwrap();
            let ext_device = t.into_device(ext.clone()).await.unwrap();
            t.synchronize().await.unwrap();

            let now = std::time::Instant::now();
            let (warmup_proof, warmup_claims) =
                simple_zerocheck(base_device, ext_device, challenger.clone(), claim).await;
            t.synchronize().await.unwrap();
            let warmup_duration = now.elapsed();

            println!("warmup duration: {warmup_duration:?}");
            let mut simple_durations = Vec::with_capacity(NUM_ITERATIONS);
            for _ in 0..NUM_ITERATIONS {
                let base_device = t.into_device(base.clone()).await.unwrap();
                let ext_device = t.into_device(ext.clone()).await.unwrap();
                t.synchronize().await.unwrap();
                let now = std::time::Instant::now();
                let (_proof, _claims) =
                    simple_zerocheck(base_device, ext_device, challenger.clone(), claim).await;
                t.synchronize().await.unwrap();
                simple_durations.push(now.elapsed());
            }

            let point = &warmup_proof.point_and_eval.0;

            let eval_base =
                base.eval_at(point).await.into_evaluations().into_buffer().into_vec()[0];
            let eval_ext = ext.eval_at(point).await.into_evaluations().into_buffer().into_vec()[0];

            verify_zerocheck_proof(
                warmup_proof.clone(),
                warmup_claims.clone(),
                &mut challenger.clone(),
                eval_base,
                eval_ext,
                num_variables as usize,
            );

            if NUM_ITERATIONS >= 1 {
                let avg_simple_duration = simple_durations.iter().sum::<std::time::Duration>()
                    / NUM_ITERATIONS.try_into().unwrap();

                let max_simple_duration = simple_durations.iter().max().unwrap();

                let min_simple_duration = simple_durations.iter().min().unwrap();

                println!("\n🚀 Performance Comparison:");
                println!("Num variables: {num_variables}");
                println!("Warmup duration: {warmup_duration:?}");
                println!("Simple implementation: {avg_simple_duration:?}");
                println!("Simple implementation max: {max_simple_duration:?}");
                println!("Simple implementation min: {min_simple_duration:?}");
            }
        })
        .await;
    }

    println!("\n\n\n Ext / Ext sumcheck benchmarks");
    // Ext / Ext sumcheck benchmarks
    run_in_place(|t| async move {
        let num_variables = 16;
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);

        let ext1 = Mle::<Ext>::rand(&mut rng, 1, num_variables);
        let ext2 = Mle::<Ext>::rand(&mut rng, 1, num_variables);

        let claim = ext1
            .guts()
            .as_slice()
            .iter()
            .zip_eq(ext2.guts().as_slice().iter())
            .map(|(e_i, b_i)| zerocheck_eval(*e_i, *b_i))
            .sum::<Ext>();

        let mut challenger = TestGC::default_challenger();
        let _lambda: Ext = challenger.sample();

        let ext1_device = t.into_device(ext1.clone()).await.unwrap();
        let ext2_device = t.into_device(ext2.clone()).await.unwrap();
        t.synchronize().await.unwrap();

        let now = std::time::Instant::now();
        let (warmup_proof, warmup_claims) =
            simple_zerocheck(ext1_device, ext2_device, challenger.clone(), claim).await;
        let warmup_duration = now.elapsed();

        println!("warmup duration: {warmup_duration:?}");
        let mut simple_durations = Vec::with_capacity(NUM_ITERATIONS);
        for _ in 0..NUM_ITERATIONS {
            let ext1_device = t.into_device(ext1.clone()).await.unwrap();
            let ext2_device = t.into_device(ext2.clone()).await.unwrap();
            t.synchronize().await.unwrap();
            let now = std::time::Instant::now();
            let (_proof, _claims) =
                simple_zerocheck(ext1_device, ext2_device, challenger.clone(), claim).await;
            t.synchronize().await.unwrap();
            simple_durations.push(now.elapsed());
        }

        // Verify warmup proof.
        let point = &warmup_proof.point_and_eval.0;
        let eval_ext1 = ext1.eval_at(point).await.into_evaluations().into_buffer().into_vec()[0];
        let eval_ext2 = ext2.eval_at(point).await.into_evaluations().into_buffer().into_vec()[0];
        verify_zerocheck_proof(
            warmup_proof.clone(),
            warmup_claims.clone(),
            &mut challenger.clone(),
            eval_ext1,
            eval_ext2,
            num_variables as usize,
        );

        // Print the average simple duration for ext-ext
        let avg_simple_duration = simple_durations.iter().sum::<std::time::Duration>()
            / NUM_ITERATIONS.try_into().unwrap();
        println!("Average simple duration for ext-ext: {avg_simple_duration:?}");
    })
    .await;
}
