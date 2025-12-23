use std::{ffi::c_void, time::Duration};

use csl_cuda::{CudaError, TaskScope};
use csl_experimental::look_ahead::{Hadamard, RoundParams};
use csl_utils::{Ext, Felt, TestGC};
use slop_algebra::{
    interpolate_univariate_polynomial, AbstractExtensionField, AbstractField, Field,
    UnivariatePolynomial,
};
use slop_alloc::{Buffer, ToHost};
use slop_challenger::{CanObserve, CanSample, FieldChallenger, IopCtx};
use slop_sumcheck::{partially_verify_sumcheck_proof, PartialSumcheckProof};
use slop_tensor::Tensor;

#[inline]
fn interpolate(result: Buffer<Ext>, claim: Ext) -> UnivariatePolynomial<Ext> {
    let [component_eval_zero, component_eval_half] = result.into_vec().try_into().unwrap();
    let eval_zero = component_eval_zero;
    let eval_half = component_eval_half;
    let eval_one = claim - eval_zero;
    interpolate_univariate_polynomial(
        &[
            Ext::from_canonical_u16(0),
            Ext::from_canonical_u16(1),
            Ext::from_canonical_u16(2).inverse(),
        ],
        &[eval_zero, eval_one, eval_half * Felt::from_canonical_u16(4).inverse()],
    )
}

#[inline]
fn populate_restrict_eq(alpha: Ext, t: &TaskScope) {
    let eq_res = [Ext::one() - alpha, alpha];
    unsafe {
        let maybe_err = csl_cuda::sys::v2_kernels::populate_restrict_eq_host(
            eq_res.as_ptr() as *const c_void,
            eq_res.len(),
            t.handle(),
        );
        CudaError::result_from_ffi(maybe_err).unwrap();
    }
}

fn verify_sumcheck_proof<C>(
    proof: PartialSumcheckProof<Ext>,
    claims: Vec<Ext>,
    challenger: &mut C,
    num_variables: usize,
) where
    C: FieldChallenger<Felt>,
{
    let [exp_eval_base, exp_eval_ext] = claims.clone().try_into().unwrap();

    // Check that the final claimed evaluation is the product of the two evaluations
    let claimed_eval = proof.point_and_eval.1;
    assert_eq!(claimed_eval, exp_eval_ext * exp_eval_base);

    partially_verify_sumcheck_proof::<Felt, Ext, _>(&proof, challenger, num_variables, 2).unwrap();
}

#[tokio::main]
async fn main() {
    const FIX_GROUP: usize = 2;
    const FIX_TILE: usize = 64;
    const SUM_GROUP: usize = 2;
    const NUM_POINTS: usize = 2;

    assert_eq!(FIX_GROUP, 2);

    const NUM_ITERATIONS: usize = 4;
    const WARMUP_ITERATIONS: usize = 2;

    for num_variables in [10, 16, 20, 24, 26, 27] {
        csl_cuda::run_in_place(|t| async move {
            let mut rng = rand::thread_rng();

            let mut times = Vec::with_capacity(NUM_ITERATIONS);
            for i in 0..NUM_ITERATIONS + WARMUP_ITERATIONS {
                let p_h = Tensor::<Ext>::rand(&mut rng, [1 << num_variables]);
                let q_h = Tensor::<Ext>::rand(&mut rng, [1 << num_variables]);

                let initial_claim = p_h
                    .as_slice()
                    .iter()
                    .zip(q_h.as_slice().iter())
                    .map(|(p_i, q_i)| *p_i * *q_i)
                    .sum::<Ext>();

                // Copy to device
                let p = t.into_device(p_h).await.unwrap();
                let q = t.into_device(q_h).await.unwrap();

                let hadamard = Hadamard::new(p, q);

                let mut challenger = TestGC::default_challenger();

                t.synchronize().await.unwrap();
                let time = tokio::time::Instant::now();

                let mut uni_polys = Vec::with_capacity(num_variables);
                let mut point = Vec::with_capacity(num_variables);

                // First round
                let (result, _) = hadamard
                    .round(&RoundParams {
                        fix_group: 1,
                        fix_tile: 64,
                        sum_group: SUM_GROUP,
                        num_points: NUM_POINTS,
                        store_restricted: false,
                    })
                    .await;
                let mut restricted = hadamard;
                let uni_poly = interpolate(result, initial_claim);
                challenger.observe_slice(
                    &uni_poly
                        .coefficients
                        .iter()
                        .flat_map(|x| x.as_base_slice())
                        .copied::<Felt>()
                        .collect::<Vec<_>>(),
                );
                uni_polys.push(uni_poly);
                let alpha: Ext = challenger.sample();
                populate_restrict_eq(alpha, &t);
                point.insert(0, alpha);

                for _ in 1..num_variables {
                    let store_res = true;
                    let (result, next_restricted) = restricted
                        .round(&RoundParams {
                            fix_group: FIX_GROUP,
                            fix_tile: FIX_TILE,
                            sum_group: SUM_GROUP,
                            num_points: NUM_POINTS,
                            store_restricted: store_res,
                        })
                        .await;
                    let next_restricted = next_restricted.unwrap();
                    restricted = next_restricted;
                    // Get the round claims from the last round's univariate poly messages.
                    let round_claim =
                        uni_polys.last().unwrap().eval_at_point(*point.first().unwrap());
                    let uni_poly = interpolate(result, round_claim);
                    challenger.observe_slice(
                        &uni_poly
                            .coefficients
                            .iter()
                            .flat_map(|x| x.as_base_slice())
                            .copied::<Felt>()
                            .collect::<Vec<_>>(),
                    );
                    uni_polys.push(uni_poly);
                    let alpha: Ext = challenger.sample_ext_element();
                    populate_restrict_eq(alpha, &t);
                    point.insert(0, alpha);
                }

                // Perform the final fix last variable operation to get the final base and extension evaluations.
                let Hadamard { p, q } = restricted;
                let mut p_buffer = p.into_buffer();
                let mut q_buffer = q.into_buffer();
                unsafe {
                    p_buffer.set_len(2);
                    q_buffer.set_len(2);
                }

                let p_guts = p_buffer.to_host().await.unwrap();
                let q_guts = q_buffer.to_host().await.unwrap();

                let alpha = *point.first().unwrap();
                let p_eval = *p_guts[0] + alpha * (*p_guts[1] - *p_guts[0]);
                let q_eval = *q_guts[0] + alpha * (*q_guts[1] - *q_guts[0]);

                let proof = PartialSumcheckProof {
                    univariate_polys: uni_polys.clone(),
                    claimed_sum: initial_claim,
                    point_and_eval: (
                        point.clone().into(),
                        uni_polys.last().unwrap().eval_at_point(*point.first().unwrap()),
                    ),
                };

                let (proof, final_evals) = (proof, vec![p_eval, q_eval]);

                t.synchronize().await.unwrap();
                let elapsed = time.elapsed();

                if i >= WARMUP_ITERATIONS {
                    times.push(elapsed);
                }

                let mut challenger = TestGC::default_challenger();
                verify_sumcheck_proof(proof, final_evals, &mut challenger, num_variables);
            }

            let avg_time = times.iter().sum::<Duration>() / NUM_ITERATIONS as u32;
            let std_dev = times
                .iter()
                .map(|t| (t.as_secs_f64() - avg_time.as_secs_f64()).powi(2))
                .sum::<f64>()
                / NUM_ITERATIONS as f64;
            let std_dev = Duration::from_secs_f64(std_dev.sqrt());
            println!(
                "Num variables: {num_variables}, average time: {avg_time:?}, std dev: {std_dev:?}",
            );
        })
        .await
        .await
        .unwrap();
    }
}
