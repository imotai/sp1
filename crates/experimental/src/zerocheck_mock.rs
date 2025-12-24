//! This isn't a real zerocheck -- just a proxy used for benchmarking and experimenting
//! with optimization techniques.

use csl_cuda::sys::runtime::Dim3;
use csl_cuda::sys::runtime::KernelPtr;
use csl_cuda::sys::v2_kernels::mle_fix_last_variable_koala_bear_ext_ext_zero_padding;
use csl_utils::{Ext, Felt};

// zerocheck kernels
use csl_cuda::sys::v2_kernels::zerocheck_fix_last_variable_and_sum_as_poly_base_ext_kernel;
use csl_cuda::sys::v2_kernels::zerocheck_fix_last_variable_and_sum_as_poly_ext_ext_kernel;
use csl_cuda::sys::v2_kernels::zerocheck_sum_as_poly_base_ext_kernel;
use csl_cuda::sys::v2_kernels::zerocheck_sum_as_poly_ext_ext_kernel;

use csl_cuda::TaskScope;
use csl_cuda::{args, SmallTensor};
use itertools::Itertools;
use num_bigint::BigUint;
use slop_algebra::{interpolate_univariate_polynomial, AbstractField, Field};
use slop_algebra::{AbstractExtensionField, UnivariatePolynomial};
use slop_alloc::IntoHost;
use slop_challenger::FieldChallenger;
use slop_multilinear::Mle;
use slop_multilinear::MleBaseBackend;

use slop_sumcheck::PartialSumcheckProof;
use slop_tensor::Tensor;

/// Generic helper for sum in last variable operations
async fn sum_in_last_variable<F>(
    poly_base: &Mle<F, TaskScope>,
    poly_ext: &Mle<Ext, TaskScope>,
    claim: Ext,
    kernel: unsafe extern "C" fn() -> KernelPtr,
) -> UnivariatePolynomial<Ext>
where
    F: Field,
{
    let num_variables = poly_base.num_variables();
    let num_polys = poly_base.num_polynomials();
    let scope = poly_base.backend();

    debug_assert!(num_variables >= 1);
    const BLOCK_SIZE: usize = 256;
    const STRIDE: usize = 1;

    let output_height = 1usize << (num_variables - 1);

    let grid_dim: Dim3 = (output_height.div_ceil(BLOCK_SIZE).div_ceil(STRIDE), num_polys, 1).into();

    let mut univariate_evals = Tensor::<Ext, TaskScope>::with_sizes_in(
        [3, grid_dim.y as usize, grid_dim.x as usize],
        scope.clone(),
    );
    let num_tiles = BLOCK_SIZE.checked_div(32).unwrap_or(1);
    let shared_mem = num_tiles * std::mem::size_of::<Ext>();
    let num_variables_minus_one: usize = num_variables as usize - 1;
    unsafe {
        let args = args!(
            univariate_evals.as_mut_ptr(),
            poly_base.guts().as_ptr(),
            poly_ext.guts().as_ptr(),
            num_variables_minus_one,
            num_polys
        );
        univariate_evals.assume_init();
        scope.launch_kernel(kernel(), grid_dim, BLOCK_SIZE, &args, shared_mem).unwrap();
    }

    let univariate_evals = univariate_evals.sum(2).await.sum(1).await;
    let host_evals = unsafe { univariate_evals.into_buffer().copy_into_host_vec() };
    let [component_eval_zero, component_eval_two, component_eval_four] =
        host_evals.try_into().unwrap();
    let eval_zero = component_eval_zero;
    let eval_two = component_eval_two;
    let eval_four = component_eval_four;

    let eval_one = claim - eval_zero;

    interpolate_univariate_polynomial(
        &[
            Ext::from_canonical_u16(0),
            Ext::from_canonical_u16(1),
            Ext::from_canonical_u16(2),
            Ext::from_canonical_u16(4),
        ],
        &[eval_zero, eval_one, eval_two, eval_four],
    )
}

async fn fix_last_variable<F>(
    base: Mle<F, TaskScope>,
    ext: Mle<Ext, TaskScope>,
    alpha: Ext,
    kernel: unsafe extern "C" fn() -> KernelPtr,
) -> (Mle<Ext, TaskScope>, Mle<Ext, TaskScope>)
where
    F: Field,
{
    let base = fix_last_variable_inner(&base, alpha, kernel).await;
    let ext =
        fix_last_variable_inner(&ext, alpha, mle_fix_last_variable_koala_bear_ext_ext_zero_padding)
            .await;

    (base, ext)
}

async fn fix_last_variable_inner<F>(
    mle: &Mle<F, TaskScope>,
    alpha: Ext,
    kernel: unsafe extern "C" fn() -> KernelPtr,
) -> Mle<Ext, TaskScope>
where
    F: Field,
{
    let num_polynomials = 1;
    let input_height = mle.guts().sizes()[1];
    assert!(input_height > 0);
    let output_height = input_height.div_ceil(2);
    let mut output: Tensor<Ext, TaskScope> =
        mle.backend().uninit_mle(num_polynomials, output_height);

    const BLOCK_SIZE: usize = 256;
    const STRIDE: usize = 1;
    let grid_size_x = output_height.div_ceil(BLOCK_SIZE * STRIDE);
    let grid_size_y = num_polynomials;
    let grid_size = (grid_size_x, grid_size_y, 1);

    let args =
        args!(mle.guts().as_ptr(), output.as_mut_ptr(), alpha, input_height, num_polynomials);

    unsafe {
        output.assume_init();
        mle.backend().launch_kernel(kernel(), grid_size, BLOCK_SIZE, &args, 0).unwrap();
    }

    Mle::new(output)
}

// returns (base_output, ext_output, next_univariate)
async fn fix_last_variable_and_sum_as_poly<F>(
    base: Mle<F, TaskScope>,
    ext: Mle<Ext, TaskScope>,
    alpha: Ext,
    claim: Ext,
    kernel: unsafe extern "C" fn() -> KernelPtr,
) -> (Mle<Ext, TaskScope>, Mle<Ext, TaskScope>, UnivariatePolynomial<Ext>)
where
    F: Field,
{
    let input_height = base.guts().sizes()[1];
    let output_height = input_height.div_ceil(2);
    let backend = base.backend();
    let mut base_output: Tensor<Ext, TaskScope> = backend.uninit_mle(1, output_height);
    let mut ext_output: Tensor<Ext, TaskScope> = backend.uninit_mle(1, output_height);

    let grid_size_x = output_height.div_ceil(BLOCK_SIZE * STRIDE);
    let grid_size_y = 1;
    let grid_size = (grid_size_x, grid_size_y, 1);

    const BLOCK_SIZE: usize = 256;
    const STRIDE: usize = 1;

    let num_tiles = BLOCK_SIZE.checked_div(32).unwrap_or(1);
    let shared_mem = num_tiles * std::mem::size_of::<Ext>();

    let mut univariate_evals =
        Tensor::<Ext, TaskScope>::with_sizes_in([3, grid_size.1, grid_size.0], backend.clone());

    let args = args!(
        base.guts().as_ptr(),
        ext.guts().as_ptr(),
        base_output.as_mut_ptr(),
        ext_output.as_mut_ptr(),
        alpha,
        univariate_evals.as_mut_ptr(),
        1usize,
        input_height
    );

    unsafe {
        univariate_evals.assume_init();
        base_output.assume_init();
        ext_output.assume_init();
        backend.launch_kernel(kernel(), grid_size, (BLOCK_SIZE, 1, 1), &args, shared_mem).unwrap();
    }

    // Sum the univariate evals and interpolate into a degree-2 univariate
    let univariate_evals = univariate_evals.sum(2).await.sum(1).await;
    let host_evals = unsafe { univariate_evals.into_buffer().copy_into_host_vec() };

    let [eval_zero, eval_two, eval_four] = host_evals.try_into().unwrap();

    let eval_one = claim - eval_zero;

    let uni_poly = interpolate_univariate_polynomial(
        &[
            Ext::from_canonical_u16(0),
            Ext::from_canonical_u16(1),
            Ext::from_canonical_u16(2),
            Ext::from_canonical_u16(4),
        ],
        &[eval_zero, eval_one, eval_two, eval_four],
    );

    (Mle::new(base_output), Mle::new(ext_output), uni_poly)
}

/// A simpler hadamard sumcheck. Avoids using the complex slop traits, and prioritizes a simple, readable implementation.
pub async fn zerocheck<C, F>(
    base: Mle<F, TaskScope>,
    ext: Mle<Ext, TaskScope>,
    mut challenger: C,
    initial_claim: Ext,
    base_ext_sum_as_poly_kernel: unsafe extern "C" fn() -> KernelPtr,
    base_ext_fix_and_sum_kernel: unsafe extern "C" fn() -> KernelPtr,
) -> (PartialSumcheckProof<Ext>, Vec<Ext>)
where
    C: FieldChallenger<Felt>,
    F: Field,
{
    let mut uni_polys = vec![];
    let initial_univariate =
        sum_in_last_variable::<F>(&base, &ext, initial_claim, base_ext_sum_as_poly_kernel).await;
    let coefficients = initial_univariate
        .coefficients
        .iter()
        .flat_map(|x| x.as_base_slice())
        .copied()
        .collect_vec();
    challenger.observe_slice(&coefficients);

    uni_polys.push(initial_univariate);

    let num_variables = base.num_variables();

    let alpha = challenger.sample_ext_element();

    let mut point = vec![alpha];

    // For the first round, use base-ext kernels.
    let round_claim = uni_polys.last().unwrap().eval_at_point(*point.first().unwrap());
    let (mut base, mut ext, uni_poly) = fix_last_variable_and_sum_as_poly(
        base,
        ext,
        alpha,
        round_claim,
        base_ext_fix_and_sum_kernel,
    )
    .await;

    let coefficients =
        uni_poly.coefficients.iter().flat_map(|x| x.as_base_slice()).copied().collect_vec();

    challenger.observe_slice(&coefficients);

    uni_polys.push(uni_poly);

    let alpha: Ext = challenger.sample_ext_element();
    point.insert(0, alpha);

    // The multi-variate polynomial used at the start of each sumcheck round.
    for _ in 2..num_variables as usize {
        // Get the round claims from the last round's univariate poly messages.
        let round_claim = uni_polys.last().unwrap().eval_at_point(*point.first().unwrap());

        let uni_poly;
        (base, ext, uni_poly) = fix_last_variable_and_sum_as_poly(
            base,
            ext,
            *point.first().unwrap(),
            round_claim,
            zerocheck_fix_last_variable_and_sum_as_poly_ext_ext_kernel,
        )
        .await;

        // base.backend().synchronize().await.unwrap();
        // println!("after fix_last_variable_and_sum_as_poly");
        let coefficients =
            uni_poly.coefficients.iter().flat_map(|x| x.as_base_slice()).copied().collect_vec();

        challenger.observe_slice(&coefficients);

        uni_polys.push(uni_poly);

        let alpha: Ext = challenger.sample_ext_element();
        point.insert(0, alpha);
    }

    // Perform the final fix last variable operation to get the final base and extension evaluations.
    let (base, ext) = fix_last_variable(
        base,
        ext,
        *point.first().unwrap(),
        mle_fix_last_variable_koala_bear_ext_ext_zero_padding,
    )
    .await;

    let proof = PartialSumcheckProof {
        univariate_polys: uni_polys.clone(),
        claimed_sum: initial_claim,
        point_and_eval: (
            point.clone().into(),
            uni_polys.last().unwrap().eval_at_point(*point.first().unwrap()),
        ),
    };
    let base_eval = unsafe { SmallTensor::new(base.guts()) };
    let base_eval = Ext::from_base(base_eval.into_host().await.unwrap().as_slice()[0]);
    let ext_eval = unsafe { SmallTensor::new(ext.guts()) };
    let ext_eval = ext_eval.into_host().await.unwrap().as_slice()[0];
    (proof, vec![base_eval, ext_eval])
}

pub async fn simple_zerocheck<C, F>(
    base: Mle<F, TaskScope>,
    ext: Mle<Ext, TaskScope>,
    challenger: C,
    claim: Ext,
) -> (PartialSumcheckProof<Ext>, Vec<Ext>)
where
    C: FieldChallenger<Felt>,
    F: Field,
{
    let (proof, claims) = if F::order() > BigUint::from(0x7f000001u32) {
        zerocheck(
            base,
            ext,
            challenger,
            claim,
            zerocheck_sum_as_poly_ext_ext_kernel,
            zerocheck_fix_last_variable_and_sum_as_poly_ext_ext_kernel,
        )
    } else {
        zerocheck(
            base,
            ext,
            challenger,
            claim,
            zerocheck_sum_as_poly_base_ext_kernel,
            zerocheck_fix_last_variable_and_sum_as_poly_base_ext_kernel,
        )
    }
    .await;
    (proof, claims)
}

#[cfg(test)]
mod tests {
    use csl_cuda::run_in_place;
    use csl_utils::TestGC;
    use itertools::Itertools;

    use rand::SeedableRng;
    use slop_challenger::{CanSample, IopCtx};
    use slop_multilinear::Mle;
    use slop_sumcheck::partially_verify_sumcheck_proof;

    use super::*;
    const NUM_SIMPLE_ITERATIONS: usize = 1;

    // see cuda/zerocheck/zerocheck.cu
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
        // let claimed_eval = proof.point_and_eval.1;
        // assert_eq!(claimed_eval, zerocheck_eval(exp_eval_ext, exp_eval_base));
        assert_eq!(eval_ext, exp_eval_ext);
        assert_eq!(eval_base, exp_eval_base);

        assert!(partially_verify_sumcheck_proof::<Felt, Ext, _>(
            &proof,
            challenger,
            num_variables,
            3,
        )
        .is_ok());
    }

    /// Compares our simple hadamard sumcheck implementation with the slop implementation, which is more complicated and supports batching.
    #[tokio::test]
    async fn test_zerocheck() {
        // Base / Ext sumcheck benchmarks
        for num_variables in [16, 20, 24, 27] {
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
                let mut simple_durations = Vec::with_capacity(NUM_SIMPLE_ITERATIONS);
                for _ in 0..NUM_SIMPLE_ITERATIONS {
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
                let eval_ext =
                    ext.eval_at(point).await.into_evaluations().into_buffer().into_vec()[0];

                verify_zerocheck_proof(
                    warmup_proof.clone(),
                    warmup_claims.clone(),
                    &mut challenger.clone(),
                    eval_base,
                    eval_ext,
                    num_variables as usize,
                );

                if NUM_SIMPLE_ITERATIONS >= 1 {
                    let avg_simple_duration = simple_durations.iter().sum::<std::time::Duration>()
                        / NUM_SIMPLE_ITERATIONS.try_into().unwrap();

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
            let mut simple_durations = Vec::with_capacity(NUM_SIMPLE_ITERATIONS);
            for _ in 0..NUM_SIMPLE_ITERATIONS {
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
            let eval_ext1 =
                ext1.eval_at(point).await.into_evaluations().into_buffer().into_vec()[0];
            let eval_ext2 =
                ext2.eval_at(point).await.into_evaluations().into_buffer().into_vec()[0];
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
                / NUM_SIMPLE_ITERATIONS.try_into().unwrap();
            println!("Average simple duration for ext-ext: {avg_simple_duration:?}");
        })
        .await;
    }
}
