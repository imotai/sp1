use csl_cuda::{
    args,
    sys::prover_clean::{
        fix_last_variable_ext_ext_kernel, padded_hadamard_fix_and_sum,
        prover_clean_jagged_fix_and_sum, prover_clean_jagged_sum_as_poly,
    },
    SmallTensor, TaskScope,
};

use itertools::Itertools;
use slop_algebra::{
    interpolate_univariate_polynomial, AbstractExtensionField, AbstractField, Field,
    UnivariatePolynomial,
};
use slop_alloc::{Backend, HasBackend, IntoHost, ToHost};
use slop_challenger::FieldChallenger;
use slop_multilinear::Mle;
use slop_sumcheck::PartialSumcheckProof;
use slop_tensor::Tensor;

use crate::{
    config::{Ext, Felt},
    DenseData, JaggedMle,
};

use super::hadamard::{fix_last_variable, fix_last_variable_and_sum_as_poly};

pub type JaggedFirstRoundPolyMle<B> = JaggedMle<JaggedFirstRoundPoly<B>, B>;

/// A sumcheck polynomial for the jagged PCS that does not materialize the jagged little polynomial
/// in the first round.
///
/// The jagged little polynomial is an MLE with the same length as the total amount of trace area
/// and with entries in the extension field. As a result, materializing it in full consumes a large
/// amount of memory. We can avoid paying that cost by only materializing the components
/// `eq(z_row, _)` and `eq(z_col, _)` in the first round.
#[derive(Debug)]
pub struct JaggedFirstRoundSumcheckPoly {
    /// Jagged first round polynomial.
    poly: JaggedFirstRoundPolyMle<TaskScope>,
    /// The total number of variables in the sumcheck.
    total_number_of_variables: u32,
}

#[derive(Debug)]
pub struct JaggedFirstRoundPoly<A: Backend = TaskScope> {
    pub base: Tensor<Felt, A>,
    pub eq_z_col: Mle<Ext, A>,
    pub eq_z_row: Mle<Ext, A>,
    pub height: usize,
}

impl<A: Backend> JaggedFirstRoundPoly<A> {
    #[inline]
    pub fn new(
        base: Tensor<Felt, A>,
        eq_z_col: Mle<Ext, A>,
        eq_z_row: Mle<Ext, A>,
        height: usize,
    ) -> Self {
        Self { base, eq_z_col, eq_z_row, height }
    }

    /// # Safety
    ///
    /// See [std::mem::MaybeUninit::assume_init].
    #[inline]
    pub unsafe fn assume_init(&mut self) {
        self.base.assume_init();
        self.eq_z_col.assume_init();
        self.eq_z_row.assume_init();
    }
}

/// We allow dead code here because this is just a wrapper for a c struct. Rust never needs to read these fields.
#[allow(dead_code)]
#[repr(C)]
pub struct JaggedFirstRoundPolyRaw {
    base: *const Felt,
    eq_z_col: *const Ext,
    eq_z_row: *const Ext,
    height: usize,
}

#[allow(dead_code)]
pub struct JaggedFirstRoundPolyMutRaw {
    base: *mut Felt,
    eq_z_col: *mut Ext,
    eq_z_row: *mut Ext,
    height: usize,
}

impl<A: Backend> DenseData<A> for JaggedFirstRoundPoly<A> {
    type DenseDataRaw = JaggedFirstRoundPolyRaw;
    type DenseDataMutRaw = JaggedFirstRoundPolyMutRaw;
    fn as_ptr(&self) -> Self::DenseDataRaw {
        JaggedFirstRoundPolyRaw {
            base: self.base.as_ptr(),
            eq_z_col: self.eq_z_col.guts().as_ptr(),
            eq_z_row: self.eq_z_row.guts().as_ptr(),
            height: self.height,
        }
    }
    fn as_mut_ptr(&mut self) -> Self::DenseDataMutRaw {
        JaggedFirstRoundPolyMutRaw {
            base: self.base.as_mut_ptr(),
            eq_z_col: self.eq_z_col.guts_mut().as_mut_ptr(),
            eq_z_row: self.eq_z_row.guts_mut().as_mut_ptr(),
            height: self.height,
        }
    }
}

async fn sum_as_poly_first_round(
    poly: &JaggedFirstRoundSumcheckPoly,
    claim: Ext,
) -> UnivariatePolynomial<Ext> {
    let circuit = &poly.poly;

    let height = circuit.dense_data.height;

    let backend = circuit.backend();

    const BLOCK_SIZE: usize = 256;
    const STRIDE: usize = 32;

    let grid_dim = height.div_ceil(BLOCK_SIZE).div_ceil(STRIDE);
    let mut output = Tensor::<Ext, TaskScope>::with_sizes_in([2, grid_dim], backend.clone());

    let num_tiles = BLOCK_SIZE.checked_div(STRIDE).unwrap_or(1);
    let shared_mem = num_tiles * std::mem::size_of::<Ext>();

    unsafe {
        output.assume_init();
        let args = args!(output.as_mut_ptr(), circuit.as_raw());
        backend
            .launch_kernel(
                prover_clean_jagged_sum_as_poly(),
                grid_dim,
                BLOCK_SIZE,
                &args,
                shared_mem,
            )
            .unwrap();
    }

    let tensor = output.sum(1).await.to_host().await.unwrap();
    let [eval_zero, eval_half] = tensor.as_slice().try_into().unwrap();

    let eval_one: slop_algebra::extension::BinomialExtensionField<slop_koala_bear::KoalaBear, 4> =
        claim - eval_zero;

    interpolate_univariate_polynomial(
        &[
            Ext::from_canonical_u16(0),
            Ext::from_canonical_u16(1),
            Ext::from_canonical_u16(2).inverse(),
        ],
        &[eval_zero, eval_one, eval_half * Ext::from_canonical_u16(4).inverse()],
    )
}

/// Fix the last variable of the first gkr layer.
async fn fix_and_sum_first_round(
    poly: JaggedFirstRoundSumcheckPoly,
    alpha: Ext,
    claim: Ext,
) -> (UnivariatePolynomial<Ext>, Mle<Ext, TaskScope>, Mle<Ext, TaskScope>) {
    let backend = poly.poly.backend();
    let height = poly.poly.dense_data.height;

    // Create a new layer
    let mut output_p: Tensor<Ext, TaskScope> = Tensor::with_sizes_in([1, height], backend.clone());
    let mut output_q: Tensor<Ext, TaskScope> = Tensor::with_sizes_in([1, height], backend.clone());

    // populate the new layer
    const BLOCK_SIZE: usize = 256;
    const STRIDE: usize = 32;
    let grid_size_x = height.div_ceil(BLOCK_SIZE * STRIDE * 2); // * 2 because we are doing 2 fixes per thread.
    let mut evaluations =
        Tensor::<Ext, TaskScope>::with_sizes_in([2, grid_size_x], backend.clone());
    let grid_size = (grid_size_x, 1, 1);
    let block_dim = BLOCK_SIZE;

    let num_tiles = BLOCK_SIZE.checked_div(STRIDE).unwrap_or(1);
    let shared_mem = num_tiles * std::mem::size_of::<Ext>();

    unsafe {
        output_p.assume_init();
        output_q.assume_init();
        evaluations.assume_init();
        let args = args!(
            evaluations.as_mut_ptr(),
            poly.poly.as_raw(),
            output_p.as_mut_ptr(),
            output_q.as_mut_ptr(),
            alpha
        );
        backend
            .launch_kernel(
                prover_clean_jagged_fix_and_sum(),
                grid_size,
                block_dim,
                &args,
                shared_mem,
            )
            .unwrap();
    }
    // backend.synchronize().await.unwrap();

    // Sum the evaluations across all dimensions.
    let evaluations = evaluations.sum(1).await;
    let evaluations = evaluations.to_host().await.unwrap();
    let [eval_zero, eval_half] = evaluations.as_slice().try_into().unwrap();

    let eval_one = claim - eval_zero;

    let uni_poly = interpolate_univariate_polynomial(
        &[
            Ext::from_canonical_u16(0),
            Ext::from_canonical_u16(1),
            Ext::from_canonical_u16(2).inverse(),
        ],
        &[eval_zero, eval_one, eval_half * Ext::from_canonical_u16(4).inverse()],
    );

    (uni_poly, Mle::new(output_p), Mle::new(output_q))
}

/// Process a univariate polynomial by observing it with the challenger and sampling the next evaluation point
#[inline]
fn process_univariate_polynomial<C>(
    uni_poly: UnivariatePolynomial<Ext>,
    challenger: &mut C,
    univariate_poly_msgs: &mut Vec<UnivariatePolynomial<Ext>>,
    point: &mut Vec<Ext>,
) -> Ext
where
    C: FieldChallenger<Felt>,
{
    let coefficients =
        uni_poly.coefficients.iter().flat_map(|x| x.as_base_slice()).copied().collect_vec();
    challenger.observe_slice(&coefficients);
    univariate_poly_msgs.push(uni_poly);
    let alpha: Ext = challenger.sample_ext_element();
    point.insert(0, alpha);
    alpha
}

pub async fn jagged_sumcheck<C>(
    poly: JaggedFirstRoundSumcheckPoly,
    challenger: &mut C,
    claim: Ext,
) -> (PartialSumcheckProof<Ext>, Vec<Ext>)
where
    C: FieldChallenger<Felt>,
{
    let num_variables = poly.total_number_of_variables;

    // The first round will process the first t variables, so we need to ensure that there are at least t variables.
    assert!(num_variables >= 1_u32);

    // The point at which the reduced sumcheck proof should be evaluated.
    let mut point = vec![];

    // The univariate poly messages.  This will be a rlc of the polys' univariate polys.
    let mut univariate_poly_msgs: Vec<UnivariatePolynomial<Ext>> = vec![];

    let uni_poly = sum_as_poly_first_round(&poly, claim).await;

    let alpha =
        process_univariate_polynomial(uni_poly, challenger, &mut univariate_poly_msgs, &mut point);
    let round_claim = univariate_poly_msgs.last().unwrap().eval_at_point(alpha);

    let (mut uni_poly, mut p, mut q) = fix_and_sum_first_round(poly, alpha, round_claim).await;

    let mut alpha =
        process_univariate_polynomial(uni_poly, challenger, &mut univariate_poly_msgs, &mut point);

    for _ in 2..num_variables as usize {
        // Get the round claims from the last round's univariate poly messages.
        let round_claim = univariate_poly_msgs.last().unwrap().eval_at_point(alpha);

        (p, q, uni_poly) = fix_last_variable_and_sum_as_poly(
            p,
            q,
            alpha,
            round_claim,
            padded_hadamard_fix_and_sum,
        )
        .await;

        alpha = process_univariate_polynomial(
            uni_poly,
            challenger,
            &mut univariate_poly_msgs,
            &mut point,
        );
    }

    let (p, q) = fix_last_variable(p, q, alpha, fix_last_variable_ext_ext_kernel).await;

    let proof = PartialSumcheckProof {
        univariate_polys: univariate_poly_msgs.clone(),
        claimed_sum: claim,
        point_and_eval: (
            point.clone().into(),
            univariate_poly_msgs.last().unwrap().eval_at_point(alpha),
        ),
    };
    let p_eval = unsafe { SmallTensor::new(p.guts()) };
    let p_eval = Ext::from_base(p_eval.into_host().await.unwrap().as_slice()[0]);
    let q_eval = unsafe { SmallTensor::new(q.guts()) };
    let q_eval = q_eval.into_host().await.unwrap().as_slice()[0];

    (proof, vec![p_eval, q_eval])
}

#[cfg(test)]
mod tests {

    use csl_cuda::{IntoDevice, TaskScope};
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    use slop_algebra::AbstractExtensionField;
    use slop_alloc::{Buffer, ToHost};
    use slop_challenger::IopCtx;
    use slop_multilinear::{Mle, MultilinearPcsChallenger};
    use slop_sumcheck::partially_verify_sumcheck_proof;

    use rand::{rngs::StdRng, SeedableRng as _};
    use slop_tensor::Tensor;

    use crate::{
        config::{Ext, Felt, GC},
        jagged_sumcheck::*,
    };

    #[tokio::test]
    async fn test_jagged_sumcheck_poly() {
        let mut rng = StdRng::seed_from_u64(2);

        // Source from an RSP block. Includes preprocessed row counts.
        let row_counts_1 = vec![
            65536_usize,
            472032,
            131072,
            4194304,
            115200,
            80736,
            1814464,
            11616,
            643776,
            997920,
            65536,
            0,
            408608,
            0,
            0,
            48128,
            79264,
            1041248,
            406880,
            0,
            2624,
            832,
            0,
            128,
            203072,
            2880,
            16,
            0,
            472032,
            131072,
            18688,
            28000,
            32,
            699040,
            376000,
            832,
            32,
            23200,
            832,
            2496,
            2496,
            56736,
            4194304,
            415328,
        ];

        let row_counts_2 = vec![
            65536_usize,
            472032,
            131072,
            4194304,
            115200,
            295040,
            1056352,
            11072,
            659168,
            1083552,
            65536,
            32,
            303168,
            0,
            0,
            21920,
            115712,
            977152,
            635040,
            256,
            1792,
            768,
            24896,
            128,
            150752,
            18944,
            16,
            0,
            472032,
            131072,
            442912,
            233728,
            32,
            348832,
            550656,
            736,
            2496,
            43968,
            960,
            1664,
            1696,
            59200,
            4194304,
            1277984,
        ];

        let column_counts = vec![
            7_usize, 16, 2, 0, 1, 34, 31, 37, 52, 46, 6, 247, 282, 61, 36, 32, 39, 49, 41, 46, 46,
            50, 45, 15, 20, 83, 60, 10, 1, 1, 66, 70, 14, 52, 41, 47, 46, 34, 33, 10, 68, 32, 0, 1,
        ];

        let test_cases = [(row_counts_1, column_counts.clone()), (row_counts_2, column_counts)];

        csl_cuda::run_in_place(|t| async move {
            // The data is organized as N consecutive tables of row count row_count[i] and column count column_count[i].
            for (i, (row_counts, column_counts)) in test_cases.iter().enumerate() {
                let mut challenger = GC::default_challenger();

                let log_max_row_count =
                    row_counts.iter().max().unwrap().next_power_of_two().ilog2();
                let num_col_variables =
                    column_counts.iter().sum::<usize>().next_power_of_two().ilog2();

                // todo: make these from rng
                let z_row = challenger.sample_point::<Ext>(log_max_row_count);
                let z_col = challenger.sample_point::<Ext>(num_col_variables);

                println!("log max row count: {}", log_max_row_count);
                println!("num col variables: {}", num_col_variables);

                let z_row = t.into_device(z_row).await.unwrap();
                let z_col = t.into_device(z_col).await.unwrap();

                let eq_z_row = Mle::<_, TaskScope>::partial_lagrange(&z_row).await;
                let eq_z_col = Mle::<_, TaskScope>::partial_lagrange(&z_col).await;

                let mut dense_size = 0;
                let mut col_index_vec = Vec::new();
                let mut start_indices_vec = Vec::with_capacity(row_counts.len() + 1);

                // This has the same length as the dense data. row[i] is the index of the row that dense_data[i] belongs to.
                let mut row = Vec::new();

                let mut columns_so_far = 0;
                for (row_count, column_count) in row_counts.iter().zip(column_counts.iter()) {
                    for i in 0..*column_count {
                        start_indices_vec.push(((dense_size + i * row_count) >> 1) as u32);
                        col_index_vec
                            .extend_from_slice(&vec![(columns_so_far + i) as u32; *row_count >> 1]);
                    }
                    dense_size += row_count * column_count;

                    // Since the data is organized in column major order, this section of the row vector is
                    // 0..row_count repeated column_count times.
                    let row_counts = (0..*row_count).collect::<Vec<_>>();
                    for _ in 0..*column_count {
                        row.extend_from_slice(&row_counts);
                    }
                    columns_so_far += column_count;
                }
                start_indices_vec.push((dense_size >> 1) as u32);

                let dense_number_of_variables = dense_size.next_power_of_two().ilog2();
                println!("total number of variables: {}", dense_number_of_variables);

                let base_host = Tensor::<Felt>::rand(&mut rng, [dense_size]);

                let eq_z_row_vec = eq_z_row.guts().as_buffer().to_host().await.unwrap().to_vec();
                let eq_z_col_vec = eq_z_col.guts().as_buffer().to_host().await.unwrap().to_vec();
                let base_host_vec = base_host.as_buffer().to_host().await.unwrap().to_vec();

                // \sum_i{base[i] eq_row(z_{row},row[i]) eq_col(z_{col},col[i])}
                let claim = (0..dense_size)
                    .into_par_iter()
                    .map(|i| {
                        let base_val = Ext::from_base(base_host_vec[i]);
                        let row_val = eq_z_row_vec[row[i]];
                        let col_val = eq_z_col_vec[col_index_vec[i >> 1] as usize];
                        base_val * (row_val * col_val)
                    })
                    .sum::<Ext>();

                let base_device = t.into_device(base_host).await.unwrap();

                let col_index = col_index_vec.clone().into_iter().collect::<Buffer<_>>();
                let col_index_device = t.into_device(col_index).await.unwrap();
                let start_indices = start_indices_vec.into_iter().collect::<Buffer<_>>();
                let start_indices_device = t.into_device(start_indices).await.unwrap();

                let jagged_first_round_poly: JaggedFirstRoundPoly =
                    JaggedFirstRoundPoly::new(base_device, eq_z_col, eq_z_row, dense_size >> 1);

                // There's no fix_last_variable for jagged sumcheck, so we can put a dummy column_heights in this test.
                let jagged_mle = JaggedFirstRoundPolyMle::new(
                    jagged_first_round_poly,
                    col_index_device,
                    start_indices_device,
                    Vec::new(),
                );

                let jagged_first_round_sumcheck_poly = JaggedFirstRoundSumcheckPoly {
                    poly: jagged_mle,
                    total_number_of_variables: dense_number_of_variables,
                };

                let mut proof_challenger = challenger.clone();
                t.synchronize().await.unwrap();

                let now = std::time::Instant::now();
                let (proof, evaluations) =
                    jagged_sumcheck(jagged_first_round_sumcheck_poly, &mut proof_challenger, claim)
                        .await;
                t.synchronize().await.unwrap();
                println!("jagged sumcheck time: {:?}", now.elapsed());

                let mut verification_challenger = challenger.clone();

                partially_verify_sumcheck_proof(
                    &proof,
                    &mut verification_challenger,
                    dense_number_of_variables as usize,
                    2,
                )
                .unwrap();

                let (point, expected_final_eval) = proof.point_and_eval;

                // Assert that the point has the expected dimension.
                assert_eq!(point.dimension() as u32, dense_number_of_variables);

                // Calculate the expected evaluations at the point.
                let [p_eval, q_eval] = evaluations.try_into().unwrap();
                let final_eval = p_eval * q_eval;

                // q_eval should be equal to Mle::from(eq_row(z_{row},row[i]) * eq_col(z_{col},col[i])).eval_at(&point)
                // In other words, for 0 < i < dense_size, view eq_row(z_{row},row[i]) * eq_col(z_{col},col[i]) as evaluations of an MLE on the boolean hypercube.
                // Then evaluate this MLE at point.
                let jagged_poly = (0..dense_size)
                    .into_par_iter()
                    .map(|i| {
                        let row_val = eq_z_row_vec[row[i]];
                        let col_val = eq_z_col_vec[col_index_vec[i >> 1] as usize];
                        row_val * col_val
                    })
                    .collect::<Vec<_>>();
                let jagged_poly_buf = jagged_poly.into_iter().collect::<Buffer<_>>();

                let jagged_poly_mle =
                    Mle::from_buffer(jagged_poly_buf).into_device_in(&t).await.unwrap();

                let point_device = point.into_device_in(&t).await.unwrap();
                let jagged_eval = jagged_poly_mle.eval_at(&point_device).await;
                let jagged_eval =
                    jagged_eval.evaluations().as_buffer().to_host().await.unwrap().as_slice()[0];
                assert_eq!(jagged_eval, q_eval, "jagged eval mismatch");

                // p_eval should be equal to Mle::from(base_vec).eval_at_point(point)
                let base_buf = base_host_vec.into_iter().collect::<Buffer<_>>();
                let base_mle = Mle::from_buffer(base_buf).into_device_in(&t).await.unwrap();
                let base_eval = base_mle.eval_at(&point_device).await;
                let base_eval =
                    base_eval.evaluations().as_buffer().to_host().await.unwrap().as_slice()[0];
                assert_eq!(base_eval, p_eval, "base eval mismatch");

                // Assert that the final eval is correct.
                assert_eq!(final_eval, expected_final_eval, "final eval mismatch");

                println!("*********** test case {} passed ***********\n", i);
            }
        })
        .await;
    }
}
