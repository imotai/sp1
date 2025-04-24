use std::sync::Arc;

use csl_cuda::{
    args,
    sys::{
        jagged::{
            jagged_baby_bear_base_ext_sum_as_poly,
            jagged_baby_bear_extension_virtual_fix_last_variable,
        },
        runtime::{Dim3, KernelPtr},
    },
    TaskScope,
};
use futures::future::join_all;
use itertools::Itertools;
use slop_algebra::{
    extension::BinomialExtensionField, interpolate_univariate_polynomial, ExtensionField, Field,
};
use slop_alloc::{Buffer, HasBackend, IntoHost};
use slop_baby_bear::BabyBear;
use slop_commit::{Message, Rounds};
use slop_jagged::{HadamardProduct, JaggedBackend, JaggedSumcheckProver, LongMle};
use slop_multilinear::{Mle, MleFixLastVariableBackend, PartialLagrangeBackend};
use slop_sumcheck::{SumcheckPolyBase, SumcheckPolyFirstRound};
use slop_tensor::{ReduceSumBackend, Tensor};

#[derive(Debug, Clone, Copy, Default, PartialEq, PartialOrd, Eq, Ord)]
pub struct VirtualJaggedSumcheckProver;

/// A sumcheck polynomial for the jagged PCS that does not materialize the jagged little polynomial
/// in the first round.
///
/// The jagged little polynomial is an MLE with the same length as the total amount of trace area
/// and with entries in the extension field. As a result, materializing it in full consumes a large
/// amount of memory. We can avoid paying that cost by only materializing the components
/// `eq(z_row, _)` and `eq(z_col, _)` in the first round.
#[derive(Debug)]
pub struct VirtualJaggedFirstRoundSumcheckPoly<F, EF> {
    /// The MLEs coming from the restacked traces.
    base: Rounds<Message<Mle<F, TaskScope>>>,
    /// Device pointer to a slice of pointers for the components.
    base_components: Rounds<Buffer<CanSendRaw<F>, TaskScope>>,
    /// The row counts for each table.
    row_data: Rounds<Arc<Vec<usize>>>,
    /// The column counts for each table.
    column_data: Rounds<Arc<Vec<usize>>>,
    /// The MLE for `eq(z_col, _)`.
    eq_z_col: Mle<EF, TaskScope>,
    /// The MLE for `eq(z_row, _)`.
    eq_z_row: Mle<EF, TaskScope>,
    /// The total number of variables in the sumcheck.
    total_number_of_variables: u32,
}

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
struct CanSendRaw<T>(pub *const T);

unsafe impl<T> Send for CanSendRaw<T> {}
unsafe impl<T> Sync for CanSendRaw<T> {}

#[derive(Debug, Clone, Copy)]
#[repr(transparent)]
struct CanSendRawMut<T>(pub *mut T);

unsafe impl<T> Send for CanSendRawMut<T> {}
unsafe impl<T> Sync for CanSendRawMut<T> {}

/// A pointer to the sum as poly kernel.
///
/// # Safety
///
/// The signature of the kernel must match the args call below.
pub unsafe trait VirtualJaggedSumAsPolyKernel<F, EF> {
    fn sum_as_poly_kernel() -> KernelPtr;
}

/// A pointer to the fix last variable kernel.
///
/// # Safety
///
/// The signature of the kernel must match the args call below.
pub unsafe trait VirtualJaggedFixLastVariableKernel<EF> {
    fn virtual_jagged_fix_last_variable_kernel() -> KernelPtr;
}

impl<F, EF> SumcheckPolyBase for VirtualJaggedFirstRoundSumcheckPoly<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    fn num_variables(&self) -> u32 {
        self.total_number_of_variables
    }
}

impl<F, EF> SumcheckPolyFirstRound<EF> for VirtualJaggedFirstRoundSumcheckPoly<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
    TaskScope: JaggedBackend<F, EF>
        + VirtualJaggedSumAsPolyKernel<F, EF>
        + ReduceSumBackend<EF>
        + VirtualJaggedFixLastVariableKernel<EF>
        + MleFixLastVariableBackend<F, EF>,
{
    type NextRoundPoly = HadamardProduct<EF, EF, TaskScope>;

    async fn sum_as_poly_in_last_t_variables(
        &self,
        claim: Option<EF>,
        t: usize,
    ) -> slop_algebra::UnivariatePolynomial<EF> {
        // We only support t = 1 for now.
        assert_eq!(t, 1);
        // Claim must be included.
        let claim = claim.unwrap();

        // We want to compute the inner product of the MLEs with the jagged little polynomial as
        // polynomials in the last variable. This is given by the expression:
        //
        //     sum(X) = \sum_{i} mle(i, X) * eq(z_row(i), X) * eq(z_col(i, X)).
        //
        // Instead of indexing on the MLEs, we could compute the contribution to this sum from each
        // individual table, which is how the sum is done below. Given an entry (row, col) for a
        // table with a certain height, the corresponding index is given by
        // `i = offset + col * height + row`. Since our MLEs are not contiguous but grouped in
        // batches according to thwe log_stacking_height, with each round having potentially
        // different batch sizes, we can compute the corresponding element mle(i,X) by indexing
        // correctly into the pointer of the batch, and then the specific index.
        //
        // **Warning**: for the computation of the index, we assume that the batch sizes are fixed.

        let mut col_offset = 0;
        let mut eval_zero = EF::zero();
        let mut eval_half = EF::zero();
        for (((row_counts, column_counts), base_component), mles) in self
            .row_data
            .iter()
            .zip_eq(self.column_data.iter())
            .zip_eq(self.base_components.iter())
            .zip_eq(self.base.iter())
        {
            // Get the components for the rounds.
            let scope = base_component.backend();
            let stacking_width: usize =
                mles.first().unwrap().num_polynomials() << mles.first().unwrap().num_variables();
            let mut offset = 0;
            for (row_count, column_count) in row_counts.iter().zip_eq(column_counts.iter()) {
                // Sum the entries corresponding to this table
                let half_height: usize = row_count.div_ceil(2);
                let col_count = *column_count;
                const BLOCK_SIZE: usize = 256;
                const ROW_STRIDE: usize = 32;
                const COL_STRIDE: usize = 2;
                let grid_dim: Dim3 = (
                    half_height.div_ceil(BLOCK_SIZE).div_ceil(ROW_STRIDE).max(1),
                    col_count.div_ceil(COL_STRIDE).max(1),
                    1,
                )
                    .into();

                let mut evaluations = Tensor::<EF, TaskScope>::with_sizes_in(
                    [2, grid_dim.y as usize, grid_dim.x as usize],
                    scope.clone(),
                );

                let num_tiles = BLOCK_SIZE.checked_div(32).unwrap_or(1);
                let shared_mem = num_tiles * std::mem::size_of::<EF>();

                unsafe {
                    evaluations.assume_init();
                    let args = args!(
                        evaluations.as_mut_ptr(),
                        base_component.as_ptr(),
                        self.eq_z_col.guts().as_ptr(),
                        self.eq_z_row.guts().as_ptr(),
                        offset,
                        col_offset,
                        half_height,
                        col_count,
                        stacking_width
                    );
                    scope
                        .launch_kernel(
                            TaskScope::sum_as_poly_kernel(),
                            grid_dim,
                            BLOCK_SIZE,
                            &args,
                            shared_mem,
                        )
                        .unwrap();
                }

                // Sum the evaluations across all dimensions.
                let evaluations = evaluations.sum(2).await.sum(1).await;
                let evaluations = evaluations.into_host().await.unwrap();

                let [val_zero, val_half] = evaluations.as_slice().try_into().unwrap();

                eval_zero += val_zero;
                eval_half += val_half;

                offset += row_count * column_count;
                col_offset += *column_count;
            }
        }

        let eval_one = claim - eval_zero;

        interpolate_univariate_polynomial(
            &[
                EF::from_canonical_u16(0),
                EF::from_canonical_u16(1),
                EF::from_canonical_u16(2).inverse(),
            ],
            &[eval_zero, eval_one, eval_half * F::from_canonical_u16(4).inverse()],
        )
    }

    async fn fix_t_variables(self, alpha: EF, t: usize) -> HadamardProduct<EF, EF, TaskScope> {
        assert_eq!(t, 1);

        let mut col_offset = 0;
        let mut jagged_polynomial = Vec::with_capacity(self.base.len());
        for (((row_counts, column_counts), base_component), mles) in self
            .row_data
            .iter()
            .zip_eq(self.column_data.iter())
            .zip_eq(self.base_components.iter())
            .zip_eq(self.base.iter())
        {
            let scope = base_component.backend();
            let mut round_components = mles
                .iter()
                .map(|mle| {
                    let mut jagged_comp = Mle::<EF, TaskScope>::uninit(
                        mle.num_polynomials(),
                        1 << (mle.num_variables() - 1),
                        scope,
                    );
                    unsafe {
                        jagged_comp.assume_init();
                    }
                    jagged_comp
                })
                .collect::<Vec<_>>();
            let jagged_components = round_components
                .iter_mut()
                .map(|mle| CanSendRawMut(mle.guts_mut().as_mut_ptr()))
                .collect::<Buffer<_>>();
            let mut jagged_components = scope.into_device(jagged_components).await.unwrap();

            // Get the components for the rounds.
            let scope = base_component.backend();
            let current_stacking_width: usize =
                mles.first().unwrap().num_polynomials() << mles.first().unwrap().num_variables();
            let stacking_width: usize = current_stacking_width >> 1;

            let mut offset = 0;
            for (row_count, column_count) in row_counts.iter().zip_eq(column_counts.iter()) {
                // Sum the entries corresponding to this table

                let half_height: usize = row_count.checked_div(2).unwrap();
                const BLOCK_SIZE: usize = 256;
                const STRIDE: usize = 32;
                let grid_dim: Dim3 =
                    (half_height.div_ceil(BLOCK_SIZE).div_ceil(STRIDE).max(32), 1, 1).into();

                let col_count = *column_count;
                unsafe {
                    let args = args!(
                        jagged_components.as_mut_ptr(),
                        self.eq_z_col.guts().as_ptr(),
                        self.eq_z_row.guts().as_ptr(),
                        alpha,
                        offset,
                        col_offset,
                        half_height,
                        col_count,
                        stacking_width
                    );
                    scope
                        .launch_kernel(
                            TaskScope::virtual_jagged_fix_last_variable_kernel(),
                            grid_dim,
                            BLOCK_SIZE,
                            &args,
                            0,
                        )
                        .unwrap();
                }
                offset += row_count * column_count;
                col_offset += *column_count;
            }
            jagged_polynomial.extend(round_components);
        }

        let base_components =
            self.base.into_iter().flatten().collect::<Message<Mle<F, TaskScope>>>();
        let log_stacking_height = base_components.first().unwrap().num_variables();
        let base = LongMle::from_message(base_components, log_stacking_height);
        let base = base.fix_last_variable(alpha).await;

        let jagged = LongMle::from_components(jagged_polynomial, base.log_stacking_height());
        HadamardProduct { base, ext: jagged }
    }
}

impl<F, EF> JaggedSumcheckProver<F, EF, TaskScope> for VirtualJaggedSumcheckProver
where
    F: Field,
    EF: ExtensionField<F>,
    TaskScope: JaggedBackend<F, EF>
        + PartialLagrangeBackend<EF>
        + VirtualJaggedSumAsPolyKernel<F, EF>
        + ReduceSumBackend<EF>
        + VirtualJaggedFixLastVariableKernel<EF>
        + MleFixLastVariableBackend<F, EF>,
{
    type Polynomial = VirtualJaggedFirstRoundSumcheckPoly<F, EF>;

    async fn jagged_sumcheck_poly(
        &self,
        base: Rounds<Message<Mle<F, TaskScope>>>,
        _jagged_params: &slop_jagged::JaggedLittlePolynomialProverParams,
        row_data: slop_commit::Rounds<std::sync::Arc<Vec<usize>>>,
        column_data: slop_commit::Rounds<std::sync::Arc<Vec<usize>>>,
        z_row: &slop_multilinear::Point<EF, TaskScope>,
        z_col: &slop_multilinear::Point<EF, TaskScope>,
    ) -> Self::Polynomial {
        let eq_z_col = Mle::<_, TaskScope>::partial_lagrange(z_col).await;
        let eq_z_row = Mle::<_, TaskScope>::partial_lagrange(z_row).await;

        let total_size = row_data
            .iter()
            .zip(column_data.iter())
            .map(|(r, c)| r.iter().zip(c.iter()).map(|(r, c)| r * c).sum::<usize>())
            .sum::<usize>();

        let total_number_of_variables = total_size.next_power_of_two().ilog2();

        tracing::debug!(
            "total_number_of_variables for jagged sumcheck: {}",
            total_number_of_variables
        );

        let base_components = base.iter().map(|message| async move {
            let backend = message.first().unwrap().backend();
            let ptrs =
                message.iter().map(|mle| CanSendRaw(mle.guts().as_ptr())).collect::<Buffer<_>>();
            backend.into_device(ptrs).await.unwrap()
        });
        let base_components = Rounds { rounds: join_all(base_components).await };

        VirtualJaggedFirstRoundSumcheckPoly {
            base,
            base_components,
            row_data,
            column_data,
            eq_z_col,
            eq_z_row,
            total_number_of_variables,
        }
    }
}

unsafe impl VirtualJaggedSumAsPolyKernel<BabyBear, BinomialExtensionField<BabyBear, 4>>
    for TaskScope
{
    fn sum_as_poly_kernel() -> KernelPtr {
        unsafe { jagged_baby_bear_base_ext_sum_as_poly() }
    }
}

unsafe impl VirtualJaggedFixLastVariableKernel<BinomialExtensionField<BabyBear, 4>> for TaskScope {
    fn virtual_jagged_fix_last_variable_kernel() -> KernelPtr {
        unsafe { jagged_baby_bear_extension_virtual_fix_last_variable() }
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use futures::prelude::*;
    use rand::thread_rng;
    use serial_test::serial;
    use slop_algebra::AbstractField;
    use slop_challenger::{CanObserve, FieldChallenger};
    use slop_commit::Rounds;
    use slop_jagged::{
        BabyBearPoseidon2TrivialEval, HadamardJaggedSumcheckProver, JaggedConfig,
        JaggedLittlePolynomialProverParams, JaggedPcsVerifier, JaggedProver, JaggedSumcheckProver,
    };
    use slop_multilinear::{Mle, MultilinearPcsChallenger, PaddedMle};
    use slop_sumcheck::SumcheckPolyFirstRound;

    use crate::{
        CudaJaggedMleGenerator, Poseidon2BabyBearJaggedCudaProverComponentsTrivialEval,
        VirtualJaggedSumcheckProver,
    };

    #[tokio::test]
    #[serial]
    async fn test_virtual_jagged_sumcheck_poly() {
        let log_blowup = 1;

        type JC = BabyBearPoseidon2TrivialEval;
        type Prover = JaggedProver<Poseidon2BabyBearJaggedCudaProverComponentsTrivialEval>;
        type F = <JC as JaggedConfig>::F;
        type EF = <JC as JaggedConfig>::EF;

        let mut rng = thread_rng();

        for (log_stacking_height, max_log_row_count) in [(10, 10), (11, 11), (16, 16), (20, 20)] {
            let row_counts_rounds = vec![
                vec![(1 << (max_log_row_count - 2)) + 8, 1 << max_log_row_count],
                vec![
                    (1 << (max_log_row_count - 6)) + 12,
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

            let hadamard_prover =
                HadamardJaggedSumcheckProver { jagged_generator: CudaJaggedMleGenerator };

            let virtual_prover = VirtualJaggedSumcheckProver;

            // Begin the commit rounds
            let mut challenger = jagged_verifier.challenger();

            csl_cuda::run_in_place(|t| async move {
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

                let num_col_variables = prover_data
                    .iter()
                    .map(|data| data.column_counts.iter().sum::<usize>())
                    .sum::<usize>()
                    .next_power_of_two()
                    .ilog2();

                let row_data =
                    prover_data.iter().map(|data| data.row_counts.clone()).collect::<Rounds<_>>();
                let column_data = prover_data
                    .iter()
                    .map(|data| data.column_counts.clone())
                    .collect::<Rounds<_>>();

                // Collect the jagged polynomial parameters.
                let params = JaggedLittlePolynomialProverParams::new(
                    prover_data
                        .iter()
                        .flat_map(|data| {
                            data.row_counts
                                .iter()
                                .copied()
                                .zip(data.column_counts.iter().copied())
                                .flat_map(|(row_count, column_count)| {
                                    std::iter::repeat(row_count).take(column_count)
                                })
                        })
                        .collect(),
                    max_log_row_count as usize,
                );

                let z_row = challenger.sample_point::<EF>(max_log_row_count);
                let z_col = challenger.sample_point::<EF>(num_col_variables);

                let z_row = t.into_device(z_row).await.unwrap();
                let z_col = t.into_device(z_col).await.unwrap();

                let all_mles = prover_data
                    .iter()
                    .map(|data| data.stacked_pcs_prover_data.interleaved_mles.clone())
                    .collect::<Rounds<_>>();

                let hadamard_poly = hadamard_prover
                    .jagged_sumcheck_poly(
                        all_mles.clone(),
                        &params,
                        row_data.clone(),
                        column_data.clone(),
                        &z_row,
                        &z_col,
                    )
                    .await;

                let virtual_poly = virtual_prover
                    .jagged_sumcheck_poly(all_mles, &params, row_data, column_data, &z_row, &z_col)
                    .await;

                // Get the sum as poly value with the zero claim (it's not correct but we only
                // want to test the function itself)
                let sum_as_poly_hadamard =
                    hadamard_poly.sum_as_poly_in_last_t_variables(Some(EF::zero()), 1).await;

                let sum_as_poly_virtual =
                    virtual_poly.sum_as_poly_in_last_t_variables(Some(EF::zero()), 1).await;

                assert_eq!(sum_as_poly_hadamard, sum_as_poly_virtual);

                // Test restrict last variable
                let alpha = challenger.sample_ext_element::<EF>();
                let eval_point = challenger.sample_point::<EF>(total_number_of_variables - 1);

                let restricted_virtual = virtual_poly.fix_t_variables(alpha, 1).await;
                let restricted_hadamard = hadamard_poly.fix_t_variables(alpha, 1).await;

                // Evaluate the restricted polynomials
                let hadamard_base_eval = restricted_hadamard.base.eval_at(&eval_point).await;
                let virtual_base_eval = restricted_virtual.base.eval_at(&eval_point).await;
                assert_eq!(hadamard_base_eval, virtual_base_eval);

                let hadamard_ext_eval = restricted_hadamard.ext.eval_at(&eval_point).await;
                let virtual_ext_eval = restricted_virtual.ext.eval_at(&eval_point).await;
                assert_eq!(hadamard_ext_eval, virtual_ext_eval);
            })
            .await
            .await
            .unwrap();
        }
    }
}
