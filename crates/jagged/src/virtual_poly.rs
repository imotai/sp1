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
use itertools::Itertools;
use slop_algebra::{
    extension::BinomialExtensionField, interpolate_univariate_polynomial, ExtensionField, Field,
};
use slop_alloc::{HasBackend, IntoHost};
use slop_baby_bear::BabyBear;
use slop_commit::Rounds;
use slop_jagged::{HadamardProduct, JaggedBackend, JaggedSumcheckProver, LongMle};
use slop_multilinear::{Mle, MleFixLastVariableBackend, PartialLagrangeBackend};
use slop_stacked::{FixedRateInterleave, InterleaveMultilinears};
use slop_sumcheck::{SumcheckPolyBase, SumcheckPolyFirstRound};
use slop_tensor::{ReduceSumBackend, Tensor};

#[derive(Debug, Clone, Copy, Default, PartialEq, PartialOrd, Eq, Ord)]
pub struct VirtualJaggedSumcheckProver;

#[derive(Debug)]
pub struct VirtualJaggedFirstRoundSumcheckPoly<F, EF> {
    base: LongMle<F, TaskScope>,
    row_data: Rounds<Arc<Vec<usize>>>,
    column_data: Rounds<Arc<Vec<usize>>>,
    eq_z_col: Mle<EF, TaskScope>,
    eq_z_row: Mle<EF, TaskScope>,
}

pub trait VirtualJaggedSumAsPolyKernel<F, EF> {
    fn sum_as_poly_kernel() -> KernelPtr;
}

pub trait VirtualJaggedFixLastVariableKernel<EF> {
    fn virtual_jagged_fix_last_variable_kernel() -> KernelPtr;
}

impl<F, EF> SumcheckPolyBase for VirtualJaggedFirstRoundSumcheckPoly<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    fn num_variables(&self) -> u32 {
        self.base.num_variables()
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
        assert!(self.base.num_components() == 1);
        assert_eq!(t, 1);

        let claim = claim.unwrap();
        let poly_base = self.base.first_component_mle();
        let scope = poly_base.backend();

        let mut offset = 0;
        let mut col_offset = 0;

        let mut eval_zero = EF::zero();
        let mut eval_half = EF::zero();
        for (row_counts, column_counts) in self.row_data.iter().zip_eq(self.column_data.iter()) {
            for (row_count, column_count) in row_counts.iter().zip_eq(column_counts.iter()) {
                // Sum the entries corresponding to this table

                let half_height: usize = row_count.div_ceil(2);
                const BLOCK_SIZE: usize = 256;
                const STRIDE: usize = 32;
                let grid_dim: Dim3 =
                    (half_height.div_ceil(BLOCK_SIZE).div_ceil(STRIDE).max(32), 1, 1).into();

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
                        poly_base.guts().as_ptr(),
                        self.eq_z_col.guts().as_ptr(),
                        self.eq_z_row.guts().as_ptr(),
                        offset,
                        col_offset,
                        half_height,
                        *column_count
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
        assert!(self.base.num_components() == 1);

        let poly_base = self.base.first_component_mle();
        let scope = poly_base.backend();

        let mut offset = 0;
        let mut col_offset = 0;

        let previous_num_variables = poly_base.num_variables();
        let num_variables = previous_num_variables - 1;
        assert!(num_variables > 0);
        let mut jagged_poly_restricted = Mle::<EF, TaskScope>::uninit(1, 1 << num_variables, scope);
        for (row_counts, column_counts) in self.row_data.iter().zip_eq(self.column_data.iter()) {
            for (row_count, column_count) in row_counts.iter().zip_eq(column_counts.iter()) {
                // Sum the entries corresponding to this table

                let half_height: usize = row_count.div_ceil(2);
                const BLOCK_SIZE: usize = 256;
                const STRIDE: usize = 32;
                let grid_dim: Dim3 =
                    (half_height.div_ceil(BLOCK_SIZE).div_ceil(STRIDE).max(32), 1, 1).into();

                unsafe {
                    let args = args!(
                        jagged_poly_restricted.guts_mut().as_mut_ptr(),
                        self.eq_z_col.guts().as_ptr(),
                        self.eq_z_row.guts().as_ptr(),
                        alpha,
                        offset,
                        col_offset,
                        half_height,
                        *column_count
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
        }

        let total_initialized_length = offset >> 1;
        unsafe {
            jagged_poly_restricted.guts_mut().as_mut_buffer().set_len(total_initialized_length);
        }
        let padding_size =
            ((1 << num_variables) - total_initialized_length) * std::mem::size_of::<EF>();
        jagged_poly_restricted.guts_mut().as_mut_buffer().write_bytes(0, padding_size).unwrap();

        let jagged = LongMle::from_components(vec![jagged_poly_restricted], num_variables);

        let base = self.base.fix_last_variable(alpha).await;

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
        base: LongMle<F, TaskScope>,
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

        let stacker = FixedRateInterleave::<F, TaskScope>::new(1);
        let restacked_mle = LongMle::from_message(
            stacker
                .interleave_multilinears(base.components().clone(), total_number_of_variables)
                .await,
            total_number_of_variables,
        );

        VirtualJaggedFirstRoundSumcheckPoly {
            base: restacked_mle,
            row_data,
            column_data,
            eq_z_col,
            eq_z_row,
        }
    }
}

impl VirtualJaggedSumAsPolyKernel<BabyBear, BinomialExtensionField<BabyBear, 4>> for TaskScope {
    fn sum_as_poly_kernel() -> KernelPtr {
        unsafe { jagged_baby_bear_base_ext_sum_as_poly() }
    }
}

impl VirtualJaggedFixLastVariableKernel<BinomialExtensionField<BabyBear, 4>> for TaskScope {
    fn virtual_jagged_fix_last_variable_kernel() -> KernelPtr {
        unsafe { jagged_baby_bear_extension_virtual_fix_last_variable() }
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use csl_cuda::TaskScope;
    use futures::prelude::*;
    use rand::thread_rng;
    use serial_test::serial;
    use slop_algebra::AbstractField;
    use slop_challenger::{CanObserve, FieldChallenger};
    use slop_commit::{Message, Rounds};
    use slop_jagged::{
        BabyBearPoseidon2, HadamardJaggedSumcheckProver, JaggedConfig,
        JaggedLittlePolynomialProverParams, JaggedPcsVerifier, JaggedProver, JaggedSumcheckProver,
        LongMle,
    };
    use slop_multilinear::{Mle, MultilinearPcsChallenger, PaddedMle};
    use slop_sumcheck::SumcheckPolyFirstRound;

    use crate::{
        CudaJaggedMleGenerator, Poseidon2BabyBearJaggedCudaProverComponents,
        VirtualJaggedSumcheckProver,
    };

    #[tokio::test]
    #[serial]
    async fn test_virtual_jagged_sumcheck_poly() {
        let log_blowup = 1;

        type JC = BabyBearPoseidon2;
        type Prover = JaggedProver<Poseidon2BabyBearJaggedCudaProverComponents>;
        type F = <JC as JaggedConfig>::F;
        type EF = <JC as JaggedConfig>::EF;

        let mut rng = thread_rng();

        for (log_stacking_height, max_log_row_count) in [(10, 10), (11, 11), (16, 16), (20, 20)] {
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

            let hadamard_prover =
                HadamardJaggedSumcheckProver { jagged_generator: CudaJaggedMleGenerator };

            let virtual_prover = VirtualJaggedSumcheckProver;

            // Begin the commit rounds
            let mut challenger = jagged_verifier.challenger();

            csl_cuda::task()
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

                    let num_col_variables = prover_data
                        .iter()
                        .map(|data| data.column_counts.iter().sum::<usize>())
                        .sum::<usize>()
                        .next_power_of_two()
                        .ilog2();

                    let row_data = prover_data
                        .iter()
                        .map(|data| data.row_counts.clone())
                        .collect::<Rounds<_>>();
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
                        .flat_map(|data| data.stacked_pcs_prover_data.interleaved_mles.clone())
                        .collect::<Message<Mle<F, TaskScope>>>();
                    let base = LongMle::from_message(all_mles, log_stacking_height);

                    let hadamard_poly = hadamard_prover
                        .jagged_sumcheck_poly(
                            base.clone(),
                            &params,
                            row_data.clone(),
                            column_data.clone(),
                            &z_row,
                            &z_col,
                        )
                        .await;

                    let virtual_poly = virtual_prover
                        .jagged_sumcheck_poly(base, &params, row_data, column_data, &z_row, &z_col)
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

                    // let restricted_hadamard = hadamard_poly.fix_t_variables(alpha, 1).await;
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
