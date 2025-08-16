use std::sync::Arc;

use csl_cuda::{
    args,
    sys::{jagged::jagged_koala_bear_extension_populate, runtime::KernelPtr},
    TaskScope,
};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use slop_algebra::{extension::BinomialExtensionField, Field};
use slop_commit::Rounds;
use slop_jagged::{JaggedLittlePolynomialProverParams, JaggedMleGenerator, LongMle};
use slop_koala_bear::KoalaBear;
use slop_multilinear::{Mle, PartialLagrangeBackend, Point};

#[derive(Debug, Clone, Default, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct CudaJaggedMleGenerator;

/// # Safety    
pub unsafe trait JaggedPopulateKernel<F: Field> {
    fn jagged_populate_kernel() -> KernelPtr;
}

unsafe impl JaggedPopulateKernel<BinomialExtensionField<KoalaBear, 4>> for CudaJaggedMleGenerator {
    fn jagged_populate_kernel() -> KernelPtr {
        unsafe { jagged_koala_bear_extension_populate() }
    }
}

impl<F> JaggedMleGenerator<F, TaskScope> for CudaJaggedMleGenerator
where
    F: Field,
    Self: JaggedPopulateKernel<F>,
    TaskScope: PartialLagrangeBackend<F>,
{
    async fn partial_jagged_multilinear(
        &self,
        _jagged_params: &JaggedLittlePolynomialProverParams,
        row_data: Rounds<Arc<Vec<usize>>>,
        column_data: Rounds<Arc<Vec<usize>>>,
        z_row: &Point<F, TaskScope>,
        z_col: &Point<F, TaskScope>,
        _num_components: usize,
    ) -> LongMle<F, TaskScope> {
        let eq_z_col = Mle::partial_lagrange(z_col).await;
        let eq_z_row = Mle::partial_lagrange(z_row).await;

        let total_size = row_data
            .iter()
            .zip(column_data.iter())
            .map(|(r, c)| r.iter().zip(c.iter()).map(|(r, c)| r * c).sum::<usize>())
            .sum::<usize>();

        let total_number_of_variables = total_size.next_power_of_two().ilog2();

        let backend = z_row.backend();
        let mut jagged_polynomial = Mle::<F, _>::uninit(1, 1 << total_number_of_variables, backend);

        let mut offset = 0;
        let mut col_offset = 0;
        for (row_counts, column_counts) in row_data.iter().zip_eq(column_data.iter()) {
            for (row_count, column_count) in row_counts.iter().zip_eq(column_counts.iter()) {
                // Populate the entries corresponding to this table
                let block_dim = 256;
                let grid_dim = row_count.div_ceil(block_dim).max(1);
                let args = args!(
                    jagged_polynomial.guts_mut().as_mut_ptr(),
                    eq_z_col.guts().as_ptr(),
                    eq_z_row.guts().as_ptr(),
                    offset,
                    col_offset,
                    *row_count,
                    *column_count
                );
                unsafe {
                    backend
                        .launch_kernel(
                            Self::jagged_populate_kernel(),
                            grid_dim,
                            block_dim,
                            &args,
                            0,
                        )
                        .unwrap();
                }
                offset += row_count * column_count;
                col_offset += *column_count;
            }
        }
        unsafe {
            jagged_polynomial.guts_mut().as_mut_buffer().set_len(offset);
        }
        let padding_size = ((1 << total_number_of_variables) - offset) * std::mem::size_of::<F>();
        jagged_polynomial.guts_mut().as_mut_buffer().write_bytes(0, padding_size).unwrap();

        LongMle::from_components(vec![jagged_polynomial], total_number_of_variables)
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use rand::thread_rng;
    use serial_test::serial;
    use slop_alloc::IntoHost;
    use slop_jagged::CpuJaggedMleGenerator;

    use super::*;

    #[tokio::test]
    #[serial]
    async fn test_jagged_populate() {
        let row_counts_rounds = vec![vec![(1 << 8) + 12, 1 << 9], vec![(1 << 10) + 12]];
        let column_counts_rounds = vec![vec![3, 2], vec![1]];
        let max_log_row_count = 11;

        type F = KoalaBear;
        type EF = BinomialExtensionField<F, 4>;

        let row_counts =
            row_counts_rounds.into_iter().map(Arc::new).collect::<Rounds<Arc<Vec<usize>>>>();
        let column_counts =
            column_counts_rounds.into_iter().map(Arc::new).collect::<Rounds<Arc<Vec<usize>>>>();

        let params = JaggedLittlePolynomialProverParams::new(
            row_counts
                .iter()
                .zip_eq(column_counts.iter())
                .flat_map(|(row_counts, column_counts)| {
                    row_counts.iter().copied().zip_eq(column_counts.iter().copied()).flat_map(
                        |(row_count, column_count)| std::iter::repeat_n(row_count, column_count),
                    )
                })
                .collect(),
            max_log_row_count,
        );

        let num_col_variables = column_counts
            .iter()
            .map(|column_counts| column_counts.iter().sum::<usize>())
            .sum::<usize>()
            .next_power_of_two()
            .ilog2();

        let num_row_variables = max_log_row_count as u32;

        let mut rng = thread_rng();
        let z_row = Point::<EF>::rand(&mut rng, num_row_variables);
        let z_col = Point::<EF>::rand(&mut rng, num_col_variables);

        let host_generator = CpuJaggedMleGenerator;

        let host_jagged_polynomial = host_generator
            .partial_jagged_multilinear(
                &params,
                row_counts.clone(),
                column_counts.clone(),
                &z_row,
                &z_col,
                1,
            )
            .await;

        let devide_generator = CudaJaggedMleGenerator;
        let device_paramerters = params.clone();
        let device_jagged_polynomial = csl_cuda::run_in_place(|t| async move {
            let z_row = t.to_device(&z_row).await.unwrap();
            let z_col = t.to_device(&z_col).await.unwrap();
            let device_jagged_polynomial = devide_generator
                .partial_jagged_multilinear(
                    &device_paramerters,
                    row_counts,
                    column_counts,
                    &z_row,
                    &z_col,
                    1,
                )
                .await;
            device_jagged_polynomial.into_host().await.unwrap()
        })
        .await
        .await
        .unwrap();

        for (host_component, device_component) in host_jagged_polynomial
            .into_components()
            .into_iter()
            .zip(device_jagged_polynomial.into_components().into_iter())
        {
            for (idx, (host_val, device_val)) in host_component
                .guts()
                .as_slice()
                .iter()
                .zip(device_component.guts().as_slice().iter())
                .enumerate()
            {
                assert_eq!(host_val, device_val, "Index: {idx:?}");
            }
        }
    }
}
