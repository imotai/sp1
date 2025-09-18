use crate::config::Ext;
use crate::zerocheck::data::DenseBuffer;
use crate::zerocheck::data::JaggedDenseMle;
use csl_cuda::args;
use csl_cuda::sys::runtime::KernelPtr;
use csl_cuda::TaskScope;
use csl_cuda::ToDevice;
use slop_algebra::Field;
use slop_alloc::Buffer;
use slop_alloc::HasBackend;
use slop_multilinear::Mle;
use slop_multilinear::Point;
use slop_tensor::Tensor;
use std::iter::once;

pub mod data;

pub async fn evaluate_jagged_fix_last_variable<F: Field>(
    jagged_mle: JaggedDenseMle<F, TaskScope>,
    value: Ext,
    input_heights: Vec<u32>,
    kernel: unsafe extern "C" fn() -> KernelPtr,
) -> (JaggedDenseMle<Ext, TaskScope>, Vec<u32>) {
    let backend = jagged_mle.dense().backend();

    let length = input_heights.iter().sum::<u32>();
    let output_heights =
        input_heights.iter().map(|height| height.div_ceil(4) * 2).collect::<Vec<u32>>();
    let new_start_idx = once(0)
        .chain(output_heights.iter().scan(0u32, |acc, x| {
            *acc += x;
            Some(*acc)
        }))
        .collect::<Vec<_>>();
    let new_total_length = *new_start_idx.last().unwrap() * 2;
    let buffer_start_idx = Buffer::from(new_start_idx);
    let output_start_idx = buffer_start_idx.to_device_in(backend).await.unwrap();

    let new_data =
        Buffer::<Ext, TaskScope>::with_capacity_in(new_total_length as usize, backend.clone());
    let new_cols = Buffer::<u32, TaskScope>::with_capacity_in(
        (new_total_length / 2) as usize,
        backend.clone(),
    );

    let mut next_jagged_mle =
        JaggedDenseMle::new(DenseBuffer { data: new_data }, new_cols, output_start_idx);

    const BLOCK_SIZE: usize = 256;
    const CHUNK_SIZE: usize = 1 << 16;
    let grid_size_x = (length as usize).div_ceil(CHUNK_SIZE).max(256);
    let grid_size = (grid_size_x, 1, 1);
    let block_dim = BLOCK_SIZE;

    unsafe {
        next_jagged_mle.dense_data.assume_init();
        next_jagged_mle.col_index.assume_init();
        let args = args!(jagged_mle.as_raw(), next_jagged_mle.as_mut_raw(), length, value);
        backend.launch_kernel(kernel(), grid_size, block_dim, &args, 0).unwrap();
    }

    (next_jagged_mle, output_heights)
}

pub async fn evaluate_jagged_mle_chunked<F: Field>(
    jagged_mle: JaggedDenseMle<F, TaskScope>,
    z_row: Point<Ext, TaskScope>,
    z_col: Point<Ext, TaskScope>,
    num_cols: usize,
    total_length: usize,
    kernel: unsafe extern "C" fn() -> KernelPtr,
) -> Tensor<Ext, TaskScope> {
    let backend = z_row.backend();

    const BLOCK_SIZE: usize = 256;
    const CHUNK_ELEMS: usize = 1 << 15;
    const MAX_COLS_SH: usize = 16;

    let n_chunks = total_length.div_ceil(CHUNK_ELEMS);
    let grid_size_x = n_chunks.min(u16::MAX as usize);
    let grid_size = (grid_size_x, 1, 1);

    // Dynamic shared memory:
    let nwarps = BLOCK_SIZE.div_ceil(32);
    let shared_reduce = nwarps * std::mem::size_of::<Ext>();
    let shared_starts = (MAX_COLS_SH + 1) * std::mem::size_of::<u32>();
    let shared_zcol = MAX_COLS_SH * std::mem::size_of::<Ext>();
    let shared_mem = shared_reduce + shared_starts + shared_zcol;

    // output one partial per block
    let mut output_evals =
        Tensor::<Ext, TaskScope>::with_sizes_in([1, grid_size.0], backend.clone());

    let z_row_lagrange = Mle::partial_lagrange(&z_row).await;
    let z_col_lagrange = Mle::partial_lagrange(&z_col).await;

    let args = args!(
        jagged_mle.as_raw(),
        z_row_lagrange.guts().as_ptr(),
        z_col_lagrange.guts().as_ptr(),
        (total_length as u32),
        (num_cols as u32),
        output_evals.as_mut_ptr()
    );

    unsafe {
        output_evals.assume_init();
        backend.launch_kernel(kernel(), grid_size, (BLOCK_SIZE, 1, 1), &args, shared_mem).unwrap();
    }

    let output_eval = output_evals.sum(1).await;
    output_eval
}

#[cfg(test)]
mod tests {
    use csl_cuda::run_in_place;
    use csl_cuda::sys::prover_clean::fix_last_variable_jagged_ext;
    use csl_cuda::sys::prover_clean::fix_last_variable_jagged_felt;
    use csl_cuda::sys::prover_clean::jagged_eval_kernel_chunked_felt;
    use rand::RngCore;
    use serial_test::serial;
    use slop_algebra::{extension::BinomialExtensionField, AbstractField};
    use slop_alloc::Buffer;
    use slop_koala_bear::KoalaBear;
    use slop_multilinear::{Mle, Point};
    use sp1_hypercube::log2_ceil_usize;

    use crate::config::Ext;
    use crate::zerocheck::data::DenseBuffer;
    use crate::zerocheck::data::JaggedDenseMle;
    use crate::zerocheck::evaluate_jagged_fix_last_variable;
    use crate::zerocheck::evaluate_jagged_mle_chunked;

    fn get_keccak_size() -> Vec<(u32, u32)> {
        vec![
            (65536, 13),
            (252960, 282),
            (119040, 2640),
            (4960, 672),
            (124000, 20),
            (0, 10),
            (472032, 17),
            (131072, 3),
            (4960, 10),
        ]
    }

    fn get_secp256k1_double_size() -> Vec<(u32, u32)> {
        vec![
            (65536, 13),
            (1016736, 282),
            (478464, 20),
            (0, 10),
            (472032, 17),
            (131072, 3),
            (59808, 1610),
            (59808, 10),
        ]
    }

    fn get_core_size() -> Vec<(u32, u32)> {
        vec![
            (291872, 34),
            (1158208, 31),
            (15328, 37),
            (551776, 52),
            (1254144, 46),
            (65536, 13),
            (64, 247),
            (295232, 282),
            (0, 61),
            (0, 36),
            (25024, 32),
            (131936, 39),
            (974144, 49),
            (664544, 41),
            (320, 46),
            (1760, 46),
            (960, 50),
            (31648, 45),
            (128, 15),
            (144832, 20),
            (23392, 83),
            (16, 60),
            (0, 10),
            (472032, 17),
            (131072, 3),
            (387488, 66),
            (206720, 70),
            (32, 14),
            (316288, 52),
            (573792, 41),
            (736, 47),
            (2592, 46),
            (49440, 34),
            (1216, 33),
            (5600, 10),
            (5664, 68),
            (61408, 32),
        ]
    }

    // Given a vector of (row_count, col_count) as input, returns
    // 1. Mle's for every table.
    // 2. Randomly generated dense data corresponding to the Mle's.
    // 3. Col Index corresponding to the dense data, for use in a JaggedMle.
    // 4. Start indices for every column, for use in a JaggedMle.
    fn get_input(
        sizes: Vec<(u32, u32)>,
    ) -> (Vec<Mle<KoalaBear>>, Vec<KoalaBear>, Vec<u32>, Vec<u32>) {
        let mut rng = rand::thread_rng();
        let sum_length = sizes.iter().map(|(a, b)| a * b).sum::<u32>();
        let mut cols = vec![0; (sum_length / 2) as usize];
        let num_cols = sizes.iter().map(|(_, b)| b).sum::<u32>();
        let mut start_idx = vec![0u32; (num_cols + 1) as usize];
        let mut col_idx: u32 = 0;
        let mut cnt: usize = 0;
        let data = (0..sum_length)
            .map(|_| KoalaBear::from_wrapped_u32(rng.next_u32()))
            .collect::<Vec<_>>();
        let mut mles = vec![];
        for (row, col) in sizes {
            assert_eq!(row % 4, 0);
            for _ in 0..col {
                mles.push(Mle::from_buffer(Buffer::from(
                    data[2 * cnt..2 * cnt + row as usize].to_vec(),
                )));
                for _ in 0..row / 2 {
                    cols[cnt] = col_idx;
                    cnt += 1;
                }
                start_idx[(col_idx + 1) as usize] = start_idx[col_idx as usize] + row / 2;
                col_idx += 1;
            }
        }
        (mles, data, cols, start_idx)
    }

    async fn mle_evaluation_test(table_sizes: Vec<(u32, u32)>) {
        let (mles, data, cols, start_idx) = get_input(table_sizes);

        let mut rng = rand::thread_rng();

        let row_variable: usize = 22;
        let col_variable = log2_ceil_usize(mles.len());
        let z_row = Point::<Ext>::rand(&mut rng, row_variable as u32);
        let z_col = Point::<Ext>::rand(&mut rng, col_variable as u32);
        let z_row_lagrange = Mle::partial_lagrange(&z_row).await;
        let z_col_lagrange = Mle::partial_lagrange(&z_col).await;

        let mut eval = BinomialExtensionField::<KoalaBear, 4>::zero();
        for (i, mle) in mles.iter().enumerate() {
            eval += mle.eval_at_eq(&z_row_lagrange).await.to_vec()[0]
                * z_col_lagrange.guts().as_slice()[i];
        }

        let data = Buffer::from(data);
        let cols = Buffer::from(cols);
        let start_idx = Buffer::from(start_idx);
        run_in_place(|t| async move {
            // Warmup iteration.
            let z_row_device = t.into_device(z_row.clone()).await.unwrap();
            let z_col_device = t.into_device(z_col.clone()).await.unwrap();
            let jagged_mle = JaggedDenseMle::new(
                DenseBuffer { data: data.clone() },
                cols.clone(),
                start_idx.clone(),
            )
            .into_device(&t)
            .await
            .unwrap();

            t.synchronize().await.unwrap();
            let evaluation = evaluate_jagged_mle_chunked(
                jagged_mle,
                z_row_device,
                z_col_device,
                mles.len(),
                data.len() / 2,
                jagged_eval_kernel_chunked_felt,
            )
            .await;
            t.synchronize().await.unwrap();

            let host_evals = unsafe { evaluation.into_buffer().copy_into_host_vec() };
            let evaluation = host_evals[0];
            assert_eq!(evaluation, eval);

            // Real iteration for benchmarking.
            let z_row_device = t.into_device(z_row.clone()).await.unwrap();
            let z_col_device = t.into_device(z_col.clone()).await.unwrap();
            let jagged_mle = JaggedDenseMle::new(
                DenseBuffer { data: data.clone() },
                cols.clone(),
                start_idx.clone(),
            )
            .into_device(&t)
            .await
            .unwrap();

            t.synchronize().await.unwrap();
            let now = std::time::Instant::now();
            let evaluation = evaluate_jagged_mle_chunked(
                jagged_mle,
                z_row_device,
                z_col_device,
                mles.len(),
                data.len() / 2,
                jagged_eval_kernel_chunked_felt,
            )
            .await;

            t.synchronize().await.unwrap();
            let elapsed = now.elapsed();

            let host_evals = unsafe { evaluation.into_buffer().copy_into_host_vec() };
            let evaluation = host_evals[0];
            assert_eq!(evaluation, eval);

            println!("elapsed jagged chunked {elapsed:?}");
        })
        .await;
    }

    // Instead of encoding all of the column evaluations as an MLE, this test directly
    // compares all column evaluations to the expected value from host.
    async fn mle_individual_evaluation_test(table_sizes: Vec<(u32, u32)>) {
        let (mles, data, cols, start_idx) = get_input(table_sizes);

        let mut input_heights = vec![];
        for i in 1..start_idx.len() {
            input_heights.push(start_idx[i] - start_idx[i - 1]);
        }

        let mut rng = rand::thread_rng();

        let row_variable: usize = 22;
        let z_row = Point::<Ext>::rand(&mut rng, row_variable as u32);
        let z_row_lagrange = Mle::partial_lagrange(&z_row).await;

        let mut eval = vec![];
        for mle in mles.iter() {
            eval.push(mle.eval_at_eq(&z_row_lagrange).await.to_vec()[0]);
        }

        let data = Buffer::from(data);
        let cols = Buffer::from(cols);
        let start_idx = Buffer::from(start_idx);
        run_in_place(|t| async move {
            let jagged_mle = JaggedDenseMle::new(
                DenseBuffer { data: data.clone() },
                cols.clone(),
                start_idx.clone(),
            )
            .into_device(&t)
            .await
            .unwrap();

            t.synchronize().await.unwrap();
            let now = std::time::Instant::now();
            let (mut jagged_mle, mut input_heights) = evaluate_jagged_fix_last_variable(
                jagged_mle,
                *z_row[row_variable - 1],
                input_heights,
                fix_last_variable_jagged_felt,
            )
            .await;

            t.synchronize().await.unwrap();
            let elapsed = now.elapsed();
            println!("time for round 1: {elapsed:?}");

            for i in (0..row_variable - 1).rev() {
                let now = std::time::Instant::now();
                (jagged_mle, input_heights) = evaluate_jagged_fix_last_variable(
                    jagged_mle,
                    *z_row[i],
                    input_heights,
                    fix_last_variable_jagged_ext,
                )
                .await;
                t.synchronize().await.unwrap();
                let elapsed = now.elapsed();
                println!("time for round {}: {elapsed:?}", row_variable - i);
            }

            let result = unsafe { jagged_mle.dense_data.data.copy_into_host_vec() };
            let mut idx = 0;
            for i in 0..eval.len() {
                if input_heights[i] == 0 {
                    assert_eq!(eval[i], Ext::zero());
                } else {
                    assert_eq!(eval[i], result[idx]);
                    idx += 4;
                }
            }
            t.synchronize().await.unwrap();

            let elapsed = now.elapsed();
            println!("time: {elapsed:?}");
        })
        .await;
    }

    #[tokio::test]
    #[serial]
    async fn test_jagged_mle_eval_keccak() {
        let table_sizes = get_keccak_size();
        mle_evaluation_test(table_sizes).await;
    }

    #[tokio::test]
    #[serial]
    async fn test_jagged_mle_eval_secp() {
        let table_sizes = get_secp256k1_double_size();
        mle_evaluation_test(table_sizes).await;
    }

    #[tokio::test]
    #[serial]
    async fn test_jagged_mle_eval_core() {
        let table_sizes = get_core_size();
        mle_evaluation_test(table_sizes).await;
    }

    #[tokio::test]
    #[serial]
    async fn test_jagged_mle_eval_individual_keccak() {
        let table_sizes = get_keccak_size();
        mle_individual_evaluation_test(table_sizes).await;
    }

    #[tokio::test]
    #[serial]
    async fn test_jagged_mle_eval_individual_secp() {
        let table_sizes = get_secp256k1_double_size();
        mle_individual_evaluation_test(table_sizes).await;
    }

    #[tokio::test]
    #[serial]
    async fn test_jagged_mle_eval_individual_core() {
        let table_sizes = get_core_size();
        mle_individual_evaluation_test(table_sizes).await;
    }
}
