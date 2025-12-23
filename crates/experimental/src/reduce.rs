use csl_cuda::{
    args,
    sys::{
        runtime::{Dim3, KernelPtr},
        v2_kernels::{reduce_kernel_ext, reduce_kernel_felt},
    },
    TaskScope,
};
use slop_tensor::Tensor;

use csl_utils::{Ext, Felt};

pub fn block_reduce_along_last_dim_with_kernel<
    T,
    const BLOCK_DIM_X: usize,
    const BLOCK_DIM_Y: usize,
    const ROW_STRIDE: usize,
>(
    input: &Tensor<T, TaskScope>,
    ker: KernelPtr,
) -> Tensor<T, TaskScope> {
    let height = input.sizes()[1];
    let width = input.sizes()[0];

    let output_height = height.div_ceil(BLOCK_DIM_X * ROW_STRIDE);

    let mut output = Tensor::<T, TaskScope>::with_sizes_in(
        [input.sizes()[0], output_height],
        input.backend().clone(),
    );

    let num_tiles = BLOCK_DIM_X / 32;
    let shared_mem = num_tiles * std::mem::size_of::<T>();

    let block_dim: Dim3 = (BLOCK_DIM_X, BLOCK_DIM_Y, 1).into();
    let grid_dim: Dim3 = (output_height, width.div_ceil(block_dim.y as usize), 1).into();

    // Call the kernel
    let t = input.backend();

    unsafe {
        let args = args!(input.as_ptr(), output.as_mut_ptr(), width, height, shared_mem);
        t.launch_kernel(ker, grid_dim, block_dim, &args, shared_mem).unwrap();
    }

    output
}

pub fn block_reduce_felt<
    const BLOCK_DIM_X: usize,
    const BLOCK_DIM_Y: usize,
    const ROW_STRIDE: usize,
>(
    input: &Tensor<Felt, TaskScope>,
) -> Tensor<Felt, TaskScope> {
    let ker = unsafe { reduce_kernel_felt() };
    block_reduce_along_last_dim_with_kernel::<Felt, BLOCK_DIM_X, BLOCK_DIM_Y, ROW_STRIDE>(
        input, ker,
    )
}

pub fn block_reduce_ext<
    const BLOCK_DIM_X: usize,
    const BLOCK_DIM_Y: usize,
    const ROW_STRIDE: usize,
>(
    input: &Tensor<Ext, TaskScope>,
) -> Tensor<Ext, TaskScope> {
    let ker = unsafe { reduce_kernel_ext() };
    block_reduce_along_last_dim_with_kernel::<Ext, BLOCK_DIM_X, BLOCK_DIM_Y, ROW_STRIDE>(input, ker)
}

pub fn reduce_with_partial_fn<T>(
    input: &Tensor<T, TaskScope>,
    mut partial_reduce: impl FnMut(&Tensor<T, TaskScope>) -> Tensor<T, TaskScope>,
) -> Tensor<T, TaskScope> {
    // Perform one block reduce.
    let mut output = partial_reduce(input);

    // Perform further reductions if necessary.
    while output.sizes()[1] > 1 {
        output = partial_reduce(&output);
    }

    let width = output.sizes()[0];
    output.reshape([width])
}

#[cfg(test)]
mod tests {
    use slop_alloc::ToHost;

    use slop_algebra::AbstractField;

    use super::*;

    #[tokio::test]
    async fn test_reduce_felt() {
        const BLOCK_DIM_X: usize = 128;
        const BLOCK_DIM_Y: usize = 2;
        const ROW_STRIDE: usize = 4;

        csl_cuda::run_in_place(|t| async move {
            // Doing multiple tests since in the first time the device is not warmed up.
            for (width, height) in [(10, 100), (100, 10000), (10, 10000), (20, 1 << 21)] {
                let mut rng = rand::thread_rng();
                let input_h = Tensor::rand(&mut rng, [width, height]);

                let input_d = t.to_device(&input_h).await.unwrap();

                let reduce_fn = |input: &Tensor<Felt, TaskScope>| {
                    block_reduce_felt::<BLOCK_DIM_X, BLOCK_DIM_Y, ROW_STRIDE>(input)
                };

                t.synchronize().await.unwrap();
                let time = tokio::time::Instant::now();
                let output_d = reduce_with_partial_fn(&input_d, reduce_fn);
                t.synchronize().await.unwrap();
                let time = time.elapsed();
                println!("reduce time: {time:?}");
                let output_h = output_d.to_host().await.unwrap();

                // Check that the output has the correct shape.
                assert_eq!(output_h.sizes(), [width]);

                // Check that the output is correct.
                for (j, out) in output_h.as_buffer().iter().enumerate() {
                    let mut acc = Felt::zero();
                    for i in 0..height {
                        acc += *input_h[[i, j]];
                    }
                    assert_eq!(*out, acc);
                }
            }
        })
        .await
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_reduce_ext() {
        const BLOCK_DIM_X: usize = 128;
        const BLOCK_DIM_Y: usize = 2;
        const ROW_STRIDE: usize = 4;

        csl_cuda::run_in_place(|t| async move {
            // Doing multiple tests since in the first time the device is not warmed up.
            for (width, height) in [(10, 100), (100, 10000), (10, 10000), (20, 1 << 21)] {
                let mut rng = rand::thread_rng();
                let input_h = Tensor::rand(&mut rng, [width, height]);

                let input_d = t.to_device(&input_h).await.unwrap();

                let reduce_fn = |input: &Tensor<Ext, TaskScope>| {
                    block_reduce_ext::<BLOCK_DIM_X, BLOCK_DIM_Y, ROW_STRIDE>(input)
                };

                t.synchronize().await.unwrap();
                let time = tokio::time::Instant::now();
                let output_d = reduce_with_partial_fn(&input_d, reduce_fn);
                t.synchronize().await.unwrap();
                let time = time.elapsed();
                println!("reduce time: {time:?}");
                let output_h = output_d.to_host().await.unwrap();

                // Check that the output has the correct shape.
                assert_eq!(output_h.sizes(), [width]);

                // Check that the output is correct.
                for (j, out) in output_h.as_buffer().iter().enumerate() {
                    let mut acc = Ext::zero();
                    for i in 0..height {
                        acc += *input_h[[i, j]];
                    }
                    assert_eq!(*out, acc);
                }
            }
        })
        .await
        .await
        .unwrap();
    }
}
