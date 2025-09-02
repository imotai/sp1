use std::ffi::c_void;

use csl_cuda::{
    args,
    sys::{
        prover_clean::{
            round_kernel_1_128_4_8_false, round_kernel_1_32_2_2_false, round_kernel_1_32_2_3_false,
            round_kernel_1_64_2_2_false, round_kernel_1_64_2_3_false, round_kernel_1_64_4_8_false,
            round_kernel_2_32_2_2_false, round_kernel_2_32_2_2_true, round_kernel_2_32_2_3_false,
            round_kernel_2_32_2_3_true, round_kernel_2_64_2_2_false, round_kernel_2_64_2_2_true,
            round_kernel_2_64_2_3_false, round_kernel_2_64_2_3_true, round_kernel_4_32_2_2_false,
            round_kernel_4_32_2_2_true, round_kernel_4_32_2_3_false, round_kernel_4_32_2_3_true,
            round_kernel_4_64_2_2_false, round_kernel_4_64_2_2_true, round_kernel_4_64_2_3_false,
            round_kernel_4_64_2_3_true, round_kernel_4_64_4_8_false, round_kernel_4_64_4_8_true,
            round_kernel_8_32_2_2_false, round_kernel_8_32_2_2_true, round_kernel_8_32_2_3_false,
            round_kernel_8_32_2_3_true, round_kernel_8_64_2_2_false, round_kernel_8_64_2_2_true,
            round_kernel_8_64_2_3_false, round_kernel_8_64_2_3_true,
        },
        runtime::KernelPtr,
    },
    CudaError, TaskScope,
};
use slop_alloc::{Buffer, ToHost};
use slop_multilinear::{Mle, Point};
use slop_tensor::Tensor;

use crate::config::Ext;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoundParams {
    pub fix_group: usize,
    pub fix_tile: usize,
    pub sum_group: usize,
    pub num_points: usize,
    pub store_restricted: bool,
}

#[inline]
pub fn populate_restrict_eq(point: &Point<Ext>, t: &TaskScope) -> Result<(), CudaError> {
    let eq_res = Mle::blocking_partial_lagrange(point);
    assert_eq!(eq_res.guts().total_len(), 1 << point.dimension());
    unsafe {
        let maybe_err = csl_cuda::sys::prover_clean::populate_restrict_eq_host(
            eq_res.guts().as_ptr() as *const c_void,
            eq_res.guts().total_len(),
            t.handle(),
        );
        CudaError::result_from_ffi(maybe_err)
    }
}

pub struct Hadamard<T> {
    pub p: Tensor<T, TaskScope>,
    pub q: Tensor<T, TaskScope>,
}

fn look_ahead_round_kernel(params: &RoundParams) -> KernelPtr {
    match (
        params.fix_group,
        params.fix_tile,
        params.sum_group,
        params.num_points,
        params.store_restricted,
    ) {
        // FIX_TILE=32 variants
        (1, 32, 2, 2, false) => unsafe { round_kernel_1_32_2_2_false() },
        (2, 32, 2, 2, true) => unsafe { round_kernel_2_32_2_2_true() },
        (2, 32, 2, 2, false) => unsafe { round_kernel_2_32_2_2_false() },
        (4, 32, 2, 2, true) => unsafe { round_kernel_4_32_2_2_true() },
        (4, 32, 2, 2, false) => unsafe { round_kernel_4_32_2_2_false() },
        (8, 32, 2, 2, true) => unsafe { round_kernel_8_32_2_2_true() },
        (8, 32, 2, 2, false) => unsafe { round_kernel_8_32_2_2_false() },
        // FIX_TILE=64 variants
        (1, 64, 2, 2, false) => unsafe { round_kernel_1_64_2_2_false() },
        (1, 64, 4, 8, false) => unsafe { round_kernel_1_64_4_8_false() },
        (2, 64, 2, 2, true) => unsafe { round_kernel_2_64_2_2_true() },
        (2, 64, 2, 2, false) => unsafe { round_kernel_2_64_2_2_false() },
        (4, 64, 2, 2, true) => unsafe { round_kernel_4_64_2_2_true() },
        (4, 64, 2, 2, false) => unsafe { round_kernel_4_64_2_2_false() },
        (8, 64, 2, 2, true) => unsafe { round_kernel_8_64_2_2_true() },
        (8, 64, 2, 2, false) => unsafe { round_kernel_8_64_2_2_false() },
        (1, 32, 2, 3, false) => unsafe { round_kernel_1_32_2_3_false() },
        (2, 32, 2, 3, true) => unsafe { round_kernel_2_32_2_3_true() },
        (2, 32, 2, 3, false) => unsafe { round_kernel_2_32_2_3_false() },
        (4, 32, 2, 3, true) => unsafe { round_kernel_4_32_2_3_true() },
        (4, 32, 2, 3, false) => unsafe { round_kernel_4_32_2_3_false() },
        (8, 32, 2, 3, true) => unsafe { round_kernel_8_32_2_3_true() },
        (8, 32, 2, 3, false) => unsafe { round_kernel_8_32_2_3_false() },
        (1, 64, 2, 3, false) => unsafe { round_kernel_1_64_2_3_false() },
        (2, 64, 2, 3, true) => unsafe { round_kernel_2_64_2_3_true() },
        (2, 64, 2, 3, false) => unsafe { round_kernel_2_64_2_3_false() },
        (4, 64, 2, 3, true) => unsafe { round_kernel_4_64_2_3_true() },
        (4, 64, 2, 3, false) => unsafe { round_kernel_4_64_2_3_false() },
        (4, 64, 4, 8, true) => unsafe { round_kernel_4_64_4_8_true() },
        (4, 64, 4, 8, false) => unsafe { round_kernel_4_64_4_8_false() },
        (8, 64, 2, 3, true) => unsafe { round_kernel_8_64_2_3_true() },
        (8, 64, 2, 3, false) => unsafe { round_kernel_8_64_2_3_false() },
        // FIX_TILE=128 variants
        (1, 128, 4, 8, false) => unsafe { round_kernel_1_128_4_8_false() },
        _ => panic!("Unsupported kernel with params: {params:?}"),
    }
}

impl Hadamard<Ext> {
    #[inline]
    pub const fn new(p: Tensor<Ext, TaskScope>, q: Tensor<Ext, TaskScope>) -> Self {
        Self { p, q }
    }

    pub fn backend(&self) -> &TaskScope {
        self.p.backend()
    }

    pub fn alloc(len: usize, backend: &TaskScope) -> Self {
        Self {
            p: Tensor::with_sizes_in([len], backend.clone()),
            q: Tensor::with_sizes_in([len], backend.clone()),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.p.sizes()[0]
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn num_variables(&self) -> u32 {
        self.len().ilog2()
    }

    pub async fn round(&self, params: &RoundParams) -> (Buffer<Ext>, Option<Self>) {
        // TODO: make configurable
        let reduced_height = self.len() / params.fix_group;
        let mut restricted = if params.store_restricted {
            Some(Self::alloc(reduced_height, self.backend()))
        } else {
            None
        };

        const BLOCK_SIZE: usize = 256;
        // let stride: usize = 8 / params.fix_group;

        let num_fix_groups = 32 / params.fix_group;
        let num_tiles_per_warp = params.fix_tile / num_fix_groups;
        let block_tile: usize = BLOCK_SIZE * num_tiles_per_warp;

        // let grid_dim = self.len().div_ceil(block_tile * stride);
        let grid_dim = self.len().div_ceil(block_tile).min(1 << 16);

        let num_warps = BLOCK_SIZE / 32;
        let shared_mem_size =
            (2 * params.fix_tile + params.num_points) * num_warps * std::mem::size_of::<Ext>();

        let mut result =
            Tensor::<Ext, _>::with_sizes_in([params.num_points, grid_dim], self.backend().clone());

        let tile_height = self.len().next_multiple_of(block_tile);

        unsafe {
            let (restricted_p, restricted_q) = restricted
                .as_mut()
                .map(|r| (r.p.as_mut_ptr(), r.q.as_mut_ptr()))
                .unwrap_or_else(|| (std::ptr::null_mut(), std::ptr::null_mut()));

            let args = args!(
                result.as_mut_ptr(),
                self.p.as_ptr(),
                self.q.as_ptr(),
                restricted_p,
                restricted_q,
                tile_height,
                self.len()
            );
            self.backend()
                .launch_kernel(
                    look_ahead_round_kernel(params),
                    grid_dim,
                    BLOCK_SIZE,
                    &args,
                    shared_mem_size,
                )
                .unwrap();
        }

        let result = result.sum(1).await;
        let result = result.into_buffer().to_host().await.unwrap();

        if let Some(ref mut restricted) = restricted {
            let res_len = restricted.len();
            unsafe {
                restricted.p.as_mut_buffer().set_len(res_len);
                restricted.q.as_mut_buffer().set_len(res_len);
            }
        }

        (result, restricted)
    }
}

#[cfg(test)]
mod tests {
    use csl_cuda::IntoDevice;
    use rand::Rng;
    use slop_alloc::IntoHost;
    use slop_multilinear::Mle;

    use super::*;

    fn sum_as_poly_in_last_variable(p_h: &Mle<Ext>, q_h: &Mle<Ext>) -> (Ext, Ext) {
        // Compute the "sum as poly in last variable"
        let eval_0 = p_h
            .guts()
            .as_buffer()
            .as_slice()
            .chunks_exact(2)
            .zip(q_h.guts().as_buffer().as_slice().chunks_exact(2))
            .map(|(p_chunk, q_chunk)| {
                let p_0 = p_chunk[0];
                let _p_1 = p_chunk[1];
                let q_0 = q_chunk[0];
                let _q_1 = q_chunk[1];

                p_0 * q_0
            })
            .sum::<Ext>();

        let eval_half = p_h
            .guts()
            .as_buffer()
            .as_slice()
            .chunks_exact(2)
            .zip(q_h.guts().as_buffer().as_slice().chunks_exact(2))
            .map(|(p_chunk, q_chunk)| {
                let p_0 = p_chunk[0];
                let p_1 = p_chunk[1];
                let q_0 = q_chunk[0];
                let q_1 = q_chunk[1];

                let p_half = p_0 + p_1;
                let q_half = q_0 + q_1;

                p_half * q_half
            })
            .sum::<Ext>();

        (eval_0, eval_half)
    }

    #[tokio::test]
    async fn test_look_ahead_round() {
        let num_variables = 10;

        let mut rng = rand::thread_rng();
        let p_h = Tensor::<Ext>::rand(&mut rng, [1 << num_variables]);
        let q_h = Tensor::<Ext>::rand(&mut rng, [1 << num_variables]);

        let p_h_mle = Mle::new(p_h.clone().reshape([1 << num_variables, 1]));
        let q_h_mle = Mle::new(q_h.clone().reshape([1 << num_variables, 1]));

        let (eval_0_first, eval_half_first) = sum_as_poly_in_last_variable(&p_h_mle, &q_h_mle);

        let alpha = rng.gen::<Ext>();

        let p_h_fixed = p_h_mle.fix_last_variable(alpha).await;
        let q_h_fixed = q_h_mle.fix_last_variable(alpha).await;

        let (eval_0, eval_half) = sum_as_poly_in_last_variable(&p_h_fixed, &q_h_fixed);

        csl_cuda::spawn(move |t| async move {
            let p_d = p_h.into_device_in(&t).await.unwrap();
            let q_d = q_h.into_device_in(&t).await.unwrap();

            let hadamard = Hadamard::new(p_d, q_d);

            let first_round_params = RoundParams {
                fix_group: 1,
                fix_tile: 64,
                sum_group: 2,
                num_points: 2,
                store_restricted: false,
            };
            let (first_result, _) = hadamard.round(&first_round_params).await;

            let eval_0_d_first = *first_result[0];
            let eval_half_d_first = *first_result[1];

            assert_eq!(eval_0_d_first, eval_0_first);
            assert_eq!(eval_half_d_first, eval_half_first);

            let second_round_params = RoundParams {
                fix_group: 2,
                fix_tile: 64,
                sum_group: 2,
                num_points: 2,
                store_restricted: true,
            };

            populate_restrict_eq(&Point::from(vec![alpha]), &t).unwrap();
            let (result, restricted) = hadamard.round(&second_round_params).await;

            let Hadamard { p, q } = restricted.unwrap();

            let p_d_fixed = p.into_host().await.unwrap();
            let q_d_fixed = q.into_host().await.unwrap();

            let eval_0_d = *result[0];
            let eval_half_d = *result[1];

            assert_eq!(p_d_fixed.as_buffer().len(), p_h_fixed.guts().as_buffer().len());

            for (p_d, p_h) in p_d_fixed.as_buffer().iter().zip(p_h_fixed.guts().as_buffer().iter())
            {
                assert_eq!(*p_d, *p_h);
            }

            assert_eq!(q_d_fixed.as_buffer().len(), q_h_fixed.guts().as_buffer().len());
            for (q_d, q_h) in q_d_fixed.as_buffer().iter().zip(q_h_fixed.guts().as_buffer().iter())
            {
                assert_eq!(*q_d, *q_h);
            }

            // Compare the evals
            assert_eq!(eval_half_d, eval_half);
            assert_eq!(eval_0_d, eval_0);
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_look_ahead_multi_fix() {
        let num_variables = 10;
        let num_fix_variables = 2;

        let mut rng = rand::thread_rng();
        let p_h = Tensor::<Ext>::rand(&mut rng, [1 << num_variables]);
        let q_h = Tensor::<Ext>::rand(&mut rng, [1 << num_variables]);

        let p_h_mle = Mle::new(p_h.clone().reshape([1 << num_variables, 1]));
        let q_h_mle = Mle::new(q_h.clone().reshape([1 << num_variables, 1]));

        let point = Point::rand(&mut rng, num_fix_variables);

        let mut p_h_fixed = p_h_mle;
        let mut q_h_fixed = q_h_mle;
        for alpha in point.iter().rev() {
            p_h_fixed = p_h_fixed.fix_last_variable(*alpha).await;
            q_h_fixed = q_h_fixed.fix_last_variable(*alpha).await;
        }

        let (eval_0, eval_half) = sum_as_poly_in_last_variable(&p_h_fixed, &q_h_fixed);

        csl_cuda::spawn(move |t| async move {
            let p_d = p_h.into_device_in(&t).await.unwrap();
            let q_d = q_h.into_device_in(&t).await.unwrap();

            let hadamard = Hadamard::new(p_d, q_d);

            let round_params = RoundParams {
                fix_group: 4,
                fix_tile: 64,
                sum_group: 2,
                num_points: 2,
                store_restricted: true,
            };

            populate_restrict_eq(&point, &t).unwrap();
            let (result, restricted) = hadamard.round(&round_params).await;

            let Hadamard { p, q } = restricted.unwrap();

            let p_d_fixed = p.into_host().await.unwrap();
            let q_d_fixed = q.into_host().await.unwrap();

            let eval_0_d = *result[0];
            let eval_half_d = *result[1];

            assert_eq!(p_d_fixed.as_buffer().len(), p_h_fixed.guts().as_buffer().len());

            for (p_d, p_h) in p_d_fixed.as_buffer().iter().zip(p_h_fixed.guts().as_buffer().iter())
            {
                assert_eq!(*p_d, *p_h);
            }

            assert_eq!(q_d_fixed.as_buffer().len(), q_h_fixed.guts().as_buffer().len());
            for (q_d, q_h) in q_d_fixed.as_buffer().iter().zip(q_h_fixed.guts().as_buffer().iter())
            {
                assert_eq!(*q_d, *q_h);
            }

            // Compare the evals
            assert_eq!(eval_half_d, eval_half);
            assert_eq!(eval_0_d, eval_0);
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    #[ignore]
    async fn bench_look_ahead_skip() {
        // let num_fix_variables = 0;

        let mut rng = rand::thread_rng();

        for num_variables in [10, 16, 20, 24, 26, 26, 27, 27, 27] {
            println!("Num variables: {num_variables}");
            for (fix_tile, fix_group, sum_group, num_points, store_restricted) in [
                (64, 1, 2, 2, false),
                (128, 1, 4, 8, false),
                (64, 2, 2, 2, true),
                (64, 4, 2, 2, true),
            ] {
                let p_h = Tensor::<Ext>::rand(&mut rng, [1 << num_variables]);
                let q_h = Tensor::<Ext>::rand(&mut rng, [1 << num_variables]);

                // let point = Point::rand(&mut rng, num_fix_variables);

                csl_cuda::spawn(move |t| async move {
                    let p_d = p_h.into_device_in(&t).await.unwrap();
                    let q_d = q_h.into_device_in(&t).await.unwrap();

                    let hadamard = Hadamard::new(p_d, q_d);

                    let round_params = RoundParams {
                        fix_group,
                        fix_tile,
                        sum_group,
                        num_points,
                        store_restricted,
                    };
                    // populate_restrict_eq(&point, &t).unwrap();
                    t.synchronize().await.unwrap();
                    let time = tokio::time::Instant::now();
                    let (result, _) = hadamard.round(&round_params).await;
                    t.synchronize().await.unwrap();
                    let duration = time.elapsed();
                    println!(
                        "Time taken for 
                        fix_tile={fix_tile}, 
                        fix_group={fix_group}, 
                        sum_group={sum_group}, 
                        num_points={num_points}: {duration:?}"
                    );

                    drop(result);
                })
                .await
                .unwrap();
            }
            println!("--------------------------------");
        }
    }
}
