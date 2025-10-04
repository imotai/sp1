use csl_cuda::{
    args,
    sys::{
        prover_clean::{sum_kernel_ext, sum_kernel_felt, sum_kernel_u32},
        runtime::KernelPtr,
    },
    TaskScope,
};
use slop_alloc::{Buffer, HasBackend};

use crate::config::{Ext, Felt};

fn sum_with_kernel<T>(
    a: &Buffer<T, TaskScope>,
    b: &Buffer<T, TaskScope>,
    ker: KernelPtr,
) -> Buffer<T, TaskScope> {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let mut c = Buffer::<T, TaskScope>::with_capacity_in(n, a.backend().clone());

    unsafe {
        c.assume_init();
        let block_dim = 256;
        let stride = 4;
        let grid_dim = n.div_ceil(block_dim * stride);
        let args = args!(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), n);
        let t = a.backend();
        t.launch_kernel(ker, grid_dim, block_dim, &args, 0).unwrap();
    }
    c
}

pub fn sum_u32(a: &Buffer<u32, TaskScope>, b: &Buffer<u32, TaskScope>) -> Buffer<u32, TaskScope> {
    sum_with_kernel(a, b, unsafe { sum_kernel_u32() })
}

pub fn sum_felt(
    a: &Buffer<Felt, TaskScope>,
    b: &Buffer<Felt, TaskScope>,
) -> Buffer<Felt, TaskScope> {
    sum_with_kernel(a, b, unsafe { sum_kernel_felt() })
}

pub fn sum_ext(a: &Buffer<Ext, TaskScope>, b: &Buffer<Ext, TaskScope>) -> Buffer<Ext, TaskScope> {
    sum_with_kernel(a, b, unsafe { sum_kernel_ext() })
}

#[cfg(test)]
mod tests {
    use rand::Rng;
    use slop_alloc::ToHost;

    use super::*;

    #[tokio::test]
    async fn test_sum_u32() {
        csl_cuda::run_in_place(|t| async move {
            let n = 10000;

            let mut rng = rand::thread_rng();
            let a_h = (0..n).map(|_| rng.gen::<u16>() as u32).collect::<Buffer<_>>();
            let b_h = (0..n).map(|_| rng.gen::<u16>() as u32).collect::<Buffer<_>>();

            let a_d = t.to_device(&a_h).await.unwrap();
            let b_d = t.to_device(&b_h).await.unwrap();

            let c_d = sum_u32(&a_d, &b_d);

            let c_h = c_d.to_host().await.unwrap();

            for ((a, b), c) in a_h.iter().zip(b_h.iter()).zip(c_h.iter()) {
                assert_eq!(a + b, *c);
            }
        })
        .await
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_sum_felt() {
        csl_cuda::run_in_place(|t| async move {
            let n = 10000;

            let mut rng = rand::thread_rng();
            let a_h = (0..n).map(|_| rng.gen::<Felt>()).collect::<Buffer<_>>();
            let b_h = (0..n).map(|_| rng.gen::<Felt>()).collect::<Buffer<_>>();

            let a_d = t.to_device(&a_h).await.unwrap();
            let b_d = t.to_device(&b_h).await.unwrap();

            let c_d = sum_felt(&a_d, &b_d);

            let c_h = c_d.to_host().await.unwrap();

            for ((a, b), c) in a_h.iter().zip(b_h.iter()).zip(c_h.iter()) {
                assert_eq!(*a + *b, *c);
            }
        })
        .await
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn test_sum_ext() {
        csl_cuda::run_in_place(|t| async move {
            let n = 1 << 24;

            let mut rng = rand::thread_rng();
            let a_h = (0..n).map(|_| rng.gen::<Ext>()).collect::<Buffer<_>>();
            let b_h = (0..n).map(|_| rng.gen::<Ext>()).collect::<Buffer<_>>();

            let a_d = t.to_device(&a_h).await.unwrap();
            let b_d = t.to_device(&b_h).await.unwrap();

            let c_d = sum_ext(&a_d, &b_d);

            let c_h = c_d.to_host().await.unwrap();

            for ((a, b), c) in a_h.iter().zip(b_h.iter()).zip(c_h.iter()) {
                assert_eq!(*a + *b, *c);
            }
        })
        .await
        .await
        .unwrap();
    }
}
