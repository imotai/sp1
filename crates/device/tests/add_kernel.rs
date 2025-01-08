use std::ffi::c_void;

use csl_device::cuda::{CudaError, DeviceBuffer, TaskScope};
use csl_sys::{algebra::addKernelu32Ptr, runtime::Dim3};
use rand::Rng;

fn launch_add_u32_kernel(
    scope: &TaskScope,
    grid_dim: impl Into<Dim3>,
    block_dim: impl Into<Dim3>,
    a: &DeviceBuffer<u32>,
    b: DeviceBuffer<u32>,
    shared_mem: usize,
) -> Result<DeviceBuffer<u32>, CudaError> {
    assert_eq!(a.len(), b.len());
    let mut c = scope.alloc::<u32>(a.len());

    let args = [
        &(a.as_ptr()) as *const _ as *mut c_void,
        &(b.as_ptr()) as *const _ as *mut c_void,
        &(c.as_mut_ptr()) as *const _ as *mut c_void,
        &a.len() as *const _ as *mut c_void,
    ];
    unsafe {
        c.set_len(a.len());
        scope.launch_kernel(addKernelu32Ptr(), grid_dim, block_dim, &args, shared_mem)?;
    }

    Ok(c)
}

#[tokio::test]
async fn test_async_add_kernel() {
    let mut rng = rand::thread_rng();

    let len = 1024;

    let a_values = (0..len).map(|_| rng.gen_range(0..100)).collect::<Vec<u32>>();
    let b_values = (0..len).map(|_| rng.gen_range(0..100)).collect::<Vec<u32>>();

    csl_device::cuda::task()
        .await
        .unwrap()
        .run(|t| async move {
            let a = DeviceBuffer::from_host_vec(a_values.clone(), t.clone()).await.unwrap();
            let b = DeviceBuffer::from_host_vec(b_values.clone(), t.clone()).await.unwrap();
            let block_dim = 256;
            let grid_dim = (a.len() as u32).div_ceil(block_dim);
            let c = launch_add_u32_kernel(&t, grid_dim, block_dim, &a, b, 0).unwrap();
            let c_host = c.to_vec().await.unwrap();
            assert_eq!(
                c_host,
                a_values.iter().zip(b_values.iter()).map(|(a, b)| a + b).collect::<Vec<_>>()
            );
        })
        .await;
}

#[test]
fn test_add_kernel() {
    let mut rng = rand::thread_rng();

    let len = 1024;

    let a_values = (0..len).map(|_| rng.gen_range(0..100)).collect::<Vec<u32>>();
    let b_values = (0..len).map(|_| rng.gen_range(0..100)).collect::<Vec<u32>>();

    csl_device::cuda::get_task().unwrap().blocking_run(move |t| {
        let a = DeviceBuffer::from_host_slice_blocking(&a_values, t.clone()).unwrap();
        let b = DeviceBuffer::from_host_slice_blocking(&b_values, t.clone()).unwrap();
        let block_dim = 256;
        let grid_dim = (a.len() as u32).div_ceil(block_dim);
        let c = launch_add_u32_kernel(&t, grid_dim, block_dim, &a, b, 0).unwrap();
        let c_host = c.to_vec_blocking().unwrap();
        assert_eq!(
            c_host,
            a_values.iter().zip(b_values.iter()).map(|(a, b)| a + b).collect::<Vec<_>>()
        );
    });
}
