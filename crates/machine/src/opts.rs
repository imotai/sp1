use csl_cuda::cuda_memory_info;
use sp1_stark::SP1ProverOpts;

/// Get the optimal options for the GPU prover automatically.
pub fn gpu_prover_opts() -> SP1ProverOpts {
    // Convert bytes to GB.
    let gb = 1024.0 * 1024.0 * 1024.0;

    // Get the amount of memory on CPU.
    let cpu_memory_gb: usize =
        ((sysinfo::System::new_all().total_memory() as f64) / gb).ceil() as usize;

    // Get the amount of memory on the GPU.
    let gpu_memory_gb: usize = (((cuda_memory_info().unwrap().1 as f64) / gb).ceil() as usize) + 4;

    // Log the memory on CPU and GPU.
    tracing::info!("cpu_memory_gb={}, gpu_memory_gb={}", cpu_memory_gb, gpu_memory_gb);

    SP1ProverOpts::gpu(cpu_memory_gb, gpu_memory_gb)
}
