use std::ops::{Deref, DerefMut};

use csl_cuda::{cuda_memory_info, TaskScope};
use sp1_core_executor::SplitOpts;
use sp1_prover::{components::SP1ProverComponents, local::LocalProverOpts, SP1ProverBuilder};
use sp1_stark::prover::ProverSemaphore;

use crate::{new_cuda_prover_sumcheck_eval, CudaSP1ProverComponents};

pub struct SP1CudaProverBuilder {
    inner: SP1ProverBuilder<CudaSP1ProverComponents>,
}

impl SP1CudaProverBuilder {
    pub fn new(scope: TaskScope) -> Self {
        // Convert bytes to GB.
        let gb = 1024.0 * 1024.0 * 1024.0;

        // Get the amount of memory on CPU.
        let cpu_memory_gb: usize =
            ((sysinfo::System::new_all().total_memory() as f64) / gb).ceil() as usize;

        // Get the amount of memory on the GPU.
        let gpu_memory_gb: usize =
            (((cuda_memory_info().unwrap().1 as f64) / gb).ceil() as usize) + 4;

        let prover_permits = ProverSemaphore::new(1);

        if gpu_memory_gb < 24 {
            panic!("Unsupported GPU memory: {}, must be at least 24GB", gpu_memory_gb);
        }

        let mut num_prover_workers = 4;

        let core_verifier = CudaSP1ProverComponents::core_verifier();
        let core_prover =
            new_cuda_prover_sumcheck_eval(core_verifier.shard_verifier().clone(), scope.clone());

        let recursion_verifier = CudaSP1ProverComponents::recursion_verifier();
        let recursion_prover = new_cuda_prover_sumcheck_eval(
            recursion_verifier.shard_verifier().clone(),
            scope.clone(),
        );

        if cpu_memory_gb <= 20 {
            num_prover_workers = 2;
        }

        // Log the memory on CPU and GPU.
        tracing::info!("cpu_memory_gb={}, gpu_memory_gb={}", cpu_memory_gb, gpu_memory_gb);

        let num_core_workers = num_prover_workers;
        let num_recursion_workers = num_prover_workers;

        let core_prover_permit = prover_permits.clone();
        let recursion_prover_permit = prover_permits.clone();
        let recursion_program_cache_size = 0;

        let inner = SP1ProverBuilder::new_single_permit(
            core_prover,
            core_prover_permit,
            num_core_workers,
            recursion_prover,
            recursion_prover_permit,
            num_recursion_workers,
            recursion_program_cache_size,
        );

        Self { inner }
    }
}

impl Deref for SP1CudaProverBuilder {
    type Target = SP1ProverBuilder<CudaSP1ProverComponents>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for SP1CudaProverBuilder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

pub fn local_gpu_opts() -> LocalProverOpts {
    let mut opts = LocalProverOpts::default();

    let log2_shard_size = 24;
    opts.core_opts.shard_size = 1 << log2_shard_size;
    opts.core_opts.shard_batch_size = 1;

    let log2_deferred_threshold = 14;
    opts.core_opts.split_opts = SplitOpts::new(1 << log2_deferred_threshold);

    opts.records_capacity_buffer = 1;
    opts.num_record_workers = 4;

    opts
}
