use std::ops::{Deref, DerefMut};

use csl_cuda::{cuda_memory_info, TaskScope};
use cslpc_tracegen::CORE_MAX_TRACE_SIZE;
use sp1_hypercube::prover::ProverSemaphore;
use sp1_prover::{
    components::SP1ProverComponents,
    local::LocalProverOpts,
    recursion::{RECURSION_LOG_BLOWUP, SHRINK_LOG_BLOWUP, WRAP_LOG_BLOWUP},
    SP1ProverBuilder, CORE_LOG_BLOWUP,
};

use crate::{
    new_cuda_prover_sumcheck_eval, new_prover_clean_prover, CudaSP1ProverComponents,
    ProverCleanSP1ProverComponents,
};

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
            panic!("Unsupported GPU memory: {gpu_memory_gb}, must be at least 24GB");
        }

        let mut num_prover_workers = 4;

        let core_verifier = CudaSP1ProverComponents::core_verifier();
        let core_prover =
            new_cuda_prover_sumcheck_eval(core_verifier.shard_verifier().clone(), scope.clone());

        let recursion_verifier = CudaSP1ProverComponents::compress_verifier();
        let recursion_prover = new_cuda_prover_sumcheck_eval(
            recursion_verifier.shard_verifier().clone(),
            scope.clone(),
        );

        let shrink_verifier = CudaSP1ProverComponents::shrink_verifier();
        let shrink_prover =
            new_cuda_prover_sumcheck_eval(shrink_verifier.shard_verifier().clone(), scope.clone());

        let wrap_verifier = CudaSP1ProverComponents::wrap_verifier();
        let wrap_prover =
            new_cuda_prover_sumcheck_eval(wrap_verifier.shard_verifier().clone(), scope.clone());

        if cpu_memory_gb <= 20 {
            num_prover_workers = 1;
        }

        // Log the memory on CPU and GPU.
        tracing::info!("cpu_memory_gb={}, gpu_memory_gb={}", cpu_memory_gb, gpu_memory_gb);

        let num_core_workers = num_prover_workers;
        let num_recursion_workers = num_prover_workers;
        let num_shrink_workers = num_prover_workers;
        let num_wrap_workers = num_prover_workers;

        let core_prover_permit = prover_permits.clone();
        let recursion_prover_permit = prover_permits.clone();
        let shrink_prover_permit = prover_permits.clone();
        let wrap_prover_permit = prover_permits.clone();
        let recursion_programs_cache_size = 4;
        let max_reduce_arity = 4;

        let inner = SP1ProverBuilder::new_single_permit(
            core_prover,
            core_prover_permit,
            num_core_workers,
            recursion_prover,
            recursion_prover_permit,
            num_recursion_workers,
            shrink_prover,
            shrink_prover_permit,
            num_shrink_workers,
            wrap_prover,
            wrap_prover_permit,
            num_wrap_workers,
            recursion_programs_cache_size,
            max_reduce_arity,
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
    opts.num_record_workers = 4;

    opts
}

pub struct SP1ProverCleanBuilder {
    inner: SP1ProverBuilder<ProverCleanSP1ProverComponents>,
}

impl SP1ProverCleanBuilder {
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
            panic!("Unsupported GPU memory: {gpu_memory_gb}, must be at least 24GB");
        }

        let mut num_prover_workers = 4;

        let core_verifier = ProverCleanSP1ProverComponents::core_verifier();
        let core_prover = new_prover_clean_prover(
            core_verifier.clone(),
            CORE_MAX_TRACE_SIZE as usize,
            CORE_LOG_BLOWUP,
            scope.clone(),
        );

        // TODO: tune this more precisely and make it a constant.
        let recursion_verifier = ProverCleanSP1ProverComponents::compress_verifier();
        let recursion_prover = new_prover_clean_prover(
            recursion_verifier.clone(),
            1 << 27,
            RECURSION_LOG_BLOWUP,
            scope.clone(),
        );

        let shrink_verifier = ProverCleanSP1ProverComponents::shrink_verifier();
        let shrink_prover = new_prover_clean_prover(
            shrink_verifier.clone(),
            1 << 24,
            SHRINK_LOG_BLOWUP,
            scope.clone(),
        );

        let wrap_verifier = ProverCleanSP1ProverComponents::wrap_verifier();
        let wrap_prover =
            new_prover_clean_prover(wrap_verifier.clone(), 1 << 25, WRAP_LOG_BLOWUP, scope.clone());

        if cpu_memory_gb <= 20 {
            num_prover_workers = 1;
        }

        // Log the memory on CPU and GPU.
        tracing::debug!("cpu_memory_gb={}, gpu_memory_gb={}", cpu_memory_gb, gpu_memory_gb);

        let num_core_workers = num_prover_workers;
        let num_recursion_workers = num_prover_workers;
        let num_shrink_workers = num_prover_workers;
        let num_wrap_workers = num_prover_workers;

        let core_prover_permit = prover_permits.clone();
        let recursion_prover_permit = prover_permits.clone();
        let shrink_prover_permit = prover_permits.clone();
        let wrap_prover_permit = prover_permits.clone();
        let recursion_programs_cache_size = 4;
        let max_reduce_arity = 4;

        let inner = SP1ProverBuilder::new_single_permit(
            core_prover,
            core_prover_permit,
            num_core_workers,
            recursion_prover,
            recursion_prover_permit,
            num_recursion_workers,
            shrink_prover,
            shrink_prover_permit,
            num_shrink_workers,
            wrap_prover,
            wrap_prover_permit,
            num_wrap_workers,
            recursion_programs_cache_size,
            max_reduce_arity,
        );

        Self { inner }
    }
}

impl Deref for SP1ProverCleanBuilder {
    type Target = SP1ProverBuilder<ProverCleanSP1ProverComponents>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for SP1ProverCleanBuilder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
