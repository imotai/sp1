use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use csl_cuda::{cuda_memory_info, TaskScope};

use sp1_core_executor::ELEMENT_THRESHOLD;
use sp1_hypercube::prover::ProverSemaphore;
use sp1_prover::{
    components::SP1ProverComponents, core::CORE_LOG_STACKING_HEIGHT, local::LocalProverOpts,
    worker::SP1WorkerBuilder, SP1ProverBuilder,
};

pub const RECURSION_TRACE_ALLOCATION: usize = 1 << 27;
pub const SHRINK_TRACE_ALLOCATION: usize = 1 << 25;

/// Taken from "Total number of Cells" when generating traces for wrap. Plus an extra 5%.
pub const WRAP_TRACE_ALLOCATION: usize = 85_376_340;

use crate::{new_cuda_prover, SP1CudaProverComponents};

pub fn local_gpu_opts() -> LocalProverOpts {
    let mut opts = LocalProverOpts::default();

    let log2_shard_size = 24;
    opts.core_opts.shard_size = 1 << log2_shard_size;
    opts.num_record_workers = 4;

    let gb = 1024.0 * 1024.0 * 1024.0;

    // Get the amount of memory on the GPU.
    let gpu_memory_gb: usize = (((cuda_memory_info().unwrap().1 as f64) / gb).ceil() as usize) + 4;

    if gpu_memory_gb < 24 {
        panic!("Unsupported GPU memory: {gpu_memory_gb}, must be at least 24GB");
    }

    let shard_threshold = if gpu_memory_gb <= 30 {
        ELEMENT_THRESHOLD - (1 << 27) - (1 << 26)
    } else {
        ELEMENT_THRESHOLD
    };

    println!("Shard threshold: {shard_threshold}");
    opts.core_opts.sharding_threshold.element_threshold = shard_threshold;

    opts.core_opts.global_dependencies_opt = true;

    opts
}

pub struct SP1CudaProverBuilder {
    inner: SP1ProverBuilder<SP1CudaProverComponents>,
}

impl SP1CudaProverBuilder {
    /// The core trace area allocated for all traces. A buffer of this size will be allocated and reused for all
    /// core traces.
    pub fn num_elts() -> usize {
        // Convert bytes to GB.
        let gb = 1024.0 * 1024.0 * 1024.0;

        // Get the amount of memory on the GPU.
        let gpu_memory_gb: usize =
            (((cuda_memory_info().unwrap().1 as f64) / gb).ceil() as usize) + 4;

        if gpu_memory_gb <= 30 {
            (ELEMENT_THRESHOLD - (1 << 26)) as usize
        } else {
            ELEMENT_THRESHOLD as usize
        }
    }

    pub async fn new(scope: TaskScope) -> Self {
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

        if cpu_memory_gb <= 20 {
            num_prover_workers = 1;
        }

        let num_core_workers = num_prover_workers;
        let num_recursion_workers = num_prover_workers;
        let num_shrink_workers = num_prover_workers;
        let num_wrap_workers = num_prover_workers;

        let num_elts = Self::num_elts() + (1 << CORE_LOG_STACKING_HEIGHT);

        let core_verifier = SP1CudaProverComponents::core_verifier();
        let core_prover =
            new_cuda_prover(core_verifier.clone(), num_elts, num_core_workers, scope.clone()).await;

        // TODO: tune this more precisely and make it a constant.
        let recursion_verifier = SP1CudaProverComponents::compress_verifier();
        let recursion_prover = new_cuda_prover(
            recursion_verifier.clone(),
            RECURSION_TRACE_ALLOCATION,
            num_recursion_workers,
            scope.clone(),
        )
        .await;

        let shrink_verifier = SP1CudaProverComponents::shrink_verifier();
        let shrink_prover = new_cuda_prover(
            shrink_verifier.clone(),
            SHRINK_TRACE_ALLOCATION,
            num_shrink_workers,
            scope.clone(),
        )
        .await;

        let wrap_verifier = SP1CudaProverComponents::wrap_verifier();
        let wrap_prover = new_cuda_prover(
            wrap_verifier.clone(),
            WRAP_TRACE_ALLOCATION,
            num_wrap_workers,
            scope.clone(),
        )
        .await;

        // Log the memory on CPU and GPU.
        tracing::debug!("cpu_memory_gb={}, gpu_memory_gb={}", cpu_memory_gb, gpu_memory_gb);

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
    type Target = SP1ProverBuilder<SP1CudaProverComponents>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for SP1CudaProverBuilder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

/// Create a [SP1CudaProverWorkerBuilder]
pub async fn cuda_worker_builder(scope: TaskScope) -> SP1WorkerBuilder<SP1CudaProverComponents> {
    // Create a prover permits, assuming a single proof happens at a time.
    let prover_permits = ProverSemaphore::new(1);

    // Get the core options.
    let opts = local_gpu_opts().core_opts;

    let num_elts = SP1CudaProverBuilder::num_elts() + (1 << CORE_LOG_STACKING_HEIGHT);

    let core_verifier = SP1CudaProverComponents::core_verifier();
    let core_prover =
        Arc::new(new_cuda_prover(core_verifier.clone(), num_elts, 4, scope.clone()).await);

    // TODO: tune this more precisely and make it a constant.
    let recursion_verifier = SP1CudaProverComponents::compress_verifier();
    let recursion_prover = Arc::new(
        new_cuda_prover(recursion_verifier.clone(), RECURSION_TRACE_ALLOCATION, 4, scope.clone())
            .await,
    );

    let shrink_verifier = SP1CudaProverComponents::shrink_verifier();
    let shrink_prover = Arc::new(
        new_cuda_prover(shrink_verifier.clone(), SHRINK_TRACE_ALLOCATION, 1, scope.clone()).await,
    );

    let wrap_verifier = SP1CudaProverComponents::wrap_verifier();
    let wrap_prover = Arc::new(
        new_cuda_prover(wrap_verifier.clone(), WRAP_TRACE_ALLOCATION, 1, scope.clone()).await,
    );

    SP1WorkerBuilder::new()
        .with_core_opts(opts)
        .with_core_air_prover(core_prover, prover_permits.clone())
        .with_compress_air_prover(recursion_prover, prover_permits.clone())
        .with_shrink_air_prover(shrink_prover, prover_permits.clone())
        .with_wrap_air_prover(wrap_prover, prover_permits)
}
