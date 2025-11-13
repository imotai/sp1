use std::env;

use sp1_core_executor::SP1CoreOpts;

use crate::worker::{
    SP1ControllerConfig, SP1CoreProverConfig, SP1ProverConfig, SP1RecursionProverConfig,
};

#[derive(Clone)]
pub struct SP1WorkerConfig {
    pub controller_config: SP1ControllerConfig,
    pub prover_config: SP1ProverConfig,
}

impl Default for SP1WorkerConfig {
    fn default() -> Self {
        // Build the core config using data from environment or default values.
        //
        // TODO: base default values on system information.

        // Build the controller config.
        let num_splicing_workers = env::var("SP1_WORKER_NUM_SPLICING_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NUM_SPLICING_WORKERS);
        let splicing_buffer_size = env::var("SP1_WORKER_SPLICING_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_SPLICING_BUFFER_SIZE);
        let max_reduce_arity = env::var("SP1_WORKER_MAX_REDUCE_ARITY")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_MAX_REDUCE_ARITY);
        let number_of_send_splice_workers_per_splice =
            env::var("SP1_WORKER_NUMBER_OF_SEND_SPLICE_WORKERS_PER_SPLICE")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(DEFAULT_NUMBER_OF_SEND_SPLICE_WORKERS_PER_SPLICE);
        let send_splice_input_buffer_size_per_splice =
            env::var("SP1_WORKER_SEND_SPLICE_INPUT_BUFFER_SIZE_PER_SPLICE")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(DEFAULT_SEND_SPLICE_INPUT_BUFFER_SIZE_PER_SPLICE);
        // Set the default global memory buffer size to twice the number of splicing workers, this means
        // no worker will be blocked when emitting global memory shards.
        let global_memory_buffer_size = env::var("SP1_WORKER_GLOBAL_MEMORY_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(2 * num_splicing_workers);

        // Use default core options as a starting point.
        let opts = SP1CoreOpts::default();
        let controller_config = SP1ControllerConfig {
            opts,
            num_splicing_workers,
            splicing_buffer_size,
            max_reduce_arity,
            number_of_send_splice_workers_per_splice,
            send_splice_input_buffer_size_per_splice,
            global_memory_buffer_size,
        };

        // Build the core prover config.
        let num_trace_executor_workers = env::var("SP1_WORKER_NUM_TRACE_EXECUTOR_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NUM_TRACE_EXECUTOR_WORKERS);
        let trace_executor_buffer_size = env::var("SP1_WORKER_TRACE_EXECUTOR_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_TRACE_EXECUTOR_BUFFER_SIZE);
        let num_core_prover_workers = env::var("SP1_WORKER_NUM_CORE_PROVER_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NUM_CORE_PROVER_WORKERS);
        let core_prover_buffer_size = env::var("SP1_WORKER_CORE_PROVER_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_CORE_PROVER_BUFFER_SIZE);
        let num_setup_workers = env::var("SP1_WORKER_NUM_SETUP_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NUM_SETUP_WORKERS);
        let setup_buffer_size = env::var("SP1_WORKER_SETUP_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_SETUP_BUFFER_SIZE);
        let normalize_program_cache_size = env::var("SP1_WORKER_NORMALIZE_PROGRAM_CACHE_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NORMALIZE_PROGRAM_CACHE_SIZE);

        let core_prover_config = SP1CoreProverConfig {
            num_trace_executor_workers,
            trace_executor_buffer_size,
            num_core_prover_workers,
            core_prover_buffer_size,
            num_setup_workers,
            setup_buffer_size,
            normalize_program_cache_size,
        };

        // Build the recursion prover config.
        let num_prepare_reduce_workers = env::var("SP1_WORKER_NUM_PREPARE_REDUCE_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NUM_PREPARE_REDUCE_WORKERS);
        let prepare_reduce_buffer_size = env::var("SP1_WORKER_PREPARE_REDUCE_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_PREPARE_REDUCE_BUFFER_SIZE);
        let num_recursion_executor_workers = env::var("SP1_WORKER_NUM_RECURSION_EXECUTOR_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NUM_RECURSION_EXECUTOR_WORKERS);
        let recursion_executor_buffer_size = env::var("SP1_WORKER_RECURSION_EXECUTOR_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_RECURSION_EXECUTOR_BUFFER_SIZE);
        let num_recursion_prover_workers = env::var("SP1_WORKER_NUM_RECURSION_PROVER_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NUM_RECURSION_PROVER_WORKERS);
        let recursion_prover_buffer_size = env::var("SP1_WORKER_RECURSION_PROVER_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_RECURSION_PROVER_BUFFER_SIZE);
        let max_compose_arity = env::var("SP1_WORKER_MAX_COMPOSE_ARITY")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_MAX_COMPOSE_ARITY);
        let vk_verification = env::var("SP1_WORKER_VK_VERIFICATION")
            .ok()
            .and_then(|s| s.parse::<bool>().ok())
            .unwrap_or(DEFAULT_VK_VERIFICATION);
        let recursion_prover_config = SP1RecursionProverConfig {
            num_prepare_reduce_workers,
            prepare_reduce_buffer_size,
            num_recursion_executor_workers,
            recursion_executor_buffer_size,
            num_recursion_prover_workers,
            recursion_prover_buffer_size,
            max_compose_arity,
            vk_verification,
        };

        let verify_intermediates = env::var("SP1_WORKER_VERIFY_INTERMEDIATES")
            .ok()
            .and_then(|s| s.parse::<bool>().ok())
            .unwrap_or(true);
        let use_fixed_pk = env::var("SP1_WORKER_USE_FIXED_PK")
            .ok()
            .and_then(|s| s.parse::<bool>().ok())
            .unwrap_or(false);

        let prover_config = SP1ProverConfig {
            core_prover_config,
            recursion_prover_config,
            verify_intermediates,
            use_fixed_pk,
        };

        // Get the local node config from parts above.
        SP1WorkerConfig { controller_config, prover_config }
    }
}

// Default values for the controller config.
pub(crate) const DEFAULT_NUM_SPLICING_WORKERS: usize = 2;
pub(crate) const DEFAULT_SPLICING_BUFFER_SIZE: usize = 2;
pub(crate) const DEFAULT_MAX_REDUCE_ARITY: usize = 4;
pub(crate) const DEFAULT_NUMBER_OF_SEND_SPLICE_WORKERS_PER_SPLICE: usize = 2;
pub(crate) const DEFAULT_SEND_SPLICE_INPUT_BUFFER_SIZE_PER_SPLICE: usize = 2;

// Default values for the core prover config.
pub(crate) const DEFAULT_NUM_TRACE_EXECUTOR_WORKERS: usize = 4;
pub(crate) const DEFAULT_TRACE_EXECUTOR_BUFFER_SIZE: usize = 4;
pub(crate) const DEFAULT_NUM_CORE_PROVER_WORKERS: usize = 4;
pub(crate) const DEFAULT_CORE_PROVER_BUFFER_SIZE: usize = 4;
pub(crate) const DEFAULT_NUM_SETUP_WORKERS: usize = 2;
pub(crate) const DEFAULT_SETUP_BUFFER_SIZE: usize = 2;
pub(crate) const DEFAULT_NORMALIZE_PROGRAM_CACHE_SIZE: usize = 5;
pub(crate) const DEFAULT_MAX_COMPOSE_ARITY: usize = 4;

// Default values for the recursion prover config.
pub(crate) const DEFAULT_NUM_PREPARE_REDUCE_WORKERS: usize = DEFAULT_NUM_RECURSION_EXECUTOR_WORKERS;
pub(crate) const DEFAULT_PREPARE_REDUCE_BUFFER_SIZE: usize = DEFAULT_RECURSION_EXECUTOR_BUFFER_SIZE;
pub(crate) const DEFAULT_NUM_RECURSION_EXECUTOR_WORKERS: usize = 4;
pub(crate) const DEFAULT_RECURSION_EXECUTOR_BUFFER_SIZE: usize = 4;
pub(crate) const DEFAULT_NUM_RECURSION_PROVER_WORKERS: usize = 8;
pub(crate) const DEFAULT_RECURSION_PROVER_BUFFER_SIZE: usize = 8;
pub(crate) const DEFAULT_VK_VERIFICATION: bool = false;
