use std::env;

use crate::worker::{SP1ControllerConfig, SP1CoreProverConfig, SP1ProverConfig};

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
        let num_splicing_workers = env::var("SP1_LOCAL_NODE_NUM_SPLICING_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NUM_SPLICING_WORKERS);
        let splicing_buffer_size = env::var("SP1_LOCAL_NODE_SPLICING_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_SPLICING_BUFFER_SIZE);
        let controller_config = SP1ControllerConfig { num_splicing_workers, splicing_buffer_size };

        // Build the core prover config.
        let num_trace_executor_workers = env::var("SP1_LOCAL_NODE_NUM_TRACE_EXECUTOR_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NUM_TRACE_EXECUTOR_WORKERS);
        let trace_executor_buffer_size = env::var("SP1_LOCAL_NODE_TRACE_EXECUTOR_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_TRACE_EXECUTOR_BUFFER_SIZE);
        let num_core_prover_workers = env::var("SP1_LOCAL_NODE_NUM_CORE_PROVER_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NUM_CORE_PROVER_WORKERS);
        let core_prover_buffer_size = env::var("SP1_LOCAL_NODE_CORE_PROVER_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_CORE_PROVER_BUFFER_SIZE);
        let num_setup_workers = env::var("SP1_LOCAL_NODE_NUM_SETUP_WORKERS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NUM_SETUP_WORKERS);
        let setup_buffer_size = env::var("SP1_LOCAL_NODE_SETUP_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_SETUP_BUFFER_SIZE);
        let core_prover_config = SP1CoreProverConfig {
            num_trace_executor_workers,
            trace_executor_buffer_size,
            num_core_prover_workers,
            core_prover_buffer_size,
            num_setup_workers,
            setup_buffer_size,
        };

        let prover_config = SP1ProverConfig { core_prover_config };

        // Get the local node config from parts above.
        SP1WorkerConfig { controller_config, prover_config }
    }
}

// Default values for the controller config.
pub(crate) const DEFAULT_NUM_SPLICING_WORKERS: usize = 2;
pub(crate) const DEFAULT_SPLICING_BUFFER_SIZE: usize = 2;

// Default values for the core prover config.
pub(crate) const DEFAULT_NUM_TRACE_EXECUTOR_WORKERS: usize = 4;
pub(crate) const DEFAULT_TRACE_EXECUTOR_BUFFER_SIZE: usize = 4;
pub(crate) const DEFAULT_NUM_CORE_PROVER_WORKERS: usize = 4;
pub(crate) const DEFAULT_CORE_PROVER_BUFFER_SIZE: usize = 4;
pub(crate) const DEFAULT_NUM_SETUP_WORKERS: usize = 2;
pub(crate) const DEFAULT_SETUP_BUFFER_SIZE: usize = 2;
