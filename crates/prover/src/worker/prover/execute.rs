use futures::{stream::FuturesUnordered, StreamExt};
use slop_futures::pipeline::{AsyncEngine, AsyncWorker, Pipeline, SubmitHandle};
use sp1_core_executor::{
    ExecutionError, ExecutionReport, GasEstimatingVM, MinimalExecutor, Program, SP1CoreOpts,
};
use sp1_core_machine::io::SP1Stdin;
use sp1_hypercube::air::PROOF_NONCE_NUM_WORDS;
use sp1_jit::TraceChunkRaw;
use sp1_primitives::io::SP1PublicValues;
use std::sync::Arc;
use tracing::Instrument;

use crate::worker::{DEFAULT_GAS_EXECUTOR_BUFFER_SIZE, DEFAULT_NUM_GAS_EXECUTOR_WORKERS};

/// Configuration for the executor.
#[derive(Debug, Clone)]
pub struct SP1ExecutorConfig {
    /// The number of gas executors.
    pub num_gas_executors: usize,
    /// The buffer size for the gas executor.
    pub gas_executor_buffer_size: usize,
}

impl Default for SP1ExecutorConfig {
    fn default() -> Self {
        let num_gas_executors = std::env::var("SP1_WORKER_NUMBER_OF_GAS_EXECUTORS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_NUM_GAS_EXECUTOR_WORKERS);
        let gas_executor_buffer_size = std::env::var("SP1_WORKER_GAS_EXECUTOR_BUFFER_SIZE")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_GAS_EXECUTOR_BUFFER_SIZE);
        Self { num_gas_executors, gas_executor_buffer_size }
    }
}

pub fn initialize_gas_engine(
    config: &SP1ExecutorConfig,
    program: Arc<Program>,
    nonce: [u32; PROOF_NONCE_NUM_WORDS],
    opts: SP1CoreOpts,
    calculate_gas: bool,
) -> GasExecutingEngine {
    let workers = (0..config.num_gas_executors)
        .map(|_| GasExecutingWorker::new(program.clone(), nonce, opts.clone(), calculate_gas))
        .collect();
    AsyncEngine::new(workers, config.gas_executor_buffer_size)
}

pub type GasExecutingEngine =
    AsyncEngine<GasExecutingTask, Result<ExecutionReport, ExecutionError>, GasExecutingWorker>;

/// A task for gas estimation on a trace chunk.
pub struct GasExecutingTask {
    pub chunk: TraceChunkRaw,
}

#[derive(Debug, Clone)]
pub struct GasExecutingWorker {
    program: Arc<Program>,
    nonce: [u32; PROOF_NONCE_NUM_WORDS],
    opts: SP1CoreOpts,
    calculate_gas: bool,
}

impl GasExecutingWorker {
    pub fn new(
        program: Arc<Program>,
        nonce: [u32; PROOF_NONCE_NUM_WORDS],
        opts: SP1CoreOpts,
        calculate_gas: bool,
    ) -> Self {
        Self { program, nonce, opts, calculate_gas }
    }
}

impl AsyncWorker<GasExecutingTask, Result<ExecutionReport, ExecutionError>> for GasExecutingWorker {
    async fn call(&self, input: GasExecutingTask) -> Result<ExecutionReport, ExecutionError> {
        if !self.calculate_gas {
            return Ok(ExecutionReport::default());
        }
        let mut gas_estimating_vm =
            GasEstimatingVM::new(&input.chunk, self.program.clone(), self.nonce, self.opts.clone());
        let report = gas_estimating_vm.execute()?;
        Ok(report)
    }
}

pub async fn execute_with_optional_gas(
    program: Arc<Program>,
    stdin: SP1Stdin,
    nonce: [u32; PROOF_NONCE_NUM_WORDS],
    calculate_gas: bool,
    opts: SP1CoreOpts,
    executor_config: SP1ExecutorConfig,
) -> anyhow::Result<(SP1PublicValues, [u8; 32], ExecutionReport)> {
    let minimal_trace_chunk_threshold =
        if calculate_gas { Some(opts.minimal_trace_chunk_threshold) } else { None };
    let gas_engine =
        initialize_gas_engine(&executor_config, program.clone(), nonce, opts, calculate_gas);

    let mut minimal_executor =
        MinimalExecutor::new(program.clone(), false, minimal_trace_chunk_threshold);

    // Feed stdin buffers to the executor
    for buf in stdin.buffer {
        minimal_executor.with_input(&buf);
    }

    // Execute the program to completion, collecting all trace chunks
    let (handle_sender, mut handle_receiver) = tokio::sync::mpsc::unbounded_channel();

    // The return values of the two tasks in the join set.
    enum ExecutorOutput {
        Report(ExecutionReport),
        PublicValues(SP1PublicValues),
    }

    let mut join_set = tokio::task::JoinSet::new();
    // Spawn a task that runs gas executors.
    join_set.spawn(async move {
        let mut report = ExecutionReport::default();
        let mut gas_handles: FuturesUnordered<SubmitHandle<GasExecutingEngine>> =
            FuturesUnordered::new();
        loop {
            tokio::select! {
                Some(result) = handle_receiver.recv() => {
                    let gas_handles_len = gas_handles.len();
                    tracing::debug!(num_gas_handles = %gas_handles_len, "Received gas handle");
                    gas_handles.push(result);

                }
                Some(result) = gas_handles.next() => {
                    let chunk_report: ExecutionReport = result.map_err(|e| anyhow::anyhow!("gas task panicked: {}", e))??;
                    let gas_handles_len = gas_handles.len();
                    tracing::debug!(num_gas_handles = %gas_handles_len, "Gas task finished.");
                    report += chunk_report;
                }

                else => {
                    tracing::debug!("No more gas handles to receive");
                    break;
                }
            }
        }
        while let Some(result) = gas_handles.next().await {
            let chunk_report = result.map_err(|e| anyhow::anyhow!("gas task panicked: {}", e))??;
            report += chunk_report;
        }
        Ok::<_, anyhow::Error>(ExecutorOutput::Report(report))
    }.instrument(tracing::debug_span!("report_accumulator")));

    // Spawn a blocking task to run the minimal executor.
    join_set.spawn_blocking(move || {
        while let Some(chunk) = minimal_executor.execute_chunk() {
            let handle = gas_engine
                .blocking_submit(GasExecutingTask { chunk })
                .map_err(|e| anyhow::anyhow!("Gas engine submission failed: {}", e))?;
            handle_sender.send(handle)?;
        }
        tracing::debug!("Minimal executor finished in {} cycles", minimal_executor.global_clk());
        let public_value_stream = minimal_executor.into_public_values_stream();
        let public_values = SP1PublicValues::from(&public_value_stream);

        tracing::info!("public_value_stream: {:?}", public_value_stream);
        Ok::<_, anyhow::Error>(ExecutorOutput::PublicValues(public_values))
    });

    // Wait for all gas calculations to complete.
    let mut final_report = ExecutionReport::default();
    let mut public_values = SP1PublicValues::default();
    while let Some(result) = join_set.join_next().await {
        let output = result??;
        match output {
            ExecutorOutput::PublicValues(pv) => public_values = pv,
            ExecutorOutput::Report(report) => final_report = report,
        }
    }

    // TODO: hash the public values.
    Ok((public_values, [0u8; 32], final_report))
}
