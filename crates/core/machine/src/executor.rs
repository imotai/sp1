use std::{
    io,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

use futures::future::try_join_all;
use hashbrown::HashSet;
use slop_algebra::PrimeField32;
use slop_futures::queue::{TryAcquireWorkerError, WorkerQueue};
use sp1_core_executor::{
    events::MemoryRecord, CompressedMemory, ExecutionError, ExecutionRecord, Program, SP1Context,
    SP1CoreOpts, SplicingVM, SplitOpts, TracingVM,
};
use sp1_hypercube::{
    prover::{MemoryPermit, MemoryPermitting},
    Machine, MachineRecord,
};
use sp1_jit::MinimalTrace;
use thiserror::Error;
use tokio::sync::{
    mpsc::{self, UnboundedSender},
    TryAcquireError,
};
use tracing::{Instrument, Level};

use sp1_core_executor::{CycleResult, MinimalExecutor, SplicedMinimalTrace, TraceChunkRaw};

use crate::{io::SP1Stdin, riscv::RiscvAir};

pub struct MachineExecutor<F: PrimeField32> {
    num_record_workers: usize,
    opts: SP1CoreOpts,
    machine: Machine<F, RiscvAir<F>>,
    memory: MemoryPermitting,
    _marker: std::marker::PhantomData<F>,
}

impl<F: PrimeField32> MachineExecutor<F> {
    pub fn new(record_buffer_size: u64, num_record_workers: usize, opts: SP1CoreOpts) -> Self {
        let machine = RiscvAir::<F>::machine();

        Self {
            num_record_workers,
            opts,
            machine,
            memory: MemoryPermitting::new(record_buffer_size),
            _marker: PhantomData,
        }
    }

    /// Get a reference to the core options.
    pub fn opts(&self) -> &SP1CoreOpts {
        &self.opts
    }

    pub async fn execute(
        &self,
        program: Arc<Program>,
        stdin: SP1Stdin,
        _context: SP1Context<'static>,
        record_tx: mpsc::UnboundedSender<(ExecutionRecord, Option<MemoryPermit>)>,
    ) -> Result<ExecutionOutput, MachineExecutorError> {
        let (chunk_tx, mut chunk_rx) = mpsc::unbounded_channel::<TraceChunkRaw>();
        let (last_record_tx, mut last_record_rx) =
            tokio::sync::mpsc::channel::<(ExecutionRecord, [MemoryRecord; 32])>(1);
        let deferred = Arc::new(Mutex::new(ExecutionRecord::new(program.clone())));
        let mut record_worker_channels = Vec::with_capacity(self.num_record_workers);

        // todo: use page protection
        let split_opts = SplitOpts::new(&self.opts, program.instructions.len(), false);

        tracing::debug!("starting {} record worker channels", self.num_record_workers);
        let mut handles = Vec::with_capacity(self.num_record_workers);
        for i in 0..self.num_record_workers {
            let (tx, mut rx) = mpsc::unbounded_channel::<RecordTask>();
            record_worker_channels.push(tx);
            let machine = self.machine.clone();
            let opts = self.opts.clone();
            let program = program.clone();
            let record_tx = record_tx.clone();
            let deferred: Arc<Mutex<ExecutionRecord>> = deferred.clone();
            let last_record_tx = last_record_tx.clone();
            let permitting = self.memory.clone();

            handles.push(tokio::task::spawn(
                async move {
                    while let Some(task) = rx.recv().await {
                        let RecordTask { chunk } = task;
                        tracing::trace!("tracing chunk with worker: {}", i);

                        // Assume a record is 4Gb for now.
                        let permit = permitting.acquire(2 * 1024 * 1024 * 1024).await.unwrap();

                        let program = program.clone();
                        let opts = opts.clone();
                        let (done, mut record, registers) = tokio::task::spawn_blocking({
                            let program = program.clone();
                            let opts = opts.clone();

                            move || {
                                let _debug_span =
                                    tracing::trace_span!("tracing chunk blocking task").entered();
                                trace_chunk(program, opts, chunk)
                            }
                        })
                        .await
                        .expect("error: trace chunk task panicked")
                        .expect("todo: handle error");

                        if done {
                            tracing::trace!("last record");

                            // If this is the last record, we have special handling for the memory
                            // events.
                            last_record_tx.send((record, registers)).await.unwrap();
                        } else {
                            tracing::trace!("defferring record");

                            let deferred_records = {
                                let mut deferred = deferred.lock().unwrap();
                                defer::<F>(&mut record, &mut deferred, &split_opts, opts, done)
                            };
                            start_prove(
                                machine.clone(),
                                record_tx.clone(),
                                Some(permit),
                                record,
                                deferred_records,
                            )
                            .await;
                        };
                    }

                    tracing::trace!("tracing worker finished");
                }
                .instrument(tracing::debug_span!("tracing worker")),
            ));
        }

        let record_worker_channels = Arc::new(WorkerQueue::new(record_worker_channels));

        let minimal_executor_handle = tokio::task::spawn_blocking({
            let program = program.clone();

            move || {
                let _debug_span = tracing::debug_span!("minimal executor task").entered();
                let mut minimal_executor = MinimalExecutor::tracing(program.clone(), None);

                for buf in stdin.buffer {
                    minimal_executor.with_input(&buf);
                }

                tracing::trace!("Starting minimal executor");
                while let Some(chunk) = minimal_executor.execute_chunk() {
                    tracing::trace!("program is done?: {}", minimal_executor.is_done());
                    tracing::trace!(
                        "mem reads chunk size bytes {}",
                        chunk.num_mem_reads() * std::mem::size_of::<sp1_jit::MemValue>() as u64
                    );

                    chunk_tx.send(chunk).unwrap();
                }
                tracing::trace!("minimal executor finished");

                minimal_executor
            }
        });

        let touched_addresses = tokio::task::spawn({
            let program: Arc<Program> = program.clone();
            async move {
                let mut splicing_handles = Vec::new();
                let touched_addresses = Arc::new(Mutex::new(HashSet::new()));
                while let Some(chunk) = chunk_rx.recv().await {
                    let splicing_handle = tokio::task::spawn_blocking({
                        let program = program.clone();
                        let touched_addresses = touched_addresses.clone();
                        let record_worker_channels = record_worker_channels.clone();

                        move || {
                            generate_chunks(
                                program,
                                chunk,
                                record_worker_channels,
                                touched_addresses,
                            )
                        }
                    });

                    splicing_handles.push(splicing_handle);
                }

                try_join_all(splicing_handles).await.expect("error: splicing tasks panicked");

                touched_addresses
            }
            .instrument(tracing::debug_span!("splitting task"))
        })
        .await
        .unwrap();

        // Wait for the minimal executor to finish.
        let minimal_executor = minimal_executor_handle.await.unwrap();
        tracing::debug!("Minimal executor finished");

        // Wait for the record workers to finish.
        try_join_all(handles).await.expect("error: tracing tasks panicked");
        tracing::debug!("All record workers finished");

        // Wait for the last record to be traced.
        let (mut last_record, final_registers) = last_record_rx.recv().await.unwrap();
        tracing::info!(
            "last_record.public_values.committed_value_digest: {:?}",
            last_record.public_values.committed_value_digest
        );
        let deferred_records = tracing::trace_span!("emit globals").in_scope(|| {
            let touched_addresses = std::mem::take(&mut *touched_addresses.lock().unwrap());

            // Insert the global memory events into the last record.
            minimal_executor.emit_globals(&mut last_record, final_registers, touched_addresses);

            let mut deferred = deferred.lock().unwrap();
            // let mut state = state.lock().unwrap();
            tracing::trace_span!("defer last shard").in_scope(|| {
                defer::<F>(&mut last_record, &mut deferred, &split_opts, self.opts.clone(), true)
            })
        });

        start_prove(self.machine.clone(), record_tx, None, last_record, deferred_records).await;

        Ok(ExecutionOutput {
            cycles: minimal_executor.global_clk(),
            public_value_stream: minimal_executor.into_public_values_stream(),
        })
    }
}

#[derive(Error, Debug)]
pub enum MachineExecutorError {
    #[error("Failed to execute program: {0}")]
    ExecutionError(ExecutionError),
    #[error("IO error: {0}")]
    IoError(io::Error),
    #[error("Serialization error: {0}")]
    SerializationError(bincode::Error),
    #[error("Executor is already closed")]
    ExecutorClosed,
    #[error("Task failed: {0:?}")]
    ExecutorPanicked(#[from] tokio::task::JoinError),
    #[error("Failed to send record to prover channel")]
    ProverChannelClosed,
}

/// The output of the machine executor.
pub struct ExecutionOutput {
    pub public_value_stream: Vec<u8>,
    pub cycles: u64,
}

struct RecordTask {
    chunk: SplicedMinimalTrace<TraceChunkRaw>,
}

/// Generate the chunks (corresponding to shards) and send them to the record workers.
#[tracing::instrument(name = "generate_chunks", skip_all)]
fn generate_chunks(
    program: Arc<Program>,
    chunk: TraceChunkRaw,
    record_worker_channels: Arc<WorkerQueue<UnboundedSender<RecordTask>>>,
    all_touched_addresses: Arc<Mutex<HashSet<u64>>>,
) -> Result<(), ExecutionError> {
    let mut touched_addresses = CompressedMemory::new();
    let mut vm = SplicingVM::new(&chunk, program.clone(), &mut touched_addresses);

    let start_num_mem_reads = chunk.num_mem_reads();

    let mut last_splice = SplicedMinimalTrace::new_full_trace(chunk.clone());
    loop {
        tracing::debug!("starting new shard at clk: {} at pc: {}", vm.core.clk(), vm.core.pc());

        match vm.execute().expect("todo: handle result") {
            CycleResult::ShardBoundry => {
                // Note: Chunk implentations should always be cheap to clone.
                if let Some(spliced) = vm.splice(chunk.clone()) {
                    tracing::trace!("shard ended at clk: {}", vm.core.clk());
                    tracing::trace!("shard ended at pc: {}", vm.core.pc());
                    tracing::trace!("shard ended at global clk: {}", vm.core.global_clk());
                    tracing::trace!("shard ended with {} mem reads left ", vm.core.mem_reads.len());

                    // Set the last splice clk.
                    last_splice.set_last_clk(vm.core.clk());
                    last_splice.set_last_mem_reads_idx(
                        start_num_mem_reads as usize - vm.core.mem_reads.len(),
                    );

                    let splice_to_send = std::mem::replace(&mut last_splice, spliced);
                    send_spliced_trace_blocking(record_worker_channels.clone(), splice_to_send);
                } else {
                    tracing::trace!("trace ended at clk: {}", vm.core.clk());
                    tracing::trace!("trace ended at pc: {}", vm.core.pc());
                    tracing::trace!("trace ended at global clk: {}", vm.core.global_clk());
                    tracing::trace!("trace ended with {} mem reads left ", vm.core.mem_reads.len());

                    last_splice.set_last_clk(vm.core.clk());
                    last_splice.set_last_mem_reads_idx(
                        start_num_mem_reads as usize - vm.core.mem_reads.len(),
                    );

                    send_spliced_trace_blocking(record_worker_channels.clone(), last_splice);

                    break;
                }
            }
            CycleResult::Done(true) => {
                last_splice.set_last_clk(vm.core.clk());
                last_splice.set_last_mem_reads_idx(chunk.num_mem_reads() as usize);

                send_spliced_trace_blocking(record_worker_channels.clone(), last_splice);

                break;
            }
            CycleResult::Done(false) | CycleResult::TraceEnd => {
                // Note: Trace ends get mapped to shard boundaries.
                unreachable!("The executor should never return an imcomplete program without a shard boundary");
            }
        }
    }

    // Append the touched addresses from this chunk to the globally tracked touched addresses.
    tracing::trace!("extending all_touched_addresses with touched_addresses");
    all_touched_addresses.lock().unwrap().extend(touched_addresses.is_set().into_iter());

    Ok(())
}

/// Trace a single [`SplicedMinimalTrace`] (corresponding to a shard) and return the execution
/// record.
#[tracing::instrument(
    level = Level::DEBUG,
    name = "trace_chunk",
    skip_all,
)]
fn trace_chunk(
    program: Arc<Program>,
    _opts: SP1CoreOpts,
    chunk: SplicedMinimalTrace<TraceChunkRaw>,
) -> Result<(bool, ExecutionRecord, [MemoryRecord; 32]), ExecutionError> {
    let mut vm = TracingVM::new(&chunk, program);
    let status = vm.execute()?;
    tracing::trace!("chunk ended at clk: {}", vm.core.clk());
    tracing::trace!("chunk ended at pc: {}", vm.core.pc());

    let mut record = std::mem::take(&mut vm.record);
    let pv = record.public_values;

    // Handle the case where `COMMIT` or `COMMIT_DEFERRED_PROOFS` happens across last two shards.
    //
    // todo: does this actually work in the new regieme? what if the shard is stopped due to the clk
    // limit? if so, does that mean this could be wrong? its unclear!
    if status.is_shard_boundry() && (pv.commit_syscall == 1 || pv.commit_deferred_syscall == 1) {
        tracing::trace!("commit syscall or commit deferred proofs across last two shards");

        loop {
            // Execute until we get a done status.
            if vm.execute()?.is_done() {
                let pv = vm.record.public_values;

                // Update the record.
                record.public_values.commit_syscall = 1;
                record.public_values.commit_deferred_syscall = 1;
                record.public_values.committed_value_digest = pv.committed_value_digest;
                record.public_values.deferred_proofs_digest = pv.deferred_proofs_digest;

                break;
            }
        }
    }

    Ok((status.is_done(), record, *vm.core.registers()))
}

#[tracing::instrument(name = "defer", skip_all)]
fn defer<F: PrimeField32>(
    record: &mut ExecutionRecord,
    deferred: &mut ExecutionRecord,
    split_opts: &SplitOpts,
    opts: SP1CoreOpts,
    done: bool,
) -> Vec<ExecutionRecord> {
    let state = &mut record.public_values;
    state.is_execution_shard = 1;

    let initial_timestamp_high = (state.initial_timestamp >> 24) as u32;
    let initial_timestamp_low = (state.initial_timestamp & 0xFFFFFF) as u32;
    let last_timestamp_high = (state.last_timestamp >> 24) as u32;
    let last_timestamp_low = (state.last_timestamp & 0xFFFFFF) as u32;

    state.initial_timestamp_inv = if state.initial_timestamp == 1 {
        0
    } else {
        F::from_canonical_u32(initial_timestamp_high + initial_timestamp_low - 1)
            .inverse()
            .as_canonical_u32()
    };

    state.last_timestamp_inv = F::from_canonical_u32(last_timestamp_high + last_timestamp_low - 1)
        .inverse()
        .as_canonical_u32();

    if initial_timestamp_high == last_timestamp_high {
        state.is_timestamp_high_eq = 1;
    } else {
        state.is_timestamp_high_eq = 0;
        state.inv_timestamp_high = (F::from_canonical_u32(last_timestamp_high)
            - F::from_canonical_u32(initial_timestamp_high))
        .inverse()
        .as_canonical_u32();
    }

    if initial_timestamp_low == last_timestamp_low {
        state.is_timestamp_low_eq = 1;
    } else {
        state.is_timestamp_low_eq = 0;
        state.inv_timestamp_low = (F::from_canonical_u32(last_timestamp_low)
            - F::from_canonical_u32(initial_timestamp_low))
        .inverse()
        .as_canonical_u32();
    }
    state.is_first_execution_shard = (state.initial_timestamp == 1) as u32;

    // Defer events that are too expensive to include in every shard.
    deferred.append(&mut record.defer(&opts.retained_events_presets));

    let can_pack_global_memory = done
        && record.estimated_trace_area <= split_opts.pack_trace_threshold
        && deferred.global_memory_initialize_events.len() <= split_opts.combine_memory_threshold
        && deferred.global_memory_finalize_events.len() <= split_opts.combine_memory_threshold
        && deferred.global_page_prot_initialize_events.len()
            <= split_opts.combine_page_prot_threshold
        && deferred.global_page_prot_finalize_events.len()
            <= split_opts.combine_page_prot_threshold;

    // See if any deferred shards are ready to be committed to.
    let deferred_records = deferred.split(done, record, can_pack_global_memory, split_opts);
    tracing::trace!("split deffered into {} records", deferred_records.len());

    deferred_records
}

/// Generate the dependencies and send the records to the prover channel.
#[tracing::instrument(name = "start_prove", skip_all)]
async fn start_prove<F: PrimeField32>(
    machine: Machine<F, RiscvAir<F>>,
    record_tx: mpsc::UnboundedSender<(ExecutionRecord, Option<MemoryPermit>)>,
    permit: Option<MemoryPermit>,
    mut record: ExecutionRecord,
    mut deferred_records: Vec<ExecutionRecord>,
) {
    tracing::debug!("num deferred records: {:#?}", deferred_records.len());

    // Generate the dependencies.
    tokio::task::spawn_blocking({
        let machine = machine.clone();
        let record_tx = record_tx.clone();

        move || {
            let record_iter = std::iter::once(&mut record);
            machine.generate_dependencies(record_iter, None);
            machine.generate_dependencies(deferred_records.iter_mut(), None);

            // Send the records to the output channel.
            record_tx.send((record, permit)).unwrap();

            // If there are deferred records, send them to the output channel.
            for record in deferred_records {
                record_tx.send((record, None)).unwrap();
            }
        }
    })
    .await
    .expect("failed to send records");
}

/// Send the splice trace to a available record worker.
#[tracing::instrument(name = "send_spliced_trace_blocking", skip_all)]
fn send_spliced_trace_blocking(
    record_worker_channels: Arc<WorkerQueue<UnboundedSender<RecordTask>>>,
    chunk: SplicedMinimalTrace<TraceChunkRaw>,
) {
    loop {
        match record_worker_channels.clone().try_pop() {
            Ok(worker) => {
                worker.send(RecordTask { chunk }).unwrap();
                break;
            }
            Err(TryAcquireWorkerError(TryAcquireError::NoPermits)) => {
                std::hint::spin_loop();
            }
            Err(_) => {
                panic!("failed to send spliced trace to record worker");
            }
        }
    }
}
