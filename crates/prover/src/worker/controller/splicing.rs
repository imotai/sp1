use std::sync::Arc;

use futures::StreamExt;
use hashbrown::HashMap;
use opentelemetry::Context;
use slop_futures::pipeline::{AsyncEngine, AsyncWorker};
use sp1_core_executor::{
    CompressedMemory, CycleResult, ExecutionError, Program, SP1CoreOpts, SplicedMinimalTrace,
    SplicingVM, SplitOpts,
};
use sp1_hypercube::air::{ShardBoundary, ShardRange};
use sp1_jit::{MinimalTrace, TraceChunkRaw};
use sp1_prover_types::{Artifact, ArtifactClient, TaskStatus, TaskType};
use tokio::{sync::mpsc, task::JoinSet};
use tracing::Instrument;

use crate::worker::{
    DeferredEvents, DeferredMessage, FinalVmState, FinalVmStateLock, ProofData, ProofId,
    ProveShardTaskRequest, RawTaskRequest, RequesterId, SendSpliceTask, SpawnProveOutput, TaskId,
    TouchedAddresses, TraceData, WorkerClient,
};

pub type SplicingEngine<A, W> =
    AsyncEngine<SplicingTask, Result<(), ExecutionError>, SplicingWorker<A, W>>;

/// A task for splicing a trace into single shard chunks.
pub struct SplicingTask {
    pub program: Arc<Program>,
    pub chunk: TraceChunkRaw,
    pub elf_artifact: Artifact,
    pub num_deferred_proofs: usize,
    pub common_input_artifact: Artifact,
    pub all_touched_addresses: TouchedAddresses,
    pub final_vm_state: FinalVmStateLock,
    pub prove_shard_tx: mpsc::UnboundedSender<ProofData>,
    pub proof_id: ProofId,
    pub parent_id: Option<TaskId>,
    pub parent_context: Option<Context>,
    pub requester_id: RequesterId,
    pub opts: SP1CoreOpts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SplicingWorker<A, W> {
    artifact_client: A,
    worker_client: W,
}

impl<A, W> SplicingWorker<A, W>
where
    A: ArtifactClient,
    W: WorkerClient,
{
    pub fn new(artifact_client: A, worker_client: W) -> Self {
        Self { artifact_client, worker_client }
    }
}

impl<A, W> SplicingWorker<A, W>
where
    A: ArtifactClient,
    W: WorkerClient,
{
    async fn create_core_proving_task(
        &self,
        elf_artifact: Artifact,
        common_input_artifact: Artifact,
        proof_id: ProofId,
        requester_id: RequesterId,
        parent_id: Option<TaskId>,
        parent_context: Option<Context>,
        range: ShardRange,
        trace_data: TraceData,
    ) -> Result<SpawnProveOutput, ExecutionError> {
        let worker_client = self.worker_client.clone();
        let artifact_client = self.artifact_client.clone();
        let record_artifact =
            artifact_client.create_artifact().map_err(|e| ExecutionError::Other(e.to_string()))?;

        // Make a deferred marker task. This is used for the worker to send
        // its deferred record back to the controller.
        let deferred_marker_task = match &trace_data {
            TraceData::Core(_) => worker_client
                .submit_task(
                    TaskType::MarkerDeferredRecord,
                    RawTaskRequest {
                        inputs: vec![],
                        outputs: vec![],
                        proof_id: proof_id.clone(),
                        parent_id: None,
                        parent_context: None,
                        requester_id: requester_id.clone(),
                    },
                )
                .await
                .map_err(|e| ExecutionError::Other(e.to_string()))?,
            TraceData::Memory(_) => TaskId::new("memory dummy deferred marker task id"),
            TraceData::Precompile(_, _) => TaskId::new("precompile dummy deferred marker task id"),
        };

        let deferred_output_artifact = match &trace_data {
            TraceData::Core(_) => artifact_client
                .create_artifact()
                .map_err(|e| ExecutionError::Other(e.to_string()))?,
            TraceData::Memory(_) => {
                Artifact::from("dummy global memory deferred output artifact".to_string())
            }
            TraceData::Precompile(_, _) => {
                Artifact::from("dummy precompile deferred output artifact".to_string())
            }
        };

        let deferred_message = match trace_data {
            TraceData::Core(_) => Some(DeferredMessage {
                task_id: deferred_marker_task.clone(),
                record: deferred_output_artifact.clone(),
            }),
            _ => None,
        };

        artifact_client
            .upload(&record_artifact, trace_data)
            .await
            .map_err(|e| ExecutionError::Other(e.to_string()))?;

        let deferred_marker_task = Artifact::from(deferred_marker_task.to_string());

        // Allocate an artifact for the proof
        let proof_artifact = artifact_client.create_artifact().map_err(|_| {
            ExecutionError::Other("failed to create shard proof artifact".to_string())
        })?;

        let request = ProveShardTaskRequest {
            proof_id: proof_id.clone(),
            elf: elf_artifact,
            common_input: common_input_artifact,
            record: record_artifact,
            output: proof_artifact.clone(),
            deferred_marker_task,
            deferred_output: deferred_output_artifact,
            parent_id,
            parent_context,
            requester_id,
        };

        let task = request.into_raw().map_err(|e| ExecutionError::Other(e.to_string()))?;

        // Send the task to the worker.
        let task_id = worker_client
            .submit_task(TaskType::ProveShard, task)
            .await
            .expect("failed to send task");
        let proof_data = ProofData { task_id, range, proof: proof_artifact };
        Ok(SpawnProveOutput { deferred_message, proof_data })
    }
}

impl<A, W> AsyncWorker<SplicingTask, Result<(), ExecutionError>> for SplicingWorker<A, W>
where
    A: ArtifactClient,
    W: WorkerClient,
{
    async fn call(&self, input: SplicingTask) -> Result<(), ExecutionError> {
        let SplicingTask {
            program,
            chunk,
            all_touched_addresses,
            final_vm_state,
            elf_artifact,
            common_input_artifact,
            num_deferred_proofs,
            prove_shard_tx,
            proof_id,
            parent_id,
            parent_context,
            requester_id,
            opts,
        } = input;
        let split_opts = SplitOpts::new(&opts, program.instructions.len(), false);

        let (splicing_tx, mut splicing_rx) = mpsc::channel::<SendSpliceTask>(1);
        // TODO: what size should the buffer be? maybe even unbounded?
        let (deferred_marker_tx, mut deferred_marker_rx) =
            mpsc::unbounded_channel::<DeferredMessage>();

        let mut join_set = JoinSet::<Result<(), ExecutionError>>::new();

        // Spawn the task to spawn the prove shard tasks.
        join_set.spawn({
            let self_clone = self.clone();
            let elf_artifact = elf_artifact.clone();
            let common_input_artifact = common_input_artifact.clone();
            let proof_id = proof_id.clone();
            let requester_id = requester_id.clone();
            let parent_id_clone = parent_id.clone();
            let parent_context_clone = parent_context.clone();
            let prove_shard_tx = prove_shard_tx.clone();
            async move {
                while let Some(task) = splicing_rx.recv().await {
                    let SendSpliceTask { chunk, range } = task;
                    let chunk_bytes =
                        bincode::serialize(&chunk).expect("failed to serialize record");
                    let data = TraceData::Core(chunk_bytes);

                    let SpawnProveOutput { deferred_message, proof_data } = self_clone
                        .create_core_proving_task(
                            elf_artifact.clone(),
                            common_input_artifact.clone(),
                            proof_id.clone(),
                            requester_id.clone(),
                            parent_id_clone.clone(),
                            parent_context_clone.clone(),
                            range,
                            data,
                        )
                        .await
                        .expect("failed to create core proving task");

                    // Send the task id
                    prove_shard_tx.send(proof_data).expect("failed to send task id");
                    // Send the deferred message to the deferred marker receiver.
                    if let Some(deferred_message) = deferred_message {
                        deferred_marker_tx
                            .send(deferred_message)
                            .expect("failed to send deferred marker");
                    }
                    // Check for deferred records that can be sent and spawn the prove shard tasks
                    // for them.
                }

                Ok(())
            }
        });

        // Spawn the task that splices the trace.
        join_set.spawn_blocking(move || {
            let mut touched_addresses = CompressedMemory::new();
            let mut vm = SplicingVM::new(&chunk, program.clone(), &mut touched_addresses, opts);

            let start_num_mem_reads = chunk.num_mem_reads();
            let mut last_splice = SplicedMinimalTrace::new_full_trace(chunk.clone());
                let mut boundary = ShardBoundary {
                    timestamp: vm.core.clk(),
                    initialized_address: 0,
                    finalized_address: 0,
                    initialized_page_index: 0,
                    finalized_page_index: 0,
                    deferred_proof: num_deferred_proofs as u64,
                };
            loop {
                tracing::debug!("starting new shard at clk: {} at pc: {}", vm.core.clk(), vm.core.pc());
                match vm.execute()? {
                    CycleResult::ShardBoundry => {
                        // Note: Chunk implentations should always be cheap to clone.
                        if let Some(spliced) = vm.splice(chunk.clone()) {
                            tracing::trace!("shard ended at clk: {}", vm.core.clk());
                            tracing::trace!("shard ended at pc: {}", vm.core.pc());
                            tracing::trace!("shard ended at global clk: {}", vm.core.global_clk());
                            tracing::trace!(
                                "shard ended with {} mem reads left ",
                                vm.core.mem_reads.len()
                            );
                            // Get the end boundary of the shard.
                            let end = ShardBoundary {
                                timestamp: vm.core.clk(),
                                initialized_address: 0,
                                finalized_address: 0,
                                initialized_page_index: 0,
                                finalized_page_index: 0,
                                deferred_proof: num_deferred_proofs as u64,
                            };
                            // Get the range of the shard.
                            let range = (boundary..end).into();
                            // Update the boundary to the end of the shard.
                            boundary = end;

                            // Set the last splice clk.
                            last_splice.set_last_clk(vm.core.clk());
                            last_splice.set_last_mem_reads_idx(
                                start_num_mem_reads as usize - vm.core.mem_reads.len(),
                            );
                            let splice_to_send = std::mem::replace(&mut last_splice, spliced);

                            // TODO: handle errors.
                            splicing_tx.blocking_send(SendSpliceTask { chunk: splice_to_send, range }).expect("failed to send splicing task");
                        } else {
                            tracing::trace!("trace ended at clk: {}", vm.core.clk());
                            tracing::trace!("trace ended at pc: {}", vm.core.pc());
                            tracing::trace!("trace ended at global clk: {}", vm.core.global_clk());
                            tracing::trace!(
                                "trace ended with {} mem reads left ",
                                vm.core.mem_reads.len()
                            );
                            // Get the end boundary of the shard.
                            let end = ShardBoundary {
                                timestamp: vm.core.clk(),
                                initialized_address: 0,
                                finalized_address: 0,
                                initialized_page_index: 0,
                                finalized_page_index: 0,
                                deferred_proof: num_deferred_proofs as u64,
                            };
                            // Get the range of the shard.
                            let range = (boundary..end).into();

                            last_splice.set_last_clk(vm.core.clk());
                            last_splice.set_last_mem_reads_idx(
                                start_num_mem_reads as usize - vm.core.mem_reads.len(),
                            );

                            splicing_tx.blocking_send(SendSpliceTask { chunk: last_splice, range }).expect("failed to send splicing task");

                            break;
                        }
                    }
                    CycleResult::Done(true) => {
                        last_splice.set_last_clk(vm.core.clk());
                        last_splice.set_last_mem_reads_idx(chunk.num_mem_reads() as usize);

                        // Get the end boundary of the shard.
                        let end = ShardBoundary {
                            timestamp: vm.core.clk(),
                            initialized_address: 0,
                            finalized_address: 0,
                            initialized_page_index: 0,
                            finalized_page_index: 0,
                            deferred_proof: num_deferred_proofs as u64,
                        };
                        // Get the range of the shard.
                        let range = (boundary..end).into();

                        // Get the last state of the vm execution and set the global final vm state to
                        // this value.
                        let final_state = FinalVmState::new(&vm.core);
                        final_vm_state.set(final_state);

                        // Send the last splice.
                        splicing_tx.blocking_send(SendSpliceTask { chunk: last_splice, range }).expect("failed to send splicing task");

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
            all_touched_addresses.extend(touched_addresses.is_set().into_iter());

            Ok(())
           });

        // Spawn the task that waits for deferred records, accumulates them, and creates tasks to
        // prove them.
        join_set.spawn({
            let self_clone = self.clone();
            let elf_artifact_clone = elf_artifact.clone();
            let common_input_artifact_clone = common_input_artifact.clone();
            let proof_id_clone = proof_id.clone();
            let requester_id_clone = requester_id.clone();
            let parent_id_clone = parent_id.clone();
            let parent_context_clone = parent_context.clone();
            let prove_shard_tx = prove_shard_tx.clone();
            async move {
                let artifact_client = self_clone.artifact_client.clone();
                let worker_client = self_clone.worker_client.clone();

                let mut join_set = JoinSet::new();
                let task_data_map = Arc::new(tokio::sync::Mutex::new(HashMap::new()));

                // This subscriber monitors for deferred marker task completion
                let (subscriber, mut event_stream) = worker_client
                    .subscriber(proof_id.clone())
                    .await
                    .map_err(|e| ExecutionError::Other(e.to_string()))?
                    .stream();
                join_set.spawn({
                    let task_data_map = task_data_map.clone();
                    async move {
                        while let Some(deferred_message) = deferred_marker_rx.recv().await {
                            tracing::trace!(
                                "received deferred message with task id {:?}",
                                deferred_message.task_id
                            );
                            let DeferredMessage { task_id, record: deferred_events } =
                                deferred_message;
                            task_data_map.lock().await.insert(task_id.clone(), deferred_events);
                            subscriber
                                .subscribe(task_id.clone())
                                .map_err(|e| ExecutionError::Other(e.to_string()))?;
                        }
                        Ok::<_, ExecutionError>(())
                    }
                });

                join_set.spawn({
                    async move {
                        let mut deferred_accumulator = DeferredEvents::empty();
                        while let Some((task_id, status)) = event_stream.next().await {
                            if status != TaskStatus::Succeeded {
                                return Err(ExecutionError::Other(format!(
                                    "deferred marker task failed: {}",
                                    task_id
                                )));
                            }
                            let deferred_events_artifact =
                                task_data_map.lock().await.remove(&task_id).ok_or_else(|| {
                                    ExecutionError::Other(format!(
                                        "task data not found for task id: {}",
                                        task_id
                                    ))
                                })?;
                            let deferred_events: DeferredEvents = artifact_client
                                .download(&deferred_events_artifact)
                                .await
                                .map_err(|e| ExecutionError::Other(e.to_string()))?;

                            deferred_accumulator.append(deferred_events, &artifact_client).await;
                            let new_shards = deferred_accumulator
                                .split(false, split_opts, &artifact_client)
                                .await;

                            for shard in new_shards {
                                let SpawnProveOutput { deferred_message, proof_data } = self_clone
                                    .create_core_proving_task(
                                        elf_artifact_clone.clone(),
                                        common_input_artifact_clone.clone(),
                                        proof_id_clone.clone(),
                                        requester_id_clone.clone(),
                                        parent_id_clone.clone(),
                                        parent_context_clone.clone(),
                                        ShardRange::deferred(),
                                        shard,
                                    )
                                    .await
                                    .map_err(|e| ExecutionError::Other(e.to_string()))?;

                                if deferred_message.is_some() {
                                    return Err(ExecutionError::Other(
                                        "deferred message is not none".to_string(),
                                    ));
                                }
                                prove_shard_tx.send(proof_data).expect("failed to send task id");
                            }
                        }
                        let final_shards = deferred_accumulator
                            .split(true, split_opts, &artifact_client)
                            .instrument(tracing::info_span!("split last"))
                            .await;
                        for shard in final_shards {
                            let SpawnProveOutput { deferred_message, proof_data } = self_clone
                                .create_core_proving_task(
                                    elf_artifact_clone.clone(),
                                    common_input_artifact_clone.clone(),
                                    proof_id_clone.clone(),
                                    requester_id_clone.clone(),
                                    parent_id_clone.clone(),
                                    parent_context_clone.clone(),
                                    ShardRange::deferred(),
                                    shard,
                                )
                                .await
                                .expect("failed to create core proving task");

                            assert!(deferred_message.is_none());
                            prove_shard_tx.send(proof_data).expect("failed to send task id");
                        }
                        Ok::<_, ExecutionError>(())
                    }
                });

                join_set.join_all().await.into_iter().collect::<Result<(), ExecutionError>>()?;
                Ok::<(), ExecutionError>(())
            }
        });
        // Wait for the tasks to finish and collect the errors.
        join_set.join_all().await.into_iter().collect::<Result<(), ExecutionError>>()?;

        Ok(())
    }
}
