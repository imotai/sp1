mod compress;
mod core;

pub use compress::*;
pub use core::*;
use opentelemetry::Context;
use sp1_core_machine::{executor::ExecutionOutput, io::SP1Stdin};
use sp1_hypercube::{air::PublicValues, ShardProof};
use sp1_primitives::{io::SP1PublicValues, SP1GlobalContext};
use sp1_prover_types::{
    network_base_types::ProofMode, Artifact, ArtifactClient, TaskStatus, TaskType,
};
use std::{borrow::Borrow, sync::Arc};
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinSet,
};
use tracing::Instrument;

use crate::{
    worker::{ProofId, RawTaskRequest, RequesterId, TaskError, TaskId, WorkerClient},
    CoreSC, SP1CoreProof, SP1CoreProofData, SP1VerifyingKey,
};

#[derive(Clone)]
pub struct SP1ControllerConfig {
    pub num_splicing_workers: usize,
    pub splicing_buffer_size: usize,
    pub max_reduce_arity: usize,
}

pub struct SP1Controller<A, W> {
    splicing_engine: Arc<SplicingEngine<A, W>>,
    max_reduce_arity: usize,
    pub(crate) artifact_client: A,
    pub(crate) worker_client: W,
}

impl<A, W> SP1Controller<A, W>
where
    A: ArtifactClient,
    W: WorkerClient,
{
    pub fn new(config: SP1ControllerConfig, artifact_client: A, worker_client: W) -> Self {
        let splicing_workers = (0..config.num_splicing_workers)
            .map(|_| SplicingWorker::new(artifact_client.clone(), worker_client.clone()))
            .collect();
        let splicing_engine =
            Arc::new(SplicingEngine::new(splicing_workers, config.splicing_buffer_size));
        let reduce_batch_size = config.max_reduce_arity;

        Self {
            splicing_engine,
            max_reduce_arity: reduce_batch_size,
            artifact_client,
            worker_client,
        }
    }

    fn executor(
        &self,
        elf: Artifact,
        stdin: Arc<SP1Stdin>,
        common_input: Artifact,
        num_deferred_proofs: usize,
        proof_id: ProofId,
        parent_id: Option<TaskId>,
        parent_context: Option<Context>,
        requester_id: RequesterId,
        sender: mpsc::UnboundedSender<ProofData>,
    ) -> SP1CoreExecutor<A, W> {
        SP1CoreExecutor::new(
            self.splicing_engine.clone(),
            elf,
            stdin,
            common_input,
            num_deferred_proofs,
            proof_id,
            parent_id,
            parent_context,
            requester_id,
            sender,
            self.artifact_client.clone(),
            self.worker_client.clone(),
        )
    }

    fn collect_core_proofs(
        &self,
        proof_id: ProofId,
        stdin: Arc<SP1Stdin>,
        outputs: Vec<Artifact>,
        mut core_proof_rx: mpsc::UnboundedReceiver<ProofData>,
        execution_output_rx: oneshot::Receiver<ExecutionOutput>,
        join_set: &mut JoinSet<Result<(), TaskError>>,
    ) {
        // Spawn a task to gather the proofs
        let (proofs_tx, proofs_rx) = oneshot::channel();
        join_set.spawn({
            let worker_client = self.worker_client.clone();
            let artifact_client = self.artifact_client.clone();
            async move {
                let subscriber = worker_client.subscriber(proof_id).await?.per_task();
                let mut shard_proofs = Vec::new();
                while let Some(proof_data) = core_proof_rx.recv().await {
                    let ProofData { task_id, proof, .. } = proof_data;
                    // Wait for the task to finish
                    let status = subscriber
                        .wait_task(task_id.clone())
                        .await
                        .map_err(|e| TaskError::Fatal(e.into()))?;
                    assert_eq!(status, TaskStatus::Succeeded);
                    // Download the proof
                    let proof = artifact_client
                        .download::<ShardProof<SP1GlobalContext, CoreSC>>(&proof)
                        .await?;
                    shard_proofs.push(proof);
                }
                proofs_tx.send(shard_proofs).ok();
                Ok(())
            }
        });

        // Task to wait for the proofs and execution output and upload the output.
        join_set.spawn({
            let artifact_client = self.artifact_client.clone();
            async move {
                // Wait for the proofs to finish
                let mut shard_proofs = proofs_rx.await.map_err(|e| TaskError::Fatal(e.into()))?;
                shard_proofs.sort_by_key(|shard_proof| {
                    let public_values: &PublicValues<[_; 4], [_; 3], [_; 4], _> =
                        shard_proof.public_values.as_slice().borrow();
                    public_values.range()
                });

                // Wait for the execution output to finish
                let execution_output =
                    execution_output_rx.await.map_err(|e| TaskError::Fatal(e.into()))?;

                let ExecutionOutput { public_value_stream, cycles } = execution_output;
                let public_values = SP1PublicValues::from(&public_value_stream);
                let proof = SP1CoreProof {
                    proof: SP1CoreProofData(shard_proofs),
                    stdin: stdin.as_ref().clone(),
                    public_values,
                    cycles,
                };

                // Upload the proof
                artifact_client.upload(&outputs[0], proof).await?;

                Ok(())
            }
        });
    }

    pub async fn run(&self, request: RawTaskRequest) -> Result<ExecutionOutput, TaskError> {
        let RawTaskRequest { inputs, outputs, proof_id, parent_id, parent_context, requester_id } =
            request;

        let elf = inputs[0].clone();
        let stdin_artifact = inputs[1].clone();
        let mode_artifact = inputs[2].clone().to_id();
        let parsed = mode_artifact.parse::<i32>().map_err(|e| TaskError::Fatal(e.into()))?;
        let mode = ProofMode::try_from(parsed).map_err(|e| TaskError::Fatal(e.into()))?;

        tracing::info!("mode: {:?}", mode);

        // For now, assume no deferred proofs
        let deferred_digest = [0; 8];
        let num_deferred_proofs = 0usize;

        // Create a setup task and wait for the vk
        let vk_artifact = self.artifact_client.create_artifact()?;
        let setup_request = RawTaskRequest {
            inputs: vec![elf.clone()],
            outputs: vec![vk_artifact.clone()],
            proof_id: proof_id.clone(),
            parent_id: parent_id.clone(),
            parent_context: parent_context.clone(),
            requester_id: requester_id.clone(),
        };

        // TODO: do setup and start downloading stdin and execute concurrently.
        tracing::trace!("submitting setup task");
        let setup_id = self.worker_client.submit_task(TaskType::SetupVkey, setup_request).await?;
        // Wait for the setup task to finish
        let subscriber = self.worker_client.subscriber(proof_id.clone()).await?.per_task();
        let status = subscriber
            .wait_task(setup_id)
            .instrument(tracing::debug_span!("setup task"))
            .await
            .map_err(|e| TaskError::Fatal(e.into()))?;
        if status != TaskStatus::Succeeded {
            return Err(TaskError::Fatal(anyhow::anyhow!("setup task failed")));
        }
        tracing::trace!("setup task succeeded");
        // Download the vk and stdin
        let (vk, stdin) = tokio::try_join!(
            self.artifact_client.download::<SP1VerifyingKey>(&vk_artifact),
            self.artifact_client.download_stdin::<SP1Stdin>(&stdin_artifact),
        )?;
        let stdin = Arc::new(stdin);
        // Create the common input
        let common_input = CommonProverInput { vk, mode, deferred_digest, num_deferred_proofs };
        // Upload the common input
        let common_input_artifact =
            self.artifact_client.create_artifact().expect("failed to create common input artifact");
        self.artifact_client.upload(&common_input_artifact, common_input.clone()).await?;

        let (core_proof_tx, core_proof_rx) = mpsc::unbounded_channel();

        let executor = self.executor(
            elf,
            stdin.clone(),
            common_input_artifact,
            num_deferred_proofs,
            proof_id.clone(),
            parent_id.clone(),
            parent_context.clone(),
            requester_id.clone(),
            core_proof_tx,
        );

        let mut join_set = JoinSet::<Result<(), TaskError>>::new();
        let (execution_output_tx, execution_output_rx) = oneshot::channel();
        match mode {
            ProofMode::Core => {
                self.collect_core_proofs(
                    proof_id,
                    stdin,
                    outputs,
                    core_proof_rx,
                    execution_output_rx,
                    &mut join_set,
                );
            }
            ProofMode::Compressed => {
                let reducer = SP1ReduceController::new(
                    self.max_reduce_arity,
                    self.artifact_client.clone(),
                    self.worker_client.clone(),
                );
                join_set.spawn(async move {
                    reducer
                        .reduce(
                            proof_id,
                            parent_id,
                            parent_context,
                            requester_id,
                            outputs,
                            core_proof_rx,
                            execution_output_rx,
                        )
                        .await?;
                    Ok(())
                });
            }
            _ => {
                unimplemented!("proof mode not supported: {:?}", mode)
            }
        }

        // Wait for the executor to finish
        let result = executor.execute().await?;
        execution_output_tx.send(result.clone()).ok();
        tracing::trace!("executor finished");

        // Wait for all the tasks to finish
        while let Some(result) = join_set.join_next().await {
            result.map_err(|e| TaskError::Fatal(e.into()))??;
        }

        Ok(result)
    }
}
