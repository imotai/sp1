use std::sync::Arc;

use sp1_hypercube::prover::ProverSemaphore;
use sp1_prover_types::{
    ArtifactClient, ArtifactType, InMemoryArtifactClient, TaskStatus, TaskType,
};
use tokio::{sync::mpsc, task::JoinSet};

use crate::{
    components::{CoreProver, RecursionProver, WrapProver},
    worker::{
        LocalWorkerClient, LocalWorkerClientChannels, RawTaskRequest, SP1LocalNode, SP1NodeInner,
        SP1WorkerBuilder, TaskMetadata, WorkerClient,
    },
    SP1ProverComponents,
};

pub struct SP1LocalNodeBuilder<C: SP1ProverComponents> {
    pub worker_builder: SP1WorkerBuilder<C, InMemoryArtifactClient, LocalWorkerClient>,
    pub channels: LocalWorkerClientChannels,
}

impl<C: SP1ProverComponents> SP1LocalNodeBuilder<C> {
    /// Creates a new local node builder with a default worker client builder.
    pub fn new() -> Self {
        Self::from_worker_client_builder(SP1WorkerBuilder::new())
    }

    /// Creates a new local node builder from a worker client builder.
    ///
    /// This method can be used to initialize a node from a worker client builder that has already
    /// been configured with the desired prover components.
    pub fn from_worker_client_builder(builder: SP1WorkerBuilder<C>) -> Self {
        let artifact_client = InMemoryArtifactClient::new();
        let (worker_client, channels) = LocalWorkerClient::init();
        let worker_builder =
            builder.with_artifact_client(artifact_client).with_worker_client(worker_client);
        Self { worker_builder, channels }
    }

    /// Sets the core air prover to the worker client builder.
    pub fn with_core_air_prover(
        mut self,
        core_air_prover: Arc<CoreProver<C>>,
        permit: ProverSemaphore,
    ) -> Self {
        self.worker_builder = self.worker_builder.with_core_air_prover(core_air_prover, permit);
        self
    }

    /// Sets the compress air prover to the worker client builder.
    pub fn with_compress_air_prover(
        mut self,
        compress_air_prover: Arc<RecursionProver<C>>,
        permit: ProverSemaphore,
    ) -> Self {
        self.worker_builder =
            self.worker_builder.with_compress_air_prover(compress_air_prover, permit);
        self
    }

    /// Sets the shrink air prover to the worker client builder.
    pub fn with_shrink_air_prover(
        mut self,
        shrink_air_prover: Arc<RecursionProver<C>>,
        permit: ProverSemaphore,
    ) -> Self {
        self.worker_builder = self.worker_builder.with_shrink_air_prover(shrink_air_prover, permit);
        self
    }

    /// Sets the wrap air prover to the worker client builder.
    pub fn with_wrap_air_prover(
        mut self,
        wrap_air_prover: Arc<WrapProver<C>>,
        permit: ProverSemaphore,
    ) -> Self {
        self.worker_builder = self.worker_builder.with_wrap_air_prover(wrap_air_prover, permit);
        self
    }

    pub async fn build(self) -> anyhow::Result<SP1LocalNode> {
        // Destructure the builder.
        let Self { worker_builder, mut channels } = self;
        // Get the core options from the worker builder.
        let opts = worker_builder.core_opts().clone();

        // Build the node.
        let worker = worker_builder.build().await?;

        // Spawn tasks to handle all the requests. We must spawn a handler for each task type to
        // avoid blocking the main thread by not having processed the input channel.

        // Spawn the controller handler
        tokio::task::spawn({
            let mut controller_rx = channels.task_receivers.remove(&TaskType::Controller).unwrap();
            let worker = worker.clone();
            async move {
                while let Some((task_id, request)) = controller_rx.recv().await {
                    // Run the controller task
                    if let Err(e) = worker.controller().run(request.clone()).await {
                        tracing::error!("Controller: task failed: {e:?}");
                        continue;
                    }

                    // Complete the task
                    if let Err(e) = worker
                        .worker_client()
                        .complete_task(
                            request.context.proof_id,
                            task_id,
                            TaskMetadata { gpu_time: None },
                        )
                        .await
                    {
                        tracing::error!("Controller: marking task as complete failed: {e:?}");
                    }

                    // Remove all the inputs from the task
                    for input in request.inputs {
                        if let Err(e) = worker
                            .artifact_client()
                            .delete(&input, ArtifactType::UnspecifiedArtifactType)
                            .await
                        {
                            tracing::error!("Controller: deleting input artifact failed: {e:?}");
                        }
                    }
                }
            }
        });

        // Spawn the setup handler
        tokio::task::spawn({
            let mut setup_rx = channels.task_receivers.remove(&TaskType::SetupVkey).unwrap();
            let worker = worker.clone();
            let worker_client = worker.worker_client().clone();
            async move {
                let mut task_set = JoinSet::new();
                let (task_tx, mut task_rx) = mpsc::unbounded_channel();
                loop {
                    tokio::select! {
                        Some((id, request)) = setup_rx.recv() => {
                            let RawTaskRequest { inputs, outputs, context } = request;
                            let proof_id = context.proof_id.clone();
                            let elf = inputs[0].clone();
                            let output = outputs[0].clone();
                            // TODO: handle errors
                            let handle = worker.prover_engine().submit_setup(id, elf, output).await.unwrap();
                            let proof_id = proof_id.clone();
                            let tx = task_tx.clone();
                            task_set.spawn(async move {
                                let task_id = handle.await.unwrap().unwrap();
                                tx.send((proof_id, task_id, TaskStatus::Succeeded)).ok();
                            });

                        }

                        Some((proof_id, (id, task_metadata), status)) = task_rx.recv() => {
                            assert_eq!(status, TaskStatus::Succeeded);
                            worker_client.complete_task(proof_id, id, task_metadata).await.unwrap();
                        }
                        else => {
                            break;
                        }
                    }
                }
            }
        });

        // Spawn the prove shard handler
        tokio::task::spawn({
            let mut core_prover_rx = channels.task_receivers.remove(&TaskType::ProveShard).unwrap();
            let worker = worker.clone();
            let worker_client = worker.worker_client().clone();
            async move {
                let mut task_set = JoinSet::new();
                let (task_tx, mut task_rx) = mpsc::unbounded_channel();

                loop {
                    tokio::select! {
                        Some((id, request)) = core_prover_rx.recv() => {
                            let proof_id = request.context.proof_id.clone();
                            let handle = worker
                                .prover_engine()
                                .submit_prove_core_shard(
                                    request,
                                )
                                .await
                                .unwrap();
                            let proof_id = proof_id.clone();
                            let tx = task_tx.clone();
                            let task_id = id;
                            task_set.spawn(async move {
                                match handle.await {
                                    Ok(Ok(task_metadata)) => {
                                        tx.send((proof_id, (task_id, task_metadata), TaskStatus::Succeeded)).ok();
                                    }
                                    Ok(Err(e)) => {
                                        tracing::error!("Failed to prove core shard: {:?}", e);
                                    }
                                    Err(e) => {
                                        tracing::error!("Panicked in prove core shard: {:?}", e);
                                    }
                                }
                            });
                        }

                        Some((proof_id, (task_id, task_metadata), status)) = task_rx.recv() => {
                            assert_eq!(status, TaskStatus::Succeeded);
                            worker_client.complete_task(proof_id, task_id, task_metadata).await.unwrap();
                        }
                        else => {
                            break;
                        }
                    }
                }
            }
        });

        // Spawn the recursion reduce handler
        tokio::task::spawn({
            let mut recursion_reduce_rx =
                channels.task_receivers.remove(&TaskType::RecursionReduce).unwrap();
            let worker = worker.clone();
            let worker_client = worker.worker_client().clone();
            async move {
                let mut task_set = JoinSet::new();
                let (task_tx, mut task_rx) = mpsc::unbounded_channel();
                loop {
                    tokio::select! {
                        Some((id, request)) = recursion_reduce_rx.recv() => {
                            let proof_id = request.context.proof_id.clone();
                            let handle = worker.prover_engine().submit_recursion_reduce(request).await.unwrap();
                            let tx = task_tx.clone();
                            task_set.spawn(async move {
                                match handle.await {
                                    Ok(Ok(task_metadata)) => {
                                        tx.send((proof_id, (id, task_metadata), TaskStatus::Succeeded)).ok();
                                    }
                                    Ok(Err(e)) => {
                                        tracing::error!("Failed to reduce recursion: {:?}", e);
                                    }
                                    Err(e) => {
                                        tracing::error!("Panicked in reduce recursion: {:?}", e);
                                    }
                                }
                            });
                        }

                        Some((proof_id, (task_id, task_metadata), status)) = task_rx.recv() => {
                            assert_eq!(status, TaskStatus::Succeeded);
                            worker_client.complete_task(proof_id, task_id, task_metadata).await.unwrap();
                        }
                        else => {
                            break;
                        }
                    }
                }
            }
        });

        tokio::task::spawn({
            let mut recursion_deferred_rx =
                channels.task_receivers.remove(&TaskType::RecursionDeferred).unwrap();
            async move { while let Some((_task_id, _request)) = recursion_deferred_rx.recv().await {} }
        });

        tokio::task::spawn({
            let mut marker_deferred_task_rx =
                channels.task_receivers.remove(&TaskType::MarkerDeferredRecord).unwrap();
            async move { while let Some((_task_id, _request)) = marker_deferred_task_rx.recv().await {} }
        });

        // Spawn the shrink wrap handler
        tokio::task::spawn({
            let mut shrink_wrap_rx = channels.task_receivers.remove(&TaskType::ShrinkWrap).unwrap();
            async move { while let Some((_task_id, _request)) = shrink_wrap_rx.recv().await {} }
        });

        // Spawn the plonk wrap handler
        tokio::task::spawn({
            let mut plonk_wrap_rx = channels.task_receivers.remove(&TaskType::PlonkWrap).unwrap();
            async move { while let Some((_task_id, _request)) = plonk_wrap_rx.recv().await {} }
        });

        // Spawn the groth16 wrap handler
        tokio::task::spawn({
            let mut groth16_wrap_rx =
                channels.task_receivers.remove(&TaskType::Groth16Wrap).unwrap();
            async move { while let Some((_task_id, _request)) = groth16_wrap_rx.recv().await {} }
        });

        // Get the verifier, artifact client, and worker client from the worker
        let verifier = worker.verifier().clone();
        let artifact_client = worker.artifact_client().clone();
        let worker_client = worker.worker_client().clone();
        let inner = Arc::new(SP1NodeInner { artifact_client, worker_client, verifier, opts });
        Ok(SP1LocalNode { inner })
    }
}
