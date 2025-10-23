use std::sync::Arc;

use sp1_hypercube::prover::ProverSemaphore;
use sp1_prover_types::{
    ArtifactClient, ArtifactType, InMemoryArtifactClient, TaskStatus, TaskType,
};
use tokio::{sync::mpsc, task::JoinSet};

use crate::{
    components::{CoreProver, RecursionProver, WrapProver},
    verify::SP1Verifier,
    worker::{
        LocalWorkerClient, LocalWorkerClientChannels, RawTaskRequest, SP1LocalNode,
        SP1WorkerBuilder, TaskMetadata, WorkerClient,
    },
    SP1ProverComponents,
};

pub struct SP1LocalNodeBuilder<C: SP1ProverComponents> {
    worker_builder: SP1WorkerBuilder<InMemoryArtifactClient, LocalWorkerClient, C>,
    channels: LocalWorkerClientChannels,
}

impl<C: SP1ProverComponents> SP1LocalNodeBuilder<C> {
    pub fn new() -> Self {
        let artifact_client = InMemoryArtifactClient::new();
        let (worker_client, channels) = LocalWorkerClient::init();
        let worker_builder = SP1WorkerBuilder::new(artifact_client, worker_client);
        Self { worker_builder, channels }
    }

    pub fn with_core_air_prover(
        mut self,
        core_air_prover: Arc<CoreProver<C>>,
        permit: ProverSemaphore,
    ) -> Self {
        self.worker_builder = self.worker_builder.with_core_air_prover(core_air_prover, permit);
        self
    }

    pub fn with_compress_air_prover(
        mut self,
        compress_air_prover: Arc<RecursionProver<C>>,
        permit: ProverSemaphore,
    ) -> Self {
        self.worker_builder =
            self.worker_builder.with_compress_air_prover(compress_air_prover, permit);
        self
    }

    pub fn with_shrink_air_prover(
        mut self,
        shrink_air_prover: Arc<RecursionProver<C>>,
        permit: ProverSemaphore,
    ) -> Self {
        self.worker_builder = self.worker_builder.with_shrink_air_prover(shrink_air_prover, permit);
        self
    }

    pub fn with_wrap_air_prover(
        mut self,
        wrap_air_prover: Arc<WrapProver<C>>,
        permit: ProverSemaphore,
    ) -> Self {
        self.worker_builder = self.worker_builder.with_wrap_air_prover(wrap_air_prover, permit);
        self
    }

    pub fn build(self) -> anyhow::Result<SP1LocalNode> {
        // Destructure the builder.
        let Self { worker_builder, mut channels } = self;

        // Build the node.
        let worker = worker_builder.build()?;

        // Spawn tasks to handle all the requests

        // Spawn the controller handler
        tokio::task::spawn({
            let mut controller_rx = channels.task_receivers.remove(&TaskType::Controller).unwrap();
            let worker = worker.clone();
            async move {
                while let Some((task_id, request)) = controller_rx.recv().await {
                    // Run the controller task
                    worker.controller().run(request.clone()).await.unwrap();

                    // Complete the task
                    worker
                        .worker_client()
                        .complete_task(request.proof_id, task_id, TaskMetadata { gpu_time: None })
                        .await
                        .unwrap();

                    // Remove all the inputs from the task
                    for input in request.inputs {
                        worker
                            .artifact_client()
                            .delete(&input, ArtifactType::UnspecifiedArtifactType)
                            .await
                            .unwrap();
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
                            let RawTaskRequest { inputs, outputs, proof_id, .. } = request;
                            let elf = inputs[0].clone();
                            let output = outputs[0].clone();
                            // TODO: handle errors
                            let handle = worker.prover_engine().submit_setup(id, elf, output).await.unwrap();
                            let proof_id = proof_id.clone();
                            let tx = task_tx.clone();
                            task_set.spawn(async move {
                                let task_id = handle.await.unwrap();
                                tx.send((proof_id, task_id, TaskStatus::Succeeded)).ok();
                            });

                        }

                        Some((proof_id, id, status)) = task_rx.recv() => {
                            assert_eq!(status, TaskStatus::Succeeded);
                            worker_client.complete_task(proof_id, id, TaskMetadata { gpu_time: None }).await.unwrap();
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
                            let RawTaskRequest { inputs, outputs, proof_id, .. } = request;

                            let elf = inputs[0].clone();
                            let common_input = inputs[1].clone();
                            let record = inputs[2].clone();
                            let output = outputs[0].clone();

                            let handle = worker.prover_engine().submit_prove_core_shard(id, elf, common_input, record, output).await.unwrap();
                            let proof_id = proof_id.clone();
                            let tx = task_tx.clone();
                            task_set.spawn(async move {
                                let task_id = handle.await.unwrap();
                                tx.send((proof_id, task_id, TaskStatus::Succeeded)).ok();
                            });
                        }

                        Some((proof_id, task_id, status)) = task_rx.recv() => {
                            assert_eq!(status, TaskStatus::Succeeded);
                            worker_client.complete_task(proof_id, task_id, TaskMetadata { gpu_time: None }).await.unwrap();
                        }
                        else => {
                            break;
                        }
                    }
                }
            }
        });

        // Create the verifier
        let core_verifier = C::core_verifier();
        let compress_verifier = C::compress_verifier();
        let shrink_verifier = C::shrink_verifier();
        let wrap_verifier = C::wrap_verifier();
        let verifier = SP1Verifier {
            core: core_verifier,
            compress: compress_verifier,
            shrink: shrink_verifier,
            wrap: wrap_verifier,
            // TODO: get the actual values
            recursion_vk_root: Default::default(),
            // TODO: get the actual values
            recursion_vk_map: Default::default(),
            vk_verification: true,
            shrink_vk: None,
            wrap_vk: None,
        };

        let artifact_client = worker.artifact_client().clone();
        let worker_client = worker.worker_client().clone();

        Ok(SP1LocalNode { artifact_client, worker_client, verifier })
    }
}
