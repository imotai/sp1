use std::sync::Arc;

use slop_futures::pipeline::SubmitError;
use sp1_core_executor::SP1CoreOpts;
use sp1_hypercube::prover::ProverSemaphore;
use sp1_prover_types::{Artifact, ArtifactClient};

use crate::{
    components::{CoreProver, RecursionProver},
    worker::{
        CoreProveSubmitHandle, RawTaskRequest, ReduceSubmitHandle, SP1CoreProver,
        SP1CoreProverConfig, SP1RecursionProver, SP1RecursionProverConfig, SetupSubmitHandle,
        SetupTask, TaskError, TaskId, WorkerClient,
    },
    SP1ProverComponents,
};

#[derive(Clone)]
pub struct SP1ProverConfig {
    pub core_prover_config: SP1CoreProverConfig,
    pub recursion_prover_config: SP1RecursionProverConfig,
}

pub struct SP1ProverEngine<A, W, C: SP1ProverComponents> {
    pub core_prover: SP1CoreProver<A, W, C>,
    pub recursion_prover: SP1RecursionProver<A, C>,
}

impl<A: ArtifactClient, W: WorkerClient, C: SP1ProverComponents> SP1ProverEngine<A, W, C> {
    pub async fn new(
        config: SP1ProverConfig,
        opts: SP1CoreOpts,
        artifact_client: A,
        worker_client: W,
        core_prover_and_permits: (Arc<CoreProver<C>>, ProverSemaphore),
        recursion_prover_and_permits: (Arc<RecursionProver<C>>, ProverSemaphore),
    ) -> Self {
        let recursion_prover = SP1RecursionProver::new(
            config.recursion_prover_config,
            artifact_client.clone(),
            recursion_prover_and_permits.0,
            recursion_prover_and_permits.1,
        )
        .await;

        let core_prover = SP1CoreProver::new(
            config.core_prover_config,
            opts,
            artifact_client,
            worker_client,
            core_prover_and_permits.0,
            core_prover_and_permits.1,
            recursion_prover.clone(),
        );
        Self { core_prover, recursion_prover }
    }

    pub async fn submit_prove_core_shard(
        &self,
        request: RawTaskRequest,
    ) -> Result<CoreProveSubmitHandle<A, W, C>, TaskError> {
        self.core_prover.submit_prove_shard(request).await
    }

    pub async fn submit_setup(
        &self,
        id: TaskId,
        elf: Artifact,
        output: Artifact,
    ) -> Result<SetupSubmitHandle<A, C>, SubmitError> {
        let handle = self.core_prover.submit_setup(SetupTask { id, elf, output }).await?;
        Ok(handle)
    }

    pub async fn submit_recursion_reduce(
        &self,
        request: RawTaskRequest,
    ) -> Result<ReduceSubmitHandle<A, C>, TaskError> {
        self.recursion_prover.submit_recursion_reduce(request).await
    }
}
