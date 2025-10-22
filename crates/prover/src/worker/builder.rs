use std::{env, sync::Arc};

use sp1_hypercube::prover::ProverSemaphore;
use sp1_prover_types::ArtifactClient;

use crate::{
    components::{CoreProver, RecursionProver, WrapProver},
    verify::SP1Verifier,
    worker::{SP1Controller, SP1ProverEngine, SP1Worker, SP1WorkerConfig, WorkerClient},
    SP1ProverComponents,
};

pub struct SP1WorkerBuilder<A, W, C: SP1ProverComponents> {
    config: SP1WorkerConfig,
    core_air_prover_and_permits: Option<(Arc<CoreProver<C>>, ProverSemaphore)>,
    compress_air_prover_and_permits: Option<(Arc<RecursionProver<C>>, ProverSemaphore)>,
    shrink_air_prover_and_permits: Option<(Arc<RecursionProver<C>>, ProverSemaphore)>,
    wrap_air_prover_and_permits: Option<(Arc<WrapProver<C>>, ProverSemaphore)>,
    #[allow(dead_code)]
    vk_map_path: String,
    artifact_client: A,
    worker_client: W,
}

impl<A: ArtifactClient, W: WorkerClient, C: SP1ProverComponents> SP1WorkerBuilder<A, W, C> {
    pub fn new(artifact_client: A, worker_client: W) -> Self {
        let config = SP1WorkerConfig::default();

        let vk_map_path =
            env::var("SP1_LOCAL_NODE_VK_MAP_PATH").unwrap_or("./src/vk_map.bin".to_string());

        Self {
            config,
            core_air_prover_and_permits: None,
            compress_air_prover_and_permits: None,
            shrink_air_prover_and_permits: None,
            wrap_air_prover_and_permits: None,
            vk_map_path,
            artifact_client,
            worker_client,
        }
    }

    pub fn with_core_air_prover(
        mut self,
        core_air_prover: Arc<CoreProver<C>>,
        permit: ProverSemaphore,
    ) -> Self {
        self.core_air_prover_and_permits = Some((core_air_prover, permit));
        self
    }

    pub fn with_compress_air_prover(
        mut self,
        compress_air_prover: Arc<RecursionProver<C>>,
        permit: ProverSemaphore,
    ) -> Self {
        self.compress_air_prover_and_permits = Some((compress_air_prover, permit));
        self
    }

    pub fn with_shrink_air_prover(
        mut self,
        shrink_air_prover: Arc<RecursionProver<C>>,
        permit: ProverSemaphore,
    ) -> Self {
        self.shrink_air_prover_and_permits = Some((shrink_air_prover, permit));
        self
    }

    pub fn with_wrap_air_prover(
        mut self,
        wrap_air_prover: Arc<WrapProver<C>>,
        permit: ProverSemaphore,
    ) -> Self {
        self.wrap_air_prover_and_permits = Some((wrap_air_prover, permit));
        self
    }

    pub fn build(self) -> anyhow::Result<SP1Worker<A, W, C>> {
        // Destructure the builder.
        let Self {
            config,
            core_air_prover_and_permits,
            compress_air_prover_and_permits: _,
            shrink_air_prover_and_permits: _,
            wrap_air_prover_and_permits: _,
            vk_map_path: _,
            artifact_client,
            worker_client,
        } = self;

        let controller = SP1Controller::new(
            config.controller_config,
            artifact_client.clone(),
            worker_client.clone(),
        );

        // Create the prover engine
        let core_air_prover_and_permits = core_air_prover_and_permits
            .ok_or(anyhow::anyhow!("Core air prover and permit are required"))?;
        let prover_engine = SP1ProverEngine::new(
            config.prover_config,
            artifact_client.clone(),
            worker_client.clone(),
            core_air_prover_and_permits,
        );

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

        Ok(SP1Worker::new(controller, prover_engine, verifier))
    }
}
