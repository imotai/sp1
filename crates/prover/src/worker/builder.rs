use std::sync::Arc;

use sp1_core_executor::SP1CoreOpts;
use sp1_hypercube::prover::{CpuShardProver, ProverSemaphore};
use sp1_prover_types::{ArtifactClient, InMemoryArtifactClient};

use crate::{
    components::{CoreProver, RecursionProver, WrapProver},
    verify::SP1Verifier,
    worker::{
        LocalWorkerClient, SP1Controller, SP1ProverEngine, SP1Worker, SP1WorkerConfig, WorkerClient,
    },
    CpuSP1ProverComponents, SP1ProverComponents,
};

pub struct SP1WorkerBuilder<
    C: SP1ProverComponents,
    A = InMemoryArtifactClient,
    W = LocalWorkerClient,
> {
    config: SP1WorkerConfig,
    core_air_prover_and_permits: Option<(Arc<CoreProver<C>>, ProverSemaphore)>,
    compress_air_prover_and_permits: Option<(Arc<RecursionProver<C>>, ProverSemaphore)>,
    shrink_air_prover_and_permits: Option<(Arc<RecursionProver<C>>, ProverSemaphore)>,
    wrap_air_prover_and_permits: Option<(Arc<WrapProver<C>>, ProverSemaphore)>,
    artifact_client: Option<A>,
    worker_client: Option<W>,
}

impl<C: SP1ProverComponents> SP1WorkerBuilder<C> {
    pub fn new() -> Self {
        let config = SP1WorkerConfig::default();

        Self {
            config,
            core_air_prover_and_permits: None,
            compress_air_prover_and_permits: None,
            shrink_air_prover_and_permits: None,
            wrap_air_prover_and_permits: None,
            artifact_client: None,
            worker_client: None,
        }
    }
}

impl<C: SP1ProverComponents, A, W> SP1WorkerBuilder<C, A, W> {
    /// Set the artifact client.
    #[must_use]
    pub fn with_artifact_client<B: ArtifactClient>(
        self,
        artifact_client: B,
    ) -> SP1WorkerBuilder<C, B, W> {
        let SP1WorkerBuilder {
            config,
            core_air_prover_and_permits,
            compress_air_prover_and_permits,
            shrink_air_prover_and_permits,
            wrap_air_prover_and_permits,
            artifact_client: _,
            worker_client,
        } = self;

        SP1WorkerBuilder {
            config,
            core_air_prover_and_permits,
            compress_air_prover_and_permits,
            shrink_air_prover_and_permits,
            wrap_air_prover_and_permits,
            artifact_client: Some(artifact_client),
            worker_client,
        }
    }

    /// Set the worker client.
    #[must_use]
    pub fn with_worker_client<V: WorkerClient>(
        self,
        worker_client: V,
    ) -> SP1WorkerBuilder<C, A, V> {
        let SP1WorkerBuilder {
            config,
            core_air_prover_and_permits,
            compress_air_prover_and_permits,
            shrink_air_prover_and_permits,
            wrap_air_prover_and_permits,
            artifact_client,
            worker_client: _,
        } = self;

        SP1WorkerBuilder {
            config,
            core_air_prover_and_permits,
            compress_air_prover_and_permits,
            shrink_air_prover_and_permits,
            wrap_air_prover_and_permits,
            artifact_client,
            worker_client: Some(worker_client),
        }
    }

    /// Set the core air prover.
    #[must_use]
    pub fn with_core_air_prover(
        mut self,
        core_air_prover: Arc<CoreProver<C>>,
        permit: ProverSemaphore,
    ) -> Self {
        self.core_air_prover_and_permits = Some((core_air_prover, permit));
        self
    }

    /// Set the compress air prover.
    #[must_use]
    pub fn with_compress_air_prover(
        mut self,
        compress_air_prover: Arc<RecursionProver<C>>,
        permit: ProverSemaphore,
    ) -> Self {
        self.compress_air_prover_and_permits = Some((compress_air_prover, permit));
        self
    }

    /// Set the shrink air prover.
    #[must_use]
    pub fn with_shrink_air_prover(
        mut self,
        shrink_air_prover: Arc<RecursionProver<C>>,
        permit: ProverSemaphore,
    ) -> Self {
        self.shrink_air_prover_and_permits = Some((shrink_air_prover, permit));
        self
    }

    /// Set the wrap air prover.
    #[must_use]
    pub fn with_wrap_air_prover(
        mut self,
        wrap_air_prover: Arc<WrapProver<C>>,
        permit: ProverSemaphore,
    ) -> Self {
        self.wrap_air_prover_and_permits = Some((wrap_air_prover, permit));
        self
    }

    /// Set the core options.
    #[must_use]
    pub fn with_core_opts(self, opts: SP1CoreOpts) -> Self {
        self.with_config(|config| config.controller_config.opts = opts)
    }

    /// Get the core options from the builder.
    #[must_use]
    pub fn core_opts(&self) -> &SP1CoreOpts {
        &self.config.controller_config.opts
    }

    /// Mutate the worker config.
    #[must_use]
    pub fn with_config(mut self, f: impl FnOnce(&mut SP1WorkerConfig)) -> Self {
        f(&mut self.config);
        self
    }

    /// Build the worker.
    pub async fn build(self) -> anyhow::Result<SP1Worker<A, W, C>>
    where
        A: ArtifactClient,
        W: WorkerClient,
    {
        // Destructure the builder.
        let Self {
            config,
            core_air_prover_and_permits,
            compress_air_prover_and_permits,
            shrink_air_prover_and_permits,
            wrap_air_prover_and_permits,
            artifact_client,
            worker_client,
        } = self;

        let artifact_client =
            artifact_client.ok_or(anyhow::anyhow!("Artifact client is required"))?;
        let worker_client = worker_client.ok_or(anyhow::anyhow!("Worker client is required"))?;

        let opts = config.controller_config.opts.clone();

        // Create the prover engine
        let core_air_prover_and_permits = core_air_prover_and_permits
            .ok_or(anyhow::anyhow!("Core air prover and permit are required"))?;

        let compress_air_prover_and_permits = compress_air_prover_and_permits
            .ok_or(anyhow::anyhow!("Compress air prover and permit are required"))?;

        let shrink_air_prover_and_permits = shrink_air_prover_and_permits
            .ok_or(anyhow::anyhow!("Shrink air prover and permit are required"))?;

        let wrap_air_prover_and_permits = wrap_air_prover_and_permits
            .ok_or(anyhow::anyhow!("Wrap air prover and permit are required"))?;

        let prover_engine = SP1ProverEngine::new(
            config.prover_config,
            opts,
            artifact_client.clone(),
            worker_client.clone(),
            core_air_prover_and_permits,
            compress_air_prover_and_permits,
            shrink_air_prover_and_permits,
            wrap_air_prover_and_permits,
        )
        .await;

        // Create the verifier
        let core_verifier = C::core_verifier();
        let compress_verifier = C::compress_verifier();
        let shrink_verifier = C::shrink_verifier();
        let wrap_verifier = C::wrap_verifier();
        let recursion_vk_root = prover_engine.recursion_prover.recursion_vk_root();
        let recursion_vk_map = prover_engine.recursion_prover.recursion_vk_map().clone();
        let vk_verification = prover_engine.recursion_prover.vk_verification();
        let verifier = SP1Verifier {
            core: core_verifier,
            compress: compress_verifier,
            shrink: shrink_verifier,
            wrap: wrap_verifier,
            recursion_vk_root,
            recursion_vk_map,
            vk_verification,
            shrink_vk: prover_engine.recursion_prover.shrink_prover.verifying_key.clone(),
            wrap_vk: prover_engine.recursion_prover.wrap_prover.verifying_key.clone(),
        };

        let controller = SP1Controller::new(
            config.controller_config,
            artifact_client.clone(),
            worker_client.clone(),
            verifier.clone(),
        );

        Ok(SP1Worker::new(controller, prover_engine, verifier))
    }

    /// Set the path to the vk map. By default, the prover will use `prover/vk_map.bin` and this
    /// should only be changed for testing purposes.
    #[cfg(feature = "experimental")]
    pub fn with_vk_map_path(self, vk_map_path: String) -> SP1WorkerBuilder<C, A, W> {
        let SP1WorkerBuilder {
            mut config,
            core_air_prover_and_permits,
            compress_air_prover_and_permits,
            shrink_air_prover_and_permits,
            wrap_air_prover_and_permits,
            artifact_client,
            worker_client,
        } = self;

        config.prover_config.recursion_prover_config =
            config.prover_config.recursion_prover_config.with_vk_map_path(vk_map_path);

        SP1WorkerBuilder {
            config,
            core_air_prover_and_permits,
            compress_air_prover_and_permits,
            shrink_air_prover_and_permits,
            wrap_air_prover_and_permits,
            artifact_client,
            worker_client,
        }
    }

    /// Turn off vk verification for recursion proofs.
    #[cfg(feature = "experimental")]
    pub fn without_vk_verification(self) -> SP1WorkerBuilder<C, A, W> {
        let SP1WorkerBuilder {
            mut config,
            core_air_prover_and_permits,
            compress_air_prover_and_permits,
            shrink_air_prover_and_permits,
            wrap_air_prover_and_permits,
            artifact_client,
            worker_client,
        } = self;

        config.prover_config.recursion_prover_config =
            config.prover_config.recursion_prover_config.without_vk_verification();

        SP1WorkerBuilder {
            config,
            core_air_prover_and_permits,
            compress_air_prover_and_permits,
            shrink_air_prover_and_permits,
            wrap_air_prover_and_permits,
            artifact_client,
            worker_client,
        }
    }
}

/// Create a [SP1WorkerBuilder] for a CPU worker.
pub fn cpu_worker_builder() -> SP1WorkerBuilder<CpuSP1ProverComponents> {
    // Create the prover permits, setting it to having 4 provers.
    let prover_permits = ProverSemaphore::new(4);

    // Get the core options.
    let opts = SP1CoreOpts::default();

    let core_verifier = CpuSP1ProverComponents::core_verifier();
    let core_air_prover = Arc::new(CpuShardProver::new(core_verifier.shard_verifier().clone()));

    let recursion_verifier = CpuSP1ProverComponents::compress_verifier();
    let recursion_air_prover =
        Arc::new(CpuShardProver::new(recursion_verifier.shard_verifier().clone()));

    let shrink_verifier = CpuSP1ProverComponents::shrink_verifier();
    let shrink_prover = Arc::new(CpuShardProver::new(shrink_verifier.shard_verifier().clone()));

    let wrap_verifier = CpuSP1ProverComponents::wrap_verifier();
    let wrap_prover = Arc::new(CpuShardProver::new(wrap_verifier.shard_verifier().clone()));

    let artifact_client = InMemoryArtifactClient::new();
    let (worker_client, _) = LocalWorkerClient::init();

    SP1WorkerBuilder::new()
        .with_artifact_client(artifact_client)
        .with_worker_client(worker_client)
        .with_core_opts(opts)
        .with_core_air_prover(core_air_prover, prover_permits.clone())
        .with_compress_air_prover(recursion_air_prover, prover_permits.clone())
        .with_shrink_air_prover(shrink_prover, prover_permits.clone())
        .with_wrap_air_prover(wrap_prover, prover_permits)
}
