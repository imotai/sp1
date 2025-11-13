use std::{future::Future, sync::Arc};

use slop_challenger::IopCtx;
use sp1_hypercube::{
    air::MachineAir,
    prover::{AirProver, ProverPermit, ProverSemaphore, ProvingKey},
    Chip, Machine, MachineConfig, MachineVerifyingKey, ShardProof,
};

/// A prover for an AIR.
pub trait AirProverWorker<
    GC: IopCtx,
    C: MachineConfig<GC>,
    Air: MachineAir<GC::F>,
    P: AirProver<GC, C, Air>,
>: 'static + Send + Sync
{
    /// Setup from a program.
    ///
    /// The setup phase produces a verifying key.
    #[allow(clippy::type_complexity)]
    fn setup(
        &self,
        program: Arc<Air::Program>,
        setup_permits: ProverSemaphore,
    ) -> impl Future<Output = (Arc<ProvingKey<GC, C, Air, P>>, MachineVerifyingKey<GC, C>)> + Send;

    /// Get the machine.
    fn machine(&self) -> &Machine<GC::F, Air>;

    /// Setup and prove a shard.
    fn setup_and_prove_shard(
        &self,
        program: Arc<Air::Program>,
        record: Air::Record,
        vk: Option<MachineVerifyingKey<GC, C>>,
        prover_permits: ProverSemaphore,
        challenger: &mut GC::Challenger,
    ) -> impl Future<Output = (MachineVerifyingKey<GC, C>, ShardProof<GC, C>, ProverPermit)> + Send;

    /// Setup and prove a shard.
    fn prove_shard_with_pk(
        &self,
        pk: Arc<ProvingKey<GC, C, Air, P>>,
        record: Air::Record,
        prover_permits: ProverSemaphore,
        challenger: &mut GC::Challenger,
    ) -> impl Future<Output = (ShardProof<GC, C>, ProverPermit)> + Send;

    /// Get all the chips in the machine.
    fn all_chips(&self) -> &[Chip<GC::F, Air>] {
        self.machine().chips()
    }
}

impl<GC, C, Air, P> AirProverWorker<GC, C, Air, P> for P
where
    GC: IopCtx,
    C: MachineConfig<GC>,
    Air: MachineAir<GC::F>,
    P: AirProver<GC, C, Air>,
{
    async fn setup(
        &self,
        program: Arc<Air::Program>,
        setup_permits: ProverSemaphore,
    ) -> (Arc<ProvingKey<GC, C, Air, P>>, MachineVerifyingKey<GC, C>) {
        let (preprocessed, vk) = self.setup(program, setup_permits).await;
        (preprocessed.pk, vk)
    }

    /// Get the machine.
    fn machine(&self) -> &Machine<GC::F, Air> {
        AirProver::machine(self)
    }

    /// Setup and prove a shard.
    async fn setup_and_prove_shard(
        &self,
        program: Arc<Air::Program>,
        record: Air::Record,
        vk: Option<MachineVerifyingKey<GC, C>>,
        prover_permits: ProverSemaphore,
        challenger: &mut GC::Challenger,
    ) -> (MachineVerifyingKey<GC, C>, ShardProof<GC, C>, ProverPermit) {
        AirProver::setup_and_prove_shard(self, program, record, vk, prover_permits, challenger)
            .await
    }

    /// Prove a shard from a given pk.
    async fn prove_shard_with_pk(
        &self,
        pk: Arc<ProvingKey<GC, C, Air, P>>,
        record: Air::Record,
        prover_permits: ProverSemaphore,
        challenger: &mut GC::Challenger,
    ) -> (ShardProof<GC, C>, ProverPermit) {
        AirProver::prove_shard_with_pk(self, pk, record, prover_permits, challenger).await
    }
}
