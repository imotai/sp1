use std::{future::Future, sync::Arc};

use slop_challenger::IopCtx;
use sp1_hypercube::{
    air::MachineAir,
    prover::{AirProver, ProverPermit, ProverSemaphore},
    Chip, Machine, MachineConfig, MachineVerifyingKey, ShardProof,
};

/// A prover for an AIR.
pub trait AirProverWorker<GC: IopCtx, C: MachineConfig<GC>, Air: MachineAir<GC::F>>:
    'static + Send + Sync
{
    /// Setup from a program.
    ///
    /// The setup phase produces a verifying key.
    fn setup(
        &self,
        program: Arc<Air::Program>,
        setup_permits: ProverSemaphore,
    ) -> impl Future<Output = MachineVerifyingKey<GC, C>> + Send;

    /// Get the machine.
    fn machine(&self) -> &Machine<GC::F, Air>;

    /// Setup and prove a shard.
    fn setup_and_prove_shard(
        &self,
        program: Arc<Air::Program>,
        record: Air::Record,
        vk: Option<MachineVerifyingKey<GC, C>>,
        prover_permits: ProverSemaphore,
        buffer_ptr: Option<usize>,
        challenger: &mut GC::Challenger,
    ) -> impl Future<Output = (MachineVerifyingKey<GC, C>, ShardProof<GC, C>, ProverPermit)> + Send;

    /// Get all the chips in the machine.
    fn all_chips(&self) -> &[Chip<GC::F, Air>] {
        self.machine().chips()
    }
}

impl<GC, C, Air, P> AirProverWorker<GC, C, Air> for P
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
    ) -> MachineVerifyingKey<GC, C> {
        let (_, vk) = self.setup(program, setup_permits).await;
        vk
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
        buffer_ptr: Option<usize>,
        challenger: &mut GC::Challenger,
    ) -> (MachineVerifyingKey<GC, C>, ShardProof<GC, C>, ProverPermit) {
        AirProver::setup_and_prove_shard(
            self,
            program,
            record,
            vk,
            prover_permits,
            buffer_ptr,
            challenger,
        )
        .await
    }
}
