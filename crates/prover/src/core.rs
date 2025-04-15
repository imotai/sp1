use std::sync::Arc;

use slop_baby_bear::BabyBear;
use slop_futures::handle::TaskHandle;
use slop_jagged::{JaggedConfig, JaggedProverComponents};
use sp1_core_executor::{ExecutionRecord, Program};
use sp1_core_machine::riscv::RiscvAir;
use sp1_stark::{
    prover::{
        CoreProofShape, MachineProver, MachineProverComponents, MachineProverError,
        MachineProvingKey,
    },
    Machine, MachineVerifier, MachineVerifyingKey, ShardProof, ShardVerifier,
};

use crate::{error::RecursionPrrogramError, shapes::SP1RecursionShape, CoreSC, SP1VerifyingKey};

pub struct SP1CoreProver<C: CoreProverComponents> {
    prover: MachineProver<C>,
}

const CORE_LOG_BLOWUP: usize = 1;
const CORE_LOG_STACKING_HEIGHT: u32 = 21;
const CORE_MAX_LOG_ROW_COUNT: usize = 22;

pub trait CoreProverComponents:
    MachineProverComponents<
    Config = CoreSC,
    Program = Program,
    F = <CoreSC as JaggedConfig>::F,
    Record = sp1_core_executor::ExecutionRecord,
    Air = RiscvAir<<CoreSC as JaggedConfig>::F>,
    Challenger = <CoreSC as JaggedConfig>::Challenger,
    PcsProverComponents: JaggedProverComponents<Config = CoreSC>,
>
{
    /// The default verifier for the core prover.
    ///
    /// Thew verifier fixes the parameters of the underlying proof system.
    fn verifier() -> MachineVerifier<CoreSC, RiscvAir<BabyBear>> {
        let core_log_blowup = CORE_LOG_BLOWUP;
        let core_log_stacking_height = CORE_LOG_STACKING_HEIGHT;
        let core_max_log_row_count = CORE_MAX_LOG_ROW_COUNT;

        let machine = RiscvAir::machine();

        let core_verifier = ShardVerifier::from_basefold_parameters(
            core_log_blowup,
            core_log_stacking_height,
            core_max_log_row_count,
            machine.clone(),
        );

        MachineVerifier::new(core_verifier)
    }
}

impl<C> CoreProverComponents for C where
    C: MachineProverComponents<
        Config = CoreSC,
        Program = Program,
        F = <CoreSC as JaggedConfig>::F,
        Record = sp1_core_executor::ExecutionRecord,
        Air = RiscvAir<<CoreSC as JaggedConfig>::F>,
        Challenger = <CoreSC as JaggedConfig>::Challenger,
        PcsProverComponents: JaggedProverComponents<Config = CoreSC>,
    >
{
}

impl<C: CoreProverComponents> SP1CoreProver<C> {
    #[inline]
    #[must_use]
    pub fn new(prover: MachineProver<C>) -> Self {
        Self { prover }
    }

    /// Get the number of workers for the core prover
    #[must_use]
    #[inline]
    pub fn num_prover_workers(&self) -> usize {
        self.prover.num_workers()
    }

    pub fn machine(&self) -> &Machine<C::F, RiscvAir<C::F>> {
        self.prover.machine()
    }

    /// Setup the core prover
    #[must_use]
    pub async fn setup(
        &self,
        elf: &[u8],
    ) -> (MachineProvingKey<C>, C::Program, MachineVerifyingKey<CoreSC>) {
        let program = <C as MachineProverComponents>::Program::from(elf).unwrap();
        let (pk, vk) = self.prover.setup(Arc::new(program.clone()), None).await.unwrap();
        (pk, program, vk)
    }

    /// Prove a core shard
    #[inline]
    #[must_use]
    pub fn prove_shard(
        &self,
        pk: Arc<MachineProvingKey<C>>,
        record: ExecutionRecord,
    ) -> TaskHandle<ShardProof<CoreSC>, MachineProverError> {
        self.prover.prove_shard(pk.clone(), record)
    }

    pub fn verifier(&self) -> &MachineVerifier<C::Config, RiscvAir<C::F>> {
        self.prover.verifier()
    }

    /// Setup and prove a core shard
    ///
    /// The prover will compute the `pk` from the `vk` if exists, or will compute the setup and
    /// prove the shard in one step.
    #[inline]
    #[must_use]
    pub fn setup_and_prove_shard(
        &self,
        program: Arc<Program>,
        vk: Option<SP1VerifyingKey>,
        record: ExecutionRecord,
    ) -> TaskHandle<(MachineVerifyingKey<CoreSC>, ShardProof<CoreSC>), MachineProverError> {
        self.prover.setup_and_prove_shard(program, vk.map(|vk| vk.vk), record)
    }

    /// Get the shape of the core shard
    #[inline]
    pub fn core_shape_from_record(
        &self,
        record: &ExecutionRecord,
    ) -> Option<CoreProofShape<C::F, C::Air>> {
        self.prover.shape_from_record(record)
    }

    /// Get the witness for a core shard
    pub fn recursion_shape(
        &self,
        record: &ExecutionRecord,
    ) -> Result<SP1RecursionShape, RecursionPrrogramError> {
        let proof_shape = self
            .prover
            .shape_from_record(record)
            .ok_or(RecursionPrrogramError::InvalidRecordShape)?;

        Ok(SP1RecursionShape {
            proof_shapes: vec![proof_shape],
            max_log_row_count: self.prover.max_log_row_count(),
            log_blowup: self.prover.verifier().fri_config().log_blowup,
            log_stacking_height: self.prover.log_stacking_height() as usize,
        })
    }
}
