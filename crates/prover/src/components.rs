use slop_jagged::SP1OuterConfig;
use sp1_core_executor::HEIGHT_THRESHOLD;
use sp1_core_machine::riscv::RiscvAir;
use sp1_hypercube::{
    prover::{
        CpuMachineProverComponents, MachineProverComponents, SP1CpuJaggedProverComponents,
        SP1OuterCpuJaggedProverComponents,
    },
    MachineVerifier, SP1CoreJaggedConfig, ShardVerifier,
};
use sp1_primitives::{
    fri_params::{core_fri_config, recursion_fri_config, shrink_fri_config, wrap_fri_config},
    SP1Field, SP1GlobalContext, SP1OuterGlobalContext,
};
use sp1_recursion_circuit::machine::InnerVal;
use sp1_verifier::compressed::{RECURSION_LOG_STACKING_HEIGHT, RECURSION_MAX_LOG_ROW_COUNT};
use static_assertions::const_assert;

pub const CORE_LOG_STACKING_HEIGHT: u32 = 21;
pub const CORE_MAX_LOG_ROW_COUNT: usize = 22;

const_assert!(HEIGHT_THRESHOLD <= (1 << CORE_MAX_LOG_ROW_COUNT));

/// The configuration for the core prover.
pub type CoreSC = SP1CoreJaggedConfig;

/// The configuration for the inner prover.
pub type InnerSC = SP1CoreJaggedConfig;

/// The configuration for the outer prover.
pub type OuterSC = SP1OuterConfig;

use sp1_recursion_machine::RecursionAir;

const COMPRESS_DEGREE: usize = 3;
const SHRINK_DEGREE: usize = 3;
const WRAP_DEGREE: usize = 3;

pub type CompressAir<F> = RecursionAir<F, COMPRESS_DEGREE, 2>;
pub type ShrinkAir<F> = RecursionAir<F, SHRINK_DEGREE, 2>;
pub type WrapAir<F> = RecursionAir<F, WRAP_DEGREE, 1>;

pub const RECURSION_LOG_TRACE_AREA: usize = 27;
const SHRINK_LOG_STACKING_HEIGHT: u32 = 18;
pub(crate) const SHRINK_MAX_LOG_ROW_COUNT: usize = 19;

pub(crate) const WRAP_LOG_STACKING_HEIGHT: u32 = 21;

pub trait CoreProverComponents:
    MachineProverComponents<SP1GlobalContext, Config = CoreSC, Air = RiscvAir<SP1Field>>
{
    /// The default verifier for the core prover.
    ///
    /// The verifier fixes the parameters of the underlying proof system.
    fn verifier() -> MachineVerifier<SP1GlobalContext, CoreSC, RiscvAir<SP1Field>> {
        let core_log_stacking_height = CORE_LOG_STACKING_HEIGHT;
        let core_max_log_row_count = CORE_MAX_LOG_ROW_COUNT;

        let machine = RiscvAir::machine();

        let core_verifier = ShardVerifier::from_basefold_parameters(
            core_fri_config(),
            core_log_stacking_height,
            core_max_log_row_count,
            machine.clone(),
        );

        MachineVerifier::new(core_verifier)
    }
}

impl<C> CoreProverComponents for C where
    C: MachineProverComponents<SP1GlobalContext, Config = CoreSC, Air = RiscvAir<SP1Field>>
{
}

pub trait RecursionProverComponents:
    MachineProverComponents<SP1GlobalContext, Config = InnerSC, Air = CompressAir<SP1Field>>
{
    fn verifier() -> MachineVerifier<SP1GlobalContext, InnerSC, CompressAir<InnerVal>> {
        let compress_log_stacking_height = RECURSION_LOG_STACKING_HEIGHT;
        let compress_max_log_row_count = RECURSION_MAX_LOG_ROW_COUNT;

        let machine = CompressAir::<SP1Field>::compress_machine();
        let recursion_shard_verifier = ShardVerifier::from_basefold_parameters(
            recursion_fri_config(),
            compress_log_stacking_height,
            compress_max_log_row_count,
            machine.clone(),
        );

        MachineVerifier::new(recursion_shard_verifier)
    }

    fn shrink_verifier() -> MachineVerifier<SP1GlobalContext, InnerSC, ShrinkAir<InnerVal>> {
        let shrink_log_stacking_height = SHRINK_LOG_STACKING_HEIGHT;
        let shrink_max_log_row_count = SHRINK_MAX_LOG_ROW_COUNT;

        let machine = CompressAir::<SP1Field>::shrink_machine();
        let recursion_shard_verifier = ShardVerifier::from_basefold_parameters(
            shrink_fri_config(),
            shrink_log_stacking_height,
            shrink_max_log_row_count,
            machine.clone(),
        );

        MachineVerifier::new(recursion_shard_verifier)
    }
}

pub trait WrapProverComponents:
    MachineProverComponents<SP1OuterGlobalContext, Config = OuterSC, Air = WrapAir<SP1Field>>
{
    fn wrap_verifier() -> MachineVerifier<SP1OuterGlobalContext, OuterSC, WrapAir<InnerVal>> {
        let wrap_log_stacking_height = WRAP_LOG_STACKING_HEIGHT;
        let wrap_max_log_row_count = RECURSION_MAX_LOG_ROW_COUNT;

        let machine = WrapAir::<SP1Field>::wrap_machine();
        let wrap_shard_verifier = ShardVerifier::from_basefold_parameters(
            wrap_fri_config(),
            wrap_log_stacking_height,
            wrap_max_log_row_count,
            machine.clone(),
        );

        MachineVerifier::new(wrap_shard_verifier)
    }
}

impl<C> RecursionProverComponents for C where
    C: MachineProverComponents<SP1GlobalContext, Config = InnerSC, Air = CompressAir<SP1Field>>
{
}

impl<C> WrapProverComponents for C where
    C: MachineProverComponents<SP1OuterGlobalContext, Config = OuterSC, Air = WrapAir<SP1Field>>
{
}

pub type CoreProver<C> = <<C as SP1ProverComponents>::CoreComponents as MachineProverComponents<
    SP1GlobalContext,
>>::Prover;

pub type RecursionProver<C> =
    <<C as SP1ProverComponents>::RecursionComponents as MachineProverComponents<
        SP1GlobalContext,
    >>::Prover;

pub type WrapProver<C> = <<C as SP1ProverComponents>::WrapComponents as MachineProverComponents<
    SP1OuterGlobalContext,
>>::Prover;

pub trait SP1ProverComponents: Send + Sync + 'static {
    /// The prover for making SP1 core proofs.
    type CoreComponents: CoreProverComponents;
    /// The prover for making SP1 recursive proofs.
    type RecursionComponents: RecursionProverComponents;
    type WrapComponents: WrapProverComponents;

    fn core_verifier() -> MachineVerifier<SP1GlobalContext, CoreSC, RiscvAir<SP1Field>> {
        <Self::CoreComponents as CoreProverComponents>::verifier()
    }

    fn compress_verifier() -> MachineVerifier<SP1GlobalContext, InnerSC, CompressAir<InnerVal>> {
        <Self::RecursionComponents as RecursionProverComponents>::verifier()
    }

    fn shrink_verifier() -> MachineVerifier<SP1GlobalContext, InnerSC, CompressAir<InnerVal>> {
        <Self::RecursionComponents as RecursionProverComponents>::shrink_verifier()
    }

    fn wrap_verifier() -> MachineVerifier<SP1OuterGlobalContext, OuterSC, WrapAir<InnerVal>> {
        <Self::WrapComponents as WrapProverComponents>::wrap_verifier()
    }
}

pub struct CpuSP1ProverComponents;

impl SP1ProverComponents for CpuSP1ProverComponents {
    type CoreComponents = CpuMachineProverComponents<
        SP1GlobalContext,
        SP1CpuJaggedProverComponents,
        RiscvAir<SP1Field>,
    >;
    type RecursionComponents = CpuMachineProverComponents<
        SP1GlobalContext,
        SP1CpuJaggedProverComponents,
        CompressAir<SP1Field>,
    >;
    type WrapComponents = CpuMachineProverComponents<
        SP1OuterGlobalContext,
        SP1OuterCpuJaggedProverComponents,
        WrapAir<SP1Field>,
    >;
}
