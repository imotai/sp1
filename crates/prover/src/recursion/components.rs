use sp1_hypercube::{prover::MachineProverComponents, MachineVerifier, ShardVerifier};
use sp1_primitives::{
    fri_params::{recursion_fri_config, shrink_fri_config, wrap_fri_config},
    SP1Field, SP1GlobalContext, SP1OuterGlobalContext,
};
use sp1_recursion_circuit::machine::InnerVal;
use sp1_verifier::compressed::{RECURSION_LOG_STACKING_HEIGHT, RECURSION_MAX_LOG_ROW_COUNT};

use crate::{CompressAir, InnerSC, OuterSC, WrapAir};

pub const RECURSION_LOG_TRACE_AREA: usize = 27;
const SHRINK_LOG_STACKING_HEIGHT: u32 = 18;
pub(crate) const SHRINK_MAX_LOG_ROW_COUNT: usize = 19;

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

    fn shrink_verifier() -> MachineVerifier<SP1GlobalContext, InnerSC, CompressAir<InnerVal>> {
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
        let wrap_log_stacking_height = RECURSION_LOG_STACKING_HEIGHT;
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
