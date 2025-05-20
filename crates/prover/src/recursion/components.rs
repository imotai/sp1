use slop_baby_bear::BabyBear;
use slop_jagged::JaggedConfig;
use sp1_recursion_circuit::machine::InnerVal;
use sp1_recursion_executor::{ExecutionRecord, RecursionProgram};
use sp1_stark::{prover::MachineProverComponents, MachineVerifier, ShardVerifier};

use crate::{CompressAir, InnerSC, OuterSC, WrapAir};

const RECURSION_LOG_BLOWUP: usize = 1;
const RECURSION_LOG_STACKING_HEIGHT: u32 = 20;
pub(crate) const RECURSION_MAX_LOG_ROW_COUNT: usize = 20;

const SHRINK_LOG_BLOWUP: usize = 2;
const WRAP_LOG_BLOWUP: usize = 4;

pub trait RecursionProverComponents:
    MachineProverComponents<
    F = <InnerSC as JaggedConfig>::F,
    Config = InnerSC,
    Record = ExecutionRecord<<InnerSC as JaggedConfig>::F>,
    Program = RecursionProgram<<InnerSC as JaggedConfig>::F>,
    Challenger = <InnerSC as JaggedConfig>::Challenger,
    Air = CompressAir<<InnerSC as JaggedConfig>::F>,
>
{
    fn verifier() -> MachineVerifier<InnerSC, CompressAir<InnerVal>> {
        let compress_log_blowup = RECURSION_LOG_BLOWUP;
        let compress_log_stacking_height = RECURSION_LOG_STACKING_HEIGHT;
        let compress_max_log_row_count = RECURSION_MAX_LOG_ROW_COUNT;

        let machine = CompressAir::<BabyBear>::machine_wide_with_all_chips();
        let recursion_shard_verifier = ShardVerifier::from_basefold_parameters(
            compress_log_blowup,
            compress_log_stacking_height,
            compress_max_log_row_count,
            machine.clone(),
        );

        MachineVerifier::new(recursion_shard_verifier)
    }

    fn shrink_verifier() -> MachineVerifier<InnerSC, CompressAir<InnerVal>> {
        let shrink_log_blowup = SHRINK_LOG_BLOWUP;
        let shrink_log_stacking_height = RECURSION_LOG_STACKING_HEIGHT;
        let shrink_max_log_row_count = RECURSION_MAX_LOG_ROW_COUNT;

        let machine = CompressAir::<BabyBear>::machine_wide_with_all_chips();
        let recursion_shard_verifier = ShardVerifier::from_basefold_parameters(
            shrink_log_blowup,
            shrink_log_stacking_height,
            shrink_max_log_row_count,
            machine.clone(),
        );

        MachineVerifier::new(recursion_shard_verifier)
    }
}

pub trait WrapProverComponents:
    MachineProverComponents<
    F = <OuterSC as JaggedConfig>::F,
    Config = OuterSC,
    Record = ExecutionRecord<<OuterSC as JaggedConfig>::F>,
    Program = RecursionProgram<<OuterSC as JaggedConfig>::F>,
    Challenger = <OuterSC as JaggedConfig>::Challenger,
    Air = WrapAir<<OuterSC as JaggedConfig>::F>,
>
{
    fn wrap_verifier() -> MachineVerifier<OuterSC, WrapAir<InnerVal>> {
        let wrap_log_blowup = WRAP_LOG_BLOWUP;
        let wrap_log_stacking_height = RECURSION_LOG_STACKING_HEIGHT;
        let wrap_max_log_row_count = RECURSION_MAX_LOG_ROW_COUNT;

        let machine = WrapAir::<BabyBear>::machine_wide_with_all_chips();
        let wrap_shard_verifier = ShardVerifier::from_basefold_parameters(
            wrap_log_blowup,
            wrap_log_stacking_height,
            wrap_max_log_row_count,
            machine.clone(),
        );

        MachineVerifier::new(wrap_shard_verifier)
    }
}

impl<C> RecursionProverComponents for C where
    C: MachineProverComponents<
        F = <InnerSC as JaggedConfig>::F,
        Config = InnerSC,
        Record = ExecutionRecord<<InnerSC as JaggedConfig>::F>,
        Program = RecursionProgram<<InnerSC as JaggedConfig>::F>,
        Challenger = <InnerSC as JaggedConfig>::Challenger,
        Air = CompressAir<<InnerSC as JaggedConfig>::F>,
    >
{
}

impl<C> WrapProverComponents for C where
    C: MachineProverComponents<
        F = <OuterSC as JaggedConfig>::F,
        Config = OuterSC,
        Record = ExecutionRecord<<OuterSC as JaggedConfig>::F>,
        Program = RecursionProgram<<OuterSC as JaggedConfig>::F>,
        Challenger = <OuterSC as JaggedConfig>::Challenger,
        Air = WrapAir<<OuterSC as JaggedConfig>::F>,
    >
{
}
