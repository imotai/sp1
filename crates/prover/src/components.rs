use slop_jagged::{
    JaggedConfig, JaggedProverComponents, Poseidon2BabyBearJaggedCpuProverComponents,
};
use sp1_core_executor::Program;
use sp1_core_machine::riscv::RiscvAir;
use sp1_recursion_executor::{ExecutionRecord, RecursionProgram};
// use sp1_stark::{CpuProver, MachineProver, StarkGenericConfig};
use sp1_stark::prover::{CpuProverComponents, MachineProverComponents};

use crate::{CompressAir, CoreSC, InnerSC, ShrinkAir};

pub trait SP1ProverComponents: Send + Sync + 'static {
    type JaggedBasefoldConfig: JaggedProverComponents<Config = CoreSC>;
    /// The prover for making SP1 core proofs.
    type CoreProverComponents: MachineProverComponents<
        Config = CoreSC,
        Program = Program,
        F = <CoreSC as JaggedConfig>::F,
        Record = sp1_core_executor::ExecutionRecord,
        Air = RiscvAir<<CoreSC as JaggedConfig>::F>,
        Challenger = <CoreSC as JaggedConfig>::Challenger,
        PcsProverComponents = Self::JaggedBasefoldConfig,
    >;
    /// The prover for making SP1 recursive proofs.
    type CompressProverComponents: MachineProverComponents<
        F = <InnerSC as JaggedConfig>::F,
        Config = InnerSC,
        Record = ExecutionRecord<<InnerSC as JaggedConfig>::F>,
        Program = RecursionProgram<<InnerSC as JaggedConfig>::F>,
        Challenger = <InnerSC as JaggedConfig>::Challenger,
        Air = CompressAir<<InnerSC as JaggedConfig>::F>,
    >;
    /// The prover for shrinking compressed proofs.
    type ShrinkProverComponents: MachineProverComponents<
        Config = InnerSC,
        Air = ShrinkAir<<InnerSC as JaggedConfig>::F>,
    >;

    // /// The prover for wrapping compressed proofs into SNARK-friendly field elements.
    // type WrapProver: MachineProver<OuterSC, WrapAir<<OuterSC as StarkGenericConfig>::Val>>
    //     + Send
    //     + Sync;
}

// ShardProver<CpuProverComponents<JaggedBasefoldProverComponents<Poseidon2BabyBear16BasefoldCpuProverComponents, HadamardJaggedSumcheckProver<CpuJaggedMleGenerator>, JaggedEvalSumcheckProver<BabyBear>>, RiscvAir<BabyBear>>>

pub struct CpuSP1ProverComponents;

impl SP1ProverComponents for CpuSP1ProverComponents {
    type JaggedBasefoldConfig = Poseidon2BabyBearJaggedCpuProverComponents;
    type CoreProverComponents = CpuProverComponents<
        Poseidon2BabyBearJaggedCpuProverComponents,
        RiscvAir<<CoreSC as JaggedConfig>::F>,
    >;
    type CompressProverComponents = CpuProverComponents<
        Poseidon2BabyBearJaggedCpuProverComponents,
        CompressAir<<InnerSC as JaggedConfig>::F>,
    >;
    type ShrinkProverComponents = CpuProverComponents<
        Poseidon2BabyBearJaggedCpuProverComponents,
        ShrinkAir<<InnerSC as JaggedConfig>::F>,
    >; //CpuProver<InnerSC, ShrinkAir<<InnerSC as StarkGenericConfig>::Val>>;
       //     type WrapProver = CpuProver<OuterSC, WrapAir<<OuterSC as StarkGenericConfig>::Val>>;
}
