use slop_baby_bear::BabyBear;
use slop_jagged::{JaggedConfig, Poseidon2BabyBearJaggedCpuProverComponents};
use sp1_core_machine::riscv::RiscvAir;
use sp1_recursion_circuit::machine::InnerVal;
use sp1_stark::{
    prover::{CpuProverComponents, MachineProverComponents},
    MachineVerifier,
};

use crate::{
    core::CoreProverComponents, recursion::RecursionProverComponents, CompressAir, CoreSC, InnerSC,
    ShrinkAir,
};

pub struct SP1Config {}

pub trait SP1ProverComponents: Send + Sync + 'static {
    /// The prover for making SP1 core proofs.
    type CoreComponents: CoreProverComponents;
    /// The prover for making SP1 recursive proofs.
    type RecursionComponents: RecursionProverComponents;

    /// The prover for shrinking compressed proofs.
    type ShrinkProverComponents: MachineProverComponents<
        Config = InnerSC,
        Air = ShrinkAir<<InnerSC as JaggedConfig>::F>,
    >;

    fn core_verifier() -> MachineVerifier<CoreSC, RiscvAir<BabyBear>> {
        <Self::CoreComponents as CoreProverComponents>::verifier()
    }

    fn recursion_verifier() -> MachineVerifier<InnerSC, CompressAir<InnerVal>> {
        <Self::RecursionComponents as RecursionProverComponents>::verifier()
    }

    // /// The prover for wrapping compressed proofs into SNARK-friendly field elements.
    // type WrapProver: MachineProver<OuterSC, WrapAir<<OuterSC as StarkGenericConfig>::Val>>
    //     + Send
    //     + Sync;
}

// ShardProver<CpuProverComponents<JaggedBasefoldProverComponents<Poseidon2BabyBear16BasefoldCpuProverComponents, HadamardJaggedSumcheckProver<CpuJaggedMleGenerator>, JaggedEvalSumcheckProver<BabyBear>>, RiscvAir<BabyBear>>>

pub struct CpuSP1ProverComponents;

impl SP1ProverComponents for CpuSP1ProverComponents {
    type CoreComponents = CpuProverComponents<
        Poseidon2BabyBearJaggedCpuProverComponents,
        RiscvAir<<CoreSC as JaggedConfig>::F>,
    >;
    type RecursionComponents = CpuProverComponents<
        Poseidon2BabyBearJaggedCpuProverComponents,
        CompressAir<<InnerSC as JaggedConfig>::F>,
    >;
    type ShrinkProverComponents = CpuProverComponents<
        Poseidon2BabyBearJaggedCpuProverComponents,
        ShrinkAir<<InnerSC as JaggedConfig>::F>,
    >;
}
