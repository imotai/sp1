use slop_baby_bear::BabyBear;
use slop_jagged::{
    JaggedConfig, Poseidon2BabyBearJaggedCpuProverComponents,
    Poseidon2Bn254JaggedCpuProverComponents,
};
use sp1_core_machine::riscv::RiscvAir;
use sp1_recursion_circuit::machine::InnerVal;
use sp1_stark::{
    prover::{CpuMachineProverComponents, MachineProverComponents},
    MachineVerifier,
};

use crate::{
    core::CoreProverComponents,
    recursion::{RecursionProverComponents, WrapProverComponents},
    CompressAir, CoreSC, InnerSC, OuterSC, WrapAir,
};

pub struct SP1Config {}

pub type CoreProver<C> =
    <<C as SP1ProverComponents>::CoreComponents as MachineProverComponents>::Prover;

pub type RecursionProver<C> =
    <<C as SP1ProverComponents>::RecursionComponents as MachineProverComponents>::Prover;

pub type WrapProver<C> =
    <<C as SP1ProverComponents>::WrapComponents as MachineProverComponents>::Prover;

pub trait SP1ProverComponents: Send + Sync + 'static {
    /// The prover for making SP1 core proofs.
    type CoreComponents: CoreProverComponents;
    /// The prover for making SP1 recursive proofs.
    type RecursionComponents: RecursionProverComponents;
    type WrapComponents: WrapProverComponents;

    fn core_verifier() -> MachineVerifier<CoreSC, RiscvAir<BabyBear>> {
        <Self::CoreComponents as CoreProverComponents>::verifier()
    }

    fn recursion_verifier() -> MachineVerifier<InnerSC, CompressAir<InnerVal>> {
        <Self::RecursionComponents as RecursionProverComponents>::verifier()
    }

    fn shrink_verifier() -> MachineVerifier<InnerSC, CompressAir<InnerVal>> {
        <Self::RecursionComponents as RecursionProverComponents>::shrink_verifier()
    }

    fn wrap_verifier() -> MachineVerifier<OuterSC, WrapAir<InnerVal>> {
        <Self::WrapComponents as WrapProverComponents>::wrap_verifier()
    }
}

// ShardProver<CpuProverComponents<JaggedBasefoldProverComponents<Poseidon2BabyBear16BasefoldCpuProverComponents, HadamardJaggedSumcheckProver<CpuJaggedMleGenerator>, JaggedEvalSumcheckProver<BabyBear>>, RiscvAir<BabyBear>>>

pub struct CpuSP1ProverComponents;

impl SP1ProverComponents for CpuSP1ProverComponents {
    type CoreComponents = CpuMachineProverComponents<
        Poseidon2BabyBearJaggedCpuProverComponents,
        RiscvAir<<CoreSC as JaggedConfig>::F>,
    >;
    type RecursionComponents = CpuMachineProverComponents<
        Poseidon2BabyBearJaggedCpuProverComponents,
        CompressAir<<InnerSC as JaggedConfig>::F>,
    >;
    type WrapComponents = CpuMachineProverComponents<
        Poseidon2Bn254JaggedCpuProverComponents,
        WrapAir<<OuterSC as JaggedConfig>::F>,
    >;
}
