use csl_jagged::Poseidon2BabyBearJaggedCudaProverComponents;
use csl_machine::CudaProverComponents;
use slop_jagged::JaggedConfig;
use sp1_core_machine::riscv::RiscvAir;
use sp1_prover::{components::SP1ProverComponents, CompressAir, CoreSC, InnerSC, ShrinkAir};

pub struct CudaSP1ProverComponents;

impl SP1ProverComponents for CudaSP1ProverComponents {
    type JaggedBasefoldConfig = Poseidon2BabyBearJaggedCudaProverComponents;
    type CoreProverComponents = CudaProverComponents<
        Poseidon2BabyBearJaggedCudaProverComponents,
        RiscvAir<<CoreSC as JaggedConfig>::F>,
    >;
    type CompressProverComponents = CudaProverComponents<
        Poseidon2BabyBearJaggedCudaProverComponents,
        CompressAir<<InnerSC as JaggedConfig>::F>,
    >;
    type ShrinkProverComponents = CudaProverComponents<
        Poseidon2BabyBearJaggedCudaProverComponents,
        ShrinkAir<<InnerSC as JaggedConfig>::F>,
    >; //CpuProver<InnerSC, ShrinkAir<<InnerSC as StarkGenericConfig>::Val>>;
       //     type WrapProver = CpuProver<OuterSC, WrapAir<<OuterSC as StarkGenericConfig>::Val>>;
}
