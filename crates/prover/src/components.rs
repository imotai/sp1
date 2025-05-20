use std::marker::PhantomData;

use csl_air::{air_block::BlockAir, SymbolicProverFolder};
use csl_cuda::TaskScope;
use csl_jagged::{
    Poseidon2BabyBearJaggedCudaProverComponents, Poseidon2Bn254JaggedCudaProverComponents,
};
use csl_logup_gkr::LogupGkrCudaProverComponents;
use csl_tracegen::{CudaTraceGenerator, CudaTracegenAir};
use csl_zerocheck::ZerocheckEvalProgramProverData;
use serde::{Deserialize, Serialize};
use slop_algebra::extension::BinomialExtensionField;
use slop_baby_bear::BabyBear;
use slop_jagged::{DefaultJaggedProver, JaggedConfig, JaggedProver, JaggedProverComponents};
use sp1_core_machine::riscv::RiscvAir;
use sp1_prover::{components::SP1ProverComponents, CompressAir, InnerSC, OuterSC, WrapAir};
use sp1_stark::{
    air::MachineAir,
    prover::{MachineProverComponents, ShardProver, ZerocheckAir},
    GkrProverImpl, MachineConfig, ShardVerifier,
};

pub struct CudaSP1ProverComponents;

impl SP1ProverComponents for CudaSP1ProverComponents {
    type CoreComponents =
        CudaProverComponents<Poseidon2BabyBearJaggedCudaProverComponents, RiscvAir<BabyBear>>;
    type RecursionComponents = CudaProverComponents<
        Poseidon2BabyBearJaggedCudaProverComponents,
        CompressAir<<InnerSC as JaggedConfig>::F>,
    >;
    type WrapComponents = CudaProverComponents<
        Poseidon2Bn254JaggedCudaProverComponents,
        WrapAir<<OuterSC as JaggedConfig>::F>,
    >;
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CudaProverComponents<PcsComponents, A>(PhantomData<(A, PcsComponents)>);

pub type CudaProver<PcsComponents, A> = ShardProver<CudaProverComponents<PcsComponents, A>>;

impl<JC, A> MachineProverComponents for CudaProverComponents<JC, A>
where
    JC: JaggedProverComponents<
        F = BabyBear,
        EF = BinomialExtensionField<BabyBear, 4>,
        A = TaskScope,
    >,
    A: CudaTracegenAir<JC::F> + ZerocheckAir<JC::F, JC::EF> + std::fmt::Debug,
{
    type F = JC::F;
    type EF = JC::EF;
    type Program = <A as MachineAir<JC::F>>::Program;
    type Record = <A as MachineAir<JC::F>>::Record;
    type Air = A;
    type B = TaskScope;

    type Commitment = JC::Commitment;

    type Challenger = JC::Challenger;

    type Config = JC::Config;

    type TraceGenerator = CudaTraceGenerator<Self::F, A>;

    type ZerocheckProverData = ZerocheckEvalProgramProverData<Self::F, Self::EF, A>;

    type PcsProverComponents = JC;

    type GkrProver =
        GkrProverImpl<LogupGkrCudaProverComponents<Self::F, Self::EF, A, Self::Challenger>>;
}

pub fn new_cuda_prover_sumcheck_eval<C, Comp, A>(
    verifier: ShardVerifier<C, A>,
    scope: TaskScope,
) -> CudaProver<Comp, A>
where
    C: MachineConfig<F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>>,
    Comp: JaggedProverComponents<A = TaskScope, Config = C, F = C::F, EF = C::EF>
        + DefaultJaggedProver,
    A: MachineAir<C::F>
        + CudaTracegenAir<C::F>
        + for<'a> BlockAir<SymbolicProverFolder<'a>>
        + ZerocheckAir<C::F, C::EF>
        + std::fmt::Debug,
{
    let ShardVerifier { pcs_verifier, machine } = verifier;
    let pcs_prover = JaggedProver::from_verifier(&pcs_verifier);
    let airs = machine.chips().iter().map(|chip| chip.air.clone()).collect::<Vec<_>>();
    let trace_generator = CudaTraceGenerator::new_in(machine, scope.clone());
    let zerocheck_data = ZerocheckEvalProgramProverData::new(&airs, scope);
    let logup_gkr_prover = LogupGkrCudaProverComponents::default_prover();
    CudaProver {
        trace_generator,
        logup_gkr_prover,
        zerocheck_prover_data: zerocheck_data,
        pcs_prover,
    }
}
