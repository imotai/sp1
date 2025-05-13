use std::marker::PhantomData;

use csl_air::{air_block::BlockAir, SymbolicProverFolder};
use csl_cuda::TaskScope;
use csl_jagged::Poseidon2BabyBearJaggedCudaProverComponents;
use csl_logup_gkr::LogupGkrCudaProverComponents;
use csl_tracegen::{CudaTraceGenerator, CudaTracegenAir};
use csl_zerocheck::ZerocheckEvalProgramProverData;
use serde::{Deserialize, Serialize};
use slop_algebra::extension::BinomialExtensionField;
use slop_baby_bear::BabyBear;
use slop_jagged::{JaggedConfig, JaggedProver, JaggedProverComponents};
use sp1_core_machine::riscv::RiscvAir;
use sp1_prover::{components::SP1ProverComponents, CompressAir, InnerSC, ShrinkAir};
use sp1_stark::{
    air::MachineAir,
    prover::{MachineProverComponents, ShardProver, ZerocheckAir},
    BabyBearPoseidon2, GkrProverImpl, ShardVerifier,
};

pub struct CudaSP1ProverComponents;

impl SP1ProverComponents for CudaSP1ProverComponents {
    type CoreComponents =
        CudaProverComponents<Poseidon2BabyBearJaggedCudaProverComponents, RiscvAir<BabyBear>>;
    type RecursionComponents = CudaProverComponents<
        Poseidon2BabyBearJaggedCudaProverComponents,
        CompressAir<<InnerSC as JaggedConfig>::F>,
    >;
    type ShrinkProverComponents = CudaProverComponents<
        Poseidon2BabyBearJaggedCudaProverComponents,
        ShrinkAir<<InnerSC as JaggedConfig>::F>,
    >;
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CudaProverComponents<PcsComponents, A>(PhantomData<(A, PcsComponents)>);

pub type CudaProver<PcsComponents, A> = ShardProver<CudaProverComponents<PcsComponents, A>>;

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

impl<A> MachineProverComponents
    for CudaProverComponents<Poseidon2BabyBearJaggedCudaProverComponents, A>
where
    A: CudaTracegenAir<F> + ZerocheckAir<F, EF> + std::fmt::Debug,
{
    type F = F;
    type EF = EF;
    type Program = <A as MachineAir<F>>::Program;
    type Record = <A as MachineAir<F>>::Record;
    type Air = A;
    type B = TaskScope;

    type Commitment =
        <Poseidon2BabyBearJaggedCudaProverComponents as JaggedProverComponents>::Commitment;

    type Challenger =
        <Poseidon2BabyBearJaggedCudaProverComponents as JaggedProverComponents>::Challenger;

    type Config = <Poseidon2BabyBearJaggedCudaProverComponents as JaggedProverComponents>::Config;

    type TraceGenerator = CudaTraceGenerator<F, A>;

    type ZerocheckProverData = ZerocheckEvalProgramProverData<Self::F, Self::EF, A>;

    type PcsProverComponents = Poseidon2BabyBearJaggedCudaProverComponents;

    type GkrProver =
        GkrProverImpl<LogupGkrCudaProverComponents<Self::F, Self::EF, A, Self::Challenger>>;
}

pub fn new_cuda_prover_sumcheck_eval<A>(
    verifier: ShardVerifier<BabyBearPoseidon2, A>,
    scope: TaskScope,
) -> CudaProver<Poseidon2BabyBearJaggedCudaProverComponents, A>
where
    A: CudaTracegenAir<F>
        + for<'a> BlockAir<SymbolicProverFolder<'a>>
        + ZerocheckAir<F, EF>
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
