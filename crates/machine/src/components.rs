use std::marker::PhantomData;

use csl_air::SymbolicProverFolder;
use csl_cuda::TaskScope;
use csl_jagged::Poseidon2BabyBearJaggedCudaProverComponents;
use csl_zerocheck::ZerocheckEvalProgramProverData;
use serde::{Deserialize, Serialize};
use slop_air::Air;
use slop_algebra::extension::BinomialExtensionField;
use slop_baby_bear::BabyBear;
use slop_jagged::{JaggedProver, JaggedProverComponents};
use sp1_stark::{
    air::MachineAir,
    prover::{DefaultTraceGenerator, MachineProverComponents, ShardProver, ZerocheckAir},
    BabyBearPoseidon2, ShardVerifier,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CudaProverComponents<PcsComponents, A>(PhantomData<(A, PcsComponents)>);

pub type CudaProver<PcsComponents, A> = ShardProver<CudaProverComponents<PcsComponents, A>>;

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

impl<A> MachineProverComponents
    for CudaProverComponents<Poseidon2BabyBearJaggedCudaProverComponents, A>
where
    A: MachineAir<F> + ZerocheckAir<F, EF> + std::fmt::Debug,
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

    type TraceGenerator = DefaultTraceGenerator<F, A, TaskScope>;

    type ZerocheckProverData = ZerocheckEvalProgramProverData<Self::F, Self::EF, A>;

    type PcsProverComponents = Poseidon2BabyBearJaggedCudaProverComponents;
}

pub fn new_cuda_prover<A>(
    verifier: ShardVerifier<BabyBearPoseidon2, A>,
    scope: TaskScope,
) -> CudaProver<Poseidon2BabyBearJaggedCudaProverComponents, A>
where
    A: MachineAir<F>
        + ZerocheckAir<F, EF>
        + for<'a> Air<SymbolicProverFolder<'a>>
        + std::fmt::Debug,
{
    let ShardVerifier { pcs_verifier, machine } = verifier;
    let pcs_prover = JaggedProver::from_verifier(&pcs_verifier);
    let airs = machine.chips().iter().map(|chip| chip.air.clone()).collect::<Vec<_>>();
    let trace_generator = DefaultTraceGenerator::new_in(machine, scope.clone());
    let zerocheck_data = ZerocheckEvalProgramProverData::new(&airs, scope);
    CudaProver { pcs_prover, zerocheck_prover_data: zerocheck_data, trace_generator }
}
