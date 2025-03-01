use std::marker::PhantomData;

use p3_uni_stark::SymbolicAirBuilder;
use serde::{Deserialize, Serialize};
use slop_air::Air;
use slop_algebra::extension::BinomialExtensionField;
use slop_alloc::CpuBackend;
use slop_baby_bear::BabyBear;
use slop_basefold_prover::BasefoldProverComponents;
use slop_jagged::{
    JaggedBasefoldProverComponents, JaggedPcsVerifier, JaggedProver, JaggedProverComponents,
    Poseidon2BabyBearJaggedCpuProverComponents,
};

use crate::{air::MachineAir, BabyBearPoseidon2, ConstraintSumcheckFolder, Machine, ShardVerifier};

use super::{
    DefaultTraceGenerator, MachineProver, MachineProverComponents, ZerocheckAir,
    ZerocheckCpuProverData,
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CpuProverComponents<PcsComponents, A>(PhantomData<(A, PcsComponents)>);

pub type CpuProver<PcsComponents, A> = MachineProver<CpuProverComponents<PcsComponents, A>>;

impl<A, PcsComponents> MachineProverComponents for CpuProverComponents<PcsComponents, A>
where
    // F: Field,
    // EF: ExtensionField<F>,
    PcsComponents: JaggedProverComponents<A = CpuBackend>,
    A: MachineAir<PcsComponents::F>
        + Air<SymbolicAirBuilder<PcsComponents::F>>
        + for<'b> Air<
            ConstraintSumcheckFolder<'b, PcsComponents::F, PcsComponents::F, PcsComponents::EF>,
        > + for<'b> Air<
            ConstraintSumcheckFolder<'b, PcsComponents::F, PcsComponents::EF, PcsComponents::EF>,
        > + MachineAir<PcsComponents::F>,
{
    type F = PcsComponents::F;
    type EF = PcsComponents::EF;
    type Program = <A as MachineAir<PcsComponents::F>>::Program;
    type Record = <A as MachineAir<PcsComponents::F>>::Record;
    type Air = A;
    type B = CpuBackend;

    type Commitment = <PcsComponents as JaggedProverComponents>::Commitment;

    type Challenger = <PcsComponents as JaggedProverComponents>::Challenger;

    type Config = <PcsComponents as JaggedProverComponents>::Config;

    type TraceGenerator = DefaultTraceGenerator<PcsComponents::F, A, CpuBackend>;

    type ZerocheckProverData = ZerocheckCpuProverData<A>;

    type PcsProverComponents = PcsComponents;
}

impl<A> CpuProver<Poseidon2BabyBearJaggedCpuProverComponents, A>
where
    A: ZerocheckAir<BabyBear, BinomialExtensionField<BabyBear, 4>>,
{
    pub fn new(verifier: ShardVerifier<BabyBearPoseidon2, A>) -> Self {
        let ShardVerifier { pcs_verifier, machine } = verifier;
        let pcs_prover = JaggedProver::from_verifier(&pcs_verifier);
        let trace_generator = DefaultTraceGenerator::new(machine);
        let zerocheck_data = ZerocheckCpuProverData::default();
        Self { pcs_prover, zerocheck_prover_data: zerocheck_data, trace_generator }
    }
}
