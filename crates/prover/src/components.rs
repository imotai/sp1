use std::{collections::BTreeMap, marker::PhantomData, sync::Arc};

use csl_air::{air_block::BlockAir, SymbolicProverFolder};
use csl_cuda::TaskScope;
use csl_jagged::{
    Poseidon2Bn254JaggedCudaProverComponents, Poseidon2KoalaBearJaggedCudaProverComponents,
};
use csl_logup_gkr::LogupGkrCudaProverComponents;
use csl_tracegen::{CudaTraceGenerator, CudaTracegenAir};
use csl_zerocheck::ZerocheckEvalProgramProverData;
use serde::{Deserialize, Serialize};
use slop_algebra::extension::BinomialExtensionField;
use slop_challenger::IopCtx;
use slop_jagged::{DefaultJaggedProver, JaggedProver, JaggedProverComponents};
use slop_koala_bear::{KoalaBear, KoalaBearDegree4Duplex};
use sp1_core_machine::riscv::RiscvAir;
use sp1_hypercube::{
    air::MachineAir,
    prover::{
        MachineProverComponents, ProvingKey, ShardProver, ShardProverComponents, ZerocheckAir,
    },
    GkrProverImpl, MachineConfig, ShardVerifier,
};
use sp1_primitives::{SP1GlobalContext, SP1OuterGlobalContext};
use sp1_prover::{components::SP1ProverComponents, CompressAir, WrapAir};

pub struct CudaSP1ProverComponents;

impl SP1ProverComponents for CudaSP1ProverComponents {
    type CoreComponents = CudaMachineProverComponents<
        KoalaBearDegree4Duplex,
        Poseidon2KoalaBearJaggedCudaProverComponents,
        RiscvAir<KoalaBear>,
    >;
    type RecursionComponents = CudaMachineProverComponents<
        KoalaBearDegree4Duplex,
        Poseidon2KoalaBearJaggedCudaProverComponents,
        CompressAir<<SP1GlobalContext as IopCtx>::F>,
    >;
    type WrapComponents = CudaMachineProverComponents<
        SP1OuterGlobalContext,
        Poseidon2Bn254JaggedCudaProverComponents,
        WrapAir<<SP1OuterGlobalContext as IopCtx>::F>,
    >;
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CudaShardProverComponents<GC, PcsComponents, A>(PhantomData<(GC, A, PcsComponents)>);

pub type CudaProver<GC, PcsComponents, A> =
    ShardProver<GC, CudaShardProverComponents<GC, PcsComponents, A>>;

impl<JC, A, GC> ShardProverComponents<GC> for CudaShardProverComponents<GC, JC, A>
where
    GC: IopCtx<F = KoalaBear, EF = BinomialExtensionField<KoalaBear, 4>>,
    JC: JaggedProverComponents<GC, A = TaskScope>,
    A: CudaTracegenAir<GC::F> + ZerocheckAir<GC::F, GC::EF> + std::fmt::Debug,
{
    type Program = <A as MachineAir<GC::F>>::Program;
    type Record = <A as MachineAir<GC::F>>::Record;
    type Air = A;
    type B = TaskScope;

    type Config = JC::Config;

    type TraceGenerator = CudaTraceGenerator<GC::F, A>;

    type ZerocheckProverData = ZerocheckEvalProgramProverData<GC::F, GC::EF, A>;

    type PcsProverComponents = JC;

    type GkrProver = GkrProverImpl<GC, LogupGkrCudaProverComponents<GC, A>>;
}

pub fn new_cuda_prover_sumcheck_eval<GC, C, Comp, A>(
    verifier: ShardVerifier<GC, C, A>,
    scope: TaskScope,
) -> CudaProver<GC, Comp, A>
where
    GC: IopCtx<F = KoalaBear, EF = BinomialExtensionField<KoalaBear, 4>>,
    C: MachineConfig<GC>,
    Comp: JaggedProverComponents<GC, A = TaskScope, Config = C> + DefaultJaggedProver<GC>,
    A: MachineAir<GC::F>
        + CudaTracegenAir<GC::F>
        + for<'a> BlockAir<SymbolicProverFolder<'a>>
        + ZerocheckAir<GC::F, GC::EF>
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CudaMachineProverComponents<GC, PcsComponents, A>(PhantomData<(GC, A, PcsComponents)>);

impl<GC, JC, A> MachineProverComponents<GC> for CudaMachineProverComponents<GC, JC, A>
where
    GC: IopCtx<F = KoalaBear, EF = BinomialExtensionField<KoalaBear, 4>>,
    JC: JaggedProverComponents<GC, A = TaskScope>,
    A: CudaTracegenAir<GC::F> + ZerocheckAir<GC::F, GC::EF> + std::fmt::Debug,
{
    type Config = JC::Config;
    type Prover = ShardProver<GC, CudaShardProverComponents<GC, JC, A>>;
    type Air = A;

    fn preprocessed_table_heights(
        pk: Arc<ProvingKey<GC, Self::Config, Self::Air, Self::Prover>>,
    ) -> BTreeMap<String, usize> {
        pk.preprocessed_data
            .preprocessed_traces
            .iter()
            .map(|(k, v)| (k.clone(), v.num_real_entries()))
            .collect()
    }
}
