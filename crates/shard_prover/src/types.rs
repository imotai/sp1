use std::{collections::BTreeSet, sync::Arc};

use csl_utils::{Ext, Felt};
use slop_challenger::IopCtx;
use slop_jagged::JaggedConfig;
use sp1_hypercube::{
    air::MachineAir,
    prover::{AirProver, ProverPermit, ProvingKey},
    Chip,
};

use crate::CudaShardProverComponents;

/// A collection of main traces with a permit.
#[allow(clippy::type_complexity)]
pub struct ShardData<GC: IopCtx<F = Felt, EF = Ext>, PC: CudaShardProverComponents<GC>>
where
    crate::CudaShardProver<GC, PC>: AirProver<GC, PC::C, PC::Air>,
{
    /// Main trace data
    pub main_trace_data: MainTraceData<GC, PC::Air, PC::C, crate::CudaShardProver<GC, PC>>,
}

pub struct MainTraceData<
    GC: IopCtx<F = Felt, EF = Ext>,
    Air: MachineAir<GC::F>,
    C: JaggedConfig<GC>,
    Prover: AirProver<GC, C, Air>,
> {
    /// The traces.
    pub traces: Arc<ProvingKey<GC, C, Air, Prover>>,
    /// The public values.
    pub public_values: Vec<GC::F>,
    /// The shape cluster corresponding to the traces.
    pub shard_chips: BTreeSet<Chip<GC::F, Air>>,
    /// A permit for a prover resource.
    pub permit: ProverPermit,
}
