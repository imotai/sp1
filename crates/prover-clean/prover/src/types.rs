use std::{collections::BTreeSet, sync::Arc};

use cslpc_utils::{Ext, Felt};
use slop_challenger::IopCtx;
use sp1_hypercube::{
    air::MachineAir,
    prover::{AirProver, ProverPermit, ProvingKey},
    Chip, MachineConfig,
};

use crate::ProverCleanProverComponents;

/// A collection of main traces with a permit.
#[allow(clippy::type_complexity)]
pub struct ShardData<GC: IopCtx<F = Felt, EF = Ext>, PC: ProverCleanProverComponents<GC>>
where
    crate::CudaShardProver<GC, PC>: AirProver<GC, PC::C, PC::Air>,
{
    /// The proving key.
    pub pk: Arc<ProvingKey<GC, PC::C, PC::Air, crate::CudaShardProver<GC, PC>>>,
    /// Main trace data
    pub main_trace_data: MainTraceData<GC, PC::Air, PC::C, crate::CudaShardProver<GC, PC>>,
}

pub struct MainTraceData<
    GC: IopCtx<F = Felt, EF = Ext>,
    Air: MachineAir<GC::F>,
    C: MachineConfig<GC>,
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
