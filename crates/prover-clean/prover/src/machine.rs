use std::{collections::BTreeMap, marker::PhantomData, sync::Arc};

use csl_basefold::DeviceGrindingChallenger;
use csl_cuda::TaskScope;
use cslpc_utils::{Ext, Felt};
use slop_basefold::BasefoldProof;
use slop_challenger::{FromChallenger, IopCtx};
use slop_jagged::JaggedConfig;
use slop_multilinear::MultilinearPcsVerifier;
use slop_stacked::StackedPcsProof;
use sp1_hypercube::prover::{MachineProverComponents, ProvingKey};

use crate::{CudaShardProver, ProverCleanProverComponents};

/// Machine prover components for the prover-clean implementation.
#[derive(Debug, Clone, Copy)]
pub struct ProverCleanMachineProverComponents<GC, PC>(PhantomData<(GC, PC)>);

impl<GC, PC> MachineProverComponents<GC> for ProverCleanMachineProverComponents<GC, PC>
where
    GC: IopCtx<F = Felt, EF = Ext>,
    PC: ProverCleanProverComponents<GC>,
    GC::Challenger: DeviceGrindingChallenger<Witness = GC::F>,
    GC::Challenger: cslpc_basefold::DeviceGrindingChallenger<Witness = GC::F>,
    GC::Challenger: slop_challenger::FieldChallenger<
        <GC::Challenger as slop_challenger::GrindingChallenger>::Witness,
    >,
    StackedPcsProof<BasefoldProof<GC>, GC::EF>:
        Into<<<PC::C as JaggedConfig<GC>>::PcsVerifier as MultilinearPcsVerifier<GC>>::Proof>,
    TaskScope: csl_jagged::BranchingProgramKernel<
        GC::F,
        GC::EF,
        <GC::Challenger as DeviceGrindingChallenger>::OnDeviceChallenger,
    >,
    <GC::Challenger as DeviceGrindingChallenger>::OnDeviceChallenger:
        FromChallenger<GC::Challenger, TaskScope> + Clone,
{
    type Config = PC::C;
    type Air = PC::Air;
    type Prover = CudaShardProver<GC, PC>;

    async fn preprocessed_table_heights(
        pk: Arc<ProvingKey<GC, Self::Config, Self::Air, Self::Prover>>,
    ) -> BTreeMap<String, usize> {
        // Access through pk.preprocessed_data which is of type CudaShardProverData
        let preprocessed_data = pk.preprocessed_data.lock().await;
        preprocessed_data
            .preprocessed_traces
            .dense()
            .preprocessed_table_index
            .iter()
            .map(|(name, offset)| (name.clone(), offset.poly_size))
            .collect()
    }
}
