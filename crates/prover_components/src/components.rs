use csl_air::{air_block::BlockAir, codegen_cuda_eval, SymbolicProverFolder};
use csl_cuda::{PinnedBuffer, TaskScope, ToDevice};

use csl_basefold::FriCudaProver;
use csl_merkle_tree::{CudaTcsProver, Poseidon2Bn254CudaProver, Poseidon2KoalaBear16CudaProver};
use csl_shard_prover::{CudaMachineProverComponents, CudaShardProver, CudaShardProverComponents};
use csl_tracegen::CudaTracegenAir;
use slop_algebra::extension::BinomialExtensionField;
use slop_basefold::BasefoldVerifier;
use slop_challenger::IopCtx;
use slop_futures::queue::WorkerQueue;
use slop_jagged::{JaggedBasefoldConfig, SP1OuterConfig};
use slop_koala_bear::{KoalaBear, KoalaBearDegree4Duplex};
use sp1_core_machine::riscv::RiscvAir;
use sp1_hypercube::{air::MachineAir, prover::ZerocheckAir, SP1CoreJaggedConfig};
use sp1_primitives::{SP1GlobalContext, SP1OuterGlobalContext};
use sp1_prover::{components::SP1ProverComponents, CompressAir, WrapAir};
use std::{collections::BTreeMap, marker::PhantomData, sync::Arc};

pub struct SP1CudaProverComponents;

impl SP1ProverComponents for SP1CudaProverComponents {
    type CoreComponents =
        CudaMachineProverComponents<KoalaBearDegree4Duplex, CudaProverCoreComponents>;
    type RecursionComponents =
        CudaMachineProverComponents<KoalaBearDegree4Duplex, CudaProverRecursionComponents>;
    type WrapComponents =
        CudaMachineProverComponents<SP1OuterGlobalContext, CudaProverWrapComponents>;
}

/// Core prover components for the CUDA prover.
pub struct CudaProverCoreComponents;

impl CudaShardProverComponents<KoalaBearDegree4Duplex> for CudaProverCoreComponents {
    type P = Poseidon2KoalaBear16CudaProver;
    type Air = RiscvAir<KoalaBear>;
    type C = SP1CoreJaggedConfig;
}

/// Recursion prover components for the CUDA prover.
pub struct CudaProverRecursionComponents;

impl CudaShardProverComponents<KoalaBearDegree4Duplex> for CudaProverRecursionComponents {
    type P = Poseidon2KoalaBear16CudaProver;
    type Air = CompressAir<<SP1GlobalContext as IopCtx>::F>;
    type C = SP1CoreJaggedConfig;
}

/// Wrap prover components for the CUDA prover.
pub struct CudaProverWrapComponents;

impl CudaShardProverComponents<SP1OuterGlobalContext> for CudaProverWrapComponents {
    type P = Poseidon2Bn254CudaProver;
    type Air = WrapAir<<SP1OuterGlobalContext as IopCtx>::F>;
    type C = SP1OuterConfig;
}

pub async fn new_cuda_prover<GC, PC>(
    verifier: sp1_hypercube::MachineVerifier<GC, JaggedBasefoldConfig<GC>, PC::Air>,
    max_trace_size: usize,
    num_workers: usize,
    scope: TaskScope,
) -> CudaShardProver<GC, PC>
where
    GC: IopCtx<F = KoalaBear, EF = BinomialExtensionField<KoalaBear, 4>>,
    PC: CudaShardProverComponents<GC>,
    PC::P: CudaTcsProver<GC> + Default,
    PC::Air: CudaTracegenAir<GC::F>
        + for<'a> BlockAir<SymbolicProverFolder<'a>>
        + ZerocheckAir<GC::F, GC::EF>
        + std::fmt::Debug,
{
    let machine = verifier.machine().clone();

    let mut cache = BTreeMap::new();
    for chip in machine.chips() {
        let result = codegen_cuda_eval(chip.air.as_ref());
        cache.insert(chip.air.name().to_string(), result);
    }

    let log_stacking_height = verifier.log_stacking_height();
    let max_log_row_count = verifier.max_log_row_count();

    // Create the basefold prover from the verifier's PCS config
    // TODO: get this straight from the verifier.
    let basefold_verifier = BasefoldVerifier::<GC>::new(*verifier.fri_config(), 2);

    let basefold_prover = FriCudaProver::<GC, PC::P, GC::F>::new(
        PC::P::default(),
        basefold_verifier.fri_config,
        log_stacking_height,
    );

    let mut all_interactions = BTreeMap::new();

    for chip in machine.chips().iter() {
        let host_interactions = csl_logup_gkr::Interactions::new(chip.sends(), chip.receives());
        let device_interactions =
            csl_logup_gkr::Interactions::to_device_in(&host_interactions, &scope).await.unwrap();
        all_interactions.insert(chip.name().to_string(), Arc::new(device_interactions));
    }

    let mut trace_buffers = Vec::with_capacity(num_workers);
    for _ in 0..num_workers {
        let pinned_buffer = PinnedBuffer::<GC::F>::with_capacity(max_trace_size);
        trace_buffers.push(pinned_buffer);
    }

    CudaShardProver {
        trace_buffers: Arc::new(WorkerQueue::new(trace_buffers)),
        all_interactions,
        all_zerocheck_programs: cache,
        max_log_row_count: max_log_row_count as u32,
        basefold_prover,
        max_trace_size,
        machine,
        backend: scope,
        _marker: PhantomData,
    }
}
