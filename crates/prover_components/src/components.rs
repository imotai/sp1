use std::{collections::BTreeMap, sync::Arc};

use csl_air::{air_block::BlockAir, codegen_cuda_eval, SymbolicProverFolder};
use csl_challenger::{DuplexChallenger, MultiField32Challenger};
use csl_cuda::{PinnedBuffer, TaskScope};

use csl_basefold::FriCudaProver;
use csl_merkle_tree::{CudaTcsProver, Poseidon2Bn254CudaProver, Poseidon2KoalaBear16CudaProver};
use csl_shard_prover::{CudaShardProver, CudaShardProverComponents};
use csl_tracegen::CudaTracegenAir;
use slop_algebra::extension::BinomialExtensionField;
use slop_basefold::BasefoldVerifier;
use slop_bn254::Bn254Fr;
use slop_challenger::IopCtx;
use slop_futures::queue::WorkerQueue;
use slop_koala_bear::{KoalaBear, KoalaBearDegree4Duplex};
use sp1_core_machine::riscv::RiscvAir;
use sp1_hypercube::{air::MachineAir, prover::ZerocheckAir, SP1InnerPcs, SP1OuterPcs, SP1SC};
use sp1_primitives::{SP1GlobalContext, SP1OuterGlobalContext};
use sp1_prover::{CompressAir, SP1ProverComponents, WrapAir};

pub struct SP1CudaProverComponents;

impl SP1ProverComponents for SP1CudaProverComponents {
    type CoreProver = CudaShardProver<KoalaBearDegree4Duplex, CudaProverCoreComponents>;
    type RecursionProver = CudaShardProver<KoalaBearDegree4Duplex, CudaProverRecursionComponents>;
    type WrapProver = CudaShardProver<SP1OuterGlobalContext, CudaProverWrapComponents>;
}

/// Core prover components for the CUDA prover.
pub struct CudaProverCoreComponents;

impl CudaShardProverComponents<KoalaBearDegree4Duplex> for CudaProverCoreComponents {
    type P = Poseidon2KoalaBear16CudaProver;
    type Air = RiscvAir<KoalaBear>;
    type C = SP1InnerPcs;
    type DeviceChallenger = DuplexChallenger<KoalaBear, TaskScope>;
}

/// Recursion prover components for the CUDA prover.
pub struct CudaProverRecursionComponents;

impl CudaShardProverComponents<KoalaBearDegree4Duplex> for CudaProverRecursionComponents {
    type P = Poseidon2KoalaBear16CudaProver;
    type Air = CompressAir<<SP1GlobalContext as IopCtx>::F>;
    type C = SP1InnerPcs;
    type DeviceChallenger = DuplexChallenger<KoalaBear, TaskScope>;
}

/// Wrap prover components for the CUDA prover.
pub struct CudaProverWrapComponents;

impl CudaShardProverComponents<SP1OuterGlobalContext> for CudaProverWrapComponents {
    type P = Poseidon2Bn254CudaProver;
    type Air = WrapAir<<SP1OuterGlobalContext as IopCtx>::F>;
    type C = SP1OuterPcs;
    type DeviceChallenger = MultiField32Challenger<KoalaBear, Bn254Fr, TaskScope>;
}

pub async fn new_cuda_prover<GC, PC>(
    verifier: sp1_hypercube::MachineVerifier<GC, SP1SC<GC, PC::Air>>,
    max_trace_size: usize,
    num_workers: usize,
    recompute_first_layer: bool,
    scope: TaskScope,
) -> CudaShardProver<GC, PC>
where
    GC: IopCtx<F = KoalaBear, EF = BinomialExtensionField<KoalaBear, 4>>,
    PC: CudaShardProverComponents<GC>,
    PC::P: CudaTcsProver<GC>,
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

    let tcs_prover = PC::P::new(&scope);
    let basefold_prover = FriCudaProver::<GC, PC::P, GC::F>::new(
        tcs_prover,
        basefold_verifier.fri_config,
        log_stacking_height,
    );

    let mut all_interactions = BTreeMap::new();

    for chip in machine.chips().iter() {
        let host_interactions = csl_logup_gkr::Interactions::new(chip.sends(), chip.receives());
        let device_interactions = host_interactions.copy_to_device(&scope).unwrap();
        all_interactions.insert(chip.name().to_string(), Arc::new(device_interactions));
    }

    let mut trace_buffers = Vec::with_capacity(num_workers);
    for _ in 0..num_workers {
        let pinned_buffer = PinnedBuffer::<GC::F>::with_capacity(max_trace_size);
        trace_buffers.push(pinned_buffer);
    }

    let trace_buffers = Arc::new(WorkerQueue::new(trace_buffers));
    CudaShardProver::<GC, PC>::new(
        trace_buffers,
        max_log_row_count as u32,
        basefold_prover,
        machine,
        max_trace_size,
        scope,
        all_interactions,
        cache,
        recompute_first_layer,
        recompute_first_layer,
    )
}
