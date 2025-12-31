use std::sync::Arc;

use tokio::sync::Mutex;

use csl_challenger::DeviceGrindingChallenger;
use csl_cuda::TaskScope;
use csl_jagged_tracegen::setup_tracegen_permit;
use csl_jagged_tracegen::CudaShardProverData;
use csl_utils::{Ext, Felt, JaggedTraceMle};
use slop_challenger::{FromChallenger, IopCtx};
use slop_multilinear::MultilinearPcsVerifier;
use slop_stacked::StackedBasefoldProof;
use sp1_hypercube::{
    air::{MachineAir, MachineProgram},
    prover::{PreprocessedData, ProverSemaphore, ProvingKey},
    septic_digest::SepticDigest,
    MachineVerifyingKey,
};

use crate::{CudaShardProver, CudaShardProverComponents};

impl<GC: IopCtx<F = Felt, EF = Ext>, PC: CudaShardProverComponents<GC>> CudaShardProver<GC, PC>
where
    GC::Challenger: DeviceGrindingChallenger<Witness = GC::F>,
    GC::Challenger: csl_basefold::DeviceGrindingChallenger<Witness = GC::F>,
    GC::Challenger: slop_challenger::FieldChallenger<
        <GC::Challenger as slop_challenger::GrindingChallenger>::Witness,
    >,
    StackedBasefoldProof<GC>: Into<<PC::C as MultilinearPcsVerifier<GC>>::Proof>,
    TaskScope: csl_jagged_assist::BranchingProgramKernel<
        GC::F,
        GC::EF,
        <GC::Challenger as DeviceGrindingChallenger>::OnDeviceChallenger,
    >,
    <GC::Challenger as DeviceGrindingChallenger>::OnDeviceChallenger:
        FromChallenger<GC::Challenger, TaskScope> + Clone,
{
    /// Setup from a program with a specific initial global cumulative sum.
    pub async fn setup_with_initial_global_cumulative_sum(
        &self,
        program: Arc<<PC::Air as MachineAir<GC::F>>::Program>,
        initial_global_cumulative_sum: SepticDigest<GC::F>,
        setup_permits: ProverSemaphore,
    ) -> (PreprocessedData<ProvingKey<GC, PC::C, PC::Air, Self>>, MachineVerifyingKey<GC, PC::C>)
    {
        let pc_start = program.pc_start();
        let enable_untrusted_programs = program.enable_untrusted_programs();

        let buffer = self.get_buffer().await;

        let (preprocessed_data, permit) = setup_tracegen_permit(
            &self.machine,
            program,
            &buffer,
            self.max_trace_size,
            self.basefold_prover.log_height,
            self.max_log_row_count,
            setup_permits,
            &self.backend,
        )
        .await;

        let (pk, vk) = self
            .setup_from_preprocessed_data_and_traces(
                pc_start,
                initial_global_cumulative_sum,
                preprocessed_data,
                enable_untrusted_programs,
            )
            .await;

        let pk = Mutex::new(pk);

        let pk = ProvingKey { vk: vk.clone(), preprocessed_data: pk };

        let pk = Arc::new(pk);

        (PreprocessedData { pk, permit }, vk)
    }

    /// Setup from preprocessed data and traces.
    pub async fn setup_from_preprocessed_data_and_traces(
        &self,
        pc_start: [GC::F; 3],
        initial_global_cumulative_sum: SepticDigest<GC::F>,
        preprocessed_traces: JaggedTraceMle<Felt, TaskScope>,
        enable_untrusted_programs: GC::F,
    ) -> (CudaShardProverData<GC, PC::Air>, MachineVerifyingKey<GC, PC::C>) {
        // Commit to the preprocessed traces, if there are any.
        let (preprocessed_commit, preprocessed_data) = csl_commit::commit_multilinears(
            &preprocessed_traces,
            self.max_log_row_count,
            true,
            &self.basefold_prover,
        )
        .await
        .unwrap();

        let vk = MachineVerifyingKey {
            pc_start,
            initial_global_cumulative_sum,
            preprocessed_commit,
            enable_untrusted_programs,
            marker: std::marker::PhantomData,
        };

        let pk = CudaShardProverData::new(preprocessed_traces, preprocessed_data);

        (pk, vk)
    }
}
