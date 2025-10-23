use std::{collections::BTreeMap, sync::Arc};

use tokio::sync::Mutex;

use csl_basefold::DeviceGrindingChallenger;
use csl_cuda::TaskScope;
use cslpc_tracegen::{setup_tracegen, CudaShardProverData};
use cslpc_utils::{Ext, Felt, JaggedTraceMle};
use slop_algebra::AbstractField;
use slop_basefold::BasefoldProof;
use slop_challenger::{FromChallenger, IopCtx};
use slop_jagged::JaggedConfig;
use slop_multilinear::MultilinearPcsVerifier;
use slop_stacked::StackedPcsProof;
use sp1_hypercube::{
    air::{MachineAir, MachineProgram},
    prover::{PreprocessedData, ProverSemaphore, ProvingKey},
    septic_digest::SepticDigest,
    ChipDimensions, MachineVerifyingKey,
};

use crate::{CudaShardProver, ProverCleanProverComponents};

impl<GC: IopCtx<F = Felt, EF = Ext>, PC: ProverCleanProverComponents<GC>> CudaShardProver<GC, PC>
where
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
        let (preprocessed_data, permit) = setup_tracegen(
            &self.machine,
            program,
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
        let (preprocessed_commit, preprocessed_data) = cslpc_commit::commit_multilinears(
            &preprocessed_traces,
            self.max_log_row_count,
            true,
            &self.basefold_prover,
        )
        .await
        .unwrap();

        let preprocessed_chip_information = preprocessed_traces
            .dense()
            .preprocessed_table_index
            .iter()
            .map(|(name, trace_offset)| {
                (
                    name.to_owned(),
                    ChipDimensions {
                        height: GC::F::from_canonical_usize(trace_offset.poly_size),
                        num_polynomials: GC::F::from_canonical_usize(trace_offset.num_polys),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();

        let vk = MachineVerifyingKey {
            pc_start,
            initial_global_cumulative_sum,
            preprocessed_commit,
            preprocessed_chip_information,
            enable_untrusted_programs,
            marker: std::marker::PhantomData,
        };

        let pk = CudaShardProverData::new(preprocessed_traces, preprocessed_data);

        (pk, vk)
    }
}
