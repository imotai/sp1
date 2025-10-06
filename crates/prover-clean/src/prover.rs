use std::{marker::PhantomData, sync::Arc};

use csl_cuda::TaskScope;
use slop_challenger::IopCtx;
use slop_commit::Rounds;
use slop_jagged::{JaggedConfig, JaggedPcsProof, JaggedProverError};
use slop_multilinear::{Evaluations, Point};
use sp1_hypercube::{
    air::MachineAir,
    prover::{AirProver, PreprocessedData, ProverPermit, ProverSemaphore, ProvingKey},
    Machine, MachineConfig, MachineVerifyingKey, ShardProof,
};
use thiserror::Error;

use crate::tracegen::JaggedTraceMle;

/// A prover for the hypercube STARK, given a configuration.
pub struct CudaShardProver<GC: IopCtx> {
    _marker: PhantomData<GC>,
}

pub struct CudaShardProverData<GC: IopCtx> {
    /// The preprocessed traces.
    pub preprocessed_traces: Arc<JaggedTraceMle<GC::F, TaskScope>>,
    /// The pcs data for the preprocessed traces.
    /// TODO: fill this out.
    pub preprocessed_data: (),
}

impl<GC: IopCtx, Config: MachineConfig<GC>, Air: MachineAir<GC::F>> AirProver<GC, Config, Air>
    for CudaShardProver<GC>
{
    type PreprocessedData = CudaShardProverData<GC>;

    fn machine(&self) -> &Machine<GC::F, Air> {
        todo!()
    }

    /// Setup a shard, using a verifying key if provided.
    async fn setup_from_vk(
        &self,
        _program: Arc<Air::Program>,
        _vk: Option<MachineVerifyingKey<GC, Config>>,
        _prover_permits: ProverSemaphore,
    ) -> (PreprocessedData<ProvingKey<GC, Config, Air, Self>>, MachineVerifyingKey<GC, Config>)
    {
        todo!()
    }

    /// Setup and prove a shard.
    async fn setup_and_prove_shard(
        &self,
        _program: Arc<Air::Program>,
        _record: Air::Record,
        _vk: Option<MachineVerifyingKey<GC, Config>>,
        _prover_permits: ProverSemaphore,
        _challenger: &mut GC::Challenger,
    ) -> (MachineVerifyingKey<GC, Config>, ShardProof<GC, Config>, ProverPermit) {
        todo!()
    }

    /// Prove a shard with a given proving key.
    async fn prove_shard_with_pk(
        &self,
        _pk: Arc<ProvingKey<GC, Config, Air, Self>>,
        _record: Air::Record,
        _prover_permits: ProverSemaphore,
        _challenger: &mut GC::Challenger,
    ) -> (ShardProof<GC, Config>, ProverPermit) {
        todo!()
    }
}

// An error type for cuda jagged prover
#[derive(Debug, Error)]
pub enum CudaShardProverError {}

pub struct CudaJaggedProverData<GC: IopCtx> {
    _marker: PhantomData<GC>,
}

impl<GC: IopCtx> CudaShardProver<GC> {
    /// Commit to a batch of padded multilinears.
    ///
    /// The jagged polynomial commitments scheme is able to commit to sparse polynomials having
    /// very few or no real rows.
    /// **Note** the padding values will be ignored and treated as though they are zero.
    pub async fn commit_multilinears(
        &self,
        _multilinears: JaggedTraceMle<GC::F, TaskScope>,
        _use_preprocessed_data: bool,
    ) -> Result<(GC::Digest, CudaJaggedProverData<GC>), JaggedProverError<CudaShardProverError>>
    {
        todo!()
    }

    /// Prove trusted evaluations.
    pub async fn prove_trusted_evaluations<C: JaggedConfig<GC>>(
        &self,
        _eval_point: Point<GC::EF>,
        _evaluation_claims: Rounds<Evaluations<GC::EF, TaskScope>>,
        _prover_data: (),
        _challenger: &mut GC::Challenger,
    ) -> Result<JaggedPcsProof<GC, C>, JaggedProverError<CudaShardProverError>> {
        todo!()
    }
}
