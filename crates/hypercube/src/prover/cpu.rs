use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use slop_algebra::extension::BinomialExtensionField;
use slop_challenger::IopCtx;
use slop_jagged::{DefaultJaggedProver, JaggedProver};
use slop_multilinear::MultilinearPcsVerifier;
use slop_stacked::StackedPcsProver;
use sp1_primitives::{SP1Field, SP1GlobalContext};

use super::{
    DefaultTraceGenerator, MachineProverBuilder, ProverSemaphore, ShardProver, ZerocheckAir,
};
use crate::{
    prover::SP1MerkleTreeProver, GkrProverImpl, InnerSC, LogupGkrCpuTraceGenerator, SP1Pcs,
    ShardContextImpl, ShardVerifier,
};

type SC<GC, Verifier, A> = ShardContextImpl<GC, Verifier, A>;
type MachineProverBuilderFromVerifier<GC, Verifier, PcsComponents, A> =
    MachineProverBuilder<GC, SC<GC, Verifier, A>, CpuShardProver<GC, Verifier, PcsComponents, A>>;

/// A CPU shard prover.
pub type CpuShardProver<GC, Verifier, PcsComponents, A> =
    ShardProver<GC, SC<GC, Verifier, A>, PcsComponents>;
/// A CPU prover builder.
pub struct CpuProverBuilder<GC, Verifier, PcsComponents, A>
where
    GC: IopCtx,
    Verifier: MultilinearPcsVerifier<GC>,
    PcsComponents: DefaultJaggedProver<GC, Verifier>,
    A: ZerocheckAir<GC::F, GC::EF>,
{
    inner: MachineProverBuilderFromVerifier<GC, Verifier, PcsComponents, A>,
}

impl<GC, Verifier, PcsComponents, A> Deref for CpuProverBuilder<GC, Verifier, PcsComponents, A>
where
    GC: IopCtx,
    Verifier: MultilinearPcsVerifier<GC>,
    PcsComponents: DefaultJaggedProver<GC, Verifier>,
    A: ZerocheckAir<GC::F, GC::EF>,
{
    type Target = MachineProverBuilder<
        GC,
        SC<GC, Verifier, A>,
        CpuShardProver<GC, Verifier, PcsComponents, A>,
    >;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<GC, Verifier, PcsComponents, A> DerefMut for CpuProverBuilder<GC, Verifier, PcsComponents, A>
where
    GC: IopCtx,
    Verifier: MultilinearPcsVerifier<GC>,
    PcsComponents: DefaultJaggedProver<GC, Verifier>,
    A: ZerocheckAir<GC::F, GC::EF>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<GC, Verifier, A, PcsComponents> CpuShardProver<GC, Verifier, PcsComponents, A>
where
    GC: IopCtx,
    Verifier: MultilinearPcsVerifier<GC>,
    PcsComponents: DefaultJaggedProver<GC, Verifier>,
    A: ZerocheckAir<GC::F, GC::EF>,
{
    /// Create a new CPU prover.
    #[must_use]
    pub fn new(verifier: ShardVerifier<GC, ShardContextImpl<GC, Verifier, A>>) -> Self {
        // Construct the shard prover.
        let ShardVerifier { jagged_pcs_verifier: pcs_verifier, machine } = verifier;
        let pcs_prover = JaggedProver::from_verifier(&pcs_verifier);
        let trace_generator = DefaultTraceGenerator::new(machine);
        let logup_gkr_trace_generator = LogupGkrCpuTraceGenerator::default();
        let logup_gkr_prover = GkrProverImpl::new(logup_gkr_trace_generator);

        Self { trace_generator, logup_gkr_prover, pcs_prover }
    }
}

impl<A>
    CpuProverBuilder<
        SP1GlobalContext,
        SP1Pcs<SP1GlobalContext>,
        StackedPcsProver<SP1MerkleTreeProver, SP1GlobalContext>,
        A,
    >
where
    A: ZerocheckAir<SP1Field, BinomialExtensionField<SP1Field, 4>>,
{
    /// Create a new CPU prover builder from a verifier, having a single worker with a single
    /// permit.
    #[must_use]
    pub fn simple(verifier: ShardVerifier<SP1GlobalContext, InnerSC<A>>) -> Self {
        let shard_prover = Arc::new(CpuShardProver::new(verifier.clone()));
        let prover_permits = ProverSemaphore::new(1);

        Self {
            inner: MachineProverBuilder::new(verifier, vec![prover_permits], vec![shard_prover]),
        }
    }

    /// Create a new CPU prover builder from a verifier.
    #[must_use]
    pub fn new(
        verifier: ShardVerifier<SP1GlobalContext, InnerSC<A>>,
        prover_permits: ProverSemaphore,
    ) -> Self {
        let shard_prover = Arc::new(CpuShardProver::new(verifier.clone()));

        Self {
            inner: MachineProverBuilder::new(verifier, vec![prover_permits], vec![shard_prover]),
        }
    }
}
