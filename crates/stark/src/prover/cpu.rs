use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use p3_uni_stark::SymbolicAirBuilder;
use serde::{Deserialize, Serialize};
use slop_air::Air;
use slop_algebra::extension::BinomialExtensionField;
use slop_alloc::CpuBackend;
use slop_baby_bear::BabyBear;
use slop_jagged::{
    DefaultJaggedProver, JaggedConfig, JaggedProver, JaggedProverComponents,
    Poseidon2BabyBearJaggedCpuProverComponents,
};

use crate::{
    air::MachineAir, prover::MachineProverComponents, BabyBearPoseidon2, ConstraintSumcheckFolder,
    GkrProverImpl, LogupGkrCpuProverComponents, LogupGkrCpuRoundProver, LogupGkrCpuTraceGenerator,
    ShardVerifier,
};

use super::{
    DefaultTraceGenerator, MachineProver, MachineProverBuilder, ProverSemaphore, ShardProver,
    ShardProverComponents, ZerocheckAir, ZerocheckCpuProverData,
};

/// The components of a CPU shard prover.
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct CpuShardProverComponents<PcsComponents, A>(PhantomData<(A, PcsComponents)>);

/// The components of a CPU prover.
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct CpuMachineProverComponents<PcsComponents, A>(PhantomData<(A, PcsComponents)>);

impl<PcsComponents, A> MachineProverComponents for CpuMachineProverComponents<PcsComponents, A>
where
    PcsComponents: JaggedProverComponents<A = CpuBackend>,
    A: std::fmt::Debug
        + MachineAir<PcsComponents::F>
        + Air<SymbolicAirBuilder<PcsComponents::F>>
        + for<'b> Air<
            ConstraintSumcheckFolder<'b, PcsComponents::F, PcsComponents::F, PcsComponents::EF>,
        > + for<'b> Air<
            ConstraintSumcheckFolder<'b, PcsComponents::F, PcsComponents::EF, PcsComponents::EF>,
        > + MachineAir<PcsComponents::F>,
{
    type F = PcsComponents::F;
    type EF = PcsComponents::EF;
    type Config = <PcsComponents as JaggedProverComponents>::Config;
    type Air = A;
    type Prover = ShardProver<CpuShardProverComponents<PcsComponents, A>>;
}

/// A CPU prover.
pub type CpuProver<PcsComponents, A> = MachineProver<CpuShardProverComponents<PcsComponents, A>>;
/// A CPU shard prover.
pub type CpuShardProver<PcsComponents, A> = ShardProver<CpuShardProverComponents<PcsComponents, A>>;
/// A CPU prover builder.
pub struct CpuProverBuilder<PcsComponents, A>
where
    PcsComponents: JaggedProverComponents<A = CpuBackend>,
    A: std::fmt::Debug
        + MachineAir<PcsComponents::F>
        + Air<SymbolicAirBuilder<PcsComponents::F>>
        + for<'b> Air<
            ConstraintSumcheckFolder<'b, PcsComponents::F, PcsComponents::F, PcsComponents::EF>,
        > + for<'b> Air<
            ConstraintSumcheckFolder<'b, PcsComponents::F, PcsComponents::EF, PcsComponents::EF>,
        > + MachineAir<PcsComponents::F>,
{
    inner: MachineProverBuilder<CpuMachineProverComponents<PcsComponents, A>>,
}

impl<PcsComponents, A> Deref for CpuProverBuilder<PcsComponents, A>
where
    PcsComponents: JaggedProverComponents<A = CpuBackend>,
    A: std::fmt::Debug
        + MachineAir<PcsComponents::F>
        + Air<SymbolicAirBuilder<PcsComponents::F>>
        + for<'b> Air<
            ConstraintSumcheckFolder<'b, PcsComponents::F, PcsComponents::F, PcsComponents::EF>,
        > + for<'b> Air<
            ConstraintSumcheckFolder<'b, PcsComponents::F, PcsComponents::EF, PcsComponents::EF>,
        > + MachineAir<PcsComponents::F>,
{
    type Target = MachineProverBuilder<CpuMachineProverComponents<PcsComponents, A>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<PcsComponents, A> DerefMut for CpuProverBuilder<PcsComponents, A>
where
    PcsComponents: JaggedProverComponents<A = CpuBackend>,
    A: std::fmt::Debug
        + MachineAir<PcsComponents::F>
        + Air<SymbolicAirBuilder<PcsComponents::F>>
        + for<'b> Air<
            ConstraintSumcheckFolder<'b, PcsComponents::F, PcsComponents::F, PcsComponents::EF>,
        > + for<'b> Air<
            ConstraintSumcheckFolder<'b, PcsComponents::F, PcsComponents::EF, PcsComponents::EF>,
        > + MachineAir<PcsComponents::F>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<A, PcsComponents> ShardProverComponents for CpuShardProverComponents<PcsComponents, A>
where
    PcsComponents: JaggedProverComponents<A = CpuBackend>,
    A: std::fmt::Debug
        + MachineAir<PcsComponents::F>
        + Air<SymbolicAirBuilder<PcsComponents::F>>
        + for<'b> Air<
            ConstraintSumcheckFolder<'b, PcsComponents::F, PcsComponents::F, PcsComponents::EF>,
        > + for<'b> Air<
            ConstraintSumcheckFolder<'b, PcsComponents::F, PcsComponents::EF, PcsComponents::EF>,
        > + MachineAir<PcsComponents::F>,
{
    type F = PcsComponents::F;
    type EF = PcsComponents::EF;
    type Program = <A as MachineAir<PcsComponents::F>>::Program;
    type Record = <A as MachineAir<PcsComponents::F>>::Record;
    type Air = A;
    type B = CpuBackend;

    type Commitment = <PcsComponents as JaggedProverComponents>::Commitment;

    type Challenger = <PcsComponents as JaggedProverComponents>::Challenger;

    type Config = <PcsComponents as JaggedProverComponents>::Config;

    type TraceGenerator = DefaultTraceGenerator<PcsComponents::F, A, CpuBackend>;

    type ZerocheckProverData = ZerocheckCpuProverData<A>;

    type GkrProver = GkrProverImpl<
        LogupGkrCpuProverComponents<
            PcsComponents::F,
            PcsComponents::EF,
            A,
            <PcsComponents as JaggedProverComponents>::Challenger,
        >,
    >;

    type PcsProverComponents = PcsComponents;
}

impl<Comp, A, Config> CpuShardProver<Comp, A>
where
    Config: JaggedConfig + Sync,
    Comp: JaggedProverComponents<A = CpuBackend, Config = Config, F = Config::F, EF = Config::EF>
        + DefaultJaggedProver,
    A: ZerocheckAir<Config::F, Config::EF> + std::fmt::Debug,
{
    /// Create a new CPU prover.
    #[must_use]
    pub fn new(verifier: ShardVerifier<Config, A>) -> Self {
        // Construct the shard prover.
        let ShardVerifier { pcs_verifier, machine } = verifier;
        let pcs_prover = JaggedProver::from_verifier(&pcs_verifier);
        let trace_generator = DefaultTraceGenerator::new(machine);
        let zerocheck_data = ZerocheckCpuProverData::default();
        let logup_gkr_trace_generator = LogupGkrCpuTraceGenerator::default();
        let logup_gkr_prover =
            GkrProverImpl::new(logup_gkr_trace_generator, LogupGkrCpuRoundProver);

        Self {
            trace_generator,
            logup_gkr_prover,
            zerocheck_prover_data: zerocheck_data,
            pcs_prover,
        }
    }
}

impl<A> CpuProverBuilder<Poseidon2BabyBearJaggedCpuProverComponents, A>
where
    A: ZerocheckAir<BabyBear, BinomialExtensionField<BabyBear, 4>> + std::fmt::Debug,
{
    // /// Create a new CPU prover builder from a verifier and resource options.
    // #[must_use]
    // pub fn from_verifier(verifier: ShardVerifier<BabyBearPoseidon2, A>, opts: SP1CoreOpts) ->
    // Self {     let shard_prover = Arc::new(CpuShardProver::new(verifier.clone()));
    //     let prover_permits = Arc::new(Semaphore::new(opts.shard_batch_size));

    //     MachineProverBuilder::new(verifier, vec![prover_permits], vec![shard_prover])
    //         .num_workers(opts.trace_gen_workers)
    // }

    /// Create a new CPU prover builder from a verifier, having a single worker with a single
    /// permit.
    #[must_use]
    pub fn simple(verifier: ShardVerifier<BabyBearPoseidon2, A>) -> Self {
        let shard_prover = Arc::new(CpuShardProver::new(verifier.clone()));
        let prover_permits = ProverSemaphore::new(1);

        Self {
            inner: MachineProverBuilder::new(verifier, vec![prover_permits], vec![shard_prover]),
        }
    }

    /// Create a new CPU prover builder from a verifier.
    #[must_use]
    pub fn new(
        verifier: ShardVerifier<BabyBearPoseidon2, A>,
        prover_permits: ProverSemaphore,
    ) -> Self {
        let shard_prover = Arc::new(CpuShardProver::new(verifier.clone()));

        Self {
            inner: MachineProverBuilder::new(verifier, vec![prover_permits], vec![shard_prover]),
        }
    }
}
