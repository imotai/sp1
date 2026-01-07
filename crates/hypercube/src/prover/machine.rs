use itertools::Itertools;
use slop_air::BaseAir;
use slop_algebra::PrimeField32;
use slop_challenger::IopCtx;
use slop_futures::queue::WorkerQueue;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use crate::{
    air::MachineAir,
    prover::{shard::AirProver, CoreProofShape, PcsProof, ProvingKey},
    Machine, MachineProof, MachineVerifier, MachineVerifierConfigError, MachineVerifyingKey,
    ShardContext, ShardProof, ShardVerifier,
};

use super::{PreprocessedData, ProverSemaphore};

/// The type of program this prover can make proofs for.
pub type Program<GC, SC> =
    <<SC as ShardContext<GC>>::Air as MachineAir<<GC as IopCtx>::F>>::Program;

/// The execution record for this prover.
pub type Record<GC, SC> = <<SC as ShardContext<GC>>::Air as MachineAir<<GC as IopCtx>::F>>::Record;

/// A builder for a machine prover.
pub struct MachineProverBuilder<GC: IopCtx, SC: ShardContext<GC>, C: AirProver<GC, SC>> {
    verifier: MachineVerifier<GC, SC>,
    base_workers: Vec<Arc<C>>,
    worker_permits: Vec<ProverSemaphore>,
    num_workers: Vec<usize>,
}

/// A machine prover.
pub struct MachineProver<GC: IopCtx, SC: ShardContext<GC>, C: AirProver<GC, SC>> {
    base_workers: Vec<Arc<C>>,
    worker_permits: Vec<ProverSemaphore>,
    worker_queue: Arc<WorkerQueue<usize>>,
    verifier: MachineVerifier<GC, SC>,
}

impl<GC: IopCtx, SC: ShardContext<GC>, C: AirProver<GC, SC>> MachineProverBuilder<GC, SC, C> {
    /// Crate a new builder for a machine prover.
    ///
    /// The builder is constructed from different groups of workers, each sharing their own permits.
    /// In practice, those permits can come from the same semaphore or different ones.
    pub fn new(
        shard_verifier: ShardVerifier<GC, SC>,
        worker_permits: Vec<ProverSemaphore>,
        base_workers: Vec<Arc<C>>,
    ) -> Self {
        assert!(
            base_workers.len() == worker_permits.len(),
            "base workers and their corresponding permits must have the same length"
        );
        let num_base_workers = base_workers.len();
        Self {
            verifier: MachineVerifier::new(shard_verifier),
            base_workers,
            worker_permits,
            num_workers: vec![1; num_base_workers],
        }
    }

    /// Create a new builder for a machine prover with a single kind.
    #[inline]
    #[must_use]
    pub fn new_single_kind(
        shard_verifier: ShardVerifier<GC, SC>,
        shard_prover: C,
        permits: ProverSemaphore,
    ) -> Self {
        let base_workers = vec![Arc::new(shard_prover)];
        let worker_permits = vec![permits];
        Self::new(shard_verifier, worker_permits, base_workers)
    }

    /// Set the number of workers for a given base kind.
    pub fn num_workers_for_base_kind(&mut self, base_kind: usize, num_workers: usize) -> &mut Self {
        self.num_workers[base_kind] = num_workers;
        self
    }

    /// Set the number of workers for each base kind.
    pub fn num_workers_per_kind(&mut self, num_workers_per_kind: Vec<usize>) -> &mut Self {
        self.num_workers = num_workers_per_kind;
        self
    }

    /// Set the number of workers for all base kinds.
    pub fn num_workers(&mut self, num_workers: usize) -> &mut Self {
        self.num_workers = vec![num_workers; self.base_workers.len()];
        self
    }

    /// Build the machine prover.
    pub fn build(&mut self) -> MachineProver<GC, SC, C> {
        // For each base worker, repeat it the number of times specified by the number of workers.
        let mut worker_queue: Vec<usize> = Vec::new();
        for ((idx, _), num_workers) in
            self.base_workers.iter().enumerate().zip_eq(self.num_workers.iter())
        {
            worker_queue.extend(std::iter::repeat_n(idx, *num_workers));
        }

        MachineProver {
            base_workers: self.base_workers.clone(),
            worker_permits: self.worker_permits.clone(),
            worker_queue: Arc::new(WorkerQueue::new(worker_queue)),
            verifier: self.verifier.clone(),
        }
    }
}

impl<GC: IopCtx, SC: ShardContext<GC>, C: AirProver<GC, SC>> MachineProver<GC, SC, C> {
    /// Verify a machine proof.
    pub fn verify(
        &self,
        vk: &MachineVerifyingKey<GC>,
        proof: &MachineProof<GC, PcsProof<GC, SC>>,
    ) -> Result<(), MachineVerifierConfigError<GC, SC::Config>>
    where
        GC::F: PrimeField32,
    {
        self.verifier.verify(vk, proof)
    }

    /// Get the number of workers.
    #[must_use]
    #[inline]
    pub fn num_workers(&self) -> usize {
        self.base_workers.len()
    }

    /// Get the verifier.
    #[must_use]
    #[inline]
    pub fn verifier(&self) -> &MachineVerifier<GC, SC> {
        &self.verifier
    }

    /// Get a new challenger.
    #[must_use]
    #[inline]
    pub fn challenger(&self) -> GC::Challenger {
        self.verifier.challenger()
    }

    /// Get the machine.
    #[must_use]
    #[inline]
    pub fn machine(&self) -> &Machine<GC::F, SC::Air> {
        self.verifier.machine()
    }

    /// Get the maximum log row count.
    #[must_use]
    pub fn max_log_row_count(&self) -> usize {
        self.verifier.max_log_row_count()
    }

    /// Get the log stacking height.
    #[must_use]
    pub fn log_stacking_height(&self) -> u32 {
        self.verifier.log_stacking_height()
    }

    /// Given a record, compute the shape of the resulting shard proof.
    pub fn shape_from_record(
        &self,
        record: &Record<GC, SC>,
    ) -> Option<CoreProofShape<GC::F, SC::Air>> {
        let log_stacking_height = self.verifier.log_stacking_height() as usize;
        let max_log_row_count = self.verifier.max_log_row_count();
        let airs = self.machine().chips();
        let shard_chips: BTreeSet<_> =
            airs.iter().filter(|air| air.included(record)).cloned().collect();
        let preprocessed_multiple = shard_chips
            .iter()
            .map(|air| air.preprocessed_width() * air.num_rows(record).unwrap_or_default())
            .sum::<usize>()
            .div_ceil(1 << log_stacking_height);
        let main_multiple = shard_chips
            .iter()
            .map(|air| air.width() * air.num_rows(record).unwrap_or_default())
            .sum::<usize>()
            .div_ceil(1 << log_stacking_height);

        let main_padding_cols = (main_multiple * (1 << log_stacking_height)
            - shard_chips
                .iter()
                .map(|air| air.width() * air.num_rows(record).unwrap_or_default())
                .sum::<usize>())
        .div_ceil(1 << max_log_row_count);

        let preprocessed_padding_cols = (preprocessed_multiple * (1 << log_stacking_height)
            - shard_chips
                .iter()
                .map(|air| air.preprocessed_width() * air.num_rows(record).unwrap_or_default())
                .sum::<usize>())
        .div_ceil(1 << max_log_row_count);

        let shard_chips = self.machine().smallest_cluster(&shard_chips).cloned()?;
        Some(CoreProofShape {
            shard_chips,
            preprocessed_multiple,
            main_multiple,
            preprocessed_padding_cols,
            main_padding_cols,
        })
    }

    /// Call setup on an available worker.
    #[inline]
    #[must_use]
    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all, name = "machine_setup")]
    pub async fn setup(
        &self,
        program: Arc<Program<GC, SC>>,
        vk: Option<MachineVerifyingKey<GC>>,
    ) -> (PreprocessedData<ProvingKey<GC, SC, C>>, MachineVerifyingKey<GC>) {
        // Get a worker from the queue.
        let worker =
            self.worker_queue.clone().pop().await.expect("no workers for setup, this is a bug.");

        // Copy the worker index.
        let idx = *worker;

        self.base_workers[idx].setup_from_vk(program, vk, self.worker_permits[idx].clone()).await
    }

    /// Call `prove_shard` on an available worker.
    #[inline]
    #[must_use]
    #[tracing::instrument(skip_all, name = "machine_prove_shard")]
    pub async fn prove_shard(
        &self,
        pk: Arc<ProvingKey<GC, SC, C>>,
        record: Record<GC, SC>,
    ) -> ShardProof<GC, PcsProof<GC, SC>> {
        // Get a worker from the queue.
        let worker =
            self.worker_queue.clone().pop().await.expect("no workers for setup, this is a bug.");

        // Copy the worker index.
        let idx = *worker;

        let mut challenger = self.challenger();
        pk.vk.observe_into(&mut challenger);

        let (proof, _) = self.base_workers[idx]
            .prove_shard_with_pk(pk, record, self.worker_permits[idx].clone(), &mut challenger)
            .await;

        // Return the proof.
        proof
    }

    /// Call `setup_and_prove_shard` on an available worker.
    #[inline]
    #[must_use]
    #[allow(clippy::type_complexity)]
    #[tracing::instrument(skip_all, name = "machine_setup_and_prove_shard")]
    pub async fn setup_and_prove_shard(
        &self,
        program: Arc<Program<GC, SC>>,
        vk: Option<MachineVerifyingKey<GC>>,
        record: Record<GC, SC>,
    ) -> (MachineVerifyingKey<GC>, ShardProof<GC, PcsProof<GC, SC>>) {
        // Get a worker from the queue.
        let worker =
            self.worker_queue.clone().pop().await.expect("no workers for setup, this is a bug.");

        // Copy the worker index.
        let idx = *worker;

        let mut challenger = self.challenger();

        let (vk, proof, _) = self.base_workers[idx]
            .setup_and_prove_shard(
                program,
                record,
                vk,
                self.worker_permits[idx].clone(),
                &mut challenger,
            )
            .await;

        (vk, proof)
    }

    /// A function to extract preprocessed table heights from the pk.
    pub async fn preprocessed_table_heights(
        &self,
        pk: Arc<ProvingKey<GC, SC, C>>,
    ) -> BTreeMap<String, usize> {
        C::preprocessed_table_heights(pk).await
    }
}
