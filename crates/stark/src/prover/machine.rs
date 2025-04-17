use futures::{
    prelude::*,
    stream::{AbortHandle, AbortRegistration, Abortable},
};
use itertools::Itertools;
use slop_air::{Air, BaseAir};
use thiserror::Error;

use std::{collections::BTreeSet, sync::Arc};

use slop_futures::{handle::TaskHandle, queue::WorkerQueue};

use tokio::sync::{mpsc, oneshot};

use crate::{
    air::MachineAir,
    prover::CoreProofShape,
    prover::{ProverPermit, ProverPermits},
    Machine, MachineProof, MachineVerifier, MachineVerifierError, MachineVerifyingKey, ShardProof,
    ShardVerifier, VerifierConstraintFolder,
};

use super::{MachineProverComponents, MachineProvingKey, ProverSemaphore, ShardProver};

/// A builder for a machine prover.
pub struct MachineProverBuilder<C: MachineProverComponents> {
    verifier: MachineVerifier<C::Config, C::Air>,
    base_workers: Vec<Arc<ShardProver<C>>>,
    worker_permits: Vec<ProverSemaphore>,
    num_workers: Vec<usize>,
}

struct MachineProverInner<C: MachineProverComponents> {
    /// The task channel.
    task_tx: mpsc::UnboundedSender<Task<C>>,
    /// The verifier.
    verifier: MachineVerifier<C::Config, C::Air>,
    /// The number of workers.
    num_workers: usize,
}

/// A machine prover.
pub struct MachineProver<C: MachineProverComponents> {
    /// The task channel.
    inner: Arc<MachineProverInner<C>>,
}

/// An error for a machine prover request.
#[derive(Debug, Clone, Error)]
pub enum MachineProverError {
    /// The task failed.
    #[error("Machine prover failed")]
    TaskFailed,
    /// The prover is already closed.
    #[error("Machine prover is already closed")]
    ProverClosed,
    /// The task was aborted.
    #[error("Task was aborted")]
    TaskAborted,
    /// The prover panicked.
    #[error("Prover panicked")]
    ProverPanicked(#[from] oneshot::error::RecvError),
}

enum Task<C: MachineProverComponents> {
    Setup(SetupTask<C>),
    Prove(ProveTask<C>),
    SetupAndProve(SetupAndProveTask<C>),
}

#[allow(clippy::type_complexity)]
struct SetupTask<C: MachineProverComponents> {
    program: Arc<C::Program>,
    vk: Option<MachineVerifyingKey<C::Config>>,
    output_tx: oneshot::Sender<
        Result<(ProvingKey<C>, MachineVerifyingKey<C::Config>), MachineProverError>,
    >,
    abort_registration: AbortRegistration,
}

struct ProveTask<C: MachineProverComponents> {
    pk: Arc<MachineProvingKey<C>>,
    record: C::Record,
    proof_tx: oneshot::Sender<Result<ShardProof<C::Config>, MachineProverError>>,
    abort_registration: AbortRegistration,
}

#[allow(clippy::type_complexity)]
struct SetupAndProveTask<C: MachineProverComponents> {
    program: Arc<C::Program>,
    vk: Option<MachineVerifyingKey<C::Config>>,
    record: C::Record,
    proof_tx: oneshot::Sender<
        Result<(MachineVerifyingKey<C::Config>, ShardProof<C::Config>), MachineProverError>,
    >,
    abort_registration: AbortRegistration,
}

/// A proving key with a permit.
pub struct ProvingKey<C: MachineProverComponents> {
    /// The underlying proving key.
    pk: MachineProvingKey<C>,
    /// The permit for the prover.
    permit: ProverPermit,
}

impl<C: MachineProverComponents> ProvingKey<C> {
    /// Create a new proving key with a permit.
    #[must_use]
    #[inline]
    pub fn new(pk: MachineProvingKey<C>, permit: ProverPermit) -> Self {
        Self { pk, permit }
    }

    /// Get the underlying proving key.
    ///
    /// # Safety
    /// the caller must ensure that the resource
    #[must_use]
    #[inline]
    pub unsafe fn into_inner(self) -> MachineProvingKey<C> {
        self.pk
    }

    /// Get the underlying proving key and permit.
    ///
    /// # Safety
    /// todo
    #[must_use]
    #[inline]
    pub unsafe fn into_parts(self) -> (MachineProvingKey<C>, ProverPermit) {
        (self.pk, self.permit)
    }
}
async fn setup_from_vk<C: MachineProverComponents>(
    prover: &ShardProver<C>,
    program: Arc<C::Program>,
    vk: Option<MachineVerifyingKey<C::Config>>,
    prover_permits: ProverPermits,
) -> (MachineProvingKey<C>, MachineVerifyingKey<C::Config>, ProverPermit) {
    if let Some(vk) = vk {
        let initial_global_cumulative_sum = vk.initial_global_cumulative_sum;
        prover
            .setup_with_initial_global_cumulative_sum(
                program,
                initial_global_cumulative_sum,
                prover_permits,
            )
            .await
    } else {
        prover.setup(program, prover_permits).await
    }
}

impl<C: MachineProverComponents> MachineProverBuilder<C> {
    /// Crate a new builder for a machine prover.
    ///
    /// The builder is constructed from different groups of workers, each sharing their own permits.
    /// In practice, those permits can come from the same semaphore or different ones.
    pub fn new(
        shard_verifier: ShardVerifier<C::Config, C::Air>,
        worker_permits: Vec<ProverSemaphore>,
        base_workers: Vec<Arc<ShardProver<C>>>,
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
        shard_verifier: ShardVerifier<C::Config, C::Air>,
        shard_prover: ShardProver<C>,
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
    pub fn build(&mut self) -> MachineProver<C> {
        // Initialize the task channel.
        let (task_tx, mut task_rx) = mpsc::unbounded_channel();

        // Spawn the workers tasks and initialize the worker channel.
        let total_num_workers = self.num_workers.iter().sum();
        let mut prover_worker_channels = Vec::with_capacity(total_num_workers);
        for ((base_worker, &num_workers), permits) in self
            .base_workers
            .iter()
            .zip_eq(self.num_workers.iter())
            .zip_eq(self.worker_permits.iter())
        {
            for _ in 0..num_workers {
                // Initialize a channel for sending shard data to this trace worker.
                let (tx, mut rx) = mpsc::unbounded_channel::<Task<C>>();
                prover_worker_channels.push(tx);
                let worker = base_worker.clone();
                let prover_permits = permits.clone();
                let verifier = self.verifier.clone();
                tokio::spawn(async move {
                    while let Some(task) = rx.recv().await {
                        match task {
                            Task::Setup(task) => {
                                let SetupTask { program, vk, output_tx, abort_registration } = task;
                                let setup_result = Abortable::new(
                                    setup_from_vk(&worker, program, vk, prover_permits.pending()),
                                    abort_registration,
                                )
                                .map_err(|_| MachineProverError::TaskAborted)
                                .map_ok(|(pk, vk, permit)| {
                                    let pk = ProvingKey { pk, permit };
                                    (pk, vk)
                                })
                                .await;
                                // Send the output to the channel if it's still open, otherwise drop
                                // it.
                                output_tx.send(setup_result).ok();
                            }
                            Task::Prove(task) => {
                                let ProveTask { pk, record, proof_tx, abort_registration } = task;
                                let shard_proof_result = Abortable::new(
                                    async {
                                        // Create a challenger.
                                        let mut challenger = verifier.challenger();
                                        // Observe the preprocessed information.
                                        pk.observe_into(&mut challenger);
                                        // Prove the shard.
                                        worker
                                            .prove_shard(
                                                &pk,
                                                record,
                                                prover_permits.pending(),
                                                &mut challenger,
                                            )
                                            .await
                                    },
                                    abort_registration,
                                )
                                .map_err(|_| MachineProverError::TaskAborted)
                                .map_ok(|(proof, _)| proof)
                                .await;
                                proof_tx.send(shard_proof_result).ok();
                            }
                            Task::SetupAndProve(task) => {
                                let SetupAndProveTask {
                                    program,
                                    vk,
                                    record,
                                    proof_tx,
                                    abort_registration,
                                } = task;
                                let result = Abortable::new(
                                    async {
                                        let (vk, shard_proof, _) = {
                                            // Setup
                                            let (pk, vk, permit) = setup_from_vk(
                                                &worker,
                                                program,
                                                vk,
                                                prover_permits.pending(),
                                            )
                                            .await;
                                            // Create a challenger.
                                            let mut challenger = verifier.challenger();
                                            // Observe the preprocessed information.
                                            pk.observe_into(&mut challenger);
                                            // Prove the shard.
                                            let permit = ProverPermits::ready(permit);
                                            let (shard_proof, permit) = worker
                                                .prove_shard(&pk, record, permit, &mut challenger)
                                                .await;
                                            drop(pk);
                                            (vk, shard_proof, permit)
                                        };
                                        (vk, shard_proof)
                                    },
                                    abort_registration,
                                )
                                .map_err(|_| MachineProverError::TaskAborted)
                                .await;
                                proof_tx.send(result).ok();
                            }
                        }
                    }
                });
            }
        }

        // Spawn the entrypoint task.
        tokio::spawn(async move {
            let prover_worker_channels = Arc::new(WorkerQueue::new(prover_worker_channels));
            while let Some(task) = task_rx.recv().await {
                // Get a prover worker from the queue.
                let prover_worker = prover_worker_channels.clone().pop().await.unwrap();
                // Send the task to the prover worker.
                prover_worker.send(task).unwrap();
            }
        });

        let inner = Arc::new(MachineProverInner {
            task_tx,
            verifier: self.verifier.clone(),
            num_workers: total_num_workers,
        });

        MachineProver { inner }
    }
}

impl<C: MachineProverComponents> MachineProver<C> {
    /// Verify a machine proof.
    pub fn verify(
        &self,
        vk: &MachineVerifyingKey<C::Config>,
        proof: &MachineProof<C::Config>,
    ) -> Result<(), MachineVerifierError<C::Config>>
    where
        C::Air: for<'a> Air<VerifierConstraintFolder<'a, C::Config>>,
    {
        self.inner.verifier.verify(vk, proof)
    }

    /// Get the number of workers.
    #[must_use]
    #[inline]
    pub fn num_workers(&self) -> usize {
        self.inner.num_workers
    }

    /// Get the verifier.
    #[must_use]
    #[inline]
    pub fn verifier(&self) -> &MachineVerifier<C::Config, C::Air> {
        &self.inner.verifier
    }

    /// Get a new challenger.
    #[must_use]
    #[inline]
    pub fn challenger(&self) -> C::Challenger {
        self.inner.verifier.challenger()
    }

    /// Get the machine.
    #[must_use]
    #[inline]
    pub fn machine(&self) -> &Machine<C::F, C::Air> {
        self.inner.verifier.machine()
    }

    /// Get the maximum log row count.
    #[must_use]
    pub fn max_log_row_count(&self) -> usize {
        self.inner.verifier.max_log_row_count()
    }

    /// Get the log stacking height.
    #[must_use]
    pub fn log_stacking_height(&self) -> u32 {
        self.inner.verifier.log_stacking_height()
    }

    /// Given a record, compute the shape of the resulting shard proof.
    pub fn shape_from_record(&self, record: &C::Record) -> Option<CoreProofShape<C::F, C::Air>> {
        let log_stacking_height = self.inner.verifier.log_stacking_height() as usize;
        let airs = self.machine().chips();
        let shard_chips: BTreeSet<_> =
            airs.iter().filter(|air| air.included(record)).cloned().collect();
        let shard_chips = self.machine().smallest_cluster(&shard_chips).cloned()?;
        let preprocessed_multiple = airs
            .iter()
            .map(|air| air.preprocessed_width() * air.num_rows(record).unwrap_or_default())
            .sum::<usize>()
            .div_ceil(log_stacking_height);
        let main_multiple = airs
            .iter()
            .map(|air| air.width() * air.num_rows(record).unwrap_or_default())
            .sum::<usize>()
            .div_ceil(log_stacking_height);
        Some(CoreProofShape { shard_chips, preprocessed_multiple, main_multiple })
    }

    /// Given a proof, compute its shape.
    // pub fn shape_from_proof(&self, proof: &ShardProof<C::Config>) -> CoreProofShape<C::F, C::Air>
    // {     let shard_chips = self
    //         .machine()
    //         .chips()
    //         .iter()
    //         .filter(|air| proof.shard_chips.contains(&air.name()))
    //         .cloned()
    //         .collect::<BTreeSet<_>>();
    //     debug_assert_eq!(shard_chips.len(), proof.shard_chips.len());

    //     let preprocessed_multiple =
    //         proof.evaluation_proof.stacked_pcs_proof.batch_evaluations.rounds[0].
    // round_evaluations             [0]
    //         .num_polynomials();
    //     let main_multiple = proof.evaluation_proof.stacked_pcs_proof.batch_evaluations.rounds[1]
    //         .round_evaluations[0]
    //         .num_polynomials();

    //     CoreProofShape { shard_chips, preprocessed_multiple, main_multiple }
    // }

    /// Send a setup task to the machine prover.
    #[inline]
    #[must_use]
    pub fn setup(
        &self,
        program: Arc<C::Program>,
        vk: Option<MachineVerifyingKey<C::Config>>,
    ) -> TaskHandle<(ProvingKey<C>, MachineVerifyingKey<C::Config>), MachineProverError> {
        let (output_tx, output_rx) = oneshot::channel();
        // Crate an abort handle for this task.
        let (abort_handle, abort_registration) = AbortHandle::new_pair();
        let task = Task::Setup(SetupTask { program, vk, output_tx, abort_registration });
        self.inner.task_tx.send(task).unwrap();

        TaskHandle::new(output_rx, abort_handle)
    }

    /// Send a prove task to the machine prover.
    #[inline]
    #[must_use]
    pub fn prove_shard(
        &self,
        pk: Arc<MachineProvingKey<C>>,
        record: C::Record,
    ) -> TaskHandle<ShardProof<C::Config>, MachineProverError> {
        let (output_tx, output_rx) = oneshot::channel();
        // Create an abortable task.
        let (abort_handle, abort_registration) = AbortHandle::new_pair();
        let task = Task::Prove(ProveTask { pk, record, proof_tx: output_tx, abort_registration });
        // Since the prover is in scope, the channel should be open so we can unwrap the send error.
        self.inner.task_tx.send(task).unwrap();
        // Crate an abort handle for this task.
        TaskHandle::new(output_rx, abort_handle)
    }

    /// Setup and prove a shard.
    #[inline]
    #[must_use]
    #[allow(clippy::type_complexity)]
    pub fn setup_and_prove_shard(
        &self,
        program: Arc<C::Program>,
        vk: Option<MachineVerifyingKey<C::Config>>,
        record: C::Record,
    ) -> TaskHandle<(MachineVerifyingKey<C::Config>, ShardProof<C::Config>), MachineProverError>
    {
        let (output_tx, output_rx) = oneshot::channel();
        // Create an abortable task.
        let (abort_handle, abort_registration) = AbortHandle::new_pair();
        let task = Task::SetupAndProve(SetupAndProveTask {
            program,
            vk,
            record,
            proof_tx: output_tx,
            abort_registration,
        });
        // Since the prover is in scope, the channel should be open so we can unwrap the send error.
        self.inner.task_tx.send(task).unwrap();
        // Crate an abort handle for this task.
        TaskHandle::new(output_rx, abort_handle)
    }
}
