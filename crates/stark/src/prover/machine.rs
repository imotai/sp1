use std::{marker::PhantomData, sync::Arc};

use slop_futures::queue::WorkerQueue;
use thiserror::Error;
use tokio::sync::{
    mpsc::{self, Receiver, UnboundedSender},
    Semaphore,
};
use tracing::Instrument;

use crate::{SP1ProverOpts, ShardProof};

use super::{MachineProverComponents, MachineProvingKey, ShardData, ShardProver};

/// The options for a machine prover.
pub struct MachineProverOpts {
    /// The number of prover workers.
    pub num_prover_workers: usize,
    /// The number of trace workers.
    pub num_trace_workers: usize,
    /// The capacity of the shard data channel.
    pub shard_data_channel_capacity: usize,
}

impl Default for MachineProverOpts {
    fn default() -> Self {
        let default_opts = SP1ProverOpts::default();
        Self {
            num_prover_workers: default_opts.core_opts.trace_gen_workers,
            num_trace_workers: default_opts.core_opts.trace_gen_workers,
            shard_data_channel_capacity: default_opts.core_opts.records_and_traces_channel_capacity,
        }
    }
}

/// A prover for a machine.
pub struct MachineProver<C: MachineProverComponents> {
    /// The trace workers.
    trace_workers: Arc<WorkerQueue<Arc<ShardProver<C>>>>,
    /// The proof workers.
    prove_workers: Arc<WorkerQueue<Arc<ShardProver<C>>>>,
    /// The options.
    opts: MachineProverOpts,
}

/// An error that occurs during the prover of a machine.
#[derive(Debug, Error)]
pub enum MachineProverError<C> {
    /// An dummy error.
    #[error("Error")]
    Error(PhantomData<C>),
}

impl<C: MachineProverComponents> MachineProver<C> {
    /// Create a new machine prover.
    #[must_use]
    pub fn new(opts: MachineProverOpts, shard_prover: &Arc<ShardProver<C>>) -> Self {
        let trace_workers =
            (0..opts.num_trace_workers).map(|_| shard_prover.clone()).collect::<Vec<_>>();
        let prove_workers =
            (0..opts.num_prover_workers).map(|_| shard_prover.clone()).collect::<Vec<_>>();
        Self {
            trace_workers: Arc::new(WorkerQueue::new(trace_workers)),
            prove_workers: Arc::new(WorkerQueue::new(prove_workers)),
            opts,
        }
    }

    /// Produce a stream of shard proofs.
    pub async fn prove_stream(
        &self,
        pk: Arc<MachineProvingKey<C>>,
        mut records: Receiver<C::Record>,
        proofs: UnboundedSender<ShardProof<C::Config>>,
        mut challenger: C::Challenger,
    ) -> Result<(), MachineProverError<C>> {
        // Observe the proving key.
        pk.observe_into(&mut challenger);

        // Wait for records to arrive, and for each of them, spawn a trace generation task.
        let trace_workers = self.trace_workers.clone();
        // Set up a channel for the shard data.
        let (data_tx, mut data_rx) =
            mpsc::channel::<ShardData<C::F, C::Air, C::B>>(self.opts.shard_data_channel_capacity);

        let prover_permits = Arc::new(Semaphore::new(self.opts.num_prover_workers));
        let _records_handle = tokio::spawn(async move {
            while let Some(record) = records.recv().await {
                // Get a trace worker.
                let shard_prover = trace_workers.clone().pop().await.unwrap();
                let data_tx = data_tx.clone();
                let prover_permits = prover_permits.clone();
                tokio::spawn(async move {
                    // Generate the traces.
                    let shard_data = shard_prover
                        .generate_traces(record, prover_permits)
                        .instrument(tracing::debug_span!("generate traces"))
                        .await;
                    data_tx.send(shard_data).await.unwrap();
                });
            }
            drop(data_tx);
        });

        // Get a channel to signal the end of the proof generation.
        let (done_tx, mut done_rx) = mpsc::unbounded_channel();
        // Wait for the shard data to arrive, and for each of them, spawn a proof task.
        while let Some(data) = data_rx.recv().await {
            // Get a proof worker.
            let shard_prover = self.prove_workers.clone().pop().await.unwrap();
            let pk = pk.clone();
            let mut challenger = challenger.clone();
            let output = proofs.clone();
            let done_tx = done_tx.clone();
            // Spawn a task to generate the proof and send it.
            let _prover_handle = tokio::spawn(async move {
                let proof = shard_prover
                    .prove_shard(&pk, data, &mut challenger)
                    .instrument(tracing::debug_span!("prove shard"))
                    .await;
                output.send(proof).unwrap();
                done_tx.send(()).unwrap();
            });
        }
        drop(done_tx);

        // Wait for the proof generation to finish.
        while done_rx.recv().await.is_some() {}

        Ok(())
    }
}
