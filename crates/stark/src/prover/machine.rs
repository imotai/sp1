// use std::{marker::PhantomData, sync::Arc};

// use thiserror::Error;
// use tokio::sync::mpsc::{self, Receiver, UnboundedSender};

// use crate::{Chip, Machine, ShardProof};

// use super::{MachineProverComponents, MachineProvingKey, ShardData, ShardProver};

use std::{marker::PhantomData, sync::Arc};

use slop_futures::queue::WorkerQueue;
use thiserror::Error;
use tokio::sync::mpsc::{self, Receiver, UnboundedSender};
use tracing::Instrument;

use crate::ShardProof;

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
            mpsc::channel::<ShardData<C::F, C::B>>(self.opts.shard_data_channel_capacity);
        let _records_handle = tokio::spawn(async move {
            while let Some(record) = records.recv().await {
                // Get a trace worker.
                let shard_prover = trace_workers.clone().pop().await.unwrap();
                let data_tx = data_tx.clone();
                tokio::spawn(async move {
                    // Generate the traces.
                    shard_prover
                        .generate_traces(record, &data_tx)
                        .instrument(tracing::debug_span!("generate traces"))
                        .await;
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
                tracing::info!("proving");
                let proof = shard_prover.prove_shard(&pk, data, &mut challenger).await;
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

// #[derive(Debug, Error)]
// pub enum MachineProverError<C> {
//     #[error("error")]
//     Error(PhantomData<C>),
// }

// impl<C: MachineProverComponents> MachineProver<C> {
//     #[inline]
//     pub const fn new(opts: MachineProverOpts, shard_prover: ShardProver<C>) -> Self {
//         Self { shard_prover, opts }
//     }

//     #[inline]
//     pub fn chips(&self) -> &[Chip<C::F, C::Air>] {
//         self.shard_prover.chips()
//     }

//     #[inline]
//     pub fn machine(&self) -> &Machine<C::F, C::Air> {
//         self.shard_prover.machine()
//     }

//     #[allow(clippy::too_many_arguments)]
//     pub async fn prove_stream(
//         self: Arc<Self>,
//         pk: Arc<MachineProvingKey<C>>,
//         records: Receiver<C::Record>,
//         proof_tx: UnboundedSender<ShardProof<C::Config>>,
//         mut challenger: C::Challenger,
//     ) -> Result<(), MachineProverError<C>> {
//         // Spawn the trace generation tasks.
//         let (data_tx, data_rx) =
//             mpsc::channel::<ShardData<C::F, C::B>>(self.opts.shard_data_channel_capacity);
//         let records_rx = Arc::new(records);
//         for _ in 0..opts.trace_gen_workers {
//             let data_tx = data_tx.clone();
//             let prover = prover.clone();
//             let records_rx = records_rx.clone();
//             tokio::task::spawn(async move {
//                 loop {
//                     let mut received = { records_rx.lock().await };
//                     if let Some(record) = received.recv().await {
//                         let shard = record.public_values.shard;
//                         prover
//                             .generate_traces(record, &data_tx)
//                             .instrument(tracing::debug_span!("generate traces", shard = shard))
//                             .await;
//                     } else {
//                         break;
//                     }
//                 }
//             });
//         }
//         drop(data_tx);

//         // Spawn the proving tasks.

//         // Observe the proving key.
//         pk.observe_into(&mut challenger);

//         let prover = prover.clone();
//         let proof_tx = proof_tx.clone();
//         let data_rx = Arc::new(tokio::sync::Mutex::new(data_rx));
//         let mut prover_handles = vec![];
//         for _ in 0..opts.shard_batch_size {
//             let pk = pk.clone();
//             let mut challenger = challenger.clone();
//             let proof_tx = proof_tx.clone();
//             let data_rx = data_rx.clone();
//             let prover = prover.clone();
//             let pk = pk.clone();
//             let handle = tokio::task::spawn(async move {
//                 loop {
//                     let received = { data_rx.lock().await.recv().await };
//                     if let Some(data) = received {
//                         let pv: &PublicValues<Word<F>, F> = data.public_values.as_slice().borrow();
//                         let shard = pv.shard.as_canonical_u32();
//                         let time = tokio::time::Instant::now();
//                         let proof = prover
//                             .prove_shard(&pk, data, &mut challenger)
//                             .instrument(tracing::debug_span!("prove shard", shard = shard))
//                             .await;
//                         tracing::info!("prove shard {} took {:?}", shard, time.elapsed());
//                         proof_tx.send(proof).unwrap();
//                     } else {
//                         break;
//                     }
//                 }
//             });
//             prover_handles.push(handle);
//         }
//         drop(proof_tx);

//         // Wait for the prover handles to finish.
//         for handle in prover_handles {
//             handle.await.unwrap();
//         }

//         Ok(())
//     }
// }
