use std::{
    collections::{BTreeMap, VecDeque},
    sync::{Arc, Mutex},
};

use futures::future::try_join_all;
use hashbrown::HashMap;
use sp1_hypercube::{
    air::{ShardBoundary, ShardRange},
    SP1RecursionProof,
};
use sp1_primitives::SP1GlobalContext;
use sp1_prover_types::{Artifact, ArtifactClient, ArtifactId, ArtifactType, TaskStatus, TaskType};
use sp1_recursion_circuit::machine::SP1ShapedWitnessValues;
use tokio::{sync::mpsc, task::JoinSet};
use tracing::Instrument;

use crate::{
    worker::{ProofData, ReduceTaskRequest, TaskContext, TaskError, TaskId, WorkerClient},
    InnerSC, SP1CircuitWitness, SP1CompressWitness,
};

pub struct CompressTask {
    pub witness: SP1CompressWitness,
}

#[derive(Debug, Clone)]
pub struct RecursionProof {
    pub shard_range: ShardRange,
    pub proof: Artifact,
}

#[derive(Clone, Debug)]
pub struct RangeProofs {
    pub shard_range: ShardRange,
    pub proofs: VecDeque<RecursionProof>,
}

impl RangeProofs {
    pub fn new(shard_range: ShardRange, proofs: VecDeque<RecursionProof>) -> Self {
        Self { shard_range, proofs }
    }

    pub fn as_artifacts(self) -> impl Iterator<Item = Artifact> + Send + Sync {
        let range_artifact = Artifact::from(
            serde_json::to_string(&self.shard_range).expect("Failed to serialize shard range"),
        );
        std::iter::once(range_artifact).chain(self.proofs.into_iter().flat_map(|proof| {
            let range_str =
                serde_json::to_string(&proof.shard_range).expect("Failed to serialize shard range");
            let range_artifact = Artifact::from(range_str);
            let proof_artifact = proof.proof;
            [range_artifact, proof_artifact]
        }))
    }

    pub fn from_artifacts(artifacts: &[Artifact]) -> Result<Self, TaskError> {
        if artifacts.len() % 2 != 1 || artifacts.len() <= 1 {
            return Err(TaskError::Fatal(anyhow::anyhow!(
                "Invalid number of artifacts: {:?}",
                artifacts.len()
            )));
        }
        let shard_range =
            serde_json::from_str(artifacts[0].id()).map_err(|e| TaskError::Fatal(e.into()))?;
        let proofs = artifacts[1..]
            .chunks_exact(2)
            .map(|chunk| -> Result<RecursionProof, TaskError> {
                let shard_range =
                    serde_json::from_str(chunk[0].id()).map_err(|e| TaskError::Fatal(e.into()))?;
                let proof = chunk[1].clone();
                Ok(RecursionProof { shard_range, proof })
            })
            .collect::<Result<VecDeque<RecursionProof>, TaskError>>()?;
        Ok(RangeProofs { shard_range, proofs })
    }

    pub fn len(&self) -> usize {
        self.proofs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.proofs.is_empty()
    }

    pub fn push_right(&mut self, proof: RecursionProof) {
        assert_eq!(proof.shard_range.end(), self.shard_range.start());
        self.shard_range = (proof.shard_range.start()..self.shard_range.end()).into();
        self.proofs.push_front(proof);
    }

    pub fn push_left(&mut self, proof: RecursionProof) {
        assert_eq!(proof.shard_range.start(), self.shard_range.end());
        self.shard_range = (self.shard_range.start()..proof.shard_range.end()).into();
        self.proofs.push_back(proof);
    }

    pub fn split_off(&mut self, at: usize) -> Option<Self> {
        if at >= self.proofs.len() {
            return None;
        }
        // Split the proofs off at the given index.
        let proofs = self.proofs.split_off(at);
        // Get the range of the proofs.
        let range = {
            let at_start_range = proofs.front().unwrap().shard_range.start();
            let at_end_range = proofs.iter().last().unwrap().shard_range.end();
            at_start_range..at_end_range
        }
        .into();
        // Get the new range of the self.
        let new_self_range = {
            let at_start_range = self.proofs.front().unwrap().shard_range.start();
            let at_end_range = self.proofs.iter().last().unwrap().shard_range.end();
            at_start_range..at_end_range
        };
        // Update the shard range of the self.
        self.shard_range = new_self_range.into();
        // Return the new proofs.
        Some(Self { shard_range: range, proofs })
    }

    pub fn push_both(&mut self, middle: RecursionProof, right: Self) {
        assert_eq!(middle.shard_range.start(), self.shard_range.end());
        assert_eq!(right.shard_range.start(), middle.shard_range.end());
        // Push the middle to the queue.
        self.proofs.push_back(middle);
        // Append the right proofs to the queue.
        for proof in right.proofs {
            self.proofs.push_back(proof);
        }
        // Update the shard range.
        self.shard_range = (self.shard_range.start()..right.shard_range.end()).into();
    }

    pub fn range(&self) -> ShardRange {
        self.shard_range
    }

    pub async fn download_witness(
        self,
        is_complete: bool,
        artifact_client: &impl ArtifactClient,
    ) -> Result<SP1CircuitWitness, TaskError> {
        // Download the proofs
        let proofs = try_join_all(self.proofs.iter().map(|proof| async {
            let downloaded_proof = artifact_client
                .download::<SP1RecursionProof<SP1GlobalContext, InnerSC>>(&proof.proof)
                .await?;
            // Delete the proof artifact.
            artifact_client.try_delete(&proof.proof, ArtifactType::UnspecifiedArtifactType).await?;

            Ok::<_, TaskError>(downloaded_proof)
        }))
        .await?;

        let vks_and_proofs =
            proofs.into_iter().map(|proof| (proof.vk, proof.proof)).collect::<Vec<_>>();
        let witness = SP1ShapedWitnessValues { vks_and_proofs, is_complete };
        let witness = SP1CircuitWitness::Compress(witness);
        Ok(witness)
    }
}

/// An enum marking which sibling was found.
enum Sibling {
    Left(RangeProofs),
    Right(RangeProofs),
    Both(RangeProofs, RangeProofs),
}

pub struct CompressTree {
    map: BTreeMap<ShardBoundary, RangeProofs>,
    batch_size: usize,
}

impl CompressTree {
    pub fn new(batch_size: usize) -> Self {
        Self { map: BTreeMap::new(), batch_size }
    }

    /// Insert a new range of proofs into the tree.
    fn insert(&mut self, proofs: RangeProofs) {
        self.map.insert(proofs.shard_range.start(), proofs);
    }

    /// Get the sibling of a proof.
    fn sibling(&mut self, proof: &RecursionProof) -> Option<Sibling> {
        // Check for a left sibling
        if let Some(previous) =
            self.map.range(ShardBoundary::initial()..=proof.shard_range.start()).next_back()
        {
            let (start, proofs) = previous;
            let start = *start;
            let proofs = proofs.clone();

            if proofs.shard_range.end() == proof.shard_range.start() {
                let left = self.map.remove(&start).unwrap();
                // Check for a right sibling.
                if let Some(right) = self.map.remove(&proof.shard_range.end()) {
                    return Some(Sibling::Both(left, right));
                } else {
                    return Some(Sibling::Left(left));
                }
            }
        }
        // If there is no left sibling, check for a right sibling.
        if let Some(right) = self.map.remove(&proof.shard_range.end()) {
            return Some(Sibling::Right(right));
        }

        // No sibling found.
        None
    }

    fn is_complete(
        &self,
        range: &ShardRange,
        pending_tasks: usize,
        full_range: &Option<ShardRange>,
    ) -> bool {
        tracing::debug!(
            "Checking if complete: Pending tasks: {:?}, map is empty: {:?}, full range: {:?}",
            pending_tasks,
            self.map.is_empty(),
            full_range.as_ref().is_some_and(|full| range == full)
        );
        (pending_tasks == 0)
            && self.map.is_empty()
            && full_range.as_ref().is_some_and(|full| range == full)
    }

    /// Reduce the proofs into the tree until the batch size is reached.
    ///
    /// ### Inputs
    ///
    /// - `full_range_rx`: A receiver for the full range of proofs.
    /// - `proofs_rx`: A receiver for the proofs to reduce.
    /// - `recursion_executors`: A queue of executors to use to execute the proofs.
    /// - `pending_tasks`: The number of pending tasks that are already running.
    ///
    /// **Remark**: it's important to keep track of the number of pending tasks because the shard
    /// ranges only cover timestamp ranges but do not cover how many precomputed proofs are in the
    /// tree.
    ///
    /// ### Outputs
    ///
    /// - A vector of proofs that have been reduced.
    ///
    /// ### Notes
    ///
    /// This function will terminate when the batch size is reached or when the full range is
    /// reached and proven.
    pub async fn reduce_proofs(
        &mut self,
        context: TaskContext,
        output: Artifact,
        mut core_proofs_rx: mpsc::UnboundedReceiver<ProofData>,
        artifact_client: &impl ArtifactClient,
        worker_client: &impl WorkerClient,
    ) -> Result<(), TaskError> {
        // Populate the recursion proofs into the tree until we reach the reduce batch size.

        // Create a subscriber for core proof tasks.
        let (core_proofs_subscriber, mut core_proofs_event_stream) =
            worker_client.subscriber(context.proof_id.clone()).await?.stream();
        let core_proof_map = Arc::new(Mutex::new(HashMap::<TaskId, RecursionProof>::new()));
        // Keep track of the full range of proofs.
        let mut full_range: Option<ShardRange> = None;
        // Keep track of the max range of proofs that have been processed.
        let mut max_range = ShardBoundary::initial()..ShardBoundary::initial();
        // Keep track of the number of pending tasks.
        let mut pending_tasks = 0;
        // Create a channel to send the proofs to the proof queue.
        let (proof_tx, mut proof_rx) = mpsc::unbounded_channel::<RecursionProof>();
        // Create a subscriber for the reduction tasks.
        let (subscriber, mut event_stream) =
            worker_client.subscriber(context.proof_id.clone()).await?.stream();
        let mut proof_map = HashMap::<TaskId, RecursionProof>::new();

        let mut join_set = JoinSet::<Result<(), TaskError>>::new();

        let (num_core_proofs_tx, mut num_core_proofs_rx) = mpsc::channel(1);
        // Spawn a task to process the incoming core proofs and subscribe to them.
        join_set.spawn({
            let core_proof_map = core_proof_map.clone();
            async move {
                let mut num_core_proofs = 0;
                while let Some(proof_data) = core_proofs_rx.recv().await {
                    core_proofs_subscriber
                        .subscribe(proof_data.task_id.clone())
                        .map_err(|e| TaskError::Fatal(e.into()))?;
                    let proof =
                        RecursionProof { shard_range: proof_data.range, proof: proof_data.proof };
                    core_proof_map.lock().unwrap().insert(proof_data.task_id, proof);
                    num_core_proofs += 1;
                }
                tracing::info!(
                    "All core proofs received: number of core proofs: {:?}",
                    num_core_proofs
                );
                num_core_proofs_tx.send(num_core_proofs).await.ok();
                Ok(())
            }
            .instrument(tracing::debug_span!("Core proof processing"))
        });

        let mut num_core_proofs_completed = 0;
        let mut num_core_proofs: Option<usize> = None;
        let mut last_core_proof = None;
        loop {
            tokio::select! {
                Some(num_proofs) = num_core_proofs_rx.recv() => {
                    num_core_proofs = Some(num_proofs);
                    // If all core proofs have been completed, set the full range to the max range
                    // and send the last core proof to the proof queue.
                    if num_core_proofs_completed == num_proofs {
                        full_range = Some(max_range.clone().into());
                        // Send the last core proof to the proof queue if it hasn't been sent yet
                        // by the core proof event stream receive task below.
                        if let Some(proof) = last_core_proof.take() {
                            proof_tx.send(proof).map_err(|_| TaskError::Fatal(anyhow::anyhow!("Compress tree panicked")))?;
                        }
                    }
                }
                Some(proof) = proof_rx.recv() => {
                    // Mark that this is a completed task.
                    pending_tasks -= 1;
                    if self.is_complete(&proof.shard_range, pending_tasks, &full_range) {
                        return Ok(());
                    }
                    // Check if there is a neighboring range.
                    if let Some(sibling) = self.sibling(&proof) {
                        let mut proofs = match sibling {
                            Sibling::Left(mut proofs) => {
                                proofs.push_left(proof);
                                proofs
                            }
                            Sibling::Right(mut proofs) => {
                                proofs.push_right(proof);
                                proofs
                            }
                            Sibling::Both(mut proofs, right) => {
                                proofs.push_both(proof, right);
                                proofs
                            }
                        };

                        // Check for proofs to split and put back the remainder.
                        let split = proofs.split_off(self.batch_size);
                        if let Some(split) = split {
                            self.insert(split);
                        }

                        if proofs.len() > self.batch_size {
                            tracing::error!("Proofs are larger than the batch size: {:?}", proofs.len());
                            panic!("Proofs are larger than the batch size: {:?}", proofs.len());
                        }

                        let is_complete = self.is_complete(&proofs.shard_range, pending_tasks, &full_range);
                        if proofs.len() == self.batch_size || is_complete {
                            let shard_range = proofs.shard_range;
                            // Create an artifact for the output proof.
                            let output_artifact = if is_complete { output.clone() } else { artifact_client.create_artifact()? };
                            let task_request = ReduceTaskRequest {
                                range_proofs: proofs,
                                is_complete,
                                output: output_artifact.clone(),
                                context: context.clone(),
                            };
                            let raw_task_request = task_request.into_raw()?;
                            let task_id = worker_client.submit_task(TaskType::RecursionReduce, raw_task_request).await?;
                            // Update the proof map mapping the task id to the proof.
                            proof_map.insert(task_id.clone(), RecursionProof { shard_range, proof: output_artifact });
                            // Subsctibe to the task.
                            subscriber.subscribe(task_id).map_err(|_| TaskError::Fatal(anyhow::anyhow!("Subscriver closed")))?;
                            // Updare the number of pending tasks.
                            pending_tasks += 1;
                        } else {
                            self.insert(proofs);
                        }
                    } else {
                        // If there is no neighboring range, add the proof to the tree.
                        let mut queue = VecDeque::with_capacity(self.batch_size);
                        let range = proof.shard_range;
                        queue.push_back(proof);
                        let proofs = RangeProofs::new(range, queue);
                        self.insert(proofs);
                    }
                }
                Some((task_id, TaskStatus::Succeeded)) = event_stream.recv() => {
                    let proof = proof_map.remove(&task_id);
                    if let Some(proof) = proof {
                        // Send the proof to the proof queue.
                        proof_tx.send(proof).map_err(|_| TaskError::Fatal(anyhow::anyhow!("Compress tree panicked")))?;
                    }
                    else {
                        tracing::debug!("Proof not found for task id: {}", task_id);
                    }
                }

                Some((task_id, status)) = core_proofs_event_stream.recv() => {
                    if status != TaskStatus::Succeeded {
                        return Err(
                            TaskError::Fatal
                            (anyhow::anyhow!("Core proof task {} failed", task_id))
                        );
                    }
                    // Download the proof
                    let normalize_proof = core_proof_map.lock().unwrap().remove(&task_id);
                    if let Some(normalize_proof) = normalize_proof {
                        let shard_range = &normalize_proof.shard_range;
                        let (start, end) = (shard_range.start(), shard_range.end());
                        if start < max_range.start {
                            max_range.start = start;
                        }
                        if end > max_range.end {
                            max_range.end = end;
                        }
                        // Set it as the last core proof and take the previous one.
                        let previous_core_proof = last_core_proof.take();
                        last_core_proof = Some(normalize_proof);
                        // Send the previous core proof to the proof queue, this is safe to do since
                        // we know it's not the last one.
                        if let Some(proof) = previous_core_proof {
                            // Send the proof to the proof queue.
                            proof_tx.send(proof).map_err(|_| TaskError::Fatal(anyhow::anyhow!("Compress tree panicked")))?;
                        }

                        // Mark this as a pending task for the compress tree.
                        pending_tasks += 1;
                        // Increment the number of completed core proofs.
                        num_core_proofs_completed += 1;
                        // If all core proofs have been completed, set the full range to the max
                        // range and send the last core proof to the proof queue.
                        if let Some(num_core_proofs) = num_core_proofs {
                            if num_core_proofs_completed == num_core_proofs {
                                full_range = Some(max_range.clone().into());
                                // Send the last core proof to the proof queue.
                                let last_core_proof = last_core_proof.take().unwrap();
                                proof_tx.send(last_core_proof).map_err(|_| TaskError::Fatal(anyhow::anyhow!("Compress tree panicked")))?;
                                // Close the core proofs event stream.
                                core_proofs_event_stream.close();
                            }
                        }
                    } else {
                        tracing::debug!("Core proof not found for task id: {}", task_id);
                    }
                }
                else => {
                    break;
                }
            }
        }

        Err(TaskError::Fatal(anyhow::anyhow!("todo explain this")))
    }
}
