use std::{
    collections::{BTreeMap, VecDeque},
    ops::Range,
    sync::Arc,
};

use slop_futures::queue::WorkerQueue;
use sp1_core_executor::SP1RecursionProof;
use sp1_primitives::SP1GlobalContext;
use sp1_recursion_circuit::machine::SP1ShapedWitnessValues;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::{error::SP1ProverError, InnerSC, SP1CircuitWitness};

#[derive(Debug, Clone)]
pub struct RecursionProof {
    pub shard_range: Range<usize>,
    pub proof: SP1RecursionProof<SP1GlobalContext, InnerSC>,
}

impl RecursionProof {
    fn is_complete(&self, full_range: &Range<usize>) -> bool {
        &self.shard_range == full_range
    }
}

pub struct ExecuteTask {
    pub range: Range<usize>,
    pub input: SP1CircuitWitness,
}

/// An enum marking which sibling was found.
enum Sibling {
    Left(RangeProofs),
    Right(RangeProofs),
    Both(RangeProofs, RangeProofs),
}

#[derive(Clone, Debug)]
struct RangeProofs {
    shard_range: Range<usize>,
    proofs: VecDeque<SP1RecursionProof<SP1GlobalContext, InnerSC>>,
}

impl RangeProofs {
    pub fn new(
        shard_range: Range<usize>,
        proofs: VecDeque<SP1RecursionProof<SP1GlobalContext, InnerSC>>,
    ) -> Self {
        Self { shard_range, proofs }
    }

    pub fn push_right(&mut self, proof: RecursionProof) {
        assert_eq!(proof.shard_range.end, self.shard_range.start);
        self.shard_range = proof.shard_range.start..self.shard_range.end;
        self.proofs.push_front(proof.proof);
    }

    pub fn push_left(&mut self, proof: RecursionProof) {
        assert_eq!(proof.shard_range.start, self.shard_range.end);
        self.shard_range = self.shard_range.start..proof.shard_range.end;
        self.proofs.push_back(proof.proof);
    }

    pub fn split_off(&mut self, at: usize) -> Option<Self> {
        if at >= self.proofs.len() {
            return None;
        }
        let proofs = self.proofs.split_off(at);
        let end_point = std::cmp::min(self.shard_range.end, self.shard_range.start + at);
        let split_range = end_point..self.shard_range.end;
        self.shard_range = self.shard_range.start..end_point;
        Some(Self { shard_range: split_range, proofs })
    }

    pub fn push_both(&mut self, middle: RecursionProof, right: Self) {
        assert_eq!(middle.shard_range.start, self.shard_range.end);
        assert_eq!(right.shard_range.start, middle.shard_range.end);
        // Push the middle to the queue.
        self.proofs.push_back(middle.proof);
        // Append the right proofs to the queue.
        for proof in right.proofs {
            self.proofs.push_back(proof);
        }
        // Update the shard range.
        self.shard_range = self.shard_range.start..right.shard_range.end;
    }

    pub fn len(&self) -> usize {
        self.proofs.len()
    }

    pub fn is_complete(&self, full_range: &Range<usize>) -> bool {
        &self.shard_range == full_range
    }

    pub fn into_witness(
        self,
        full_range: &Range<usize>,
    ) -> (Range<usize>, SP1ShapedWitnessValues<SP1GlobalContext, InnerSC>) {
        let is_complete = self.is_complete(full_range);
        let vks_and_proofs =
            self.proofs.into_iter().map(|proof| (proof.vk, proof.proof)).collect::<Vec<_>>();
        (self.shard_range, SP1ShapedWitnessValues { vks_and_proofs, is_complete })
    }
}

pub struct CompressTree {
    map: BTreeMap<usize, RangeProofs>,
    batch_size: usize,
}

impl CompressTree {
    /// Create a new recursion tree.
    pub fn new(batch_size: usize) -> Self {
        Self { map: BTreeMap::new(), batch_size }
    }

    /// Insert a new range of proofs into the tree.
    fn insert(&mut self, proofs: RangeProofs) {
        self.map.insert(proofs.shard_range.start, proofs);
    }

    /// Get the sibling of a proof.
    fn sibling(&mut self, proof: &RecursionProof) -> Option<Sibling> {
        // Check for a left sibling
        if let Some(previous) = self.map.range(0..proof.shard_range.start).next_back() {
            let (start, proofs) = previous;
            let start = *start;
            let proofs = proofs.clone();

            if proofs.shard_range.end == proof.shard_range.start {
                let left = self.map.remove(&start).unwrap();
                // Check for a right sibling.
                if let Some(right) = self.map.remove(&proof.shard_range.end) {
                    return Some(Sibling::Both(left, right));
                } else {
                    return Some(Sibling::Left(left));
                }
            }
        }
        // If there is no left sibling, check for a right sibling.
        if let Some(right) = self.map.remove(&proof.shard_range.end) {
            return Some(Sibling::Right(right));
        }

        // No sibling found.
        None
    }

    pub async fn reduce_proofs(
        &mut self,
        full_range: &Range<usize>,
        proofs_rx: &mut UnboundedReceiver<RecursionProof>,
        recursion_executors: Arc<WorkerQueue<UnboundedSender<ExecuteTask>>>,
    ) -> Result<Vec<SP1RecursionProof<SP1GlobalContext, InnerSC>>, SP1ProverError> {
        // Populate the recursion proofs into the tree until we reach the reduce batch size.
        while let Some(proof) = proofs_rx.recv().await {
            if proof.is_complete(full_range) {
                return Ok(vec![proof.proof]);
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

                // Check for proofs to split and put back the reminder.
                let split = proofs.split_off(self.batch_size);
                if let Some(split) = split {
                    self.insert(split);
                }

                if proofs.len() > self.batch_size {
                    tracing::error!("Proofs are larger than the batch size: {:?}", proofs.len());
                    panic!("Proofs are larger than the batch size: {:?}", proofs.len());
                }

                if proofs.len() == self.batch_size || proofs.is_complete(full_range) {
                    // Compress all the proofs into a single proof.
                    let (range, input) = proofs.into_witness(full_range);
                    let input = SP1CircuitWitness::Compress(input);
                    // Wait for an executor to be available.
                    let executor = recursion_executors.clone().pop().await.unwrap();
                    executor.send(ExecuteTask { input, range }).ok();
                } else {
                    self.insert(proofs);
                }
            } else {
                // If there is no neighboring range, add the proof to the tree.
                let RecursionProof { shard_range, proof } = proof;
                let mut queue = VecDeque::with_capacity(self.batch_size);
                queue.push_back(proof);
                let proofs = RangeProofs::new(shard_range, queue);
                self.insert(proofs);
            }
        }

        unreachable!("todo explain this")
    }
}
