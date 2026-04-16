//! Per-proof backpressure gate for the `ProveShard` producer path.
//!
//! Restores the backpressure invariant that the local node gets from
//! `ProverSemaphore`, but bounds on the resource that's actually constrained
//! in the distributed pipeline: **artifact-store memory**, not GPU slots.
//!
//! The gate is a thin glue layer between two things that already exist:
//!
//! 1. The artifact store's own [`ShardPermit`] machinery
//!    ([`ArtifactClient::acquire_shard_permit`]) — sized from `maxmemory` on
//!    the specific Redis shard the upload is hashed to. This is what actually
//!    reserves bytes.
//! 2. The worker client's [`TaskSubscriber`] — listens for task completion
//!    (Succeeded / FailedFatal / FailedRetryable), which is when the
//!    artifact-store cleans up the input and the permit can be dropped.
//!
//! Producer flow:
//!
//! ```text
//! let permit = gate.acquire(&record_artifact).await;   // bytes reserved
//! artifact_client.upload(&record_artifact, trace).await?;
//! let task_id = worker_client.submit_task(ProveShard, task).await?;
//! gate.schedule_release(task_id, permit);              // async: drop on completion
//! ```
//!
//! The permit is held for the artifact's full memory lifetime (upload →
//! task-complete), not just the upload call — so reserving bytes in advance
//! stays honest against the actual occupancy pattern in Redis.

use std::sync::Arc;

use sp1_prover_types::{ArtifactClient, ArtifactId, ShardPermit};

use crate::worker::{ProofId, TaskId, TaskSubscriber, WorkerClient};

/// Shared backpressure gate for one proof's ProveShard submissions.
///
/// Cheap to clone — the underlying state is refcounted via `Arc<GateInner>`.
/// Permit machinery lives in the `ArtifactClient`, not here; this type exists
/// only to bundle the per-proof subscriber with the permit-acquisition helper.
/// When the last clone drops, the subscriber's background pump is aborted to
/// avoid leaking a tokio task per proof.
pub struct ProveShardGate<A: ArtifactClient, W: WorkerClient> {
    inner: Arc<GateInner<A, W>>,
}

struct GateInner<A: ArtifactClient, W: WorkerClient> {
    artifact_client: A,
    subscriber: TaskSubscriber<W>,
}

impl<A: ArtifactClient, W: WorkerClient> Drop for GateInner<A, W> {
    fn drop(&mut self) {
        // Abort the subscriber's status-update pump so it doesn't leak past
        // the end of the proof. Safe no-op if already closed.
        self.subscriber.close();
    }
}

impl<A: ArtifactClient, W: WorkerClient> Clone for ProveShardGate<A, W> {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }
}

impl<A: ArtifactClient, W: WorkerClient> ProveShardGate<A, W> {
    /// Build a gate scoped to `proof_id`. Opens a per-proof task subscriber
    /// on `worker_client` so we can observe ProveShard task completions.
    pub async fn new(
        artifact_client: A,
        worker_client: W,
        proof_id: ProofId,
    ) -> anyhow::Result<Self> {
        let subscriber = worker_client.subscriber(proof_id).await?.per_task();
        Ok(Self {
            inner: Arc::new(GateInner { artifact_client, subscriber }),
        })
    }

    /// Reserve a slot for an in-flight shard upload. Delegates to the artifact
    /// store's own permit pool; blocks when the relevant shard node is full.
    pub async fn acquire(&self, artifact: &impl ArtifactId) -> ShardPermit {
        self.inner.artifact_client.acquire_shard_permit(artifact).await
    }

    /// Hand off the permit to a background task that releases it when the
    /// coordinator reports the task reached a terminal state.
    ///
    /// Releases unconditionally on any `wait_task` return (Succeeded /
    /// FailedFatal / FailedRetryable / Err). Rationale: looping on retryable
    /// statuses risks tight-looping if the coordinator's watch channel gets
    /// stuck; we accept a brief overcommit during retries as the safer
    /// tradeoff. On `FailedRetryable` specifically, the original record
    /// artifact stays in Redis until the 4-hour TTL expires — so the
    /// "overcommit" window matches retry-storm latency, not the nominal
    /// retry-and-succeed path. Monitor via metrics if tuning matters.
    pub fn schedule_release(&self, task_id: TaskId, permit: ShardPermit) {
        let subscriber = self.inner.subscriber.clone();
        tokio::spawn(async move {
            let _permit = permit; // held for the lifetime of this task
            if let Err(e) = subscriber.wait_task(task_id.clone()).await {
                tracing::warn!(%task_id, error = %e, "ProveShardGate: wait_task failed, releasing permit");
            }
            // `_permit` drops here, returning the slot to the pool.
        });
    }
}
