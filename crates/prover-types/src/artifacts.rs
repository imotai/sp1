use core::fmt;
use std::{future::Future, sync::Arc};

use anyhow::{anyhow, Result};
use futures_util::future::FutureExt;
use hashbrown::{HashMap, HashSet};
use mti::prelude::{MagicTypeIdExt, V7};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tokio::sync::{Mutex, OwnedSemaphorePermit, RwLock};
use tracing::Instrument;

use crate::utils::{await_blocking, await_scoped_vec};

/// Opaque permit returned by [`ArtifactClient::acquire_shard_permit`].
///
/// Represents a reservation of space in the underlying artifact store for
/// one in-flight shard. The caller must hold the permit from the moment
/// `upload()` is called until the downstream consumer (GPU worker processing
/// `ProveShard`) finishes — at which point the store drops the input artifact
/// and the permit can be released.
///
/// Dropping the permit releases its reservation. For stores that don't have a
/// memory ceiling (S3, in-memory), [`ShardPermit::noop`] is used as a zero-cost
/// no-op that makes the producer-side API uniform across backends.
pub struct ShardPermit {
    // `None` means "no gating in this store"; dropping it is a no-op.
    // The guard is load-bearing even though it's never read directly —
    // dropping the `ShardPermit` releases the underlying semaphore slot.
    guard: Option<OwnedSemaphorePermit>,
}

impl ShardPermit {
    /// Construct a no-op permit. Used by stores without memory constraints
    /// (S3, in-memory) so producers can call `acquire_shard_permit` uniformly
    /// without runtime cost.
    pub const fn noop() -> Self {
        Self { guard: None }
    }

    /// Wrap a real semaphore permit from a memory-bounded store (Redis).
    pub const fn new(guard: OwnedSemaphorePermit) -> Self {
        Self { guard: Some(guard) }
    }
}

impl fmt::Debug for ShardPermit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.guard {
            Some(_) => write!(f, "ShardPermit(held)"),
            None => write!(f, "ShardPermit(noop)"),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ArtifactType {
    UnspecifiedArtifactType,
    Program,
    Stdin,
    Proof,
    Groth16Circuit,
    PlonkCircuit,
}

impl fmt::Display for ArtifactType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnspecifiedArtifactType => write!(f, "UnspecifiedArtifactType"),
            Self::Program => write!(f, "Program"),
            Self::Stdin => write!(f, "Stdin"),
            Self::Proof => write!(f, "Proof"),
            Self::Groth16Circuit => write!(f, "Groth16Circuit"),
            Self::PlonkCircuit => write!(f, "PlonkCircuit"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct Artifact(pub String);

impl Artifact {
    pub fn to_id(self) -> String {
        self.0
    }
}

impl From<String> for Artifact {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl fmt::Display for Artifact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Artifact({})", self.0)
    }
}

pub trait ArtifactId: Send + Sync {
    fn id(&self) -> &str;
}

impl ArtifactId for Artifact {
    fn id(&self) -> &str {
        &self.0
    }
}

impl ArtifactId for String {
    fn id(&self) -> &str {
        self
    }
}

pub struct ArtifactBatch(Vec<Artifact>);

impl ArtifactBatch {
    pub async fn upload<T: Serialize + Send + Sync>(
        &self,
        client: &impl ArtifactClient,
        items: &[T],
    ) -> Result<()> {
        await_scoped_vec(self.0.iter().zip(items).map(|(artifact, item)| {
            let client = client.clone();
            let artifact = artifact.clone();
            async move { client.upload(&artifact, item).await }
        }))
        .await?;
        Ok(())
    }

    pub async fn download<T: DeserializeOwned + Send + Sync + 'static>(
        &self,
        client: &impl ArtifactClient,
    ) -> Result<Vec<T>> {
        let result = await_scoped_vec(self.0.iter().map(|artifact| {
            let client = client.clone();
            let artifact = artifact.clone();
            async move { client.download::<T>(&artifact).await }
        }))
        .await?
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
        Ok(result)
    }

    pub fn to_vec(self) -> Vec<Artifact> {
        self.0
    }
}

impl From<ArtifactBatch> for Vec<Artifact> {
    fn from(val: ArtifactBatch) -> Self {
        val.0
    }
}

impl From<Vec<Artifact>> for ArtifactBatch {
    fn from(value: Vec<Artifact>) -> Self {
        Self(value)
    }
}

pub trait ArtifactClient: Send + Sync + Clone + 'static {
    fn upload_raw(
        &self,
        artifact: &impl ArtifactId,
        artifact_type: ArtifactType,
        data: Vec<u8>,
    ) -> impl Future<Output = Result<()>> + Send;

    fn download_raw(
        &self,
        artifact: &impl ArtifactId,
        artifact_type: ArtifactType,
    ) -> impl Future<Output = Result<Vec<u8>>> + Send;

    fn exists(
        &self,
        artifact: &impl ArtifactId,
        artifact_type: ArtifactType,
    ) -> impl Future<Output = Result<bool>> + Send;

    fn delete(
        &self,
        artifact: &impl ArtifactId,
        artifact_type: ArtifactType,
    ) -> impl Future<Output = Result<()>> + Send;

    fn delete_batch(
        &self,
        artifacts: &[impl ArtifactId],
        artifact_type: ArtifactType,
    ) -> impl Future<Output = Result<()>> + Send;

    /// Acquire a permit to reserve space for an in-flight shard artifact.
    ///
    /// Stores with a memory ceiling (Redis) return a permit tied to a real
    /// semaphore sized from their `maxmemory`; the permit must be held until
    /// the downstream consumer deletes the artifact, at which point dropping
    /// it releases the reservation.
    ///
    /// The default implementation is a no-op [`ShardPermit::noop`], suitable
    /// for stores with no single-host memory ceiling (S3, in-memory). This
    /// lets the producer call `acquire_shard_permit` uniformly without paying
    /// any cost on S3-backed clusters.
    fn acquire_shard_permit(
        &self,
        _artifact: &impl ArtifactId,
    ) -> impl Future<Output = ShardPermit> + Send {
        async { ShardPermit::noop() }
    }

    fn try_delete(
        &self,
        artifact: &impl ArtifactId,
        artifact_type: ArtifactType,
    ) -> impl Future<Output = Result<()>> + Send {
        async move {
            if let Err(e) = self.delete(artifact, artifact_type).await {
                tracing::warn!("Failed to delete artifact {}: {:?}", artifact.id(), e);
            }
            Ok(())
        }
    }

    fn try_delete_batch(
        &self,
        artifacts: &[impl ArtifactId],
        artifact_type: ArtifactType,
    ) -> impl Future<Output = Result<()>> + Send {
        async move {
            if let Err(e) = self.delete_batch(artifacts, artifact_type).await {
                tracing::warn!("Failed to delete artifact batch: {:?}", e);
            }
            Ok(())
        }
    }

    fn upload_with_type<T: Serialize + Send + Sync>(
        &self,
        artifact: &impl ArtifactId,
        artifact_type: ArtifactType,
        item: T,
    ) -> impl Future<Output = Result<()>> + Send {
        async move {
            let data = await_blocking(move || {
                let data = bincode::serialize(&item);
                drop(item);
                data
            })
            .instrument(tracing::trace_span!("serialize"))
            .await
            .unwrap()?;
            self.upload_raw(artifact, artifact_type, data).await
        }
    }

    fn download_with_type<T: DeserializeOwned + Send + Sync + 'static>(
        &self,
        artifact: &impl ArtifactId,
        artifact_type: ArtifactType,
    ) -> impl Future<Output = Result<T>> + Send {
        async move {
            let bytes = self.download_raw(artifact, artifact_type).await?;
            let deserialized =
                tokio::task::spawn_blocking(move || bincode::deserialize(&bytes)).await??;
            Ok(deserialized)
        }
    }

    // TODO: this should not be a result.
    fn create_artifact(&self) -> Result<Artifact> {
        Ok("artifact".create_type_id::<V7>().to_string().into())
    }

    fn create_artifacts(&self, count: usize) -> Result<ArtifactBatch> {
        Ok((0..count)
            .map(|_| "artifact".create_type_id::<V7>().to_string().into())
            .collect::<Vec<_>>()
            .into())
    }

    fn upload<T: Serialize + Send + Sync>(
        &self,
        artifact: &impl ArtifactId,
        item: T,
    ) -> impl Future<Output = Result<()>> + Send {
        self.upload_with_type(artifact, ArtifactType::UnspecifiedArtifactType, item)
    }

    fn upload_proof<T: Serialize + Send + Sync>(
        &self,
        artifact: &impl ArtifactId,
        item: T,
    ) -> impl Future<Output = Result<()>> + Send {
        self.upload_with_type(artifact, ArtifactType::Proof, item)
    }

    fn upload_program(
        &self,
        artifact: &impl ArtifactId,
        item: Vec<u8>,
    ) -> impl Future<Output = Result<()>> + Send {
        self.upload_with_type(artifact, ArtifactType::Program, item)
    }

    fn download<T: DeserializeOwned + Send + Sync + 'static>(
        &self,
        artifact: &impl ArtifactId,
    ) -> impl Future<Output = Result<T>> + Send {
        self.download_with_type(artifact, ArtifactType::UnspecifiedArtifactType)
    }

    fn download_program(
        &self,
        artifact: &impl ArtifactId,
    ) -> impl Future<Output = Result<Vec<u8>>> + Send {
        self.download_with_type(artifact, ArtifactType::Program)
    }

    fn download_stdin<T: DeserializeOwned + Send + Sync + 'static>(
        &self,
        artifact: &impl ArtifactId,
    ) -> impl Future<Output = Result<T>> + Send {
        self.download_with_type(artifact, ArtifactType::Stdin)
    }

    fn download_stdin_bytes(
        &self,
        artifact: &impl ArtifactId,
    ) -> impl Future<Output = Result<Vec<u8>>> + Send {
        self.download_with_type(artifact, ArtifactType::Stdin)
    }

    /// Add task reference for an artifact
    fn add_ref(
        &self,
        _artifact: &impl ArtifactId,
        _task_id: &str,
    ) -> impl Future<Output = Result<()>> + Send {
        // Default implementation does nothing (for non-Redis clients)
        async { Ok(()) }.boxed()
    }

    /// Remove task reference and delete artifact if no references remain
    fn remove_ref(
        &self,
        _artifact: &impl ArtifactId,
        _artifact_type: ArtifactType,
        _task_id: &str,
    ) -> impl Future<Output = Result<bool>> + Send {
        async { Ok(false) }.boxed()
    }
}

#[derive(Clone)]
pub struct InMemoryArtifactClient {
    artifacts: Arc<RwLock<HashMap<String, Vec<u8>>>>,
    refs: Arc<Mutex<HashMap<String, HashSet<String>>>>,
}

impl fmt::Debug for InMemoryArtifactClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "InMemoryArtifactClient")
    }
}

impl InMemoryArtifactClient {
    pub fn new() -> Self {
        Self {
            artifacts: Arc::new(RwLock::new(HashMap::new())),
            refs: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryArtifactClient {
    fn default() -> Self {
        Self::new()
    }
}

impl ArtifactClient for InMemoryArtifactClient {
    async fn upload_raw(
        &self,
        artifact: &impl ArtifactId,
        _artifact_type: ArtifactType,
        data: Vec<u8>,
    ) -> Result<()> {
        let mut artifacts = self.artifacts.write().await;
        artifacts.insert(artifact.id().to_string(), data.clone());
        Ok(())
    }

    async fn download_raw(
        &self,
        artifact: &impl ArtifactId,
        _artifact_type: ArtifactType,
    ) -> Result<Vec<u8>> {
        let artifacts = self.artifacts.read().await;
        let bytes = artifacts.get(artifact.id()).ok_or_else(|| anyhow!("artifact not found"))?;
        Ok(bytes.clone())
    }

    async fn exists(
        &self,
        artifact: &impl ArtifactId,
        _artifact_type: ArtifactType,
    ) -> Result<bool> {
        let artifacts = self.artifacts.read().await;
        Ok(artifacts.contains_key(artifact.id()))
    }

    async fn delete(&self, artifact: &impl ArtifactId, _artifact_type: ArtifactType) -> Result<()> {
        let mut artifacts = self.artifacts.write().await;
        artifacts.remove(artifact.id());
        Ok(())
    }

    async fn delete_batch(
        &self,
        artifacts: &[impl ArtifactId],
        _artifact_type: ArtifactType,
    ) -> Result<()> {
        let mut artifact_map = self.artifacts.write().await;
        for artifact in artifacts {
            artifact_map.remove(artifact.id());
        }
        Ok(())
    }

    async fn add_ref(&self, artifact: &impl ArtifactId, task_id: &str) -> Result<()> {
        self.refs
            .lock()
            .await
            .entry(artifact.id().to_string())
            .or_default()
            .insert(task_id.to_string());
        Ok(())
    }

    async fn remove_ref(
        &self,
        artifact: &impl ArtifactId,
        artifact_type: ArtifactType,
        task_id: &str,
    ) -> Result<bool> {
        let mut ref_count = 0;
        self.refs.lock().await.entry(artifact.id().to_string()).and_modify(|refs| {
            refs.remove(task_id);
            ref_count = refs.len();
        });
        if ref_count == 0 {
            self.refs.lock().await.remove(artifact.id());
            self.delete(artifact, artifact_type).await?;
            return Ok(true);
        }
        Ok(false)
    }
}
