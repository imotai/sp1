use std::sync::Arc;

use thiserror::Error;
use tokio::sync::{AcquireError, OwnedSemaphorePermit, Semaphore};

/// A permit for the prover.
#[derive(Debug)]
pub struct ProverPermit {
    /// The underlying permit.
    #[allow(dead_code)]
    permit: OwnedSemaphorePermit,
}

/// A prover permit that is either ready or waiting to be acquired.
pub enum ProverPermits {
    /// The permit is ready to be acquired.
    Ready(ProverPermit),
    /// The permit is waiting to be acquired.
    Pending(ProverSemaphore),
}

/// A semaphore for the prover.
#[derive(Debug, Clone)]
pub struct ProverSemaphore {
    /// The underlying semaphore.
    sem: Arc<Semaphore>,
}

impl ProverSemaphore {
    /// Create a new prover semaphore with the given number of permits.
    #[must_use]
    #[inline]
    pub fn new(max_permits: usize) -> Self {
        Self { sem: Arc::new(Semaphore::new(max_permits)) }
    }

    /// Get a pending prover permit.
    #[must_use]
    #[inline]
    pub fn pending(&self) -> ProverPermits {
        ProverPermits::Pending(self.clone())
    }

    /// Acquire a permit.
    #[inline]
    pub async fn acquire(self) -> Result<ProverPermit, ProverAcquireError> {
        let permit = self.sem.acquire_owned().await?;
        Ok(ProverPermit { permit })
    }
}

/// An error that occurs when acquiring a permit.
#[derive(Debug, Error)]
#[error("failed to acquire permit")]
pub struct ProverAcquireError(#[from] AcquireError);

impl ProverPermits {
    /// Get a pending prover permit.
    #[must_use]
    #[inline]
    pub fn pending(sem: ProverSemaphore) -> Self {
        Self::Pending(sem)
    }

    /// Get a ready prover permit.
    #[must_use]
    #[inline]
    pub fn ready(permit: ProverPermit) -> Self {
        Self::Ready(permit)
    }

    /// Acquire a permit.
    #[inline]
    pub async fn acquire(self) -> Result<ProverPermit, ProverAcquireError> {
        match self {
            Self::Ready(permit) => Ok(permit),
            Self::Pending(sem) => sem.acquire().await,
        }
    }
}
