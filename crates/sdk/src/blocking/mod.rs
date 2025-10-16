//! Blocking version of the client.

mod client;
mod cpu;
mod cuda;
mod env;
mod mock;
mod prover;

pub use client::ProverClient;
pub use cpu::{builder::CpuProverBuilder, CpuProver};
pub use cuda::{builder::CudaProverBuilder, CudaProver};
pub use env::EnvProver;
pub use mock::MockProver;
pub use prover::ProveRequest;
pub use prover::Prover;

use std::{future::Future, sync::LazyLock};

/// Block on a future, and return the result.
///
/// Will panic if run within a tokio runtime. It is advised that you switch to the async api
/// if you are already using the tokio runtime directly.
pub(crate) fn block_on<T>(future: impl Future<Output = T>) -> T {
    RUNTIME_HANDLE.block_on(future)
}

/// Runtime handle, used for running async code in a blocking context.
static RUNTIME_HANDLE: LazyLock<tokio::runtime::Handle> = LazyLock::new(|| {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        return handle;
    }
    if let Ok(runtime) = tokio::runtime::Runtime::new() {
        return runtime.handle().clone();
    }
    panic!("failed to get tokio runtime handle")
});
