use std::{
    future::Future,
    marker::PhantomData,
    pin::Pin,
    task::{Context, Poll},
};

use pin_project::pin_project;
use thiserror::Error;

use tokio::sync::oneshot;

pub enum TaskPool {
    Global,
    Local(rayon::ThreadPool),
}

pub fn spawn<F, R>(func: F) -> TaskHandle<'static, R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let (tx, rx) = oneshot::channel();
    rayon::spawn(move || {
        let result = func();
        tx.send(result).ok();
    });
    TaskHandle { rx, _marker: PhantomData }
}

#[pin_project]
pub struct TaskHandle<'a, T> {
    #[pin]
    rx: oneshot::Receiver<T>,
    _marker: PhantomData<&'a T>,
}

#[derive(Error, Debug)]
#[error("TaskJoinError")]
pub struct TaskJoinError(#[from] oneshot::error::RecvError);

impl<'a, T> Future for TaskHandle<'a, T> {
    type Output = Result<T, TaskJoinError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        let rx = this.rx;
        match oneshot::Receiver::poll(rx, cx) {
            Poll::Ready(Ok(result)) => Poll::Ready(Ok(result)),
            Poll::Ready(Err(e)) => Poll::Ready(Err(TaskJoinError(e))),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[derive(Error, Debug)]
#[error("CpuTaskPoolBuilderError: {0}")]
pub struct TaskPoolBuilderError(#[from] rayon::ThreadPoolBuildError);

#[derive(Debug, Default)]
pub struct CpuTaskPoolBuilder(rayon::ThreadPoolBuilder);

impl CpuTaskPoolBuilder {
    pub fn new() -> Self {
        Self(rayon::ThreadPoolBuilder::new())
    }

    pub fn build(self) -> Result<TaskPool, TaskPoolBuilderError> {
        let pool = self.0.build()?;
        Ok(TaskPool::Local(pool))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_spawn() {
        let handle = spawn(|| (0..100).sum::<usize>());
        assert_eq!(handle.await.unwrap(), 4950);
    }
}
