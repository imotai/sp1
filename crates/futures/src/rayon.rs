use std::any::Any;
use std::sync::OnceLock;
use std::{
    backtrace::Backtrace,
    future::Future,
    marker::PhantomData,
    pin::Pin,
    task::{Context, Poll},
};

use pin_project::pin_project;
use thiserror::Error;

use tokio::sync::oneshot;

static GLOBAL_POOL: OnceLock<()> = OnceLock::new();

fn init_global_pool() {
    rayon::ThreadPoolBuilder::new().panic_handler(panic_handler).build_global().ok();
}

fn panic_handler(panic_payload: Box<dyn Any + Send>) {
    let backtrace = Backtrace::capture();

    if let Some(message) = panic_payload.downcast_ref::<&str>() {
        eprintln!("Rayon thread panic: '{}'", message);
    } else if let Some(message) = panic_payload.downcast_ref::<String>() {
        eprintln!("Rayon thread panic: '{}'", message);
    } else {
        eprintln!("Rayon thread panic with unknown payload");
    }

    eprintln!("Backtrace:\n{:?}", backtrace);

    // TODO: perhaps safer to abort the process
}

pub enum TaskPool {
    Global,
    Local(rayon::ThreadPool),
}

pub fn spawn<F, R>(func: F) -> TaskHandle<'static, R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    GLOBAL_POOL.get_or_init(init_global_pool);
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
    use core::panic;
    use tokio::sync::oneshot;

    use super::*;

    #[tokio::test]
    #[should_panic]
    #[allow(unreachable_code)]
    #[allow(unused_variables)]
    async fn test_spawn() {
        let (tx, rx) = oneshot::channel();
        spawn(move || {
            panic!("test");
            tx.send(()).unwrap();
        });
        rx.await.unwrap();
        // handle.await.unwrap();
    }
}
