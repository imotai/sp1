use crossbeam::queue::ArrayQueue;
use std::{marker::PhantomData, sync::Arc};
use thiserror::Error;
use tokio::sync::{mpsc, OwnedSemaphorePermit, Semaphore, SemaphorePermit};

use crate::{stream::StreamRef, CudaError, CudaEvent, TaskScope};

use super::CudaSend;

#[derive(Debug)]
pub struct Channel<T> {
    events: ArrayQueue<CudaEvent>,
    events_permits: Semaphore,
    _marker: PhantomData<T>,
}

#[derive(Debug, Clone)]
pub struct Sender<'chan, T> {
    chan: &'chan Channel<T>,
    tx: mpsc::Sender<(T, SemaphorePermit<'chan>, CudaEvent)>,
}

#[derive(Debug)]
pub struct Receiver<'chan, T> {
    chan: &'chan Channel<T>,
    rx: mpsc::Receiver<(T, SemaphorePermit<'chan>, CudaEvent)>,
}

#[derive(Debug)]
pub struct OwnedChannel<T> {
    events: ArrayQueue<CudaEvent>,
    events_permits: Arc<Semaphore>,
    _marker: PhantomData<T>,
}

#[derive(Debug, Clone)]
pub struct OwnedSender<T> {
    chan: Arc<OwnedChannel<T>>,
    tx: mpsc::Sender<(T, OwnedSemaphorePermit, CudaEvent)>,
}

#[derive(Debug)]
pub struct OwnedReceiver<T> {
    chan: Arc<OwnedChannel<T>>,
    rx: mpsc::Receiver<(T, OwnedSemaphorePermit, CudaEvent)>,
}

#[derive(Debug, Clone, Error)]
pub enum ChannelError {
    #[error("failed to create CUDA event: {0}")]
    EventCreationFailed(CudaError),

    #[error("failed to push event into channel")]
    PushTaskFailed,
}

impl<T> Channel<T> {
    pub fn new(capacity: usize) -> Result<Self, ChannelError> {
        let events = ArrayQueue::new(capacity);
        let sem = Semaphore::new(capacity);
        for _ in 0..capacity {
            let event = CudaEvent::create().map_err(ChannelError::EventCreationFailed)?;
            events.push(event).map_err(|_| ChannelError::PushTaskFailed)?;
        }
        Ok(Self { events, events_permits: sem, _marker: PhantomData })
    }

    pub fn split(&self) -> (Sender<T>, Receiver<T>) {
        let (tx, rx) = mpsc::channel(self.events.len());
        (Sender { chan: self, tx }, Receiver { chan: self, rx })
    }
}

impl<T> OwnedChannel<T> {
    pub fn new(capacity: usize) -> Result<Self, ChannelError> {
        let events = ArrayQueue::new(capacity);
        let sem = Arc::new(Semaphore::new(capacity));
        for _ in 0..capacity {
            let event = CudaEvent::create().map_err(ChannelError::EventCreationFailed)?;
            events.push(event).map_err(|_| ChannelError::PushTaskFailed)?;
        }
        Ok(Self { events, events_permits: sem, _marker: PhantomData })
    }

    pub fn split(self: Arc<Self>) -> (OwnedSender<T>, OwnedReceiver<T>) {
        let (tx, rx) = mpsc::channel(self.events.len());
        (OwnedSender { chan: self.clone(), tx }, OwnedReceiver { chan: self, rx })
    }
}

#[derive(Debug, Error)]
pub enum SendError<T> {
    #[error("failed to send message: {0}")]
    MessageError(#[from] mpsc::error::SendError<T>),
    #[error("CUDA error: {0}")]
    CudaError(#[from] CudaError),
}

#[derive(Debug, Error)]
pub enum TrySendError<T> {
    #[error("failed to send message: {0}")]
    MessageError(#[from] mpsc::error::TrySendError<T>),
    #[error("failed to create CUDA event: {0}")]
    EventCreationFailed(#[from] CudaError),
}

impl<'chan, T> Sender<'chan, T>
where
    T: CudaSend,
{
    pub async fn send(&self, task: &TaskScope, message: T) -> Result<(), SendError<T>> {
        // Try to reserve a slot in the channel
        let permit = self.tx.reserve().await;
        let sem_permit = self.chan.events_permits.acquire().await;

        if permit.is_err() || sem_permit.is_err() {
            return Err(mpsc::error::SendError(message).into());
        }
        // Try to reserve a slot in the semaphore
        self.send_permit(permit.unwrap(), sem_permit.unwrap(), message, task)?;
        Ok(())
    }

    #[inline]
    fn send_permit(
        &self,
        chan_permit: mpsc::Permit<'_, (T, SemaphorePermit<'chan>, CudaEvent)>,
        sem_permit: SemaphorePermit<'chan>,
        message: T,
        task: &TaskScope,
    ) -> Result<(), CudaError> {
        // Since we have the permits, we can get an event and send the message.
        let event = self.chan.events.pop().unwrap();
        unsafe { task.record_unchecked(&event)? };

        chan_permit.send((message, sem_permit, event));
        Ok(())
    }

    pub fn blocking_send(&self, task: &TaskScope, message: T) -> Result<(), SendError<T>> {
        futures::executor::block_on(self.send(task, message))
    }

    pub fn try_send(&self, task: &TaskScope, message: T) -> Result<(), TrySendError<T>> {
        // Try to reserve a slot in the channel
        let permit = self.tx.try_reserve();

        if permit.is_err() {
            match permit {
                Err(mpsc::error::TrySendError::Full(())) => {
                    return Err(mpsc::error::TrySendError::Full(message).into())
                }
                Err(mpsc::error::TrySendError::Closed(())) => {
                    return Err(mpsc::error::TrySendError::Closed(message).into())
                }
                _ => unreachable!(),
            }
        }

        let sem_permit = self.chan.events_permits.try_acquire();
        if sem_permit.is_err() {
            return Err(mpsc::error::TrySendError::Full(message).into());
        }

        self.send_permit(permit.unwrap(), sem_permit.unwrap(), message, task)?;
        Ok(())
    }
}

impl<T> OwnedSender<T>
where
    T: CudaSend,
{
    pub async fn send(&self, task: &TaskScope, message: T) -> Result<(), SendError<T>> {
        // Try to reserve a slot in the channel
        let permit = self.tx.reserve().await;
        let sem_permit = self.chan.events_permits.clone().acquire_owned().await;

        if permit.is_err() || sem_permit.is_err() {
            return Err(mpsc::error::SendError(message).into());
        }
        // Try to reserve a slot in the semaphore
        self.send_permit(permit.unwrap(), sem_permit.unwrap(), message, task)?;
        Ok(())
    }

    #[inline]
    fn send_permit(
        &self,
        chan_permit: mpsc::Permit<'_, (T, OwnedSemaphorePermit, CudaEvent)>,
        sem_permit: OwnedSemaphorePermit,
        message: T,
        task: &TaskScope,
    ) -> Result<(), CudaError> {
        // Since we have the permits, we can get an event and send the message.
        let event = self.chan.events.pop().unwrap();
        // Safety: we have a task scope, and we know that the event is valid.
        unsafe { task.record_unchecked(&event)? };

        chan_permit.send((message, sem_permit, event));
        Ok(())
    }

    pub fn blocking_send(&self, task: &TaskScope, message: T) -> Result<(), SendError<T>> {
        futures::executor::block_on(self.send(task, message))
    }

    pub fn try_send(&self, task: &TaskScope, message: T) -> Result<(), TrySendError<T>> {
        // Try to reserve a slot in the channel
        let permit = self.tx.try_reserve();

        if permit.is_err() {
            match permit {
                Err(mpsc::error::TrySendError::Full(())) => {
                    return Err(mpsc::error::TrySendError::Full(message).into())
                }
                Err(mpsc::error::TrySendError::Closed(())) => {
                    return Err(mpsc::error::TrySendError::Closed(message).into())
                }
                _ => unreachable!(),
            }
        }

        let sem_permit = self.chan.events_permits.clone().try_acquire_owned();
        if sem_permit.is_err() {
            return Err(mpsc::error::TrySendError::Full(message).into());
        }

        self.send_permit(permit.unwrap(), sem_permit.unwrap(), message, task)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Error)]
pub enum RecvError {
    #[error("Channel closed")]
    Closed,
    #[error("CUDA error: {0}")]
    CudaError(#[from] CudaError),
}

#[derive(Debug, Clone, Copy, Error)]
pub enum TryRecvError {
    #[error("failed to receive message: {0}")]
    MessageError(#[from] mpsc::error::TryRecvError),
    #[error("CUDA error: {0}")]
    CudaError(#[from] CudaError),
}

impl<'chan, T: CudaSend> Receiver<'chan, T> {
    #[inline]
    fn process_message(
        &self,
        message: T,
        event: CudaEvent,
        sem_permit: SemaphorePermit<'chan>,
        scope: &TaskScope,
    ) -> Result<T, CudaError> {
        // Wait for the event to be completed and transfer the message to the new stream
        // Safety: we know that the event is valid and the stream is valid, and the transfer
        // is safe.
        let message = unsafe {
            scope.stream.wait_unchecked(&event)?;
            message.send_to_scope(scope)
        };
        // Return the event to the queue
        self.chan.events.push(event).unwrap();
        // release the semaphore permit
        drop(sem_permit);
        Ok(message)
    }

    pub async fn recv(&mut self, scope: &TaskScope) -> Result<T, RecvError> {
        let (message, sem_permit, event) = self.rx.recv().await.ok_or(RecvError::Closed)?;
        // TODO: this doesn't work because now the channel think there is a permit available.
        let message = self.process_message(message, event, sem_permit, scope)?;
        Ok(message)
    }

    pub fn blocking_recv(&mut self, scope: &TaskScope) -> Result<T, RecvError> {
        futures::executor::block_on(self.recv(scope))
    }

    pub fn try_recv(&mut self, scope: &TaskScope) -> Result<T, TryRecvError> {
        let (message, sem_permit, event) =
            self.rx.try_recv().map_err(TryRecvError::MessageError)?;
        let message = self.process_message(message, event, sem_permit, scope)?;
        Ok(message)
    }
}

impl<T: CudaSend> OwnedReceiver<T> {
    #[inline]
    fn process_message(
        &self,
        message: T,
        event: CudaEvent,
        sem_permit: OwnedSemaphorePermit,
        scope: &TaskScope,
    ) -> Result<T, CudaError> {
        // Wait for the event to be completed and transfer the message to the new stream
        // Safety: we know that the event is valid and the stream is valid, and the transfer
        // is safe.
        let message = unsafe {
            scope.stream.wait_unchecked(&event)?;
            message.send_to_scope(scope)
        };
        // Return the event to the queue
        self.chan.events.push(event).unwrap();
        // release the semaphore permit
        drop(sem_permit);
        Ok(message)
    }

    pub async fn recv(&mut self, scope: &TaskScope) -> Result<T, RecvError> {
        let (message, sem_permit, event) = self.rx.recv().await.ok_or(RecvError::Closed)?;
        // TODO: this doesn't work because now the channel think there is a permit available.
        let message = self.process_message(message, event, sem_permit, scope)?;
        Ok(message)
    }

    pub fn try_recv(&mut self, scope: &TaskScope) -> Result<T, TryRecvError> {
        let (message, sem_permit, event) =
            self.rx.try_recv().map_err(TryRecvError::MessageError)?;
        let message = self.process_message(message, event, sem_permit, scope)?;
        Ok(message)
    }

    pub fn blocking_recv(&mut self, scope: &TaskScope) -> Result<T, RecvError> {
        futures::executor::block_on(self.recv(scope))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use slop_alloc::{Buffer, IntoHost};

    use crate::TaskPoolBuilder;

    use super::*;

    #[tokio::test]
    async fn test_channel() {
        let pool = TaskPoolBuilder::new().num_tasks(10).build().unwrap();
        let task_1 = pool.task().await.unwrap();
        let task_2 = pool.task().await.unwrap();
        let chan = Channel::<Buffer<u8, TaskScope>>::new(10).unwrap();
        let (tx, mut rx) = chan.split();

        let (handle_1, handle_2) = tokio::join!(
            task_1.run(|t| async move {
                let mut buf = t.alloc::<u8>(1000000);
                t.sleep(Duration::from_secs(2));
                buf.write_bytes(1, buf.capacity()).unwrap();
                tx.send(&t, buf).await.unwrap();
            }),
            task_2.run(|t| async move {
                let buf = rx.recv(&t).await.unwrap();
                assert_eq!(buf.into_host().await.unwrap().into_vec(), vec![1; 1000000]);
            }),
        );

        handle_1.await.unwrap();
        handle_2.await.unwrap();
    }

    #[tokio::test]
    async fn test_owned_channel() {
        let task_1 = crate::task().await.unwrap();
        let task_2 = crate::task().await.unwrap();
        let chan = Arc::new(OwnedChannel::<Buffer<u8, TaskScope>>::new(10).unwrap());
        let (tx, mut rx) = chan.split();

        let handle_2 = tokio::spawn(task_2.run(|t| async move {
            let buf = rx.recv(&t).await.unwrap();
            assert_eq!(buf.into_host().await.unwrap().into_vec(), vec![1; 1000000]);
        }));

        let handle_1 = tokio::spawn(task_1.run(|t| async move {
            let mut buf = t.alloc::<u8>(1000000);
            t.sleep(Duration::from_secs(2));
            buf.write_bytes(1, buf.capacity()).unwrap();
            tx.send(&t, buf).await.unwrap();
        }));

        handle_1.await.unwrap();
        handle_2.await.unwrap();
    }
}
