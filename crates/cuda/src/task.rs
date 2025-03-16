use std::{
    alloc::Layout,
    ffi::c_void,
    future::{Future, IntoFuture},
    mem::MaybeUninit,
    ops::Deref,
    pin::Pin,
    ptr::{self, NonNull},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, OnceLock, Weak,
    },
    task::{Context, Poll},
    time::Duration,
};

use csl_sys::runtime::{
    cuda_device_get_mem_pool, cuda_mem_pool_set_release_threshold, CudaDevice, CudaMemPool,
    CudaStreamHandle, Dim3, KernelPtr,
};
use pin_project::pin_project;
use slop_alloc::{
    mem::{CopyDirection, CopyError, DeviceMemory},
    AllocError, Allocator, Backend, Buffer, Slice,
};
use slop_futures::queue::{AcquireWorkerError, TryAcquireWorkerError, Worker, WorkerQueue};
use thiserror::Error;
use tokio::{sync::oneshot, task::JoinHandle};

use crate::{DeviceCopy, ToDevice};

use super::{
    stream::{StreamRef, INTERVAL_MS},
    sync::CudaSend,
    CudaError, CudaEvent, CudaStream, IntoDevice, StreamCallbackFuture,
};

const DEFAULT_NUM_TASKS: usize = 64;

static GLOBAL_TASK_POOL: OnceLock<Arc<TaskPool>> = OnceLock::new();

static POOL_ID: AtomicUsize = AtomicUsize::new(0);

pub struct TaskPoolBuilder {
    device: CudaDevice,
    mem_release_threshold: u64,
    capacity: Option<usize>,
}

pub(crate) fn global_task_pool() -> &'static Arc<TaskPool> {
    GLOBAL_TASK_POOL.get_or_init(|| Arc::new(TaskPoolBuilder::new().build().unwrap()))
}

/// Acquires a task from the global task pool.
pub async fn task() -> Result<OwnedTask, SpawnError> {
    let pool = global_task_pool();
    let task = pool.task().await?;
    Ok(task)
}

pub fn spawn<F, Fut>(f: F) -> JoinHandle<TaskHandle<Fut::Output>>
where
    F: FnOnce(TaskScope) -> Fut + Send + 'static,
    Fut: Future + Send + 'static,
    Fut::Output: Send + 'static,
{
    tokio::spawn(async move {
        let task = task().await.unwrap();
        task.run(f).await
    })
}

/// Attempts to get a task from the global task pool.
///
/// This function will not block, and will return an error if no task is currently available.
pub fn try_get_task() -> Result<OwnedTask, TrySpawnError> {
    let pool = global_task_pool();
    pool.try_get_task()
}

#[derive(Debug, Clone, Error)]
pub enum TaskPoolBuildError {
    #[error("failed to create CUDA stream: {0}")]
    StreamCreationFailed(CudaError),

    #[error("failed to create CUDA event: {0}")]
    EventCreationFailed(CudaError),

    #[error("failed to push task back into pool")]
    PushTaskFailed,
}

#[derive(Debug, Clone, Error)]
pub enum GlobalTaskPoolBuildError {
    #[error("failed to build global task pool")]
    BuildFailed(#[from] TaskPoolBuildError),
    #[error("global task pool already initialized")]
    AlreadyInitialized,
}

impl TaskPoolBuilder {
    pub fn new() -> Self {
        Self { capacity: None, device: CudaDevice(0), mem_release_threshold: u64::MAX }
    }

    pub fn num_tasks(mut self, num_tasks: usize) -> Self {
        self.capacity = Some(num_tasks);
        self
    }

    pub fn device(mut self, device: CudaDevice) -> Self {
        assert!(device.0 == 0, "only device 0 is supported at the moment");
        self.device = device;
        self
    }

    /// Sets the memory release threshold for the associated device.
    ///
    /// # Warning
    /// This setting will affect the memory release threshold for the entire device, not just the
    /// current task pool being built.
    pub fn mem_release_threshold(mut self, threshold: u64) -> Self {
        self.mem_release_threshold = threshold;
        self
    }

    fn allocate_new_id(&self) -> usize {
        let id = POOL_ID.fetch_add(1, Ordering::Relaxed);
        if id > usize::MAX / 2 {
            std::process::abort();
        }
        id
    }

    pub fn build(self) -> Result<TaskPool, TaskPoolBuildError> {
        let id = self.allocate_new_id();
        let num_tasks = self.capacity.unwrap_or(DEFAULT_NUM_TASKS);

        // Set the memory release threshold
        unsafe {
            let mut mem_pool = CudaMemPool(ptr::null_mut());
            CudaError::result_from_ffi(cuda_device_get_mem_pool(&mut mem_pool, self.device))
                .unwrap();
            CudaError::result_from_ffi(cuda_mem_pool_set_release_threshold(
                mem_pool,
                self.mem_release_threshold,
            ))
            .unwrap();
        };

        let mut tasks = Vec::with_capacity(num_tasks);
        for (i, _) in (0..num_tasks).enumerate() {
            let stream = CudaStream::create().map_err(TaskPoolBuildError::StreamCreationFailed)?;
            let end_event = CudaEvent::create().map_err(TaskPoolBuildError::EventCreationFailed)?;
            tasks.push(Task { owner_id: id, id: i, stream, end_event });
        }
        let inner = Arc::new(WorkerQueue::new(tasks));

        Ok(TaskPool { inner })
    }

    pub fn build_global(self) -> Result<(), GlobalTaskPoolBuildError> {
        let pool = self.build()?;
        GLOBAL_TASK_POOL
            .set(Arc::new(pool))
            .map_err(|_| GlobalTaskPoolBuildError::AlreadyInitialized)
    }
}

impl Default for TaskPoolBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct TaskPool {
    inner: Arc<WorkerQueue<Task>>,
}

pub struct OwnedTask {
    inner: Worker<Task>,
}

impl std::fmt::Debug for OwnedTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OwnedTask {{ inner: {:?} }}", self.inner.deref())
    }
}

#[derive(Debug, Error)]
#[error("failed to acquire a task from the pool")]
pub enum SpawnError {
    AcquireError(#[from] AcquireWorkerError),
}

#[derive(Debug, Error)]
#[error("failed to acquire a task from the pool")]
pub enum TrySpawnError {
    TryAcquireError(#[from] TryAcquireWorkerError),
}

impl TaskPool {
    /// Get a task from the task pool.
    pub async fn task(&self) -> Result<OwnedTask, SpawnError> {
        let worker = self.inner.clone().pop().await.map_err(SpawnError::AcquireError)?;
        Ok(OwnedTask { inner: worker })
    }

    /// Get a task from the task pool.
    ///
    /// This function will block the current thread until a task becomes available.
    pub fn try_get_task(&self) -> Result<OwnedTask, TrySpawnError> {
        let task = self.inner.clone().try_pop().map_err(TrySpawnError::TryAcquireError)?;
        Ok(OwnedTask { inner: task })
    }
}

#[derive(Debug)]
pub struct TaskScope(Weak<OwnedTask>);

impl Clone for TaskScope {
    fn clone(&self) -> Self {
        TaskScope(self.0.clone())
    }
}

impl Deref for TaskScope {
    type Target = Task;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &(*self.0.as_ptr()).inner }
    }
}

unsafe impl Backend for TaskScope {}

unsafe extern "C" fn sleep(ptr: *mut c_void) {
    let time = unsafe { Box::from_raw(ptr as *mut Duration) };
    std::thread::sleep(*time);
}

unsafe extern "C" fn sync_host(ptr: *mut c_void) {
    let tx = unsafe { Box::from_raw(ptr as *mut oneshot::Sender<bool>) };
    tx.send(true).unwrap();
}

impl TaskScope {
    /// Allocates a buffer in this scope on the device.
    ///
    /// This call is not blocking. Upon successful completion, it will return a buffer with a memory
    /// that is guaranteed to be available in the scope of the task but without any absolute
    /// guarantee relative to the host or any other task.
    ///
    /// Other tasks may try to allocate memory concurrently. In order to guarantee enough memory
    /// for all expected work, the user must ensure some limit on task calls by e.g. using a
    /// semaphore.
    #[inline]
    pub fn alloc<T>(&self, capacity: usize) -> Buffer<T, Self> {
        Buffer::with_capacity_in(capacity, self.clone())
    }

    /// Tries to allocate a buffer in this scope on the device.
    #[inline]
    pub fn try_alloc<T>(
        &self,
        capacity: usize,
    ) -> Result<Buffer<T, Self>, slop_alloc::TryReserveError> {
        Buffer::try_with_capacity_in(capacity, self.clone())
    }

    /// Launches a host function in this task.
    ///
    /// # Safety
    ///
    /// The function essentially executes an extern call in `C`. The safety assumption of an extern.  
    /// The user must ensure the pointer is valid and that the data remains valid as this call will
    /// be asynchronous.
    #[inline]
    pub unsafe fn launch_host_fn(
        &self,
        host_fn: unsafe extern "C" fn(*mut c_void),
        data: *mut c_void,
    ) -> Result<(), CudaError> {
        self.launch_host_fn_uncheked(Some(host_fn), data)
    }

    /// Launches a kernel in this task.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - The kernel ptr is valid.
    /// - The arguments are passed correctly across the FFI interface.
    /// - The data lives whitin the scope of the current task.
    pub unsafe fn launch_kernel(
        &self,
        kernel: KernelPtr,
        grid_dim: impl Into<Dim3>,
        block_dim: impl Into<Dim3>,
        args: &[*mut c_void],
        shared_mem: usize,
    ) -> Result<(), CudaError> {
        self.stream().launch_kernel(kernel, grid_dim, block_dim, args, shared_mem)
    }

    /// Sends the CUDA task to sleep for **at least** the given duration.
    ///
    /// This function will not block the calling host thread. The function does a small allocation
    /// so the sleep time might be slightly longer than the given duration for very short times.
    pub fn sleep(&self, time: Duration) {
        let time_ptr = Box::into_raw(Box::new(time));
        unsafe {
            self.launch_host_fn(sleep, time_ptr as *mut c_void).unwrap();
        }
    }

    /// Copies data between slices using CudaMemCpyAsync
    ///
    /// # Safety
    /// The caller must ensure that the data is valid and that the data remains valid as this call
    pub unsafe fn copy<T: DeviceCopy>(
        &self,
        dst: &mut Slice<T, Self>,
        src: &Slice<T, Self>,
    ) -> Result<(), CopyError> {
        dst.copy_from_slice(src, self)
    }

    /// Waits for all work enqueued so far in this task to finish.
    ///
    /// This function can be useful in case there is work to be enqueued but for some reason this
    /// work cannot be done using [Self::launch_host_fn].
    pub async fn synchronize(&self) -> Result<(), CudaError> {
        let (tx, mut rx) = oneshot::channel::<bool>();
        let mut interval = tokio::time::interval(Duration::from_millis(INTERVAL_MS));

        // Launch the host function to signal the main thread that the task is done
        let tx = Box::new(tx);
        let tx_ptr = Box::into_raw(tx);
        unsafe {
            self.launch_host_fn(sync_host, tx_ptr as *mut c_void)?;
        }

        // Wait for the host function to signal the main thread that the task is done while
        // simultaneously polling the stream in the interval to catch any errors.
        loop {
            tokio::select! {
                _ = interval.tick() => {
                     match unsafe { self.stream().query() } {
                        Ok(()) => {break;}
                        Err(CudaError::NotReady) => {}
                        Err(e) => {
                            return Err(e);
                        }

                    }
                }
                _ =&mut rx => {
                    break;
                }
            }
        }

        Ok(())
    }

    /// Joins this task into another task.
    ///
    /// The other task will wait for the current task to finish.
    #[inline]
    unsafe fn join(self, parent: &TaskScope) -> Result<(), CudaError> {
        parent.stream.wait_unchecked(&self.end_event)
    }

    /// Copies data from the host to the device.
    #[inline]
    pub async fn into_device<T: IntoDevice>(&self, data: T) -> Result<T::Output, CopyError> {
        T::into_device_in(data, self).await
    }

    #[inline]
    pub async fn to_device<T: ToDevice>(&self, data: &T) -> Result<T::Output, CopyError> {
        T::to_device_in(data, self).await
    }

    /// Waits for all work enqueued so far in this task to finish.
    ///
    /// This function can be useful in case there is work to be enqueued but for some reason this
    /// work cannot be done using [Self::launch_host_fn].
    #[inline]
    pub fn synchronize_blocking(&self) -> Result<(), CudaError> {
        // The access to the stream is safe and therefore synchronize is safe.
        unsafe { self.stream_synchronize() }
    }

    /// # Safety
    pub unsafe fn handle(&self) -> CudaStreamHandle {
        self.stream.0
    }

    pub fn owner(&self) -> TaskPool {
        TaskPool { inner: self.0.upgrade().unwrap().inner.owner().clone() }
    }

    pub fn spawn<F, Fut>(&self, f: F) -> JoinHandle<Result<Fut::Output, CudaError>>
    where
        F: FnOnce(TaskScope) -> Fut + Send + 'static,
        Fut: Future + Send + 'static,
        Fut::Output: CudaSend + 'static,
    {
        let parent = self.clone();
        tokio::spawn(async move {
            let task = parent.owner().task().await.unwrap();
            unsafe {
                // Use the task's end event to synchronize the parent task.
                // This is safe because this is the first time this task is being run so we know
                // there are no other copies that record anything on this event at the same time.
                parent.stream.record_unchecked(&task.inner.end_event)?;
                task.inner.stream.wait_unchecked(&task.inner.end_event)?
            };
            let handle = task.run(f).await;
            handle.join(&parent)
        })
    }
}

impl StreamRef for TaskScope {
    #[inline]
    unsafe fn stream(&self) -> &CudaStream {
        &self.stream
    }
}

#[derive(Debug)]
pub struct Task {
    pub(crate) owner_id: usize,
    pub(crate) id: usize,
    pub(crate) stream: CudaStream,
    end_event: CudaEvent,
}

impl PartialEq for Task {
    fn eq(&self, other: &Self) -> bool {
        self.owner_id == other.owner_id && self.id == other.id
    }
}

impl Eq for Task {}

impl StreamRef for Task {
    #[inline]
    unsafe fn stream(&self) -> &CudaStream {
        &self.stream
    }
}

impl Drop for Task {
    fn drop(&mut self) {
        unsafe {
            self.end_event.query().expect("attempting to drop a task that did not finish");
            self.stream.query().expect("attempting to drop a task that did not finish");
        }
    }
}

impl IntoFuture for Task {
    type Output = Result<(), CudaError>;
    type IntoFuture = StreamCallbackFuture<Self>;

    fn into_future(self) -> Self::IntoFuture {
        StreamCallbackFuture::new(self)
    }
}

unsafe impl Allocator for TaskScope {
    #[inline]
    unsafe fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.stream.allocate(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: the safety contract must be upheld by the caller
        self.stream.deallocate(ptr, layout)
    }
}

impl DeviceMemory for TaskScope {
    #[inline]
    unsafe fn copy_nonoverlapping(
        &self,
        src: *const u8,
        dst: *mut u8,
        size: usize,
        direction: CopyDirection,
    ) -> Result<(), CopyError> {
        self.stream.copy_nonoverlapping(src, dst, size, direction)
    }

    #[inline]
    unsafe fn write_bytes(&self, dst: *mut u8, value: u8, size: usize) -> Result<(), CopyError> {
        self.stream.write_bytes(dst, value, size)
    }
}

impl OwnedTask {
    pub fn is_finished(&self) -> Result<bool, CudaError> {
        self.inner.end_event.query().map(|()| true).or_else(|e| match e {
            CudaError::NotReady => Ok(false),
            e => Err(e),
        })
    }

    pub async fn run<F, Fut, R>(self, f: F) -> TaskHandle<R>
    where
        F: FnOnce(TaskScope) -> Fut,
        Fut: Future<Output = R>,
        R: Send,
    {
        let strong_ptr = Arc::new(self);
        let scope = TaskScope(Arc::downgrade(&strong_ptr));
        let value = f(scope.clone()).await;
        unsafe { scope.stream.record_unchecked(&scope.end_event).unwrap() };
        TaskHandle { _task: strong_ptr, scope, value }
    }
}

impl IntoFuture for TaskScope {
    type Output = Result<(), CudaError>;
    type IntoFuture = StreamCallbackFuture<Self>;

    fn into_future(self) -> Self::IntoFuture {
        StreamCallbackFuture::new(self)
    }
}

pub struct TaskHandle<T> {
    _task: Arc<OwnedTask>,
    scope: TaskScope,
    value: T,
}

impl<T> TaskHandle<T> {
    pub fn join(self, parent: &TaskScope) -> Result<T, CudaError>
    where
        T: CudaSend,
    {
        // See [TaskHandle::join] for the explanation of safety. Here this is a bit more complex,
        // but the eventual panic still applies. This is enough in most cases.
        unsafe {
            self.scope.join(parent)?;
            let value = self.value.send_to_scope(parent);
            // Return the value to the caller.
            Ok(value)
        }
    }
}

#[pin_project]
pub struct StreamHandleFuture<T> {
    #[pin]
    callback: StreamCallbackFuture<TaskScope>,
    value: MaybeUninit<T>,
}

impl<T> Future for StreamHandleFuture<T> {
    type Output = Result<T, CudaError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        this.callback.poll(cx).map(|res| {
            res.map(|_| {
                let uinit = MaybeUninit::uninit();
                let ret = std::mem::replace(this.value, uinit);
                // We assume that JoinHandleFuture is created from a JoinHandle, so the value is
                // always initialized.
                unsafe { ret.assume_init() }
            })
        })
    }
}

impl<T> IntoFuture for TaskHandle<T> {
    type Output = Result<T, CudaError>;
    type IntoFuture = StreamHandleFuture<T>;

    #[inline]
    fn into_future(self) -> Self::IntoFuture {
        StreamHandleFuture {
            callback: self.scope.into_future(),
            value: MaybeUninit::new(self.value),
        }
    }
}

#[cfg(test)]
mod tests {
    use slop_alloc::{Buffer, IntoHost};

    use crate::TaskPoolBuilder;

    #[tokio::test]
    async fn test_global_task_pool() {
        let task = crate::task().await.unwrap();
        task.run(|_| async {}).await;
    }

    #[tokio::test]
    async fn test_async_task_buffer() {
        let task = crate::task().await.unwrap();
        let values = vec![1, 2, 3, 4, 5];
        let handle =
            task.run(|t| async move { t.into_device(Buffer::from(values)).await.unwrap() }).await;

        let buffer = handle.await.unwrap();
        let values_back = buffer.into_host().await.unwrap().into_vec();
        assert_eq!(values_back, vec![1, 2, 3, 4, 5]);
    }

    #[tokio::test]
    async fn test_local_pool() {
        let num_workers = 100;
        let num_callers = 1000;
        let pool = TaskPoolBuilder::new().num_tasks(num_workers).build().unwrap();

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        for _ in 0..num_callers {
            let pool = pool.clone();
            let tx = tx.clone();
            tokio::task::spawn(async move {
                let task = pool.task().await.unwrap();
                task.run(|_| async move {
                    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                    tx.send(true).unwrap();
                })
                .await
                .await
                .unwrap();
            });
        }
        drop(tx);

        let mut count = 0;
        while let Some(flag) = rx.recv().await {
            assert!(flag);
            count += 1;
        }
        assert_eq!(count, num_callers);
    }
}
