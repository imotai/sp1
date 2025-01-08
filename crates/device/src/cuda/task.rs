use std::{
    alloc::Layout,
    ffi::c_void,
    future::{Future, IntoFuture},
    mem::{ManuallyDrop, MaybeUninit},
    ops::Deref,
    pin::Pin,
    ptr::{self, NonNull},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, OnceLock,
    },
    task::{Context, Poll},
    time::Duration,
};

use crossbeam::queue::ArrayQueue;
use csl_alloc::{AllocError, Allocator};
use csl_sys::runtime::{
    cuda_device_get_mem_pool, cuda_mem_pool_set_release_threshold, CudaDevice, CudaMemPool, Dim3,
    KernelPtr,
};
use pin_project::pin_project;
use thiserror::Error;
use tokio::sync::{
    oneshot, AcquireError, OwnedSemaphorePermit, Semaphore, SemaphorePermit, TryAcquireError,
};

use crate::{
    mem::{CopyDirection, CopyError, DeviceData, DeviceMemory},
    Buffer,
};

use super::{
    stream::StreamRef, sync::CudaSend, CudaError, CudaEvent, CudaStream, StreamCallbackFuture,
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
pub async fn task() -> Result<TaskRef<'static>, SpawnError> {
    let pool = global_task_pool();
    pool.task().await
}

/// Acquires an owned task from the global task pool.
pub async fn owned_task() -> Result<OwnedTask, SpawnError> {
    let pool = global_task_pool();
    pool.clone().owned_task().await
}

/// Gets a task from the global task pool.
///
/// This function will block the current thread until a task becomes available.
pub fn get_task() -> Result<TaskRef<'static>, SpawnError> {
    futures::executor::block_on(task())
}

/// Gets an owned task from the global task pool.
///
/// This function will block the current thread until a task becomes available.
pub fn get_owned_task() -> Result<OwnedTask, SpawnError> {
    futures::executor::block_on(owned_task())
}

/// Attempts to get a task from the global task pool.
///
/// This function will not block, and will return an error if no task is currently available.
pub fn try_get_task() -> Result<TaskRef<'static>, TrySpawnError> {
    let pool = global_task_pool();
    pool.try_get_task()
}

/// Attempts to get an owned task from the global task pool.
///
/// This function will not block, and will return an error if no task is currently available.
pub fn try_get_owned_task() -> Result<OwnedTask, TrySpawnError> {
    let pool = global_task_pool();
    pool.clone().try_get_owned_task()
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
        let tasks = ArrayQueue::new(self.capacity.unwrap_or(DEFAULT_NUM_TASKS));

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

        for (i, _) in (0..tasks.capacity()).enumerate() {
            let stream = CudaStream::create().map_err(TaskPoolBuildError::StreamCreationFailed)?;
            let end_event = CudaEvent::create().map_err(TaskPoolBuildError::EventCreationFailed)?;
            tasks
                .push(Task { owner_id: id, id: i, stream, end_event })
                .map_err(|_| TaskPoolBuildError::PushTaskFailed)?;
        }

        let permits = Arc::new(Semaphore::new(tasks.capacity()));

        Ok(TaskPool { tasks, permits })
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

#[derive(Debug)]
pub struct TaskPool {
    permits: Arc<Semaphore>,
    tasks: ArrayQueue<Task>,
}

#[derive(Debug, Error)]
#[error("failed to acquire a task from the pool")]
pub enum SpawnError {
    AcquireError(#[from] AcquireError),
    PopTaskError,
}

#[derive(Debug, Error)]
#[error("failed to acquire a task from the pool")]
pub enum TrySpawnError {
    AcquireError(#[from] TryAcquireError),
    PopTaskError,
}

impl TaskPool {
    #[inline]
    fn task_ref<'a>(&'a self, task: Task, permit: SemaphorePermit<'a>) -> TaskRef {
        let task = ManuallyDrop::new(task);
        TaskRef { task, owner: self, _permit: permit }
    }

    #[inline]
    fn task_owned(self: Arc<Self>, task: Task, permit: OwnedSemaphorePermit) -> OwnedTask {
        OwnedTask { task: ManuallyDrop::new(task), _owner: self.clone(), _permit: permit }
    }

    /// Get a task from the task pool.
    pub async fn task(&self) -> Result<TaskRef, SpawnError> {
        let permit = self.permits.acquire().await?;
        let task = self.tasks.pop().ok_or(SpawnError::PopTaskError)?;
        Ok(self.task_ref(task, permit))
    }

    /// Get an owned task from the task pool.
    pub async fn owned_task(self: Arc<Self>) -> Result<OwnedTask, SpawnError> {
        let permit = self.permits.clone().acquire_owned().await?;
        let task = self.tasks.pop().ok_or(SpawnError::PopTaskError)?;
        Ok(self.task_owned(task, permit))
    }

    /// Get a task from the task pool.
    ///
    /// This function will block the current thread until a task becomes available.
    pub fn get_task(&self) -> Result<TaskRef, SpawnError> {
        futures::executor::block_on(async { self.task().await })
    }

    /// Get an owned task from the task pool.
    ///
    /// This function will block the current thread until a task becomes available.
    pub fn get_owned_task(self: Arc<Self>) -> Result<OwnedTask, SpawnError> {
        futures::executor::block_on(async { self.owned_task().await })
    }

    /// Try to get a task from the task pool.
    ///
    /// This function will not block, and will return an error if no task is currently available.
    pub fn try_get_task(&self) -> Result<TaskRef, TrySpawnError> {
        let permit = self.permits.try_acquire()?;
        let task = self.tasks.pop().ok_or(TrySpawnError::PopTaskError)?;
        Ok(self.task_ref(task, permit))
    }

    /// Try to get an owned task from the task pool.
    ///
    /// This function will not block, and will return an error if no task is currently available.
    pub fn try_get_owned_task(self: Arc<Self>) -> Result<OwnedTask, TrySpawnError> {
        let permit = self.permits.clone().try_acquire_owned()?;
        let task = self.tasks.pop().ok_or(TrySpawnError::PopTaskError)?;
        Ok(self.task_owned(task, permit))
    }
}

impl Drop for TaskPool {
    fn drop(&mut self) {
        // Assert that all tasks have finished
        while let Some(task) = self.tasks.pop() {
            unsafe {
                task.end_event.query().expect("attempting to drop a task that did not finish");
                task.stream.query().expect("attempting to drop a task that did not finish");
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskScope(Arc<ManuallyDrop<Task>>);

impl Deref for TaskScope {
    type Target = Task;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

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
    pub fn alloc<T: DeviceData>(&self, capacity: usize) -> Buffer<T, Self> {
        Buffer::with_capcity_in(capacity, self.clone())
    }

    /// Tries to allocate a buffer in this scope on the device.
    #[inline]
    pub fn try_alloc<T: DeviceData>(
        &self,
        capacity: usize,
    ) -> Result<Buffer<T, Self>, csl_alloc::TryReserveError> {
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

    /// Waits for all work enqueued so far in this task to finish.
    ///
    /// This function can be useful in case there is work to be enqueued but for some reason this
    /// work cannot be done using [Self::launch_host_fn].
    pub async fn synchronize(&self) {
        let (tx, rx) = oneshot::channel::<bool>();
        let tx = Box::new(tx);
        let tx_ptr = Box::into_raw(tx);
        unsafe {
            self.launch_host_fn(sync_host, tx_ptr as *mut c_void).unwrap();
        }
        rx.await.unwrap();
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
}

impl StreamRef for TaskScope {
    #[inline]
    unsafe fn stream(&self) -> &CudaStream {
        &self.0.stream
    }
}

#[derive(Debug)]
pub struct Task {
    pub(crate) owner_id: usize,
    pub(crate) id: usize,
    pub(crate) stream: CudaStream,
    end_event: CudaEvent,
}

impl Task {
    #[inline]
    unsafe fn scope(&self) -> TaskScope {
        let stream = CudaStream(self.stream.0);
        let end_event = CudaEvent(self.end_event.0);
        let task =
            ManuallyDrop::new(Task { owner_id: self.owner_id, id: self.id, stream, end_event });
        TaskScope(Arc::new(task))
    }
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

impl IntoFuture for Task {
    type Output = Result<(), CudaError>;
    type IntoFuture = StreamCallbackFuture<Self>;

    fn into_future(self) -> Self::IntoFuture {
        StreamCallbackFuture::new(self)
    }
}

/// A task running on a CUDA stream.
pub struct TaskRef<'a> {
    task: ManuallyDrop<Task>,
    owner: &'a TaskPool,
    _permit: SemaphorePermit<'a>,
}

/// A task running on a CUDA stream.
pub struct OwnedTask {
    task: ManuallyDrop<Task>,
    _owner: Arc<TaskPool>,
    _permit: OwnedSemaphorePermit,
}

impl<'a> TaskRef<'a> {
    pub fn is_finished(&self) -> Result<bool, CudaError> {
        self.task.end_event.query().map(|()| true).or_else(|e| match e {
            CudaError::NotReady => Ok(false),
            e => Err(e),
        })
    }

    /// Joins this task into another task.
    ///
    /// The other task will wait for the current task to finish.
    #[inline]
    unsafe fn join(self, parent: &TaskScope) -> Result<(), CudaError> {
        parent.stream.wait_unchecked(&self.task.end_event)
    }

    #[inline]
    fn synchronize(&self) -> Result<(), CudaError> {
        self.task.end_event.synchronize()
    }

    pub fn blocking_run<F, R>(self, f: F) -> TaskHandle<'a, R>
    where
        F: FnOnce(TaskScope) -> R,
        R: CudaSend,
    {
        let scope = unsafe { self.task.scope() };
        let value = f(scope);
        unsafe { self.task.stream.record_unchecked(&self.task.end_event).unwrap() };
        TaskHandle { task: self, value }
    }

    pub async fn run<F, Fut, R>(self, f: F) -> TaskHandle<'a, R>
    where
        F: FnOnce(TaskScope) -> Fut,
        Fut: Future<Output = R>,
        R: CudaSend,
    {
        let scope = unsafe { self.task.scope() };
        let value = f(scope).await;
        unsafe { self.task.stream.record_unchecked(&self.task.end_event).unwrap() };
        TaskHandle { task: self, value }
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
        self.task.end_event.query().map(|()| true).or_else(|e| match e {
            CudaError::NotReady => Ok(false),
            e => Err(e),
        })
    }

    /// Joins this task into another task.
    ///
    /// The other task will wait for the current task to finish.
    #[inline]
    unsafe fn join(self, parent: &TaskScope) -> Result<(), CudaError> {
        parent.stream.wait_unchecked(&self.task.end_event)
    }

    #[inline]
    pub fn synchronize(&self) -> Result<(), CudaError> {
        self.task.end_event.synchronize()
    }

    pub fn blocking_run<F, R>(self, f: F) -> OwnedTaskHandle<R>
    where
        F: FnOnce(TaskScope) -> R,
        R: CudaSend,
    {
        let scope = unsafe { self.task.scope() };
        let value = f(scope);
        unsafe { self.task.stream.record_unchecked(&self.task.end_event).unwrap() };
        OwnedTaskHandle { task: self, value }
    }

    pub async fn run<F, Fut, R>(self, f: F) -> OwnedTaskHandle<R>
    where
        F: FnOnce(TaskScope) -> Fut,
        Fut: Future<Output = R>,
        R: CudaSend,
    {
        let scope = unsafe { self.task.scope() };
        let value = f(scope).await;
        unsafe { self.task.stream.record_unchecked(&self.task.end_event).unwrap() };
        OwnedTaskHandle { task: self, value }
    }
}

impl<'a> StreamRef for TaskRef<'a> {
    unsafe fn stream(&self) -> &CudaStream {
        &self.task.stream
    }
}

impl<'a> Drop for TaskRef<'a> {
    fn drop(&mut self) {
        unsafe {
            let task = ManuallyDrop::take(&mut self.task);
            self.owner.tasks.push(task).expect("failed to push task back into pool");
        }
    }
}

impl<'a> IntoFuture for TaskRef<'a> {
    type Output = Result<(), CudaError>;
    type IntoFuture = StreamCallbackFuture<Self>;

    fn into_future(self) -> Self::IntoFuture {
        StreamCallbackFuture::new(self)
    }
}

pub struct TaskHandle<'a, T> {
    task: TaskRef<'a>,
    value: T,
}

impl<'a, T> TaskHandle<'a, T> {
    pub fn join(mut self, parent: &TaskScope) -> Result<T, CudaError>
    where
        T: CudaSend,
    {
        // The only way to get a task reference is from a pool, so we know that the parent stream
        // will be destroyed or panic if dropped early. This is **potentially** unsafe in some
        // weird edge cases, but it's ok for now. For users in a global pool, this is always safe.
        unsafe {
            self.task.join(parent)?;
            self.value.change_scope(parent)
        }
        // Return the value to the caller.
        Ok(self.value)
    }

    pub fn blocking_join(self) -> Result<T, CudaError> {
        self.task.synchronize().map(|_| self.value)
    }
}

pub struct OwnedTaskHandle<T> {
    task: OwnedTask,
    value: T,
}

impl<T> OwnedTaskHandle<T> {
    pub fn join(mut self, parent: &TaskScope) -> Result<T, CudaError>
    where
        T: CudaSend,
    {
        // See [TaskHandle::join] for the explanation of safety. Here this is a bit more complex,
        // but the eventual panic still applies. This is enough in most cases.
        unsafe {
            self.task.join(parent)?;
            self.value.change_scope(parent)
        }
        // Return the value to the caller.
        Ok(self.value)
    }

    pub fn blocking_join(self) -> Result<T, CudaError> {
        self.task.synchronize().map(|_| self.value)
    }
}

#[pin_project]
pub struct StreamHandleFuture<'a, T> {
    #[pin]
    callback: StreamCallbackFuture<TaskRef<'a>>,
    value: MaybeUninit<T>,
}

impl<'a, T> Future for StreamHandleFuture<'a, T> {
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

impl<'a, T> IntoFuture for TaskHandle<'a, T> {
    type Output = Result<T, CudaError>;
    type IntoFuture = StreamHandleFuture<'a, T>;

    #[inline]
    fn into_future(self) -> Self::IntoFuture {
        StreamHandleFuture {
            callback: self.task.into_future(),
            value: MaybeUninit::new(self.value),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Buffer;

    #[tokio::test]
    async fn test_async_task_pool() {
        let task = crate::cuda::task().await.unwrap();
        task.run(|_| async {}).await;
    }

    #[test]
    fn test_blocking_task_pool() {
        let task = crate::cuda::get_task().unwrap();
        task.blocking_run(|_| ());
    }

    #[tokio::test]
    async fn test_async_task_buffer() {
        let task = crate::cuda::task().await.unwrap();
        let values = vec![1, 2, 3, 4, 5];
        let handle = task
            .run(|t| async move { Buffer::from_host_vec(values, t.clone()).await.unwrap() })
            .await;

        let buffer = handle.await.unwrap();
        let values_back = buffer.to_vec().await.unwrap();
        assert_eq!(values_back, vec![1, 2, 3, 4, 5]);
    }
}
