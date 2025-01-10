use std::{
    alloc::Layout,
    ffi::c_void,
    future::{Future, IntoFuture},
    ops::Deref,
    pin::Pin,
    ptr::{self, NonNull},
    sync::{Arc, Mutex},
    task::{Context, Poll, Waker},
};

use csl_alloc::{AllocError, Allocator};
use csl_sys::runtime::{
    cuda_event_record, cuda_free_async, cuda_launch_host_function, cuda_malloc_async,
    cuda_mem_copy_device_to_device_async, cuda_mem_copy_device_to_host_async,
    cuda_mem_copy_host_to_device_async, cuda_mem_set_async, cuda_stream_create,
    cuda_stream_destroy, cuda_stream_query, cuda_stream_synchronize, cuda_stream_wait_event,
    CudaStreamHandle, Dim3, KernelPtr, DEFAULT_STREAM,
};

use crate::mem::{CopyDirection, CopyError, DeviceMemory};

use super::{CudaError, CudaEvent};

#[derive(Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct CudaStream(pub(crate) CudaStreamHandle);

unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if self.0 != unsafe { DEFAULT_STREAM } {
            // We unwrap because any cuda error should throw here.
            CudaError::result_from_ffi(unsafe { cuda_stream_destroy(self.0) }).unwrap();
        }
    }
}

/// A [CudaStream] is a handle to a CUDA stream.
// #[derive(Debug, Clone, PartialEq, Eq, Hash)]
// #[repr(transparent)]
// pub struct UnsafeCudaStream(Arc<CudaStreamOwned>);

impl CudaStream {
    #[inline]
    pub(crate) fn create() -> Result<Self, CudaError> {
        let mut ptr = CudaStreamHandle(ptr::null_mut());
        CudaError::result_from_ffi(unsafe {
            cuda_stream_create(&mut ptr as *mut CudaStreamHandle)
        })?;
        Ok(Self(ptr))
    }

    /// # Safety
    ///
    /// TODO
    #[inline]
    unsafe fn launch_host_fn(
        &self,
        host_fn: Option<unsafe extern "C" fn(*mut c_void)>,
        data: *const c_void,
    ) -> Result<(), CudaError> {
        CudaError::result_from_ffi(unsafe { cuda_launch_host_function(self.0, host_fn, data) })
    }

    /// # Safety
    ///
    /// This function launch is asynchronous when called with the non-default stream. The caller
    /// must ensure that the data read to and written by the kernel remains valid throughout its
    /// execution.
    #[inline]
    pub unsafe fn launch_kernel(
        &self,
        kernel: KernelPtr,
        grid_dim: impl Into<Dim3>,
        block_dim: impl Into<Dim3>,
        args: &[*mut c_void],
        shared_mem: usize,
    ) -> Result<(), CudaError> {
        CudaError::result_from_ffi(csl_sys::runtime::cuda_launch_kernel(
            kernel,
            grid_dim.into(),
            block_dim.into(),
            args.as_ptr() as *mut *mut c_void,
            shared_mem,
            self.0,
        ))
    }

    #[inline]
    fn query(&self) -> Result<(), CudaError> {
        CudaError::result_from_ffi(unsafe { cuda_stream_query(self.0) })
    }

    #[inline]
    fn record(&self, event: &CudaEvent) -> Result<(), CudaError> {
        CudaError::result_from_ffi(unsafe { cuda_event_record(event.0, self.0) })
    }

    /// # Safety
    ///
    /// This function is marked unsafe because it requires the caller to ensure that the event is
    /// valid and that the stream is valid.
    #[inline]
    unsafe fn wait(&self, event: &CudaEvent) -> Result<(), CudaError> {
        CudaError::result_from_ffi(cuda_stream_wait_event(self.0, event.0))
    }

    #[inline]
    fn synchronize(&self) -> Result<(), CudaError> {
        CudaError::result_from_ffi(unsafe { cuda_stream_synchronize(self.0) })
    }
}

impl Default for CudaStream {
    fn default() -> Self {
        Self(unsafe { DEFAULT_STREAM })
    }
}

/// State shared between the future and the CUDA callback
struct CallbackState<S> {
    // Holding the stream to prevent it from being dropped
    _task: Option<S>,
    done: bool,
    result: Result<(), CudaError>,
    waker: Option<Waker>,
}

/// A future that completes once the GPU has completed all work queued in `stream` so far.
///
/// This future uses a callback to the host to check if the GPU has completed all work. This is
/// useful for waiting for the GPU to finish work before continuing on the host and avoiding
/// busy-waiting.
pub struct StreamCallbackFuture<S> {
    shared: Arc<Mutex<CallbackState<S>>>,
}

/// A future that completes once the GPU has completed all work queued in `stream` so far.
///
/// This future uses a busy-wait loop to check if the GPU has completed all work. This is useful for
/// a future waiting on stream completion with minimal overhead.
#[repr(transparent)]
pub struct StreamSpinFuture {
    stream: CudaStream,
}

pub trait StreamRef {
    unsafe fn stream(&self) -> &CudaStream;

    /// # Safety
    ///
    /// TODO
    #[inline]
    unsafe fn launch_host_fn_uncheked(
        &self,
        host_fn: Option<unsafe extern "C" fn(*mut c_void)>,
        data: *const c_void,
    ) -> Result<(), CudaError> {
        self.stream().launch_host_fn(host_fn, data)
    }

    #[inline]
    unsafe fn query(&self) -> Result<(), CudaError> {
        self.stream().query()
    }

    #[inline]
    unsafe fn record_unchecked(&self, event: &CudaEvent) -> Result<(), CudaError> {
        self.stream().record(event)
    }

    /// # Safety
    ///
    /// This function is marked unsafe because it requires the caller to ensure that the event is
    /// valid and that the stream is valid.
    #[inline]
    unsafe fn wait_unchecked(&self, event: &CudaEvent) -> Result<(), CudaError> {
        self.stream().wait(event)
    }

    #[inline]
    unsafe fn stream_synchronize(&self) -> Result<(), CudaError> {
        self.stream().synchronize()
    }
}

impl StreamRef for CudaStream {
    #[inline]
    unsafe fn stream(&self) -> &CudaStream {
        self
    }
}

impl<S> StreamCallbackFuture<S> {
    /// Creates a new future that completes once the GPU has completed
    /// all work queued in `stream` so far.
    pub fn new(task: S) -> Self
    where
        S: StreamRef,
    {
        // 1) Create an Arc<Mutex<...>> for the shared state
        let shared = Arc::new(Mutex::new(CallbackState {
            _task: None,
            done: false,
            result: Ok(()),
            waker: None,
        }));

        // 2) Convert Arc to a raw pointer for CUDA
        let ptr = Arc::as_ptr(&shared) as *mut c_void;

        // 3) Enqueue the callback on the given stream
        //    This means "when the GPU finishes all prior tasks in `stream`,
        //    call `my_host_callback(ptr)`"
        let launch_result = unsafe { task.stream().launch_host_fn(Some(waker_callback::<S>), ptr) };

        shared.lock().unwrap()._task = Some(task);

        if let Err(e) = launch_result {
            let mut state = shared.lock().unwrap();
            state.result = Err(e);
            state.done = true;
        }

        Self { shared }
    }
}

unsafe extern "C" fn waker_callback<S>(user_data: *mut c_void)
where
    S: StreamRef,
{
    // Convert the raw pointer back to our Arc<Mutex<CallbackState>>
    let shared = &*(user_data as *const Mutex<CallbackState<S>>);
    let mut state = shared.lock().unwrap();

    // Mark GPU done
    state.done = true;

    // If we have a waker, wake it so poll() is called again
    if let Some(ref waker) = state.waker {
        waker.wake_by_ref();
    }
}

impl<S> Future for StreamCallbackFuture<S> {
    type Output = Result<(), CudaError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.shared.lock().unwrap();

        if state.done {
            // GPU has reached the callback
            Poll::Ready(state.result)
        } else {
            // Not done yet, store the waker so we can wake it later
            state.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

impl Future for StreamSpinFuture {
    type Output = Result<(), CudaError>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.stream.query() {
            Ok(()) => Poll::Ready(Ok(())),
            Err(CudaError::NotReady) => {
                // Tell the scheduler to wake us up again
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Err(e) => Poll::Ready(Err(e)),
        }
    }
}

impl IntoFuture for CudaStream {
    type Output = Result<(), CudaError>;
    type IntoFuture = StreamCallbackFuture<Self>;

    fn into_future(self) -> Self::IntoFuture {
        StreamCallbackFuture::new(self)
    }
}

unsafe impl Allocator for CudaStream {
    #[inline]
    unsafe fn allocate(&self, layout: Layout) -> Result<ptr::NonNull<[u8]>, AllocError> {
        let mut ptr: *mut c_void = ptr::null_mut();
        unsafe {
            CudaError::result_from_ffi(cuda_malloc_async(
                &mut ptr as *mut *mut c_void,
                layout.size(),
                self.0,
            ))
            .map_err(|_| AllocError)?;
        };
        let ptr = ptr as *mut u8;
        Ok(NonNull::slice_from_raw_parts(NonNull::new_unchecked(ptr), layout.size()))
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, _layout: Layout) {
        unsafe {
            CudaError::result_from_ffi(cuda_free_async(ptr.as_ptr() as *mut c_void, self.0))
                .unwrap()
        }
    }
}

impl DeviceMemory for CudaStream {
    #[inline]
    unsafe fn copy_nonoverlapping(
        &self,
        src: *const u8,
        dst: *mut u8,
        size: usize,
        direction: CopyDirection,
    ) -> Result<(), CopyError> {
        let maybe_err = match direction {
            CopyDirection::HostToDevice => cuda_mem_copy_host_to_device_async(
                dst as *mut c_void,
                src as *const c_void,
                size,
                self.0,
            ),
            CopyDirection::DeviceToHost => cuda_mem_copy_device_to_host_async(
                dst as *mut c_void,
                src as *const c_void,
                size,
                self.0,
            ),
            CopyDirection::DeviceToDevice => cuda_mem_copy_device_to_device_async(
                dst as *mut c_void,
                src as *const c_void,
                size,
                self.0,
            ),
        };

        CudaError::result_from_ffi(maybe_err).map_err(|_| CopyError)
    }

    #[inline]
    unsafe fn write_bytes(&self, dst: *mut u8, value: u8, size: usize) -> Result<(), CopyError> {
        unsafe {
            CudaError::result_from_ffi(cuda_mem_set_async(dst as *mut c_void, value, size, self.0))
                .map_err(|_| CopyError)
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct UnsafeCudaStream(CudaStream);

impl UnsafeCudaStream {
    pub fn create() -> Result<Self, CudaError> {
        Ok(Self(CudaStream::create()?))
    }
}

impl Deref for UnsafeCudaStream {
    type Target = CudaStream;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl StreamRef for UnsafeCudaStream {
    #[inline]
    unsafe fn stream(&self) -> &CudaStream {
        &self.0
    }
}

impl IntoFuture for UnsafeCudaStream {
    type Output = Result<(), CudaError>;
    type IntoFuture = StreamCallbackFuture<Self>;

    fn into_future(self) -> Self::IntoFuture {
        StreamCallbackFuture::new(self)
    }
}

unsafe impl Allocator for UnsafeCudaStream {
    #[inline]
    unsafe fn allocate(&self, layout: Layout) -> Result<ptr::NonNull<[u8]>, AllocError> {
        self.0.allocate(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        self.0.deallocate(ptr, layout)
    }
}

impl DeviceMemory for UnsafeCudaStream {
    #[inline]
    unsafe fn copy_nonoverlapping(
        &self,
        src: *const u8,
        dst: *mut u8,
        size: usize,
        direction: CopyDirection,
    ) -> Result<(), CopyError> {
        self.0.copy_nonoverlapping(src, dst, size, direction)
    }

    #[inline]
    unsafe fn write_bytes(&self, dst: *mut u8, value: u8, size: usize) -> Result<(), CopyError> {
        self.0.write_bytes(dst, value, size)
    }
}

#[cfg(test)]
mod tests {
    use rand::{thread_rng, Rng};

    use crate::Buffer;

    use super::*;

    #[tokio::test]
    async fn test_stream_callback_future() {
        let stream = CudaStream::create().unwrap();
        stream.await.unwrap();
    }

    #[test]
    fn test_buffer() {
        let stream = CudaStream::create().unwrap();

        let cap = 10000;

        let mut buffer = Buffer::<u32, _>::with_capacity_in(cap, &stream);

        let mut rng = thread_rng();
        let host_values: Vec<u32> = (0..cap).map(|_| rng.gen()).collect();

        unsafe {
            buffer.extend_from_host_slice(&host_values).unwrap();
        }

        let mut back_values = Vec::with_capacity(cap);

        unsafe {
            buffer.copy_into_host(back_values.spare_capacity_mut()).unwrap();
            back_values.set_len(cap);
        }

        assert_eq!(host_values, back_values);

        // Test writing a constant zero value to the buffer.

        unsafe {
            buffer.set_len(0);
            buffer.write_bytes_uncheked(0, cap * std::mem::size_of::<u32>()).unwrap();
        }

        let mut back_values = Vec::with_capacity(cap);

        unsafe {
            buffer.copy_into_host(back_values.spare_capacity_mut()).unwrap();
            back_values.set_len(cap);
        }

        assert_eq!(back_values, vec![0; cap]);
    }
}
