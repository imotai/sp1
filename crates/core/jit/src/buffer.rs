use std::{
    io,
    marker::PhantomData,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use memmap2::MmapMut;

/// An atomic buffer that is backed by a mmap.
///
/// This buffer is intended to be written to effciently by the JIT function without
/// needing to make an extern 'C' call.
///
/// The JIT function is expected to interact with this buffer in a single threaded fashion.
///
/// It should write to the buffer, then do an atomic "release" style write to the writes counter.
#[repr(C)]
pub struct AtomicBuffer<T: Copy> {
    data: MmapMut,
    /// The number of writes to the buffer.
    ///
    /// If the highbit is set, the producer is done.
    writes: Arc<AtomicUsize>,
    /// The number of reads from the buffer.
    reads: usize,
    /// The capacity of the buffer.
    cap: usize,
    _phantom: PhantomData<T>,
}

impl<T: Copy> AtomicBuffer<T> {
    const DONE_MASK: usize = 1 << 63;

    pub fn new(cap: usize) -> io::Result<Self> {
        let data = MmapMut::map_anon(cap * std::mem::size_of::<T>())?;

        Ok(Self {
            data,
            writes: Arc::new(AtomicUsize::new(0)),
            reads: 0,
            cap,
            _phantom: PhantomData,
        })
    }

    /// The pointer to the start of the data in the buffer.
    pub fn data_ptr(&self) -> *mut T {
        self.data.as_ptr() as *mut T
    }

    /// The pointer to the write index.
    pub fn write_ptr(&self) -> *mut usize {
        self.writes.as_ref().as_ptr()
    }

    /// Pop a value from the buffer. Returns None if empty.
    ///
    /// This method will CPU block until a value is available.
    pub fn pop(&mut self) -> Option<T> {
        loop {
            let writes = self.writes.load(Ordering::Acquire);

            let done = writes & Self::DONE_MASK > 0;
            let writes = writes & !Self::DONE_MASK;

            if writes == self.reads {
                if done {
                    return None;
                } else {
                    std::hint::spin_loop();
                    continue;
                }
            }

            let val = unsafe {
                let slot = self.data_ptr().add(self.reads);
                std::ptr::read(slot)
            };

            self.reads += 1;

            return Some(val);
        }
    }
}
