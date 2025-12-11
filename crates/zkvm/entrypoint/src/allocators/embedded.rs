use alloc::alloc::{GlobalAlloc, Layout};
use critical_section::RawRestoreState;
use embedded_alloc::TlsfHeap as Heap;

#[global_allocator]
static HEAP: EmbeddedAlloc = EmbeddedAlloc;

pub static INNER_HEAP: Heap = Heap::empty();

static mut EMBEDDED_RESERVED_INPUT_START: usize = 0;

struct CriticalSection;
critical_section::set_impl!(CriticalSection);

unsafe impl critical_section::Impl for CriticalSection {
    unsafe fn acquire() -> RawRestoreState {}

    unsafe fn release(_token: RawRestoreState) {}
}

pub fn init(start: usize) {
    unsafe { EMBEDDED_RESERVED_INPUT_START = start }

    extern "C" {
        // https://lld.llvm.org/ELF/linker_script.html#sections-command
        static _end: u8;
    }
    let heap_pos: usize = unsafe { (&_end) as *const u8 as usize };
    assert!(heap_pos <= start);
    // The heap size that is available for the program is the total memory minus the reserved input
    // region and the heap position.
    let heap_size: usize = start - heap_pos;
    unsafe { INNER_HEAP.init(heap_pos, heap_size) };
}

struct EmbeddedAlloc;

unsafe impl GlobalAlloc for EmbeddedAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        INNER_HEAP.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // Deallocating reserved input region memory is not allowed.
        if (ptr as usize) >= EMBEDDED_RESERVED_INPUT_START {
            return;
        }
        // Deallocating other memory is allowed.
        INNER_HEAP.dealloc(ptr, layout)
    }
}
