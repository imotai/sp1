use crate::ExecutorConfig;

use super::{SyscallCode, SyscallContext};

#[allow(clippy::unnecessary_wraps)]
pub(crate) fn hint_len_syscall<E: ExecutorConfig>(
    ctx: &mut SyscallContext<E>,
    _: SyscallCode,
    _: u32,
    _: u32,
) -> Option<u32> {
    panic_if_input_exhausted(ctx);

    // Note: If the user supplies an input > than length 2^32, then the length returned will be
    // truncated to 32-bits. Reading from the syscall will definitely fail in that case, as the
    // BabyBear field is < 2^32.
    Some(ctx.rt.state.input_stream.front().map_or(u32::MAX, |data| data.len() as u32))
}

pub(crate) fn hint_read_syscall<E: ExecutorConfig>(
    ctx: &mut SyscallContext<E>,
    _: SyscallCode,
    ptr: u32,
    len: u32,
) -> Option<u32> {
    panic_if_input_exhausted(ctx);

    // SAFETY: The input stream is not empty, as checked above, so the back is not None
    let vec = unsafe { ctx.rt.state.input_stream.pop_front().unwrap_unchecked() };

    assert_eq!(vec.len() as u32, len, "hint input stream read length mismatch");
    assert_eq!(ptr % 4, 0, "hint read address not aligned to 4 bytes");
    // Iterate through the vec in 4-byte chunks
    for i in (0..len).step_by(4) {
        // Get each byte in the chunk
        let b1 = vec[i as usize];
        // In case the vec is not a multiple of 4, right-pad with 0s. This is fine because we
        // are assuming the word is uninitialized, so filling it with 0s makes sense.
        let b2 = vec.get(i as usize + 1).copied().unwrap_or(0);
        let b3 = vec.get(i as usize + 2).copied().unwrap_or(0);
        let b4 = vec.get(i as usize + 3).copied().unwrap_or(0);
        let word = u32::from_le_bytes([b1, b2, b3, b4]);

        // Save the data into runtime state so the runtime will use the desired data instead of
        // 0 when first reading/writing from this address.
        ctx.rt.uninitialized_memory_checkpoint.entry(ptr + i).or_insert_with(|| false);
        ctx.rt
            .state
            .uninitialized_memory
            .entry(ptr + i)
            .and_modify(|_| panic!("hint read address is initialized already"))
            .or_insert(word);
    }
    None
}

fn panic_if_input_exhausted<E: ExecutorConfig>(ctx: &SyscallContext<E>) {
    if ctx.rt.state.input_stream.is_empty() {
        panic!("hint input stream exhausted");
    }
}
