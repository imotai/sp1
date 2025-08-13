use sp1_jit::JitContext;

pub unsafe fn hint_read(ctx: &mut JitContext, ptr: u64, len: u64) -> Option<u64> {
    panic_if_input_exhausted(ctx);

    // SAFETY: The input stream is not empty, as checked above, so the back is not None
    let vec = unsafe { ctx.input_buffer().pop_front().unwrap_unchecked() };
    let mut memory = ctx.memory();

    assert_eq!(vec.len() as u64, len, "hint input stream read length mismatch");
    assert_eq!(ptr % 4, 0, "hint read address not aligned to 4 bytes");
    // Iterate through the vec in 4-byte chunks
    for i in (0..len as u32).step_by(4) {
        // Get each byte in the chunk
        let b1 = vec[i as usize];
        // In case the vec is not a multiple of 4, right-pad with 0s. This is fine because we
        // are assuming the word is uninitialized, so filling it with 0s makes sense.
        let b2 = vec.get(i as usize + 1).copied().unwrap_or(0);
        let b3 = vec.get(i as usize + 2).copied().unwrap_or(0);
        let b4 = vec.get(i as usize + 3).copied().unwrap_or(0);
        let word = u32::from_le_bytes([b1, b2, b3, b4]);

        memory.mw(ptr + i as u64, word as u64);
    }

    None
}

unsafe fn panic_if_input_exhausted(ctx: &mut JitContext) {
    if ctx.input_buffer().is_empty() {
        panic!("hint input stream exhausted");
    }
}

#[allow(clippy::unnecessary_wraps)]
pub unsafe fn hint_len(ctx: &mut JitContext, _op_a: u64, _op_b: u64) -> Option<u64> {
    let input_stream: &mut std::collections::VecDeque<Vec<u8>> = ctx.input_buffer();
    Some(input_stream.front().map_or(u64::MAX, |data| data.len() as u64))
}
