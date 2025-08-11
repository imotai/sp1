use sp1_jit::JitContext;

pub(crate) unsafe fn sha256_extend(ctx: &mut JitContext, arg1: u32, arg2: u32) -> Option<u32> {
    let w_ptr = arg1;
    assert!(arg2 == 0, "arg2 must be 0");

    let mut memory = ctx.memory();
    for i in 16..64 {
        // Read w[i-15].
        let w_i_minus_15 = memory.mr(w_ptr + (i - 15) * 4);

        // Compute `s0`.
        let s0 = w_i_minus_15.rotate_right(7) ^ w_i_minus_15.rotate_right(18) ^ (w_i_minus_15 >> 3);

        // Read w[i-2].
        let w_i_minus_2 = memory.mr(w_ptr + (i - 2) * 4);

        // Compute `s1`.
        let s1 = w_i_minus_2.rotate_right(17) ^ w_i_minus_2.rotate_right(19) ^ (w_i_minus_2 >> 10);

        // Read w[i-16].
        let w_i_minus_16 = memory.mr(w_ptr + (i - 16) * 4);

        // Read w[i-7].
        let w_i_minus_7 = memory.mr(w_ptr + (i - 7) * 4);

        // Compute `w_i`.
        let w_i = s1.wrapping_add(w_i_minus_16).wrapping_add(s0).wrapping_add(w_i_minus_7);

        // Write w[i].
        memory.mw(w_ptr + i * 4, w_i);
    }

    None
}
