use sp1_jit::JitContext;

use tiny_keccak::keccakf;

pub(crate) const STATE_SIZE: usize = 25;

// The permutation state is 25 u64's.  Our word size is 32 bits, so it is 50 words.
pub const STATE_NUM_WORDS: usize = STATE_SIZE * 2;

pub unsafe fn keccak_permute(ctx: &mut JitContext, arg1: u32, arg2: u32) -> Option<u32> {
    let state_ptr = arg1;
    if arg2 != 0 {
        panic!("Expected arg2 to be 0, got {arg2}");
    }

    let mut state = Vec::new();
    let mut memory = ctx.memory();

    let state_values = memory.mr_slice(state_ptr, STATE_NUM_WORDS);

    for values in state_values.chunks_exact(2) {
        let least_sig = values[0];
        let most_sig = values[1];
        state.push(least_sig as u64 + ((most_sig as u64) << 32));
    }

    let mut state = state.try_into().unwrap();
    keccakf(&mut state);

    let mut values_to_write = Vec::new();
    for i in 0..STATE_SIZE {
        let most_sig = ((state[i] >> 32) & 0xFFFFFFFF) as u32;
        let least_sig = (state[i] & 0xFFFFFFFF) as u32;
        values_to_write.push(least_sig);
        values_to_write.push(most_sig);
    }

    memory.mw_slice(state_ptr, values_to_write.as_slice());

    None
}
