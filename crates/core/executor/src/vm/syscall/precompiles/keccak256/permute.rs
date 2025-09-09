use crate::{
    events::{KeccakPermuteEvent, PrecompileEvent},
    syscalls::SyscallCode,
    vm::syscall::SyscallRuntime,
    TracingVM,
};

pub(crate) const STATE_SIZE: usize = 25;
pub const STATE_NUM_WORDS: usize = STATE_SIZE;

pub fn core_keccak256_permute<'a, RT: SyscallRuntime<'a>>(
    rt: &mut RT,
    _: SyscallCode,
    _: u64,
    _: u64,
) -> Option<u64> {
    // Keccak permute operates on 25 u64 words (200 bytes total)
    // Read operations: STATE_NUM_WORDS from state_ptr
    // Write operations: STATE_NUM_WORDS to state_ptr (2 write records per word: old value, new
    // value)
    rt.core_mut().mem_reads().advance(3 * STATE_NUM_WORDS);

    None
}

pub fn tracing_keccak256_permute(
    rt: &mut TracingVM<'_>,
    syscall_code: SyscallCode,
    arg1: u64,
    arg2: u64,
) -> Option<u64> {
    let state_ptr = arg1;
    if arg2 != 0 {
        panic!("Expected arg2 to be 0, got {arg2}");
    }

    let start_clk = rt.core.clk();
    let mut memory = rt.precompile_memory();

    // Read the current state (will be overwritten)
    let state_read_records = memory.mr_slice(state_ptr, STATE_NUM_WORDS);
    let pre_state: Vec<u64> = state_read_records.iter().map(|record| record.value).collect();

    memory.increment_clk(1);

    // Write the new state (we don't compute the actual result in tracing mode)
    let state_write_records = memory.mw_slice(state_ptr, STATE_NUM_WORDS);
    let post_state: Vec<u64> = state_write_records.iter().map(|record| record.value).collect();

    let event = KeccakPermuteEvent {
        clk: start_clk,
        pre_state: pre_state.as_slice().try_into().unwrap(),
        post_state: post_state.as_slice().try_into().unwrap(),
        state_read_records,
        state_write_records,
        state_addr: state_ptr,
        local_mem_access: memory.postprocess(),
        ..Default::default()
    };

    let syscall_event = rt.syscall_event(
        start_clk,
        syscall_code,
        arg1,
        arg2,
        false,
        rt.core.next_pc(),
        rt.core.exit_code(),
    );
    rt.add_precompile_event(syscall_code, syscall_event, PrecompileEvent::KeccakPermute(event));

    None
}
