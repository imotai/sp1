use crate::{
    events::{PrecompileEvent, U256xU2048MulEvent},
    syscalls::SyscallCode,
    vm::syscall::SyscallRuntime,
    TracingVM,
};

const U256_NUM_WORDS: usize = 4;
const U2048_NUM_WORDS: usize = 32;

pub(crate) fn core_u256xu2048_mul<'a, RT: SyscallRuntime<'a, true>>(
    rt: &mut RT,
    _: SyscallCode,
    _: u64,
    _: u64,
) -> Option<u64> {
    // We need to advance the memory reads by:
    // read registers [lo_ptr, hi_ptr]: 2
    // read [a, b]: U256_NUM_WORDS + U2048_NUM_WORDS
    // write to [lo, hi]: (U2048_NUM_WORDS + U256_NUM_WORDS) * 2, write records are of the form
    // (last_entry, new_entry)
    rt.core_mut()
        .mem_reads()
        .advance(U256_NUM_WORDS + U2048_NUM_WORDS + (U2048_NUM_WORDS + U256_NUM_WORDS) * 2);

    None
}

pub(crate) fn tracing_u256xu2048_mul(
    rt: &mut TracingVM<'_>,
    syscall_code: SyscallCode,
    arg1: u64,
    arg2: u64,
) -> Option<u64> {
    let clk = rt.core.clk();
    let mut memory = rt.precompile_memory();

    let a_ptr = arg1;
    let b_ptr = arg2;

    // Read lo_ptr and hi_ptr from registers X12 and X13
    let lo_ptr_memory = memory.rr(12 /* X12 */);
    let hi_ptr_memory = memory.rr(13 /* X13 */);
    let lo_ptr = lo_ptr_memory.value;
    let hi_ptr = hi_ptr_memory.value;

    // Read input values from memory records
    let a_memory_records = memory.mr_slice(a_ptr, U256_NUM_WORDS);
    let a: Vec<_> = a_memory_records.iter().map(|record| record.value).collect();

    memory.increment_clk(1);

    let b_memory_records = memory.mr_slice(b_ptr, U2048_NUM_WORDS);
    let b: Vec<_> = b_memory_records.iter().map(|record| record.value).collect();

    // Increment clk so that the write is not at the same cycle as the read
    memory.increment_clk(1);

    // Read the computed results from memory records using mw_slice
    let lo_memory_records = memory.mw_slice(lo_ptr, U2048_NUM_WORDS);
    let lo: Vec<_> = lo_memory_records.iter().map(|record| record.value).collect();

    memory.increment_clk(1);

    let hi_memory_records = memory.mw_slice(hi_ptr, U256_NUM_WORDS);
    let hi: Vec<_> = hi_memory_records.iter().map(|record| record.value).collect();

    // Create and add the event
    let event = PrecompileEvent::U256xU2048Mul(U256xU2048MulEvent {
        clk,
        a_ptr,
        a,
        b_ptr,
        b,
        lo_ptr,
        lo_ptr_memory,
        lo,
        hi_ptr,
        hi_ptr_memory,
        hi,
        a_memory_records,
        b_memory_records,
        lo_memory_records,
        hi_memory_records,
        local_mem_access: memory.postprocess(),
        ..Default::default()
    });

    let syscall_event = rt.syscall_event(
        clk,
        syscall_code,
        arg1,
        arg2,
        false,
        rt.core.next_pc(),
        rt.core.exit_code(),
    );

    rt.add_precompile_event(syscall_code, syscall_event, event);

    None
}
