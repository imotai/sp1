use crate::{
    events::{Poseidon2PrecompileEvent, PrecompileEvent},
    syscalls::SyscallCode,
    vm::syscall::SyscallRuntime,
    TracingVM,
};

pub(crate) fn core_poseidon2<'a, RT: SyscallRuntime<'a>>(
    rt: &mut RT,
    _: SyscallCode,
    _: u64,
    _: u64,
) -> Option<u64> {
    // We need to advance the memory reads by:
    // read 8 words as input
    // write 8 words as output: 8 * 2 = 16 (write records are (old, new) pairs)
    rt.core_mut().mem_reads().advance(24);

    None
}

pub(crate) fn tracing_poseidon2(
    rt: &mut TracingVM<'_>,
    syscall_code: SyscallCode,
    arg1: u64,
    arg2: u64,
) -> Option<u64> {
    assert!(arg2 == 0, "arg2 must be 0");
    assert!(arg1.is_multiple_of(8));

    let clk = rt.core.clk();
    let ptr = arg1;

    let mut memory = rt.precompile_memory();

    // Read the input values using unsafe read (since we'll overwrite them)
    memory.mr_slice_unsafe(8);

    // Write the computed results from memory records
    let output_memory_records = memory.mw_slice(ptr, 8);

    // Create and add the event
    let event = PrecompileEvent::POSEIDON2(Poseidon2PrecompileEvent {
        clk,
        ptr,
        memory_records: output_memory_records,
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
