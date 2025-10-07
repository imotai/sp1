use crate::{
    events::{PrecompileEvent, ShaExtendEvent, ShaExtendMemoryRecords},
    syscalls::SyscallCode,
    vm::syscall::SyscallRuntime,
    TracingVM,
};

pub(crate) fn core_sha256_extend<'a, RT: SyscallRuntime<'a>>(
    rt: &mut RT,
    _: SyscallCode,
    arg1: u64,
    arg2: u64,
) -> Option<u64> {
    let w_ptr = arg1;
    assert!(arg2 == 0, "arg2 must be 0");
    assert!(arg1.is_multiple_of(8));

    let core_mut = rt.core_mut();

    for i in 16..64 {
        // Read w[i-15].
        let _ = core_mut.mr(w_ptr + (i - 15) * 8);

        // Read w[i-2].
        let _ = core_mut.mr(w_ptr + (i - 2) * 8);

        // Read w[i-16].
        let _ = core_mut.mr(w_ptr + (i - 16) * 8);

        // Read w[i-7].
        let _ = core_mut.mr(w_ptr + (i - 7) * 8);

        // Write w[i].
        let _ = core_mut.mw_slice(w_ptr + i * 8, 1);
    }

    None
}

pub(crate) fn tracing_sha256_extend(
    rt: &mut TracingVM<'_>,
    syscall_code: SyscallCode,
    arg1: u64,
    arg2: u64,
) -> Option<u64> {
    let w_ptr = arg1;
    assert!(arg2 == 0, "arg2 must be 0");
    assert!(arg1.is_multiple_of(8));

    let clk = rt.core.clk();

    let mut memory = rt.precompile_memory();
    memory.increment_clk(1);
    // let mut w_i_minus_15_reads = Vec::with_capacity(48);
    // let mut w_i_minus_2_reads = Vec::with_capacity(48);
    // let mut w_i_minus_16_reads = Vec::with_capacity(48);
    // let mut w_i_minus_7_reads = Vec::with_capacity(48);

    let mut sha_extend_memory_records = Vec::with_capacity(48);
    for i in 16..64 {
        // Read w[i-15].
        let w_i_minus_15_reads = memory.mr(w_ptr + (i - 15) * 8);

        // Read w[i-2].
        let w_i_minus_2_reads = memory.mr(w_ptr + (i - 2) * 8);

        // Read w[i-16].
        let w_i_minus_16_reads = memory.mr(w_ptr + (i - 16) * 8);

        // Read w[i-7].
        let w_i_minus_7_reads = memory.mr(w_ptr + (i - 7) * 8);
        // Write w[i].
        let w_i_write = memory.mw(w_ptr + i * 8);

        memory.increment_clk(1);

        sha_extend_memory_records.push(ShaExtendMemoryRecords {
            w_i_minus_15_reads,
            w_i_minus_2_reads,
            w_i_minus_16_reads,
            w_i_minus_7_reads,
            w_i_write,
        });
    }

    // Push the SHA extend event.
    #[allow(clippy::default_trait_access)]
    let event = PrecompileEvent::ShaExtend(ShaExtendEvent {
        clk,
        w_ptr,
        local_mem_access: memory.postprocess(),
        memory_records: sha_extend_memory_records,
        page_prot_records: Default::default(),
        local_page_prot_access: Default::default(),
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
