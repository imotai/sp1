use crate::{
    events::{PrecompileEvent, ShaCompressEvent},
    syscalls::SyscallCode,
    vm::syscall::SyscallRuntime,
    TracingVM,
};

pub(crate) fn core_sha256_compress<'a, RT: SyscallRuntime<'a>>(
    rt: &mut RT,
    _: SyscallCode,
    _: u64,
    _: u64,
) -> Option<u64> {
    // We need to advance the memory reads by:
    // [p, q]: num_words * 2
    // write to [p]: num_words * 2, write records are of the form (last_entry, new_entry)
    rt.core_mut().mem_reads().advance(88);

    None
}

pub(crate) fn tracing_sha256_compress(
    rt: &mut TracingVM<'_>,
    syscall_code: SyscallCode,
    arg1: u64,
    arg2: u64,
) -> Option<u64> {
    let w_ptr = arg1;
    let h_ptr = arg2;
    assert_ne!(w_ptr, h_ptr);

    let clk = rt.core.clk();

    // Execute the "initialize" phase where we read in the h values.
    let mut memory = rt.precompile_memory();
    let mut hx = [0u32; 8];
    let mut h_read_records = Vec::new();
    for i in 0..8 {
        let record = memory.mr(h_ptr + i as u64 * 8);
        h_read_records.push(record);
        hx[i] = record.value as u32;
    }

    memory.increment_clk(1);
    let mut original_w = Vec::new();
    let mut w_i_read_records = Vec::new();
    for i in 0..64 {
        let w_i_record = memory.mr(w_ptr + i as u64 * 8);
        w_i_read_records.push(w_i_record);
        let w_i = w_i_record.value as u32;
        original_w.push(w_i);
    }

    memory.increment_clk(1);
    let mut h_write_records = Vec::new();
    for i in 0..8 {
        let record = memory.mw(h_ptr + i as u64 * 8);
        h_write_records.push(record);
    }

    // Push the SHA extend event.
    let event = PrecompileEvent::ShaCompress(ShaCompressEvent {
        clk,
        w_ptr,
        h_ptr,
        w: original_w,
        h: hx,
        h_read_records: h_read_records.try_into().unwrap(),
        w_i_read_records,
        h_write_records: h_write_records.try_into().unwrap(),
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
