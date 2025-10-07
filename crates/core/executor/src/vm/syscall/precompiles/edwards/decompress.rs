use sp1_curves::{edwards::WORDS_FIELD_ELEMENT, COMPRESSED_POINT_BYTES, NUM_BYTES_FIELD_ELEMENT};
use sp1_primitives::consts::words_to_bytes_le;

use crate::{
    events::{EdDecompressEvent, MemoryReadRecord, MemoryWriteRecord, PrecompileEvent},
    syscalls::SyscallCode,
    vm::syscall::SyscallRuntime,
    TracingVM,
};

pub(crate) fn core_edwards_decompress<'a, RT: SyscallRuntime<'a>>(
    rt: &mut RT,
    _: SyscallCode,
    arg1: u64,
    arg2: u64,
) -> Option<u64> {
    let slice_ptr = arg1;
    let sign_bit = arg2;
    assert!(slice_ptr.is_multiple_of(8), "slice_ptr must be 8-byte aligned.");
    assert!(sign_bit <= 1, "Sign bit must be 0 or 1.");

    let core_mut = rt.core_mut();
    let _ = core_mut.mr_slice(slice_ptr + (COMPRESSED_POINT_BYTES as u64), WORDS_FIELD_ELEMENT);
    let _ = core_mut.mw_slice(slice_ptr, WORDS_FIELD_ELEMENT);

    None
}

pub(crate) fn tracing_edwards_decompress(
    rt: &mut TracingVM<'_>,
    syscall_code: SyscallCode,
    arg1: u64,
    arg2: u64,
) -> Option<u64> {
    let slice_ptr = arg1;
    let sign_bit = arg2;
    assert!(slice_ptr.is_multiple_of(8), "slice_ptr must be 8-byte aligned.");
    assert!(sign_bit <= 1, "Sign bit must be 0 or 1.");

    let clk = rt.core.clk();
    let sign = sign_bit != 0;
    let mut memory = rt.precompile_memory();

    let y_memory_records_vec =
        memory.mr_slice(slice_ptr + (COMPRESSED_POINT_BYTES as u64), WORDS_FIELD_ELEMENT);
    let y_vec: Vec<_> = y_memory_records_vec.iter().map(|record| record.value).collect();
    let y_memory_records: [MemoryReadRecord; WORDS_FIELD_ELEMENT] =
        y_memory_records_vec.try_into().unwrap();
    let y_bytes: [u8; COMPRESSED_POINT_BYTES] = words_to_bytes_le(&y_vec);

    memory.increment_clk(1);

    // Write decompressed X into slice
    let x_memory_records_vec = memory.mw_slice(slice_ptr, WORDS_FIELD_ELEMENT);
    let x_vec: Vec<_> = x_memory_records_vec.iter().map(|record| record.value).collect();
    let x_memory_records: [MemoryWriteRecord; WORDS_FIELD_ELEMENT] =
        x_memory_records_vec.try_into().unwrap();
    let decompressed_x_bytes: [u8; NUM_BYTES_FIELD_ELEMENT] = words_to_bytes_le(&x_vec);

    let event = EdDecompressEvent {
        clk,
        ptr: slice_ptr,
        sign,
        y_bytes,
        decompressed_x_bytes,
        y_memory_records,
        x_memory_records,
        local_mem_access: memory.postprocess(),
        ..Default::default()
    };
    let syscall_event = rt.syscall_event(
        clk,
        syscall_code,
        arg1,
        arg2,
        false,
        rt.core.next_pc(),
        rt.core.exit_code(),
    );
    rt.add_precompile_event(syscall_code, syscall_event, PrecompileEvent::EdDecompress(event));

    None
}
