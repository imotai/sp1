use sp1_curves::{
    edwards::{EdwardsParameters, WORDS_FIELD_ELEMENT},
    params::NumWords,
    EllipticCurve, COMPRESSED_POINT_BYTES, NUM_BYTES_FIELD_ELEMENT,
};
use sp1_primitives::consts::{bytes_to_words_le, words_to_bytes_le};
use typenum::Unsigned;

use crate::{
    events::{EdDecompressEvent, MemoryReadRecord, MemoryWriteRecord, PrecompileEvent},
    syscalls::SyscallCode,
    vm::syscall::SyscallRuntime,
    TracingVM,
};

pub(crate) fn core_edwards_decompress<
    'a,
    RT: SyscallRuntime<'a>,
    E: EllipticCurve + EdwardsParameters,
>(
    rt: &mut RT,
    _: SyscallCode,
    _: u64,
    _: u64,
) -> Option<u64> {
    // We need to advance the memory reads by:
    // [p, q]: num_words * 2
    // write to [p]: num_words * 2, write records are of the form (last_entry, new_entry)
    rt.core_mut().mem_reads().advance(WORDS_FIELD_ELEMENT * 3);

    None
}

pub(crate) fn tracing_edwards_decompress<E: EllipticCurve + EdwardsParameters>(
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
        x_memory_records,
        y_memory_records,
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
