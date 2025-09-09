use sp1_curves::{
    edwards::{EdwardsParameters, WORDS_FIELD_ELEMENT},
    params::NumWords,
    EllipticCurve, COMPRESSED_POINT_BYTES, NUM_BYTES_FIELD_ELEMENT,
};
use sp1_primitives::consts::{bytes_to_words_le, words_to_bytes_le, WORD_BYTE_SIZE};
use typenum::Unsigned;

use crate::{
    events::{
        MemoryReadRecord, MemoryWriteRecord, PrecompileEvent, Uint256MulEvent, Uint256OpsEvent,
    },
    syscalls::SyscallCode,
    vm::syscall::SyscallRuntime,
    TracingVM,
};

const U256_NUM_WORDS: usize = 4;

pub(crate) fn core_uint256_ops<'a, RT: SyscallRuntime<'a>>(
    rt: &mut RT,
    _: SyscallCode,
    _: u64,
    _: u64,
) -> Option<u64> {
    // We need to advance the memory reads by:
    // [a, b, c]: U256_NUM_WORDS * 3
    // write to [d, e]: U256_NUM_WORDS * 2 * 2, write records are of the form (last_entry,
    // new_entry)
    rt.core_mut().mem_reads().advance(U256_NUM_WORDS * 7);

    None
}

pub(crate) fn tracing_uint256_ops(
    rt: &mut TracingVM<'_>,
    syscall_code: SyscallCode,
    arg1: u64,
    arg2: u64,
) -> Option<u64> {
    let clk = rt.core.clk();
    let mut memory = rt.precompile_memory();
    let op = syscall_code.uint256_op_map();

    // Read addresses - arg1 and arg2 come from the syscall, others from registers
    let a_ptr = arg1;
    let b_ptr = arg2;
    let c_ptr_memory = memory.rr(12 /* X12 */);
    let d_ptr_memory = memory.rr(13 /* X13 */);
    let e_ptr_memory = memory.rr(14 /* X14 */);
    let c_ptr = c_ptr_memory.value;
    let d_ptr = d_ptr_memory.value;
    let e_ptr = e_ptr_memory.value;

    // Read input values (8 words = 32 bytes each for uint256) and convert to BigUint
    let a_memory_records = memory.mr_slice(a_ptr, U256_NUM_WORDS);
    memory.increment_clk(1);
    let a: Vec<_> = a_memory_records.iter().map(|record| record.value).collect();
    let b_memory_records = memory.mr_slice(b_ptr, U256_NUM_WORDS);
    memory.increment_clk(1);
    let b: Vec<_> = b_memory_records.iter().map(|record| record.value).collect();
    let c_memory_records = memory.mr_slice(c_ptr, U256_NUM_WORDS);
    let c: Vec<_> = c_memory_records.iter().map(|record| record.value).collect();

    memory.increment_clk(1);

    let d_memory_records = memory.mw_slice(d_ptr, U256_NUM_WORDS);
    let d: Vec<_> = d_memory_records.iter().map(|record| record.value).collect();
    
    memory.increment_clk(1);
    
    let e_memory_records = memory.mw_slice(e_ptr, U256_NUM_WORDS);
    let e: Vec<_> = e_memory_records.iter().map(|record| record.value).collect();

    let event = PrecompileEvent::Uint256Ops(Uint256OpsEvent {
        clk,
        op,
        a_ptr,
        a: a.try_into().unwrap(),
        b_ptr,
        b: b.try_into().unwrap(),
        c_ptr,
        c: c.try_into().unwrap(),
        d_ptr,
        d: d.try_into().unwrap(),
        e_ptr,
        e: e.try_into().unwrap(),
        c_ptr_memory,
        d_ptr_memory,
        e_ptr_memory,
        a_memory_records,
        b_memory_records,
        c_memory_records,
        d_memory_records,
        e_memory_records,
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
