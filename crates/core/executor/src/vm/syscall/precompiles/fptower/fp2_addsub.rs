use sp1_curves::{
    params::NumWords,
    weierstrass::{FieldType, FpOpField},
};
use typenum::Unsigned;

use crate::{
    events::{Fp2AddSubEvent, PrecompileEvent},
    syscalls::SyscallCode,
    vm::syscall::SyscallRuntime,
    TracingVM,
};

pub fn core_fp2_add<'a, RT: SyscallRuntime<'a, true>, P: FpOpField>(
    rt: &mut RT,
    _: SyscallCode,
    _: u64,
    _: u64,
) -> Option<u64> {
    let num_words = <P as NumWords>::WordsCurvePoint::USIZE;
    // BLS12381 FP2 operations use 24 words (12 words per element, 2 elements)
    // Memory layout: [x_c0 (12 words), x_c1 (12 words)] for x
    //                [y_c0 (12 words), y_c1 (12 words)] for y
    // Read operations: 24 words from y_ptr
    // Write operations: 24 words to x_ptr (2 write records per word: old value, new value)
    rt.core_mut().mem_reads().advance(4 * num_words);

    None
}

pub fn tracing_fp2_add<P: FpOpField>(
    rt: &mut TracingVM<'_>,
    syscall_code: SyscallCode,
    arg1: u64,
    arg2: u64,
) -> Option<u64> {
    let x_ptr = arg1;
    assert!(x_ptr.is_multiple_of(8), "x_ptr must be 8-byte aligned");
    let y_ptr = arg2;
    assert!(y_ptr.is_multiple_of(8), "y_ptr must be 8-byte aligned");

    let clk = rt.core.clk();
    let num_words = <P as NumWords>::WordsCurvePoint::USIZE;

    let mut memory = rt.precompile_memory();

    // Read x (current value that will be overwritten) using mr_slice_unsafe
    // No pointer needed - just reads next num_words from memory
    let x = memory.mr_slice_unsafe(num_words);

    // Read y using mr_slice - returns records
    let y_memory_records = memory.mr_slice(y_ptr, num_words);
    let y: Vec<u64> = y_memory_records.iter().map(|record| record.value).collect();

    memory.increment_clk(1);

    // Write result to x (we don't compute the actual result in tracing mode)
    let x_memory_records = memory.mw_slice(x_ptr, num_words);

    let op = syscall_code.fp_op_map();
    let event = Fp2AddSubEvent {
        clk,
        op,
        x_ptr,
        x,
        y_ptr,
        y,
        x_memory_records,
        y_memory_records,
        local_mem_access: memory.postprocess(),
        ..Default::default()
    };

    let (syscall_code_key, precompile_event) = match P::FIELD_TYPE {
        FieldType::Bn254 => (
            match syscall_code {
                SyscallCode::BN254_FP2_ADD | SyscallCode::BN254_FP2_SUB => {
                    SyscallCode::BN254_FP2_ADD
                }
                _ => unreachable!(),
            },
            PrecompileEvent::Bn254Fp2AddSub(event),
        ),
        FieldType::Bls12381 => (
            match syscall_code {
                SyscallCode::BLS12381_FP2_ADD | SyscallCode::BLS12381_FP2_SUB => {
                    SyscallCode::BLS12381_FP2_ADD
                }
                _ => unreachable!(),
            },
            PrecompileEvent::Bls12381Fp2AddSub(event),
        ),
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
    rt.add_precompile_event(syscall_code_key, syscall_event, precompile_event);

    None
}
