use sp1_curves::{
    params::NumWords,
    weierstrass::{FieldType, FpOpField},
};
use typenum::Unsigned;

use crate::{
    events::{Fp2MulEvent, PrecompileEvent},
    syscalls::SyscallCode,
    vm::syscall::SyscallRuntime,
    TracingVM,
};

pub fn core_fp2_mul<'a, RT: SyscallRuntime<'a>, P: FpOpField>(
    rt: &mut RT,
    _: SyscallCode,
    arg1: u64,
    arg2: u64,
) -> Option<u64> {
    let x_ptr = arg1;
    assert!(x_ptr.is_multiple_of(8), "x_ptr must be 8-byte aligned");
    let y_ptr = arg2;
    assert!(y_ptr.is_multiple_of(8), "y_ptr must be 8-byte aligned");

    // let clk = rt.core.clk();
    let num_words = <P as NumWords>::WordsCurvePoint::USIZE;

    let core_mut = rt.core_mut();

    // Read x (current value that will be overwritten) using mr_slice_unsafe
    // No pointer needed - just reads next num_words from memory
    let _ = core_mut.mr_slice_unsafe(x_ptr, num_words);

    // Read y using mr_slice - returns records
    let _ = core_mut.mr_slice(y_ptr, num_words);

    // Write result to x (we don't compute the actual result in tracing mode)
    let _ = core_mut.mw_slice(x_ptr, num_words);

    None
}

pub fn tracing_fp2_mul<P: FpOpField>(
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

    let event = Fp2MulEvent {
        clk,
        x_ptr,
        x,
        y_ptr,
        y,
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

    match P::FIELD_TYPE {
        FieldType::Bn254 => rt.add_precompile_event(
            syscall_code,
            syscall_event,
            PrecompileEvent::Bn254Fp2Mul(event),
        ),
        FieldType::Bls12381 => rt.add_precompile_event(
            syscall_code,
            syscall_event,
            PrecompileEvent::Bls12381Fp2Mul(event),
        ),
    }

    None
}
