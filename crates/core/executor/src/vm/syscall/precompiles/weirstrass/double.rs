use crate::events::{EllipticCurveDoubleEvent, PrecompileEvent};
use sp1_curves::{params::NumWords, CurveType, EllipticCurve};

use crate::{syscalls::SyscallCode, vm::syscall::SyscallRuntime, TracingVM};
use typenum::Unsigned;

pub(crate) fn core_weirstrass_double<'a, RT: SyscallRuntime<'a>, E: EllipticCurve>(
    rt: &mut RT,
    _: SyscallCode,
    arg1: u64,
    _: u64,
) -> Option<u64> {
    let p_ptr: u64 = arg1;
    assert!(p_ptr.is_multiple_of(8), "p_ptr must be 8-byte aligned");

    let core_mut = rt.core_mut();

    let num_words = <E::BaseField as NumWords>::WordsCurvePoint::USIZE;

    let _ = core_mut.mr_slice_unsafe(p_ptr, num_words);

    let _ = core_mut.mw_slice(p_ptr, num_words);

    None
}

pub(crate) fn tracing_weirstrass_double<E: EllipticCurve>(
    rt: &mut TracingVM<'_>,
    syscall_code: SyscallCode,
    arg1: u64,
    arg2: u64,
) -> Option<u64> {
    let p_ptr: u64 = arg1;
    assert!(p_ptr.is_multiple_of(8), "p_ptr must be 8-byte aligned");

    let clk = rt.core.clk();

    let num_words = <E::BaseField as NumWords>::WordsCurvePoint::USIZE;
    let mut precompile_memory = rt.precompile_memory();

    let p = precompile_memory.mr_slice_unsafe(num_words);

    let p_memory_records = precompile_memory.mw_slice(p_ptr, num_words);

    let event = EllipticCurveDoubleEvent {
        clk,
        p_ptr,
        p,
        p_memory_records,
        local_mem_access: precompile_memory.postprocess(),
        ..Default::default()
    };

    let syscall_event = rt.syscall_event(
        rt.core.clk(),
        syscall_code,
        arg1,
        arg2,
        false,
        rt.core.next_pc(),
        rt.core.exit_code(),
    );

    match E::CURVE_TYPE {
        CurveType::Secp256k1 => {
            rt.add_precompile_event(
                syscall_code,
                syscall_event,
                PrecompileEvent::Secp256k1Double(event),
            );
        }
        CurveType::Secp256r1 => rt.add_precompile_event(
            syscall_code,
            syscall_event,
            PrecompileEvent::Secp256r1Double(event),
        ),
        CurveType::Bn254 => {
            rt.add_precompile_event(
                syscall_code,
                syscall_event,
                PrecompileEvent::Bn254Double(event),
            );
        }
        CurveType::Bls12381 => {
            rt.add_precompile_event(
                syscall_code,
                syscall_event,
                PrecompileEvent::Bls12381Double(event),
            );
        }
        _ => panic!("Unsupported curve"),
    }

    None
}

// /// Create an elliptic curve double event.
// ///
// /// It takes a pointer to a memory location, reads the point from memory, doubles it, and writes
// the /// result back to the memory location.
// pub fn create_ec_double_event<E: EllipticCurve, Ex: ExecutorConfig>(
//     rt: &mut SyscallContext<'_, '_, Ex>,
//     arg1: u64,
//     _: u64,
// ) -> EllipticCurveDoubleEvent {
//     let start_clk = rt.clk;
//     let p_ptr = arg1;
//     assert!(p_ptr.is_multiple_of(8), "p_ptr must be 8-byte aligned");

//     let num_words = <E::BaseField as NumWords>::WordsCurvePoint::USIZE;

//     let p = rt.slice_unsafe(p_ptr, num_words);

//     let p_affine = AffinePoint::<E>::from_words_le(&p);

//     let result_affine = E::ec_double(&p_affine);

//     let result_words = result_affine.to_words_le();

//     let p_memory_records = rt.mw_slice(p_ptr, &result_words);

//     EllipticCurveDoubleEvent {
//         clk: start_clk,
//         p_ptr,
//         p,
//         p_memory_records,
//         local_mem_access: rt.postprocess(),
//     }
// }
