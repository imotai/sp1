use num::{BigUint, One, Zero};

use sp1_curves::edwards::WORDS_FIELD_ELEMENT;
use sp1_primitives::consts::{bytes_to_words_le, words_to_bytes_le_vec, WORD_BYTE_SIZE};

use crate::{
    events::{PrecompileEvent, Uint256MulEvent},
    syscalls::{SyscallCode, SyscallContext},
    ExecutorConfig,
};

pub(crate) fn uint256_mul<E: ExecutorConfig>(
    rt: &mut SyscallContext<E>,
    syscall_code: SyscallCode,
    arg1: u64,
    arg2: u64,
) -> Option<u64> {
    let clk = rt.clk;

    let x_ptr = arg1;
    assert!(x_ptr.is_multiple_of(8), "x_ptr must be 8-byte aligned");
    let y_ptr = arg2;
    assert!(y_ptr.is_multiple_of(8), "y_ptr must be 8-byte aligned");

    // First read the words for the x value. We can read a slice_unsafe here because we write
    // the computed result to x later.
    let x = rt.slice_unsafe(x_ptr, WORDS_FIELD_ELEMENT);

    // Read the y value.
    let (y_memory_records, y) = rt.mr_slice(y_ptr, WORDS_FIELD_ELEMENT);

    // The modulus is stored after the y value. We increment the pointer by the number of words.
    let modulus_ptr = y_ptr + WORDS_FIELD_ELEMENT as u64 * WORD_BYTE_SIZE as u64;
    let (modulus_memory_records, modulus) = rt.mr_slice(modulus_ptr, WORDS_FIELD_ELEMENT);

    // Get the BigUint values for x, y, and the modulus.
    let uint256_x = BigUint::from_bytes_le(&words_to_bytes_le_vec(&x));
    let uint256_y = BigUint::from_bytes_le(&words_to_bytes_le_vec(&y));
    let uint256_modulus = BigUint::from_bytes_le(&words_to_bytes_le_vec(&modulus));

    // Perform the multiplication and take the result modulo the modulus.
    let result: BigUint = if uint256_modulus.is_zero() {
        let modulus = BigUint::one() << 256;
        (uint256_x * uint256_y) % modulus
    } else {
        (uint256_x * uint256_y) % uint256_modulus
    };

    let mut result_bytes = result.to_bytes_le();
    result_bytes.resize(32, 0u8); // Pad the result to 32 bytes.

    // Convert the result to little endian u64 words.
    let result = bytes_to_words_le::<4>(&result_bytes);

    // Increment clk so that the write is not at the same cycle as the read.
    rt.clk += 1;
    // Write the result to x and keep track of the memory records.
    let x_memory_records = rt.mw_slice(x_ptr, &result);

    let shard = rt.shard().get();
    let event = PrecompileEvent::Uint256Mul(Uint256MulEvent {
        shard,
        clk,
        x_ptr,
        x,
        y_ptr,
        y,
        modulus,
        x_memory_records,
        y_memory_records,
        modulus_memory_records,
        local_mem_access: rt.postprocess(),
    });
    let syscall_event =
        rt.rt.syscall_event(clk, syscall_code, arg1, arg2, false, rt.next_pc, rt.exit_code);
    rt.add_precompile_event(syscall_code, syscall_event, event);

    None
}
