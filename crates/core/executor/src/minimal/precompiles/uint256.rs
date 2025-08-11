use num::{BigUint, One, Zero};

use sp1_curves::edwards::WORDS_FIELD_ELEMENT;
use sp1_jit::JitContext;
use sp1_primitives::consts::{bytes_to_words_le, words_to_bytes_le_vec, WORD_BYTE_SIZE};

pub(crate) unsafe fn uint256_mul(ctx: &mut JitContext, arg1: u32, arg2: u32) -> Option<u32> {
    let x_ptr = arg1;
    if x_ptr % 4 != 0 {
        panic!();
    }
    let y_ptr = arg2;
    if y_ptr % 4 != 0 {
        panic!();
    }

    let mut memory = ctx.memory();

    // First read the words for the x value. We can read a slice_unsafe here because we write
    // the computed result to x later.
    let x = memory.mr_slice(x_ptr, WORDS_FIELD_ELEMENT);

    // Read the y value.
    let y = memory.mr_slice(y_ptr, WORDS_FIELD_ELEMENT);

    // The modulus is stored after the y value. We increment the pointer by the number of words.
    let modulus_ptr = y_ptr + WORDS_FIELD_ELEMENT as u32 * WORD_BYTE_SIZE as u32;
    let modulus = memory.mr_slice(modulus_ptr, WORDS_FIELD_ELEMENT);

    // Get the BigUint values for x, y, and the modulus.
    let uint256_x = BigUint::from_bytes_le(&words_to_bytes_le_vec(x));
    let uint256_y = BigUint::from_bytes_le(&words_to_bytes_le_vec(y));
    let uint256_modulus = BigUint::from_bytes_le(&words_to_bytes_le_vec(modulus));

    // Perform the multiplication and take the result modulo the modulus.
    let result: BigUint = if uint256_modulus.is_zero() {
        let modulus = BigUint::one() << 256;
        (uint256_x * uint256_y) % modulus
    } else {
        (uint256_x * uint256_y) % uint256_modulus
    };

    let mut result_bytes = result.to_bytes_le();
    result_bytes.resize(32, 0u8); // Pad the result to 32 bytes.

    // Convert the result to little endian u32 words.
    let result = bytes_to_words_le::<8>(&result_bytes);

    // Write the result to x and keep track of the memory records.
    memory.mw_slice(x_ptr, &result);

    None
}
