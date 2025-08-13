use num::BigUint;
use sp1_curves::{params::NumWords, weierstrass::FpOpField};
use sp1_jit::JitContext;
use sp1_primitives::consts::u64_to_u32;
use typenum::Unsigned;

use crate::events::FieldOperation;

pub(crate) unsafe fn fp_op_syscall<P: FpOpField>(
    ctx: &mut JitContext,
    arg1: u64,
    arg2: u64,
    op: FieldOperation,
) -> Option<u64> {
    let x_ptr = arg1;
    if !x_ptr.is_multiple_of(4) {
        panic!();
    }
    let y_ptr = arg2;
    if !y_ptr.is_multiple_of(4) {
        panic!();
    }

    let mut memory = ctx.memory();

    let num_words = <P as NumWords>::WordsFieldElement::USIZE;

    let x = memory.mr_slice(x_ptr, num_words);
    let y = memory.mr_slice(y_ptr, num_words);

    let x_32 = u64_to_u32(x);
    let y_32 = u64_to_u32(y);

    let modulus = &BigUint::from_bytes_le(P::MODULUS);
    let a = BigUint::from_slice(&x_32) % modulus;
    let b = BigUint::from_slice(&y_32) % modulus;

    let result = match op {
        FieldOperation::Add => (a + b) % modulus,
        FieldOperation::Sub => ((a + modulus) - b) % modulus,
        FieldOperation::Mul => (a * b) % modulus,
        _ => panic!("Unsupported operation"),
    };
    let mut result = result.to_u64_digits();
    result.resize(num_words, 0);

    memory.mw_slice(x_ptr, &result);

    None
}
