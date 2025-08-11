use num::BigUint;
use sp1_curves::{params::NumWords, weierstrass::FpOpField};
use sp1_jit::JitContext;
use typenum::Unsigned;

use crate::events::FieldOperation;

pub(crate) unsafe fn fp_op_syscall<P: FpOpField>(
    ctx: &mut JitContext,
    arg1: u32,
    arg2: u32,
    op: FieldOperation,
) -> Option<u32> {
    let x_ptr = arg1;
    if x_ptr % 4 != 0 {
        panic!();
    }
    let y_ptr = arg2;
    if y_ptr % 4 != 0 {
        panic!();
    }

    let mut memory = ctx.memory();

    let num_words = <P as NumWords>::WordsFieldElement::USIZE;

    let x = memory.mr_slice(x_ptr, num_words);
    let y = memory.mr_slice(y_ptr, num_words);

    let modulus = &BigUint::from_bytes_le(P::MODULUS);
    let a = BigUint::from_slice(x) % modulus;
    let b = BigUint::from_slice(y) % modulus;

    let result = match op {
        FieldOperation::Add => (a + b) % modulus,
        FieldOperation::Sub => ((a + modulus) - b) % modulus,
        FieldOperation::Mul => (a * b) % modulus,
        _ => panic!("Unsupported operation"),
    };
    let mut result = result.to_u32_digits();
    result.resize(num_words, 0);

    memory.mw_slice(x_ptr, &result);

    None
}
