use num::BigUint;
use sp1_curves::{params::NumWords, weierstrass::FpOpField};
use sp1_jit::JitContext;
use typenum::Unsigned;

use crate::events::FieldOperation;

pub(crate) unsafe fn fp2_addsub_syscall<P: FpOpField>(
    ctx: &mut JitContext,
    arg1: u32,
    arg2: u32,
    fp_op: FieldOperation,
) -> Option<u32> {
    let x_ptr = arg1;
    if x_ptr % 4 != 0 {
        panic!("x_ptr must be 4-byte aligned");
    }
    let y_ptr = arg2;
    if y_ptr % 4 != 0 {
        panic!("y_ptr must be 4-byte aligned");
    }

    let mut memory = ctx.memory();

    let num_words = <P as NumWords>::WordsCurvePoint::USIZE;
    let x = memory.mr_slice(x_ptr, num_words);
    let y = memory.mr_slice(y_ptr, num_words);

    let (ac0, ac1) = x.split_at(x.len() / 2);
    let (bc0, bc1) = y.split_at(y.len() / 2);

    let ac0 = &BigUint::from_slice(ac0);
    let ac1 = &BigUint::from_slice(ac1);
    let bc0 = &BigUint::from_slice(bc0);
    let bc1 = &BigUint::from_slice(bc1);
    let modulus = &BigUint::from_bytes_le(P::MODULUS);

    let (c0, c1) = match fp_op {
        FieldOperation::Add => ((ac0 + bc0) % modulus, (ac1 + bc1) % modulus),
        FieldOperation::Sub => ((ac0 + modulus - bc0) % modulus, (ac1 + modulus - bc1) % modulus),
        _ => panic!("Invalid operation"),
    };

    // Each of c0 and c1 should use the same number of words.
    // This is regardless of how many u32 digits are required to express them.
    let mut result = c0.to_u32_digits();
    result.resize(num_words / 2, 0);
    result.append(&mut c1.to_u32_digits());
    result.resize(num_words, 0);
    memory.mw_slice(x_ptr, &result);

    None
}
