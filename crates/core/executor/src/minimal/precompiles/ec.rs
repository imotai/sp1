use sp1_curves::{params::NumWords, AffinePoint, EllipticCurve};
use sp1_jit::JitContext;
use typenum::Unsigned;

/// Create an elliptic curve add event. It takes two pointers to memory locations, reads the points
/// from memory, adds them together, and writes the result back to the first memory location.
/// The generic parameter `N` is the number of u32 words in the point representation. For example,
/// for the secp256k1 curve, `N` would be 16 (64 bytes) because the x and y coordinates are 32 bytes
/// each.
pub(crate) unsafe fn ec_add<E: EllipticCurve>(ctx: &mut JitContext, arg1: u64, arg2: u64) {
    let p_ptr = arg1;
    if !p_ptr.is_multiple_of(4) {
        panic!();
    }
    let q_ptr = arg2;
    if !q_ptr.is_multiple_of(4) {
        panic!();
    }

    let mut memory = ctx.memory();
    let num_words = <E::BaseField as NumWords>::WordsCurvePoint::USIZE;

    let p = memory.mr_slice(p_ptr, num_words);
    let q = memory.mr_slice(q_ptr, num_words);

    let p_affine = AffinePoint::<E>::from_words_le(p);
    let q_affine = AffinePoint::<E>::from_words_le(q);
    let result_affine = p_affine + q_affine;

    let result_words = result_affine.to_words_le();

    memory.mw_slice(p_ptr, &result_words);
}

/// Create an elliptic curve double event.
///
/// It takes a pointer to a memory location, reads the point from memory, doubles it, and writes the
/// result back to the memory location.
pub(crate) unsafe fn ec_double<E: EllipticCurve>(ctx: &mut JitContext, arg1: u64, _: u64) {
    let p_ptr = arg1;
    if !p_ptr.is_multiple_of(4) {
        panic!();
    }

    let mut memory = ctx.memory();

    let num_words = <E::BaseField as NumWords>::WordsCurvePoint::USIZE;

    let p = memory.mr_slice(p_ptr, num_words);

    let p_affine = AffinePoint::<E>::from_words_le(p);

    let result_affine = E::ec_double(&p_affine);

    let result_words = result_affine.to_words_le();

    memory.mw_slice(p_ptr, &result_words);
}
