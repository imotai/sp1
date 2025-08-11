use sp1_curves::EllipticCurve;
use sp1_jit::JitContext;

/// Execute a weierstrass decompress syscall.
#[allow(clippy::extra_unused_type_parameters)]
pub(crate) fn weierstrass_decompress_syscall<E: EllipticCurve>(
    _ctx: &mut JitContext,
    _slice_ptr: u32,
    _sign_bit: u32,
) -> Option<u32> {
    panic!("This method should be deprecated.");
}
