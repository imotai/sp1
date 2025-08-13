use sp1_curves::EllipticCurve;
use sp1_jit::JitContext;

/// Execute a weierstrass add assign syscall.
pub(crate) unsafe fn weierstrass_add_assign_syscall<E: EllipticCurve>(
    ctx: &mut JitContext,
    arg1: u64,
    arg2: u64,
) -> Option<u64> {
    crate::minimal::precompiles::ec::ec_add::<E>(ctx, arg1, arg2);

    None
}
