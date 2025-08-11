use sp1_curves::{edwards::EdwardsParameters, EllipticCurve};
use sp1_jit::JitContext;

pub unsafe fn edwards_add<E: EdwardsParameters + EllipticCurve>(
    ctx: &mut JitContext,
    arg1: u32,
    arg2: u32,
) -> Option<u32> {
    crate::minimal::precompiles::ec::ec_add::<E>(ctx, arg1, arg2);

    None
}
