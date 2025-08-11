#![allow(clippy::unnecessary_wraps)]

use sp1_jit::JitContext;

pub unsafe fn enter_unconstrained(ctx: &mut JitContext, _: u32, _: u32) -> Option<u32> {
    ctx.enter_unconstrained().expect("Failed to enter unconstrained mode");

    Some(1)
}

pub unsafe fn exit_unconstrained(ctx: &mut JitContext, _: u32, _: u32) -> Option<u32> {
    ctx.exit_unconstrained();

    Some(0)
}
