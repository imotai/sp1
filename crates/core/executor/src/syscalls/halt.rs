use crate::ExecutorConfig;

use super::{context::SyscallContext, SyscallCode};

pub fn halt_syscall<E: ExecutorConfig>(
    ctx: &mut SyscallContext<E>,
    _: SyscallCode,
    exit_code: u32,
    _: u32,
) -> Option<u32> {
    ctx.set_next_pc(0);
    ctx.set_exit_code(exit_code);
    None
}
