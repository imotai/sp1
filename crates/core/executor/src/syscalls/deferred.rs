use crate::ExecutorConfig;

use super::{SyscallCode, SyscallContext};

#[allow(clippy::mut_mut)]
pub fn commit_deferred_proofs_syscall<E: ExecutorConfig>(
    ctx: &mut SyscallContext<E>,
    _: SyscallCode,
    word_idx: u32,
    word: u32,
) -> Option<u32> {
    ctx.rt.record.public_values.deferred_proofs_digest[word_idx as usize] = word;
    None
}
