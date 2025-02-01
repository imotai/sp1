use crate::ExecutorConfig;

use super::{SyscallCode, SyscallContext};

pub fn commit_syscall<E: ExecutorConfig>(
    ctx: &mut SyscallContext<E>,
    _: SyscallCode,
    word_idx: u32,
    public_values_digest_word: u32,
) -> Option<u32> {
    ctx.rt.record.public_values.committed_value_digest[word_idx as usize] =
        public_values_digest_word;
    None
}
