// use crate::ExecutorConfig;

// use super::{SyscallCode, SyscallContext};

use crate::{syscalls::SyscallCode, TracingVM};

pub(crate) fn commit_syscall(
    rt: &mut TracingVM<'_>,
    _: SyscallCode,
    word_idx: u64,
    public_values_digest_word: u64,
) -> Option<u64> {
    rt.record.public_values.committed_value_digest[word_idx as usize] =
        public_values_digest_word.try_into().expect("digest word should fit in u32");
    rt.record.public_values.commit_syscall = 1;

    None
}
