// use crate::ExecutorConfig;

// use super::{SyscallCode, SyscallContext};

use crate::{syscalls::SyscallCode, TracingVM};

pub(crate) fn commit_deferred_proofs_syscall(
    rt: &mut TracingVM<'_>,
    _: SyscallCode,
    word_idx: u64,
    word: u64,
) -> Option<u64> {
    rt.record.public_values.deferred_proofs_digest[word_idx as usize] = word as u32;
    rt.record.public_values.commit_deferred_syscall = 1;

    None
}
