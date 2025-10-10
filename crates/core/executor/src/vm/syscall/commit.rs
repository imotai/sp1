// use crate::ExecutorConfig;

// use super::{SyscallCode, SyscallContext};

use crate::{syscalls::SyscallCode, vm::syscall::SyscallRuntime};

pub(crate) fn commit_syscall<'a, RT: SyscallRuntime<'a>>(
    rt: &mut RT,
    _: SyscallCode,
    word_idx: u64,
    public_values_digest_word: u64,
) -> Option<u64> {
    if RT::TRACING {
        let record = rt.record_mut();

        record.public_values.committed_value_digest[word_idx as usize] =
            public_values_digest_word.try_into().expect("digest word should fit in u32");

        record.public_values.commit_syscall = 1;
    }

    None
}
