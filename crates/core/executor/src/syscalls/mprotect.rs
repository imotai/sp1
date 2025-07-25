use sp1_primitives::consts::PAGE_SIZE;

use crate::{memory::MAX_LOG_ADDR, ExecutorConfig};

use super::{context::SyscallContext, SyscallCode};

pub fn mprotect_syscall<E: ExecutorConfig>(
    ctx: &mut SyscallContext<E>,
    _: SyscallCode,
    addr: u64,
    prot: u64,
) -> Option<u64> {
    let prot: u8 = prot.try_into().expect("prot must be 8 bits");

    assert!(addr.is_multiple_of(PAGE_SIZE as u64), "addr must be page aligned");
    assert!(addr < 1 << MAX_LOG_ADDR, "addr must be less than 2^48");

    let page_idx = addr / PAGE_SIZE as u64;

    // Set the page protection.
    ctx.rt.state.page_prots.insert(page_idx, prot);

    None
}
