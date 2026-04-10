use sp1_jit::SyscallContext;

pub(crate) unsafe fn u256x2048_mul(
    _ctx: &mut impl SyscallContext,
    _arg1: u64,
    _arg2: u64,
) -> Option<u64> {
    panic!("This method should be deprecated.");
}
