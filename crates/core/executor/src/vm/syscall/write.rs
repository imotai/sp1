use sp1_jit::RiscRegister;
use sp1_primitives::consts::fd::FD_PUBLIC_VALUES;

use crate::{syscalls::SyscallCode, vm::syscall::SyscallRuntime};

pub(crate) fn write_syscall<'a, const TRACING: bool, RT: SyscallRuntime<'a, TRACING>>(
    rt: &mut RT,
    _: SyscallCode,
    arg1: u64,
    arg2: u64,
) -> Option<u64> {
    // let a2 = RiscRegister::X12;
    // let fd = arg1;
    // let buf_ptr = arg2;
    // let nbytes = rt.core().registers()[a2 as usize].value;

    // // Take the number of words we need.
    // let head = (buf_ptr & 7) as usize;
    // let nwords = (head + nbytes as usize).div_ceil(8);

    // let core_mut = rt.core_mut();
    // // Read nbytes from memory starting at write_buf.
    // let bytes = (0..nwords)
    //     .map(|_| core_mut.mr().value)
    //     .flat_map(u64::to_le_bytes)
    //     .skip(arg2 as usize % 8)
    //     .take(nbytes as usize)
    //     .collect::<Vec<u8>>();

    // let slice = bytes.as_slice();
    // if fd == 1 {
    //     let s = core::str::from_utf8(slice).unwrap();
    //     for line in s.lines() {
    //         eprintln!("stdout: {line}");
    //     }

    //     return None;
    // } else if fd == 2 {
    //     let s = core::str::from_utf8(slice).unwrap();
    //     for line in s.lines() {
    //         eprintln!("stderr: {line}");
    //     }

    //     return None;
    // } else if fd as u32 == FD_PUBLIC_VALUES {
    //     rt.push_public_values(slice);
    //     return None;
    // }

    None
}
