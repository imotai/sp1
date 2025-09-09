use crate::{syscalls::SyscallCode, vm::syscall::SyscallRuntime};

#[allow(clippy::unnecessary_wraps)]
pub fn enter_unconstrained<'a, RT: SyscallRuntime<'a>>(
    rt: &mut RT,
    _: SyscallCode,
    _: u64,
    _: u64,
) -> Option<u64> {
    // Save the current state before entering unconstrained mode
    let current_state = crate::vm::unconstrained::UnconstrainedState {
        registers: *rt.core().registers(),
        pc: rt.core().pc(),
        clk: rt.core().clk(),
        global_clk: rt.core().global_clk(),
    };

    // Store the state in CoreVM's unconstrained_state field
    rt.core_mut().unconstrained_state = Some(current_state);

    // Return 1 to indicate unconstrained mode is active
    Some(1)
}

pub fn exit_unconstrained<'a, RT: SyscallRuntime<'a>>(
    rt: &mut RT,
    _: SyscallCode,
    _: u64,
    _: u64,
) -> (u64, Option<u64>) {
    // Restore the saved state from when we entered unconstrained mode
    if let Some(saved_state) = rt.core_mut().unconstrained_state.take() {
        // Copy the saved registers back
        rt.core_mut().registers_mut().copy_from_slice(&saved_state.registers);
        rt.core_mut().set_pc(saved_state.pc);
        rt.core_mut().set_next_pc(saved_state.pc + 4);
        rt.core_mut().set_clk(saved_state.clk);
        rt.core_mut().set_next_clk(saved_state.clk + 8);
        rt.core_mut().set_global_clk(saved_state.global_clk);
    } else {
        unreachable!("Exit unconstrained called but not state is present, this is a bug.");
    }

    // Return 0 to indicate normal execution resumes
    (rt.core().clk(), Some(0))
}
