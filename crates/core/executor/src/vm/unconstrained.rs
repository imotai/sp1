use crate::events::MemoryRecord;

/// The saved state at the start of unconstrained mode.
#[derive(Default)]
pub struct UnconstrainedState {
    pub registers: [MemoryRecord; 32],
    pub pc: u64,
    pub clk: u64,
    pub global_clk: u64,
}
