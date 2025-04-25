use enum_map::EnumMap;
use hashbrown::HashMap;

use crate::RiscvAirId;

const BYTE_NUM_ROWS: u64 = 1 << 16;
const RANGE_NUM_ROWS: u64 = 1 << 17;

/// Estimates the LDE area.
#[must_use]
pub fn estimate_trace_elements(
    num_events_per_air: EnumMap<RiscvAirId, u64>,
    costs_per_air: &HashMap<RiscvAirId, u64>,
    program_size: u64,
) -> (u64, u64) {
    let mut max_height = 0;

    // Compute the byte chip contribution.
    let mut cells = BYTE_NUM_ROWS * costs_per_air[&RiscvAirId::Byte];

    // Compute the range chip contribution.
    cells += RANGE_NUM_ROWS * costs_per_air[&RiscvAirId::Range];

    // Compute the program chip contribution.
    cells += program_size * costs_per_air[&RiscvAirId::Program];

    // Compute the memory bump contribution.
    cells += 32 * costs_per_air[&RiscvAirId::MemoryBump];

    // Compute the add chip contribution.
    cells += (num_events_per_air[RiscvAirId::Add]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::Add];
    max_height = max_height.max(num_events_per_air[RiscvAirId::Add]);

    // Compute the addi chip contribution.
    cells += (num_events_per_air[RiscvAirId::Addi]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::Addi];
    max_height = max_height.max(num_events_per_air[RiscvAirId::Addi]);

    // Compute the sub chip contribution.
    cells += (num_events_per_air[RiscvAirId::Sub]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::Sub];
    max_height = max_height.max(num_events_per_air[RiscvAirId::Sub]);
    // Compute the bitwise chip contribution.
    cells += (num_events_per_air[RiscvAirId::Bitwise]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::Bitwise];
    max_height = max_height.max(num_events_per_air[RiscvAirId::Bitwise]);
    // Compute the divrem chip contribution.
    cells += (num_events_per_air[RiscvAirId::DivRem]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::DivRem];
    max_height = max_height.max(num_events_per_air[RiscvAirId::DivRem]);
    // Compute the lt chip contribution.
    cells +=
        (num_events_per_air[RiscvAirId::Lt]).next_multiple_of(32) * costs_per_air[&RiscvAirId::Lt];
    max_height = max_height.max(num_events_per_air[RiscvAirId::Lt]);
    // Compute the mul chip contribution.
    cells += (num_events_per_air[RiscvAirId::Mul]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::Mul];
    max_height = max_height.max(num_events_per_air[RiscvAirId::Mul]);
    // Compute the shift left chip contribution.
    cells += (num_events_per_air[RiscvAirId::ShiftLeft]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::ShiftLeft];
    max_height = max_height.max(num_events_per_air[RiscvAirId::ShiftLeft]);
    // Compute the shift right chip contribution.
    cells += (num_events_per_air[RiscvAirId::ShiftRight]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::ShiftRight];
    max_height = max_height.max(num_events_per_air[RiscvAirId::ShiftRight]);
    // Compute the memory local chip contribution.
    cells += (num_events_per_air[RiscvAirId::MemoryLocal]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::MemoryLocal];
    max_height = max_height.max(num_events_per_air[RiscvAirId::MemoryLocal]);
    // Compute the branch chip contribution.
    cells += (num_events_per_air[RiscvAirId::Branch]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::Branch];
    max_height = max_height.max(num_events_per_air[RiscvAirId::Branch]);
    // Compute the jal chip contribution.
    cells += (num_events_per_air[RiscvAirId::Jal]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::Jal];
    max_height = max_height.max(num_events_per_air[RiscvAirId::Jal]);
    // Compute the jalr chip contribution.
    cells += (num_events_per_air[RiscvAirId::Jalr]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::Jalr];
    max_height = max_height.max(num_events_per_air[RiscvAirId::Jalr]);
    // Compute the auipc chip contribution.
    cells += (num_events_per_air[RiscvAirId::Auipc]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::Auipc];
    max_height = max_height.max(num_events_per_air[RiscvAirId::Auipc]);
    // Compute the memory instruction chip contribution.
    cells += (num_events_per_air[RiscvAirId::LoadByte]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::LoadByte];
    max_height = max_height.max(num_events_per_air[RiscvAirId::LoadByte]);
    cells += (num_events_per_air[RiscvAirId::LoadHalf]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::LoadHalf];
    max_height = max_height.max(num_events_per_air[RiscvAirId::LoadHalf]);
    cells += (num_events_per_air[RiscvAirId::LoadWord]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::LoadWord];
    max_height = max_height.max(num_events_per_air[RiscvAirId::LoadWord]);
    cells += (num_events_per_air[RiscvAirId::LoadX0]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::LoadX0];
    max_height = max_height.max(num_events_per_air[RiscvAirId::LoadX0]);
    cells += (num_events_per_air[RiscvAirId::StoreByte]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::StoreByte];
    max_height = max_height.max(num_events_per_air[RiscvAirId::StoreByte]);
    cells += (num_events_per_air[RiscvAirId::StoreHalf]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::StoreHalf];
    max_height = max_height.max(num_events_per_air[RiscvAirId::StoreHalf]);
    cells += (num_events_per_air[RiscvAirId::StoreWord]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::StoreWord];
    max_height = max_height.max(num_events_per_air[RiscvAirId::StoreWord]);

    // Compute the syscall instruction chip contribution.
    cells += (num_events_per_air[RiscvAirId::SyscallInstrs]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::SyscallInstrs];
    max_height = max_height.max(num_events_per_air[RiscvAirId::SyscallInstrs]);

    // Compute the syscall core chip contribution.
    cells += (num_events_per_air[RiscvAirId::SyscallCore]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::SyscallCore];
    max_height = max_height.max(num_events_per_air[RiscvAirId::SyscallCore]);

    // Compute the global chip contribution.
    cells += (num_events_per_air[RiscvAirId::Global]).next_multiple_of(32)
        * costs_per_air[&RiscvAirId::Global];
    max_height = max_height.max(num_events_per_air[RiscvAirId::Global]);

    (cells, max_height)
}

/// Pads the event counts to account for the worst case jump in events across N cycles.
#[must_use]
#[allow(clippy::match_same_arms)]
pub fn pad_rv32im_event_counts(
    mut event_counts: EnumMap<RiscvAirId, u64>,
    num_cycles: u64,
) -> EnumMap<RiscvAirId, u64> {
    event_counts.iter_mut().for_each(|(k, v)| match k {
        RiscvAirId::Add => *v += num_cycles,
        RiscvAirId::Addi => *v += num_cycles,
        RiscvAirId::Sub => *v += num_cycles,
        RiscvAirId::Bitwise => *v += num_cycles,
        RiscvAirId::DivRem => *v += num_cycles,
        RiscvAirId::Lt => *v += num_cycles,
        RiscvAirId::Mul => *v += num_cycles,
        RiscvAirId::ShiftLeft => *v += num_cycles,
        RiscvAirId::ShiftRight => *v += num_cycles,
        RiscvAirId::MemoryLocal => *v += 64 * num_cycles,
        RiscvAirId::Branch => *v += num_cycles,
        RiscvAirId::Jal => *v += num_cycles,
        RiscvAirId::Jalr => *v += num_cycles,
        RiscvAirId::Auipc => *v += num_cycles,
        RiscvAirId::LoadByte => *v += num_cycles,
        RiscvAirId::LoadHalf => *v += num_cycles,
        RiscvAirId::LoadWord => *v += num_cycles,
        RiscvAirId::LoadX0 => *v += num_cycles,
        RiscvAirId::StoreByte => *v += num_cycles,
        RiscvAirId::StoreHalf => *v += num_cycles,
        RiscvAirId::StoreWord => *v += num_cycles,
        RiscvAirId::SyscallInstrs => *v += num_cycles,
        RiscvAirId::SyscallCore => *v += num_cycles,
        RiscvAirId::Global => *v += 512 * num_cycles,
        _ => (),
    });
    event_counts
}
