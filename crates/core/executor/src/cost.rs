use enum_map::EnumMap;
use hashbrown::HashMap;

use crate::RiscvAirId;

const BYTE_NUM_ROWS: u64 = 1 << 16;
const RANGE_NUM_ROWS: u64 = 1 << 17;
const MAX_PROGRAM_SIZE: u64 = 1 << 22;

/// Estimates the LDE area.
#[must_use]
pub fn estimate_tarce_elements(
    num_events_per_air: EnumMap<RiscvAirId, u64>,
    costs_per_air: &HashMap<RiscvAirId, u64>,
) -> u64 {
    // Compute the byte chip contribution.
    let mut cells = BYTE_NUM_ROWS * costs_per_air[&RiscvAirId::Byte];

    // Compute the range chip contribution.
    cells += RANGE_NUM_ROWS * costs_per_air[&RiscvAirId::Range];

    // Compute the program chip contribution.
    cells += MAX_PROGRAM_SIZE * costs_per_air[&RiscvAirId::Program];

    // Compute the add chip contribution.
    cells +=
        (num_events_per_air[RiscvAirId::Add]).next_power_of_two() * costs_per_air[&RiscvAirId::Add];

    // Compute the addi chip contribution.
    cells += (num_events_per_air[RiscvAirId::Addi]).next_power_of_two()
        * costs_per_air[&RiscvAirId::Addi];

    // Compute the sub chip contribution.
    cells +=
        (num_events_per_air[RiscvAirId::Sub]).next_power_of_two() * costs_per_air[&RiscvAirId::Sub];

    // Compute the bitwise chip contribution.
    cells += (num_events_per_air[RiscvAirId::Bitwise]).next_power_of_two()
        * costs_per_air[&RiscvAirId::Bitwise];

    // Compute the divrem chip contribution.
    cells += (num_events_per_air[RiscvAirId::DivRem]).next_power_of_two()
        * costs_per_air[&RiscvAirId::DivRem];

    // Compute the lt chip contribution.
    cells +=
        (num_events_per_air[RiscvAirId::Lt]).next_power_of_two() * costs_per_air[&RiscvAirId::Lt];

    // Compute the mul chip contribution.
    cells +=
        (num_events_per_air[RiscvAirId::Mul]).next_power_of_two() * costs_per_air[&RiscvAirId::Mul];

    // Compute the shift left chip contribution.
    cells += (num_events_per_air[RiscvAirId::ShiftLeft]).next_power_of_two()
        * costs_per_air[&RiscvAirId::ShiftLeft];

    // Compute the shift right chip contribution.
    cells += (num_events_per_air[RiscvAirId::ShiftRight]).next_power_of_two()
        * costs_per_air[&RiscvAirId::ShiftRight];

    // Compute the memory local chip contribution.
    cells += (num_events_per_air[RiscvAirId::MemoryLocal]).next_power_of_two()
        * costs_per_air[&RiscvAirId::MemoryLocal];

    // Compute the branch chip contribution.
    cells += (num_events_per_air[RiscvAirId::Branch]).next_power_of_two()
        * costs_per_air[&RiscvAirId::Branch];

    // Compute the jal chip contribution.
    cells +=
        (num_events_per_air[RiscvAirId::Jal]).next_power_of_two() * costs_per_air[&RiscvAirId::Jal];

    // Compute the jalr chip contribution.
    cells += (num_events_per_air[RiscvAirId::Jalr]).next_power_of_two()
        * costs_per_air[&RiscvAirId::Jalr];

    // Compute the auipc chip contribution.
    cells += (num_events_per_air[RiscvAirId::Auipc]).next_power_of_two()
        * costs_per_air[&RiscvAirId::Auipc];

    // Compute the memory instruction chip contribution.
    cells += (num_events_per_air[RiscvAirId::LoadByte]).next_power_of_two()
        * costs_per_air[&RiscvAirId::LoadByte];
    cells += (num_events_per_air[RiscvAirId::LoadHalf]).next_power_of_two()
        * costs_per_air[&RiscvAirId::LoadHalf];
    cells += (num_events_per_air[RiscvAirId::LoadWord]).next_power_of_two()
        * costs_per_air[&RiscvAirId::LoadWord];
    cells += (num_events_per_air[RiscvAirId::LoadX0]).next_power_of_two()
        * costs_per_air[&RiscvAirId::LoadX0];
    cells += (num_events_per_air[RiscvAirId::StoreByte]).next_power_of_two()
        * costs_per_air[&RiscvAirId::StoreByte];
    cells += (num_events_per_air[RiscvAirId::StoreHalf]).next_power_of_two()
        * costs_per_air[&RiscvAirId::StoreHalf];
    cells += (num_events_per_air[RiscvAirId::StoreWord]).next_power_of_two()
        * costs_per_air[&RiscvAirId::StoreWord];

    // Compute the syscall instruction chip contribution.
    cells += (num_events_per_air[RiscvAirId::SyscallInstrs]).next_power_of_two()
        * costs_per_air[&RiscvAirId::SyscallInstrs];

    // Compute the syscall core chip contribution.
    cells += (num_events_per_air[RiscvAirId::SyscallCore]).next_power_of_two()
        * costs_per_air[&RiscvAirId::SyscallCore];

    // Compute the global chip contribution.
    cells += (num_events_per_air[RiscvAirId::Global]).next_power_of_two()
        * costs_per_air[&RiscvAirId::Global];

    cells
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
