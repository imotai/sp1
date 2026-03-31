use core::{
    borrow::{Borrow, BorrowMut},
    mem::{size_of, MaybeUninit},
};

use hashbrown::HashMap;
use itertools::Itertools;
use slop_air::{Air, AirBuilder, BaseAir};
use slop_algebra::{AbstractField, PrimeField32};
use slop_matrix::Matrix;
use slop_maybe_rayon::prelude::*;
use sp1_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode, ExecutionRecord, Program, CLK_INC, PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_hypercube::air::MachineAir;

use crate::{
    adapter::{
        register::alu_type::ALUTypeReader,
        state::{CPUState, CPUStateInput},
    },
    air::{SP1CoreAirBuilder, SP1Operation},
    utils::next_multiple_of_32,
};

/// The number of main trace columns for `AluX0Chip`.
pub const NUM_ALU_X0_COLS: usize = size_of::<AluX0Cols<u8>>();

/// A chip that handles all ALU instructions with `rd = x0`.
///
/// Since `x0` is hardwired to zero in RISC-V, the arithmetic result is discarded.
/// This chip only verifies the instruction against the program table and performs
/// the register accesses (writing 0 to `op_a`, reading `op_b` and `op_c`).
#[derive(Default)]
pub struct AluX0Chip;

/// The column layout for `AluX0Chip`.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct AluX0Cols<T> {
    /// The current shard, timestamp, program counter of the CPU.
    pub state: CPUState<T>,

    /// The adapter to read program and register information.
    pub adapter: ALUTypeReader<T>,

    // The corresponding ALU opcode.
    pub opcode: T,

    /// Boolean to indicate whether the row is not a padding row.
    pub is_real: T,
}

impl<F> BaseAir<F> for AluX0Chip {
    fn width(&self) -> usize {
        NUM_ALU_X0_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for AluX0Chip {
    type Record = ExecutionRecord;
    type Program = Program;

    fn name(&self) -> &'static str {
        "AluX0"
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows =
            next_multiple_of_32(input.alu_x0_events.len(), input.fixed_log2_rows::<F, _>(self));
        Some(nb_rows)
    }

    fn generate_trace_into(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
        buffer: &mut [MaybeUninit<F>],
    ) {
        let chunk_size = std::cmp::max(input.alu_x0_events.len() / num_cpus::get(), 1);
        let padded_nb_rows = <AluX0Chip as MachineAir<F>>::num_rows(self, input).unwrap();
        let num_event_rows = input.alu_x0_events.len();

        unsafe {
            let padding_start = num_event_rows * NUM_ALU_X0_COLS;
            let padding_size = (padded_nb_rows - num_event_rows) * NUM_ALU_X0_COLS;
            if padding_size > 0 {
                core::ptr::write_bytes(buffer[padding_start..].as_mut_ptr(), 0, padding_size);
            }
        }

        let buffer_ptr = buffer.as_mut_ptr() as *mut F;
        let values = unsafe {
            core::slice::from_raw_parts_mut(buffer_ptr, num_event_rows * NUM_ALU_X0_COLS)
        };

        values.chunks_mut(chunk_size * NUM_ALU_X0_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_ALU_X0_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut AluX0Cols<F> = row.borrow_mut();

                    if idx < input.alu_x0_events.len() {
                        let mut byte_lookup_events = Vec::new();
                        let event = &input.alu_x0_events[idx];
                        cols.is_real = F::one();
                        cols.opcode = F::from_canonical_u32(event.0.opcode as u32);
                        cols.state.populate(&mut byte_lookup_events, event.0.clk, event.0.pc);
                        cols.adapter.populate(&mut byte_lookup_events, event.1);
                    }
                });
            },
        );
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.alu_x0_events.len() / num_cpus::get(), 1);

        let blu_batches = input
            .alu_x0_events
            .par_chunks(chunk_size)
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_ALU_X0_COLS];
                    let cols: &mut AluX0Cols<F> = row.as_mut_slice().borrow_mut();
                    blu.add_byte_lookup_event(ByteLookupEvent {
                        opcode: ByteOpcode::LTU,
                        a: 1,
                        b: event.0.opcode as u8,
                        c: 29,
                    });
                    cols.state.populate(&mut blu, event.0.clk, event.0.pc);
                    cols.adapter.populate(&mut blu, event.1);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.alu_x0_events.is_empty()
        }
    }
}

impl<AB> Air<AB> for AluX0Chip
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &AluX0Cols<AB::Var> = (*local).borrow();

        // Check that `is_real` is boolean.
        builder.assert_bool(local.is_real);

        // This chip requires `op_a == x0`.
        builder.when(local.is_real).assert_one(local.adapter.op_a_0);

        // If `is_real` is false, then `op_a_0 == 0`.
        builder.when_not(local.is_real).assert_zero(local.adapter.op_a_0);

        // Constrain the state of the CPU.
        <CPUState<AB::F> as SP1Operation<AB>>::eval(
            builder,
            CPUStateInput::new(
                local.state,
                [
                    local.state.pc[0] + AB::F::from_canonical_u32(PC_INC),
                    local.state.pc[1].into(),
                    local.state.pc[2].into(),
                ],
                AB::Expr::from_canonical_u32(CLK_INC),
                local.is_real.into(),
            ),
        );

        // Check that `0 <= opcode < 29`, which is the range of ALU `Opcode`.
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::LTU as u32),
            AB::Expr::one(),
            local.opcode.into(),
            AB::Expr::from_canonical_u32(29),
            local.is_real.into(),
        );

        // Constrain the program and register accesses.
        ALUTypeReader::<AB::F>::eval_op_a_immutable(
            builder,
            local.state.clk_high::<AB>(),
            local.state.clk_low::<AB>(),
            local.state.pc,
            local.opcode,
            // instr_field_consts is unused by the adapter (_instr_field_consts).
            [AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()],
            local.adapter,
            local.is_real.into(),
        );
    }
}
