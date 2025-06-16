use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::{Air, BaseAir};
use p3_field::{AbstractField, PrimeField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{ParallelBridge, ParallelIterator};
use sp1_core_executor::{
    events::{AluEvent, ByteLookupEvent, ByteRecord},
    ExecutionRecord, Opcode, Program, PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::air::{MachineAir, SP1AirBuilder};

use crate::{
    adapter::{register::r_type::RTypeReader, state::CPUState},
    operations::SubOperation,
    utils::{next_multiple_of_32, zeroed_f_vec},
};

/// The number of main trace columns for `SubChip`.
pub const NUM_SUB_COLS: usize = size_of::<SubCols<u8>>();

/// A chip that implements subtraction for the opcode SUB.
#[derive(Default)]
pub struct SubChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct SubCols<T> {
    /// The current shard, timestamp, program counter of the CPU.
    pub state: CPUState<T>,

    /// The adapter to read program and register information.
    pub adapter: RTypeReader<T>,

    /// Instance of `SubOperation` to handle subtraction logic in `SubChip`'s ALU operations.
    pub sub_operation: SubOperation<T>,

    /// Boolean to indicate whether the row is not a padding row.
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for SubChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Sub".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows =
            next_multiple_of_32(input.sub_events.len(), input.fixed_log2_rows::<F, _>(self));
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        // Generate the rows for the trace.
        let chunk_size = std::cmp::max(input.sub_events.len() / num_cpus::get(), 1);
        let merged_events = input.sub_events.iter().collect::<Vec<_>>();
        let padded_nb_rows = <SubChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_SUB_COLS);

        values.chunks_mut(chunk_size * NUM_SUB_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_SUB_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut SubCols<F> = row.borrow_mut();

                    if idx < merged_events.len() {
                        let mut byte_lookup_events = Vec::new();
                        let event = merged_events[idx];
                        let instruction = input.program.fetch(event.0.pc_rel);
                        self.event_to_row(&event.0, cols, &mut byte_lookup_events);
                        cols.state.populate(
                            &mut byte_lookup_events,
                            input.public_values.execution_shard as u32,
                            event.0.clk,
                            event.0.pc_rel,
                        );
                        cols.adapter.populate(&mut byte_lookup_events, instruction, event.1);
                    }
                });
            },
        );

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(values, NUM_SUB_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.sub_events.len() / num_cpus::get(), 1);

        let event_iter = input.sub_events.chunks(chunk_size);

        let blu_batches = event_iter
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_SUB_COLS];
                    let cols: &mut SubCols<F> = row.as_mut_slice().borrow_mut();
                    let instruction = input.program.fetch(event.0.pc_rel);
                    self.event_to_row(&event.0, cols, &mut blu);
                    cols.state.populate(
                        &mut blu,
                        input.public_values.execution_shard as u32,
                        event.0.clk,
                        event.0.pc_rel,
                    );
                    cols.adapter.populate(&mut blu, instruction, event.1);
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
            !shard.sub_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl SubChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField>(
        &self,
        event: &AluEvent,
        cols: &mut SubCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        cols.is_real = F::one();
        cols.sub_operation.populate(blu, event.b, event.c);
    }
}

impl<F> BaseAir<F> for SubChip {
    fn width(&self) -> usize {
        NUM_SUB_COLS
    }
}

impl<AB> Air<AB> for SubChip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &SubCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        let opcode = AB::Expr::from_f(Opcode::SUB.as_field());

        // Constrain the sub operation over `op_b` and `op_c`.
        SubOperation::<AB::F>::eval(
            builder,
            *local.adapter.b(),
            *local.adapter.c(),
            local.sub_operation,
            local.is_real.into(),
        );

        // Constrain the state of the CPU.
        // The program counter and timestamp increment by `4`.
        CPUState::<AB::F>::eval(
            builder,
            local.state,
            local.state.pc_rel + AB::F::from_canonical_u32(PC_INC),
            AB::Expr::from_canonical_u32(PC_INC),
            local.is_real.into(),
        );

        // Constrain the program and register reads.
        RTypeReader::<AB::F>::eval(
            builder,
            local.state.shard::<AB>(),
            local.state.clk::<AB>(),
            local.state.pc_rel,
            opcode,
            local.sub_operation.value,
            local.adapter,
            local.is_real.into(),
        );
    }
}
