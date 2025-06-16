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
use sp1_stark::{
    air::{MachineAir, SP1AirBuilder},
    Word,
};

use crate::{
    adapter::{register::i_type::ITypeReader, state::CPUState},
    operations::AddwOperation,
    utils::{next_multiple_of_32, zeroed_f_vec},
};

/// The number of main trace columns for `AddiChip`.
pub const NUM_ADDIW_COLS: usize = size_of::<AddiwCols<u8>>();

/// A chip that implements addition for the opcode ADDIW.
#[derive(Default)]
pub struct AddiwChip;

/// The column layout for the chip.
#[derive(AlignedBorrow, Default, Clone, Copy)]
#[repr(C)]
pub struct AddiwCols<T> {
    /// The current shard, timestamp, program counter of the CPU.
    pub state: CPUState<T>,

    /// The adapter to read program and register information.
    pub adapter: ITypeReader<T>,

    /// Instance of `AddOperation` to handle addition logic in `AddiChip`'s ALU operations.
    pub addw_operation: AddwOperation<T>,

    /// Boolean to indicate whether the row is not a padding row.
    pub is_real: T,
}

impl<F: PrimeField32> MachineAir<F> for AddiwChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "Addiw".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows =
            next_multiple_of_32(input.addiw_events.len(), input.fixed_log2_rows::<F, _>(self));
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        // Generate the rows for the trace.
        let chunk_size = std::cmp::max(input.addiw_events.len() / num_cpus::get(), 1);
        let merged_events = input.addiw_events.iter().collect::<Vec<_>>();
        let padded_nb_rows = <AddiwChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_ADDIW_COLS);

        values.chunks_mut(chunk_size * NUM_ADDIW_COLS).enumerate().par_bridge().for_each(
            |(i, rows)| {
                rows.chunks_mut(NUM_ADDIW_COLS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut AddiwCols<F> = row.borrow_mut();

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
        RowMajorMatrix::new(values, NUM_ADDIW_COLS)
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = std::cmp::max(input.addiw_events.len() / num_cpus::get(), 1);

        let event_iter = input.addiw_events.chunks(chunk_size);

        let blu_batches = event_iter
            .par_bridge()
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|event| {
                    let mut row = [F::zero(); NUM_ADDIW_COLS];
                    let cols: &mut AddiwCols<F> = row.as_mut_slice().borrow_mut();
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
            !shard.addiw_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl AddiwChip {
    /// Create a row from an event.
    fn event_to_row<F: PrimeField>(
        &self,
        event: &AluEvent,
        cols: &mut AddiwCols<F>,
        blu: &mut impl ByteRecord,
    ) {
        cols.is_real = F::one();
        // let value = (Wrapping(event.b as i32) + Wrapping(event.c as i32)).0 as u64;
        // cols.value = Word::from(value);
        cols.addw_operation.populate(blu, event.b, event.c, true);
    }
}

impl<F> BaseAir<F> for AddiwChip {
    fn width(&self) -> usize {
        NUM_ADDIW_COLS
    }
}

impl<AB> Air<AB> for AddiwChip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &AddiwCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.is_real);

        let opcode = AB::Expr::from_f(Opcode::ADDIW.as_field());

        // Constrain the add operation over `op_b` and `op_c`.
        AddwOperation::<AB::F>::eval(
            builder,
            local.adapter.b().map(|x| x.into()),
            local.adapter.c().map(|x| x.into()),
            local.addw_operation,
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

        let u16_max = AB::F::from_canonical_u32((1 << 16) - 1 as u32);

        let word: Word<AB::Expr> = Word([
            local.addw_operation.value[0].into(),
            local.addw_operation.value[1].into(),
            local.addw_operation.msb.msb * u16_max,
            local.addw_operation.msb.msb * u16_max,
        ]);

        // Constrain the program and register reads.
        ITypeReader::<AB::F>::eval(
            builder,
            local.state.shard::<AB>(),
            local.state.clk::<AB>(),
            local.state.pc_rel,
            opcode,
            word,
            local.adapter,
            local.is_real.into(),
        );
    }
}
