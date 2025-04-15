use p3_air::{Air, BaseAir};
use p3_matrix::Matrix;
use sp1_derive::AlignedBorrow;
use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use crate::{
    adapter::{register::i_type::ITypeReader, state::CPUState},
    air::SP1CoreAirBuilder,
    memory::MemoryAccessCols,
    operations::AddressOperation,
    utils::{next_power_of_two, zeroed_f_vec},
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use rayon::iter::{ParallelBridge, ParallelIterator};
use sp1_core_executor::{
    events::{ByteLookupEvent, ByteRecord, MemInstrEvent},
    ExecutionRecord, Opcode, Program, DEFAULT_PC_INC,
};

use sp1_stark::air::MachineAir;

#[derive(Default)]
pub struct LoadWordChip;

pub const NUM_LOAD_WORD_COLUMNS: usize = size_of::<LoadWordColumns<u8>>();

/// The column layout for memory load word instructions.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct LoadWordColumns<T> {
    /// The current shard, timestamp, program counter of the CPU.
    pub state: CPUState<T>,

    /// The adapter to read program and register information.
    pub adapter: ITypeReader<T>,

    /// Instance of `AddressOperation` to constrain the memory address.
    pub address_operation: AddressOperation<T>,

    /// Memory consistency columns for the memory access.
    pub memory_access: MemoryAccessCols<T>,

    /// Whether this is a real load word instruction.
    pub is_real: T,
}

impl<F> BaseAir<F> for LoadWordChip {
    fn width(&self) -> usize {
        NUM_LOAD_WORD_COLUMNS
    }
}

impl<F: PrimeField32> MachineAir<F> for LoadWordChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "LoadWord".to_string()
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let chunk_size = std::cmp::max((input.memory_load_word_events.len()) / num_cpus::get(), 1);
        let nb_rows = input.memory_load_word_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        let padded_nb_rows = next_power_of_two(nb_rows, size_log2);
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_LOAD_WORD_COLUMNS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_LOAD_WORD_COLUMNS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_LOAD_WORD_COLUMNS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut LoadWordColumns<F> = row.borrow_mut();

                    if idx < input.memory_load_word_events.len() {
                        let event = &input.memory_load_word_events[idx];
                        let instruction = input.program.fetch(event.0.pc);
                        self.event_to_row(&event.0, cols, &mut blu);
                        cols.state.populate(
                            &mut blu,
                            input.public_values.execution_shard,
                            event.0.clk,
                            event.0.pc,
                        );
                        cols.adapter.populate(&mut blu, instruction, event.1);
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(values, NUM_LOAD_WORD_COLUMNS)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.memory_load_word_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl LoadWordChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MemInstrEvent,
        cols: &mut LoadWordColumns<F>,
        blu: &mut HashMap<ByteLookupEvent, usize>,
    ) {
        // Populate memory accesses for reading from memory.
        cols.memory_access.populate(event.mem_access, blu);

        let memory_addr = cols.address_operation.populate(blu, event.b, event.c);
        debug_assert!(memory_addr % 4 == 0);

        cols.is_real = F::one();
    }
}

impl<AB> Air<AB> for LoadWordChip
where
    AB: SP1CoreAirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &LoadWordColumns<AB::Var> = (*local).borrow();

        let shard = local.state.shard::<AB>();
        let clk = local.state.clk::<AB>();

        let opcode = AB::Expr::from_canonical_u32(Opcode::LW as u32);

        builder.assert_bool(local.is_real);

        // Step 1. Compute the address, and check offsets and address bounds.
        let aligned_addr = AddressOperation::<AB::F>::eval(
            builder,
            local.adapter.b().map(Into::into),
            local.adapter.c().map(Into::into),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.is_real.into(),
            local.address_operation,
        );

        // Step 2. Read the memory address.
        builder.eval_memory_access_read(
            shard.clone(),
            clk.clone(),
            aligned_addr.clone(),
            local.memory_access,
            local.is_real,
        );

        // This chip requires `op_a != x0`.
        builder.assert_zero(local.adapter.op_a_0);

        // Constrain the state of the CPU.
        CPUState::<AB::F>::eval(
            builder,
            local.state,
            local.state.pc + AB::F::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            local.is_real.into(),
        );

        // Constrain the program and register reads.
        ITypeReader::<AB::F>::eval(
            builder,
            shard,
            clk,
            local.state.pc,
            opcode,
            local.memory_access.prev_value,
            local.adapter,
            local.is_real.into(),
        );
    }
}
