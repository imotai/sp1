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
    utils::{next_multiple_of_32, zeroed_f_vec},
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use rayon::iter::{ParallelBridge, ParallelIterator};
use sp1_core_executor::{
    events::{ByteLookupEvent, ByteRecord, MemInstrEvent},
    ExecutionRecord, Opcode, Program, CLK_INC, PC_INC,
};
use sp1_stark::{air::MachineAir, Word};

#[derive(Default)]
pub struct StoreWordChip;

pub const NUM_STORE_WORD_COLUMNS: usize = size_of::<StoreWordColumns<u8>>();

/// The column layout for memory store word instructions.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct StoreWordColumns<T> {
    /// The current shard, timestamp, program counter of the CPU.
    pub state: CPUState<T>,

    /// The adapter to read program and register information.
    pub adapter: ITypeReader<T>,

    /// Instance of `AddressOperation` to constrain the memory address.
    pub address_operation: AddressOperation<T>,

    /// Memory consistency columns for the memory access.
    pub memory_access: MemoryAccessCols<T>,

    /// Whether the offset is `0` or `4`.
    pub offset_bit: T,

    /// The value to store.
    pub store_value: Word<T>,

    /// Whether this is a real store word instruction.
    pub is_real: T,
}

impl<F> BaseAir<F> for StoreWordChip {
    fn width(&self) -> usize {
        NUM_STORE_WORD_COLUMNS
    }
}

impl<F: PrimeField32> MachineAir<F> for StoreWordChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "StoreWord".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.memory_store_word_events.len(),
            input.fixed_log2_rows::<F, _>(self),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let chunk_size = std::cmp::max((input.memory_store_word_events.len()) / num_cpus::get(), 1);
        let padded_nb_rows = <StoreWordChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_STORE_WORD_COLUMNS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_STORE_WORD_COLUMNS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_STORE_WORD_COLUMNS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut StoreWordColumns<F> = row.borrow_mut();

                    if idx < input.memory_store_word_events.len() {
                        let event = &input.memory_store_word_events[idx];
                        let instruction = input.program.fetch(event.0.pc);
                        self.event_to_row(&event.0, cols, &mut blu);
                        cols.state.populate(&mut blu, event.0.clk, event.0.pc);
                        cols.adapter.populate(&mut blu, instruction, event.1);
                    }
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_events.iter().collect_vec());

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(values, NUM_STORE_WORD_COLUMNS)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.memory_store_word_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl StoreWordChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MemInstrEvent,
        cols: &mut StoreWordColumns<F>,
        blu: &mut HashMap<ByteLookupEvent, usize>,
    ) {
        // Populate memory accesses for reading from memory.
        cols.memory_access.populate(event.mem_access, blu);

        let memory_addr = cols.address_operation.populate(blu, event.b, event.c);
        let bit = ((memory_addr >> 2) & 1) as u16;
        cols.offset_bit = F::from_canonical_u16(bit);
        cols.store_value = Word::from(event.mem_access.value());

        cols.is_real = F::one();
    }
}

impl<AB> Air<AB> for StoreWordChip
where
    AB: SP1CoreAirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &StoreWordColumns<AB::Var> = (*local).borrow();

        let clk_high = local.state.clk_high::<AB>();
        let clk_low = local.state.clk_low::<AB>();

        let opcode = AB::Expr::from_canonical_u32(Opcode::SW as u32);

        builder.assert_bool(local.is_real);

        // Step 1. Compute the address, and check offsets and address bounds.
        let aligned_addr = AddressOperation::<AB::F>::eval(
            builder,
            local.adapter.b().map(Into::into),
            local.adapter.c().map(Into::into),
            AB::Expr::zero(),
            AB::Expr::zero(),
            local.offset_bit.into(),
            local.is_real.into(),
            local.address_operation,
        );

        // Step 2. Write at the memory address.
        builder.eval_memory_access_write(
            clk_high.clone(),
            clk_low.clone(),
            &aligned_addr.map(Into::into),
            local.memory_access,
            local.store_value,
            local.is_real,
        );

        // Step 3. Constrain the write value.
        let store_limb = local.adapter.prev_a().0;
        builder.assert_eq(
            local.store_value.0[0],
            local.memory_access.prev_value.0[0]
                + (store_limb[0] - local.memory_access.prev_value.0[0])
                    * (AB::Expr::one() - local.offset_bit),
        );
        builder.assert_eq(
            local.store_value.0[1],
            local.memory_access.prev_value.0[1]
                + (store_limb[1] - local.memory_access.prev_value.0[1])
                    * (AB::Expr::one() - local.offset_bit),
        );
        builder.assert_eq(
            local.store_value.0[2],
            local.memory_access.prev_value.0[2]
                + (store_limb[0] - local.memory_access.prev_value.0[2]) * local.offset_bit,
        );
        builder.assert_eq(
            local.store_value.0[3],
            local.memory_access.prev_value.0[3]
                + (store_limb[1] - local.memory_access.prev_value.0[3]) * local.offset_bit,
        );

        // Constrain the state of the CPU.
        CPUState::<AB::F>::eval(
            builder,
            local.state,
            [
                local.state.pc[0] + AB::F::from_canonical_u32(PC_INC),
                local.state.pc[1].into(),
                local.state.pc[2].into(),
            ],
            AB::Expr::from_canonical_u32(CLK_INC),
            local.is_real.into(),
        );

        // Constrain the program and register reads.
        ITypeReader::<AB::F>::eval_op_a_immutable(
            builder,
            clk_high,
            clk_low,
            local.state.pc,
            opcode,
            local.adapter,
            local.is_real.into(),
        );
    }
}
