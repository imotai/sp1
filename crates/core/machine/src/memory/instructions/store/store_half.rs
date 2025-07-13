use slop_air::{Air, BaseAir};
use slop_matrix::Matrix;
use sp1_derive::AlignedBorrow;
use sp1_stark::Word;
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
use rayon::iter::{ParallelBridge, ParallelIterator};
use slop_algebra::{AbstractField, PrimeField32};
use slop_matrix::dense::RowMajorMatrix;
use sp1_core_executor::{
    events::{ByteLookupEvent, ByteRecord, MemInstrEvent},
    ExecutionRecord, Opcode, Program, DEFAULT_CLK_INC, DEFAULT_PC_INC,
};
use sp1_stark::air::MachineAir;

#[derive(Default)]
pub struct StoreHalfChip;

pub const NUM_STORE_HALF_COLUMNS: usize = size_of::<StoreHalfColumns<u8>>();

/// The column layout for memory store half instructions.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct StoreHalfColumns<T> {
    /// The current shard, timestamp, program counter of the CPU.
    pub state: CPUState<T>,

    /// The adapter to read program and register information.
    pub adapter: ITypeReader<T>,

    /// Instance of `AddressOperation` to constrain the memory address.
    pub address_operation: AddressOperation<T>,

    /// Memory consistency columns for the memory access.
    pub memory_access: MemoryAccessCols<T>,

    /// Whether or not the offset is `0` or `2`.
    pub offset_bit: T,

    /// The store value.
    pub store_value: Word<T>,

    /// Whether this is a store half instruction.
    pub is_real: T,
}

impl<F> BaseAir<F> for StoreHalfChip {
    fn width(&self) -> usize {
        NUM_STORE_HALF_COLUMNS
    }
}

impl<F: PrimeField32> MachineAir<F> for StoreHalfChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "StoreHalf".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.memory_store_half_events.len(),
            input.fixed_log2_rows::<F, _>(self),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let chunk_size = std::cmp::max((input.memory_store_half_events.len()) / num_cpus::get(), 1);
        let padded_nb_rows = <StoreHalfChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_STORE_HALF_COLUMNS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_STORE_HALF_COLUMNS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_STORE_HALF_COLUMNS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut StoreHalfColumns<F> = row.borrow_mut();

                    if idx < input.memory_store_half_events.len() {
                        let event = &input.memory_store_half_events[idx];
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
        RowMajorMatrix::new(values, NUM_STORE_HALF_COLUMNS)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.memory_store_half_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl StoreHalfChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MemInstrEvent,
        cols: &mut StoreHalfColumns<F>,
        blu: &mut HashMap<ByteLookupEvent, usize>,
    ) {
        // Populate memory accesses for reading from memory.
        cols.memory_access.populate(event.mem_access, blu);

        let memory_addr = cols.address_operation.populate(blu, event.b, event.c);
        debug_assert!(memory_addr.is_multiple_of(2));

        let bit = ((memory_addr >> 1) & 1) as u16;
        cols.offset_bit = F::from_canonical_u16(bit);
        cols.store_value = Word::from(event.mem_access.value());
        cols.is_real = F::one();
    }
}

impl<AB> Air<AB> for StoreHalfChip
where
    AB: SP1CoreAirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &StoreHalfColumns<AB::Var> = (*local).borrow();

        let clk_high = local.state.clk_high::<AB>();
        let clk_low = local.state.clk_low::<AB>();

        let opcode = AB::Expr::from_canonical_u32(Opcode::SH as u32);
        builder.assert_bool(local.is_real);

        // Step 1. Compute the address, and check offsets and address bounds.
        let aligned_addr = AddressOperation::<AB::F>::eval(
            builder,
            local.adapter.b().map(Into::into),
            local.adapter.c().map(Into::into),
            AB::Expr::zero(),
            local.offset_bit.into(),
            local.is_real.into(),
            local.address_operation,
        );

        // Step 2. Write the memory address.
        // The `store_value` will be constrained in Step 3.
        builder.eval_memory_access_write(
            clk_high.clone(),
            clk_low.clone(),
            aligned_addr.clone(),
            local.memory_access,
            local.store_value,
            local.is_real.into(),
        );

        // Step 3. Use the memory value to compute the write value.
        let store_limb = local.adapter.prev_a().0[0];
        builder.assert_eq(
            local.store_value.0[0],
            local.memory_access.prev_value.0[0] * local.offset_bit
                + store_limb * (AB::Expr::one() - local.offset_bit),
        );
        builder.assert_eq(
            local.store_value.0[1],
            local.memory_access.prev_value.0[1] * (AB::Expr::one() - local.offset_bit)
                + store_limb * local.offset_bit,
        );

        // Constrain the state of the CPU.
        CPUState::<AB::F>::eval(
            builder,
            local.state,
            local.state.pc + AB::F::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::from_canonical_u32(DEFAULT_CLK_INC),
            local.is_real.into(),
        );

        // Constrain the program and register reads.
        ITypeReader::<AB::F>::eval_op_a_immutable(
            builder,
            clk_high.clone(),
            clk_low.clone(),
            local.state.pc,
            opcode,
            local.adapter,
            local.is_real.into(),
        );
    }
}
