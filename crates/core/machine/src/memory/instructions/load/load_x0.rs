use crate::{
    adapter::{register::i_type::ITypeReader, state::CPUState},
    air::SP1CoreAirBuilder,
    memory::MemoryAccessCols,
    operations::AddressOperation,
    utils::{next_multiple_of_32, zeroed_f_vec},
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use rayon::iter::{ParallelBridge, ParallelIterator};
use sp1_core_executor::{
    events::{ByteLookupEvent, ByteRecord, MemInstrEvent},
    ExecutionRecord, Opcode, Program, DEFAULT_CLK_INC, DEFAULT_PC_INC,
};
use sp1_derive::AlignedBorrow;
use sp1_stark::air::MachineAir;
use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

#[derive(Default)]
pub struct LoadX0Chip;

pub const NUM_LOAD_X0_COLUMNS: usize = size_of::<LoadX0Columns<u8>>();

/// The column layout for memory load instructions with `op_a = x0`.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct LoadX0Columns<T> {
    /// The current shard, timestamp, program counter of the CPU.
    pub state: CPUState<T>,

    /// The adapter to read program and register information.
    pub adapter: ITypeReader<T>,

    /// Instance of `AddressOperation` to constrain the memory address.
    pub address_operation: AddressOperation<T>,

    /// Memory consistency columns for the memory access.
    pub memory_access: MemoryAccessCols<T>,

    /// The bit decomposition of the offset.
    pub offset_bit: [T; 2],

    /// Whether this is a load byte instruction.
    pub is_lb: T,

    /// Whether this is a load byte unsigned instruction.
    pub is_lbu: T,

    /// Whether this is a load half instruction.
    pub is_lh: T,

    /// Whether this is a load half unsigned instruction.
    pub is_lhu: T,

    /// Whether this is a load word instruction.
    pub is_lw: T,
}

impl<F> BaseAir<F> for LoadX0Chip {
    fn width(&self) -> usize {
        NUM_LOAD_X0_COLUMNS
    }
}

impl<F: PrimeField32> MachineAir<F> for LoadX0Chip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "LoadX0".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.memory_load_x0_events.len(),
            input.fixed_log2_rows::<F, _>(self),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let chunk_size = std::cmp::max((input.memory_load_x0_events.len()) / num_cpus::get(), 1);
        let padded_nb_rows = <LoadX0Chip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_LOAD_X0_COLUMNS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_LOAD_X0_COLUMNS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_LOAD_X0_COLUMNS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut LoadX0Columns<F> = row.borrow_mut();

                    if idx < input.memory_load_x0_events.len() {
                        let event = &input.memory_load_x0_events[idx];
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
        RowMajorMatrix::new(values, NUM_LOAD_X0_COLUMNS)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.memory_load_x0_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl LoadX0Chip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MemInstrEvent,
        cols: &mut LoadX0Columns<F>,
        blu: &mut HashMap<ByteLookupEvent, usize>,
    ) {
        // Populate memory accesses for reading from memory.
        cols.memory_access.populate(event.mem_access, blu);

        let memory_addr = cols.address_operation.populate(blu, event.b, event.c);
        let bit0 = (memory_addr & 1) as u16;
        let bit1 = ((memory_addr >> 1) & 1) as u16;
        cols.offset_bit[0] = F::from_canonical_u16(bit0);
        cols.offset_bit[1] = F::from_canonical_u16(bit1);

        cols.is_lb = F::from_bool(event.opcode == Opcode::LB);
        cols.is_lbu = F::from_bool(event.opcode == Opcode::LBU);
        cols.is_lh = F::from_bool(event.opcode == Opcode::LH);
        cols.is_lhu = F::from_bool(event.opcode == Opcode::LHU);
        cols.is_lw = F::from_bool(event.opcode == Opcode::LW);
    }
}

impl<AB> Air<AB> for LoadX0Chip
where
    AB: SP1CoreAirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &LoadX0Columns<AB::Var> = (*local).borrow();

        let clk_high = local.state.clk_high::<AB>();
        let clk_low = local.state.clk_low::<AB>();

        // SAFETY: All selectors `is_lb`, `is_lbu`, `is_lh`, `is_lhu`, `is_lw` are checked to be
        // boolean. Each "real" row has exactly one selector turned on, as `is_real`, the
        // sum of the selectors, is boolean. Therefore, the `opcode` matches the
        // corresponding opcode.
        let opcode = AB::Expr::from_canonical_u32(Opcode::LB as u32) * local.is_lb
            + AB::Expr::from_canonical_u32(Opcode::LBU as u32) * local.is_lbu
            + AB::Expr::from_canonical_u32(Opcode::LH as u32) * local.is_lh
            + AB::Expr::from_canonical_u32(Opcode::LHU as u32) * local.is_lhu
            + AB::Expr::from_canonical_u32(Opcode::LW as u32) * local.is_lw;
        let is_real = local.is_lb + local.is_lbu + local.is_lh + local.is_lhu + local.is_lw;
        builder.assert_bool(local.is_lb);
        builder.assert_bool(local.is_lbu);
        builder.assert_bool(local.is_lh);
        builder.assert_bool(local.is_lhu);
        builder.assert_bool(local.is_lw);
        builder.assert_bool(is_real.clone());

        // Step 1. Compute the address, and check offsets and address bounds.
        let aligned_addr = AddressOperation::<AB::F>::eval(
            builder,
            local.adapter.b().map(Into::into),
            local.adapter.c().map(Into::into),
            local.offset_bit[0].into(),
            local.offset_bit[1].into(),
            is_real.clone(),
            local.address_operation,
        );

        // Check the alignment of the address.
        builder.when(local.is_lw).assert_zero(local.offset_bit[0]);
        builder.when(local.is_lw).assert_zero(local.offset_bit[1]);
        builder.when(local.is_lh + local.is_lhu).assert_zero(local.offset_bit[0]);

        // Step 2. Read the memory address.
        builder.eval_memory_access_read(
            clk_high.clone(),
            clk_low.clone(),
            aligned_addr.clone(),
            local.memory_access,
            is_real.clone(),
        );

        // This chip is specifically for load operations with `op_a = x0`.
        builder.when(is_real.clone()).assert_one(local.adapter.op_a_0);

        // Constrain the state of the CPU.
        CPUState::<AB::F>::eval(
            builder,
            local.state,
            local.state.pc + AB::F::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::from_canonical_u32(DEFAULT_CLK_INC),
            is_real.clone(),
        );

        // Constrain the program and register reads.
        // Since `op_a = x0`, it's immutable.
        ITypeReader::<AB::F>::eval_op_a_immutable(
            builder,
            clk_high,
            clk_low,
            local.state.pc,
            opcode,
            local.adapter,
            is_real.clone(),
        );
    }
}
