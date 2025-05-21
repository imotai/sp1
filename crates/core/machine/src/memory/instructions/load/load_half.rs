use p3_air::{Air, AirBuilder, BaseAir};
use p3_matrix::Matrix;
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
    operations::{AddressOperation, U16MSBOperation},
    utils::{next_multiple_of_32, zeroed_f_vec},
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
use sp1_primitives::consts::u64_to_u16_limbs;
use sp1_stark::air::MachineAir;

#[derive(Default)]
pub struct LoadHalfChip;

pub const NUM_LOAD_HALF_COLUMNS: usize = size_of::<LoadHalfColumns<u8>>();

/// The column layout for memory load half instructions.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct LoadHalfColumns<T> {
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

    /// Aligned address. (TODO: u64 will not need later)
    pub aligned_addr: [T; 3],

    /// The selected limb value.
    pub selected_half: T,

    /// The `MSB` of the half, if the opcode is `LH`.
    pub msb: U16MSBOperation<T>,

    /// Whether this is a load half instruction.
    pub is_lh: T,

    /// Whether this is a load half unsigned instruction.
    pub is_lhu: T,
}

impl<F> BaseAir<F> for LoadHalfChip {
    fn width(&self) -> usize {
        NUM_LOAD_HALF_COLUMNS
    }
}

impl<F: PrimeField32> MachineAir<F> for LoadHalfChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "LoadHalf".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.memory_load_half_events.len(),
            input.fixed_log2_rows::<F, _>(self),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let chunk_size = std::cmp::max((input.memory_load_half_events.len()) / num_cpus::get(), 1);
        let padded_nb_rows = <LoadHalfChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_LOAD_HALF_COLUMNS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_LOAD_HALF_COLUMNS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_LOAD_HALF_COLUMNS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut LoadHalfColumns<F> = row.borrow_mut();

                    if idx < input.memory_load_half_events.len() {
                        let event = &input.memory_load_half_events[idx];
                        let instruction = input.program.fetch(event.0.pc);
                        self.event_to_row(&event.0, cols, &mut blu);
                        cols.state.populate(
                            &mut blu,
                            input.public_values.execution_shard as u32,
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
        RowMajorMatrix::new(values, NUM_LOAD_HALF_COLUMNS)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.memory_load_half_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl LoadHalfChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MemInstrEvent,
        cols: &mut LoadHalfColumns<F>,
        blu: &mut HashMap<ByteLookupEvent, usize>,
    ) {
        // Populate memory accesses for reading from memory.
        cols.memory_access.populate(event.mem_access, blu);

        // let memory_addr = cols.address_operation.populate(blu, event.b, event.c);
        let memory_addr = event.b.wrapping_add(event.c);
        debug_assert!(memory_addr % 2 == 0);

        let bit = ((memory_addr >> 2) & 1) as u16;
        let bit_1 = ((memory_addr >> 1) & 1) as u16;
        let limb_number = 2 * bit + bit_1;
        cols.offset_bit = F::from_canonical_u16(bit);
        let limb = u64_to_u16_limbs(event.mem_access.value())[limb_number as usize];
        // let limb_2 = u64_to_u16_limbs(event.mem_access.value())[limb_number as usize + 1];
        cols.selected_half = F::from_canonical_u16(limb);
        let aligned_addr = memory_addr - 4 * bit as u64 - 2 * bit_1 as u64;
        cols.aligned_addr = [
            F::from_canonical_u64(aligned_addr & 0xFFFF),
            F::from_canonical_u64((aligned_addr >> 16) & 0xFFFF),
            F::from_canonical_u64((aligned_addr >> 32) & 0xFFFF),
        ];

        if event.opcode == Opcode::LH {
            cols.is_lh = F::one();
            cols.msb.populate_msb(blu, limb);
        } else {
            cols.is_lhu = F::one();
        }
    }
}

impl<AB> Air<AB> for LoadHalfChip
where
    AB: SP1CoreAirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &LoadHalfColumns<AB::Var> = (*local).borrow();

        let shard = local.state.shard::<AB>();
        let clk = local.state.clk::<AB>();

        // SAFETY: All selectors `is_lh`, `is_lhu` are checked to be boolean.
        // Each "real" row has exactly one selector turned on, as `is_real`, the sum of the
        // selectors, is boolean. Therefore, the `opcode` matches the corresponding opcode.
        let opcode = AB::Expr::from_canonical_u32(Opcode::LH as u32) * local.is_lh
            + AB::Expr::from_canonical_u32(Opcode::LHU as u32) * local.is_lhu;
        let is_real = local.is_lh + local.is_lhu;
        // builder.assert_bool(local.is_lh);
        // builder.assert_bool(local.is_lhu);
        // builder.assert_bool(is_real.clone());

        // // Step 1. Compute the address, and check offsets and address bounds.
        // let aligned_addr = AddressOperation::<AB::F>::eval(
        //     builder,
        //     local.adapter.b().map(Into::into),
        //     local.adapter.c().map(Into::into),
        //     AB::Expr::zero(),
        //     local.offset_bit.into(),
        //     is_real.clone(),
        //     local.address_operation,
        // );

        // Step 2. Read the memory address.
        builder.eval_memory_access_read(
            shard.clone(),
            clk.clone(),
            &local.aligned_addr.map(Into::into),
            local.memory_access,
            is_real.clone(),
        );

        // // This chip requires `op_a != x0`.
        // builder.assert_zero(local.adapter.op_a_0);

        // // Step 3. Use the memory value to compute the write value for `op_a`.
        // // Select the u16 limb corresponding to the offset.
        // builder.assert_eq(
        //     local.selected_limb,
        //     local.offset_bit * local.memory_access.prev_value[1]
        //         + (AB::Expr::one() - local.offset_bit) * local.memory_access.prev_value[0],
        // );
        // // Get the MSB of the selected limb if the opcode is `LH`.
        // // If the opcode is `LHU`, the MSB is constrained to be zero.
        // builder.when(local.is_lhu).assert_zero(local.msb.msb);
        // U16MSBOperation::<AB::F>::eval_msb(
        //     builder,
        //     local.selected_limb.into(),
        //     local.msb,
        //     local.is_lh.into(),
        // );

        // Constrain the state of the CPU.
        CPUState::<AB::F>::eval(
            builder,
            local.state,
            local.state.pc + AB::F::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            is_real.clone(),
        );

        // Constrain the program and register reads.
        ITypeReader::<AB::F>::eval(
            builder,
            shard,
            clk,
            local.state.pc,
            opcode,
            Word([
                local.selected_half.into(),
                AB::Expr::from_canonical_u16(u16::MAX) * local.msb.msb,
                AB::Expr::from_canonical_u16(u16::MAX) * local.msb.msb,
                AB::Expr::from_canonical_u16(u16::MAX) * local.msb.msb,
            ]),
            local.adapter,
            is_real.clone(),
        );
    }
}
