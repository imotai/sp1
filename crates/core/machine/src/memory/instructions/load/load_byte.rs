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
    operations::AddressOperation,
    utils::{next_multiple_of_32, zeroed_f_vec},
};
use hashbrown::HashMap;
use itertools::Itertools;
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use rayon::iter::{ParallelBridge, ParallelIterator};
use sp1_core_executor::{
    events::{ByteLookupEvent, ByteRecord, MemInstrEvent},
    ByteOpcode, ExecutionRecord, Opcode, Program, DEFAULT_PC_INC,
};
use sp1_primitives::consts::u64_to_u16_limbs;
use sp1_stark::air::MachineAir;

#[derive(Default)]
pub struct LoadByteChip;

pub const NUM_LOAD_BYTE_COLUMNS: usize = size_of::<LoadByteColumns<u8>>();

/// The column layout for memory load byte instructions.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct LoadByteColumns<T> {
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

    /// The selected limb value.
    pub selected_limb: T,

    /// The lower byte of the selected limb.
    pub selected_limb_low_byte: T,

    /// The selected byte value.
    pub selected_byte: T,

    /// The `MSB` of the byte, if the opcode is `LB`.
    pub msb: T,

    /// Whether this is a load byte instruction.
    pub is_lb: T,

    /// Whether this is a load byte unsigned instruction.
    pub is_lbu: T,
}

impl<F> BaseAir<F> for LoadByteChip {
    fn width(&self) -> usize {
        NUM_LOAD_BYTE_COLUMNS
    }
}

impl<F: PrimeField32> MachineAir<F> for LoadByteChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "LoadByte".to_string()
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = next_multiple_of_32(
            input.memory_load_byte_events.len(),
            input.fixed_log2_rows::<F, _>(self),
        );
        Some(nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let chunk_size = std::cmp::max((input.memory_load_byte_events.len()) / num_cpus::get(), 1);
        let padded_nb_rows = <LoadByteChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_LOAD_BYTE_COLUMNS);

        let blu_events = values
            .chunks_mut(chunk_size * NUM_LOAD_BYTE_COLUMNS)
            .enumerate()
            .par_bridge()
            .map(|(i, rows)| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                rows.chunks_mut(NUM_LOAD_BYTE_COLUMNS).enumerate().for_each(|(j, row)| {
                    let idx = i * chunk_size + j;
                    let cols: &mut LoadByteColumns<F> = row.borrow_mut();

                    if idx < input.memory_load_byte_events.len() {
                        let event = &input.memory_load_byte_events[idx];
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
        RowMajorMatrix::new(values, NUM_LOAD_BYTE_COLUMNS)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            !shard.memory_load_byte_events.is_empty()
        }
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl LoadByteChip {
    fn event_to_row<F: PrimeField32>(
        &self,
        event: &MemInstrEvent,
        cols: &mut LoadByteColumns<F>,
        blu: &mut HashMap<ByteLookupEvent, usize>,
    ) {
        // Populate memory accesses for reading from memory.
        cols.memory_access.populate(event.mem_access, blu);

        let memory_addr = cols.address_operation.populate(blu, event.b, event.c);
        let bit0 = (memory_addr & 1) as u16;
        let bit1 = ((memory_addr >> 1) & 1) as u16;
        cols.offset_bit[0] = F::from_canonical_u16(bit0);
        cols.offset_bit[1] = F::from_canonical_u16(bit1);

        let limb = u64_to_u16_limbs(event.mem_access.value())[bit1 as usize];
        cols.selected_limb = F::from_canonical_u16(limb);
        cols.selected_limb_low_byte = F::from_canonical_u16(limb & 0xFF);
        let byte = limb.to_le_bytes()[bit0 as usize];
        cols.selected_byte = F::from_canonical_u8(byte);
        blu.add_u8_range_checks(&limb.to_le_bytes());

        if event.opcode == Opcode::LB {
            cols.is_lb = F::one();
            cols.msb = F::from_canonical_u8(byte >> 7);
            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::MSB,
                a: (byte >> 7) as u16,
                b: byte,
                c: 0,
            });
        } else {
            cols.is_lbu = F::one();
        }
    }
}

impl<AB> Air<AB> for LoadByteChip
where
    AB: SP1CoreAirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &LoadByteColumns<AB::Var> = (*local).borrow();

        let shard = local.state.shard::<AB>();
        let clk = local.state.clk::<AB>();

        // SAFETY: All selectors `is_lb`, `is_lbu` are checked to be boolean.
        // Each "real" row has exactly one selector turned on, as `is_real`, the sum of the
        // selectors, is boolean. Therefore, the `opcode` matches the corresponding opcode.
        let opcode = AB::Expr::from_canonical_u32(Opcode::LB as u32) * local.is_lb
            + AB::Expr::from_canonical_u32(Opcode::LBU as u32) * local.is_lbu;
        let is_real = local.is_lb + local.is_lbu;
        builder.assert_bool(local.is_lb);
        builder.assert_bool(local.is_lbu);
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

        // Step 2. Read the memory address.
        builder.eval_memory_access_read(
            shard.clone(),
            clk.clone(),
            aligned_addr.clone(),
            local.memory_access,
            is_real.clone(),
        );

        // This chip requires `op_a != x0`.
        builder.assert_zero(local.adapter.op_a_0);

        // Step 3. Use the memory value to compute the write value for `op_a`.
        // Select the u16 limb corresponding to the offset.
        builder.assert_eq(
            local.selected_limb,
            local.offset_bit[1] * local.memory_access.prev_value[1]
                + (AB::Expr::one() - local.offset_bit[1]) * local.memory_access.prev_value[0],
        );
        // Split the u16 limb into two bytes.
        let byte0 = local.selected_limb_low_byte;
        let byte1 = (local.selected_limb - byte0) * AB::F::from_canonical_u32(1 << 8).inverse();
        builder.slice_range_check_u8(&[byte0.into(), byte1.clone()], is_real.clone());
        // Select the u8 byte corresponding to the offset.
        builder.assert_eq(
            local.selected_byte,
            local.offset_bit[0] * byte1 + (AB::Expr::one() - local.offset_bit[0]) * byte0,
        );
        // Get the MSB of the selected byte if the opcode is `LB`.
        // If the opcode is `LBU`, the MSB is constrained to be zero.
        builder.when(local.is_lbu).assert_zero(local.msb);
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::MSB as u32),
            local.msb,
            local.selected_byte,
            AB::Expr::zero(),
            local.is_lb,
        );

        // Constrain the state of the CPU.
        CPUState::<AB::F>::eval(
            builder,
            local.state,
            local.state.pc + AB::F::from_canonical_u32(DEFAULT_PC_INC),
            AB::Expr::from_canonical_u32(DEFAULT_PC_INC),
            is_real.clone(),
        );

        // Compute the two limbs of the word to be written to `op_a`.
        let limb0 =
            local.selected_byte + AB::Expr::from_canonical_u32((1 << 16) - (1 << 8)) * local.msb;
        let limb1 = AB::Expr::from_canonical_u32((1 << 16) - 1) * local.msb;
        let limb2 = AB::Expr::from_canonical_u32((1 << 16) - 1) * local.msb;
        let limb3 = AB::Expr::from_canonical_u32((1 << 16) - 1) * local.msb;

        // Constrain the program and register reads.
        ITypeReader::<AB::F>::eval(
            builder,
            shard,
            clk,
            local.state.pc,
            opcode,
            Word([limb0, limb1, limb2, limb3]),
            local.adapter,
            is_real.clone(),
        );
    }
}
