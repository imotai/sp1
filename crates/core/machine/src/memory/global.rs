use super::MemoryChipType;
use crate::{
    air::WordAirBuilder,
    operations::{BabyBearWordRangeChecker, IsZeroOperation, LtOperationUnsigned},
    utils::next_power_of_two,
};
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator, ParallelSlice};
use sp1_core_executor::{
    events::{ByteRecord, GlobalInteractionEvent, MemoryInitializeFinalizeEvent},
    ExecutionRecord, Program,
};
use sp1_derive::AlignedBorrow;
use sp1_primitives::consts::u32_to_u16_limbs;
use sp1_stark::{
    air::{AirInteraction, InteractionScope, MachineAir, SP1AirBuilder},
    InteractionKind, Word,
};
use std::iter::once;

/// A memory chip that can initialize or finalize values in memory.
pub struct MemoryGlobalChip {
    pub kind: MemoryChipType,
}

impl MemoryGlobalChip {
    /// Creates a new memory chip with a certain type.
    pub const fn new(kind: MemoryChipType) -> Self {
        Self { kind }
    }
}

impl<F> BaseAir<F> for MemoryGlobalChip {
    fn width(&self) -> usize {
        NUM_MEMORY_INIT_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for MemoryGlobalChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        match self.kind {
            MemoryChipType::Initialize => "MemoryGlobalInit".to_string(),
            MemoryChipType::Finalize => "MemoryGlobalFinalize".to_string(),
        }
    }

    fn generate_dependencies(&self, input: &ExecutionRecord, output: &mut ExecutionRecord) {
        let mut memory_events = match self.kind {
            MemoryChipType::Initialize => input.global_memory_initialize_events.clone(),
            MemoryChipType::Finalize => input.global_memory_finalize_events.clone(),
        };

        let is_receive = match self.kind {
            MemoryChipType::Initialize => false,
            MemoryChipType::Finalize => true,
        };

        match self.kind {
            MemoryChipType::Initialize => {
                output.public_values.global_init_count += memory_events.len() as u32;
            }
            MemoryChipType::Finalize => {
                output.public_values.global_finalize_count += memory_events.len() as u32;
            }
        };

        let previous_addr = match self.kind {
            MemoryChipType::Initialize => input.public_values.previous_init_addr_word,
            MemoryChipType::Finalize => input.public_values.previous_finalize_addr_word,
        };

        memory_events.sort_by_key(|event| event.addr);

        let chunk_size = std::cmp::max(memory_events.len() / num_cpus::get(), 1);
        let indices = (0..memory_events.len()).collect::<Vec<_>>();
        let blu_batches = indices
            .par_chunks(chunk_size)
            .map(|chunk| {
                let mut blu = Vec::new();
                let mut row = [F::zero(); NUM_MEMORY_INIT_COLS];
                let cols: &mut MemoryInitCols<F> = row.as_mut_slice().borrow_mut();
                chunk.iter().for_each(|&i| {
                    let addr = memory_events[i].addr;
                    let value = memory_events[i].value;
                    let prev_addr = if i == 0 { previous_addr } else { memory_events[i - 1].addr };
                    blu.add_u16_range_checks(&u32_to_u16_limbs(value));
                    blu.add_u16_range_checks(&u32_to_u16_limbs(prev_addr));
                    blu.add_u16_range_checks(&u32_to_u16_limbs(addr));
                    cols.prev_addr_range_checker.populate(Word::from(prev_addr), &mut blu);
                    cols.addr_range_checker.populate(Word::from(addr), &mut blu);
                    if i != 0 || prev_addr != 0 {
                        cols.lt_cols.populate_unsigned(&mut blu, 1, prev_addr, addr, true);
                    }
                });
                blu
            })
            .collect::<Vec<_>>();
        output.add_byte_lookup_events(blu_batches.into_iter().flatten().collect());

        let events = memory_events.into_iter().map(|event| {
            let interaction_shard = if is_receive { event.shard } else { 0 };
            let interaction_clk = if is_receive { event.timestamp } else { 0 };
            GlobalInteractionEvent {
                message: [
                    interaction_shard,
                    interaction_clk,
                    event.addr,
                    (event.value & 0xFFFF) as u32,
                    (event.value >> 16) as u32,
                    0,
                    0,
                ],
                is_receive,
                kind: InteractionKind::Memory as u8,
            }
        });
        output.global_interaction_events.extend(events);
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = match self.kind {
            MemoryChipType::Initialize => &input.global_memory_initialize_events,
            MemoryChipType::Finalize => &input.global_memory_finalize_events,
        };
        let nb_rows = events.len();
        let size_log2 = input.fixed_log2_rows::<F, Self>(self);
        let padded_nb_rows = next_power_of_two(nb_rows, size_log2);
        Some(padded_nb_rows)
    }

    fn generate_trace(
        &self,
        input: &ExecutionRecord,
        _output: &mut ExecutionRecord,
    ) -> RowMajorMatrix<F> {
        let mut memory_events = match self.kind {
            MemoryChipType::Initialize => input.global_memory_initialize_events.clone(),
            MemoryChipType::Finalize => input.global_memory_finalize_events.clone(),
        };

        let previous_addr = match self.kind {
            MemoryChipType::Initialize => input.public_values.previous_init_addr_word,
            MemoryChipType::Finalize => input.public_values.previous_finalize_addr_word,
        };

        memory_events.sort_by_key(|event| event.addr);
        let mut rows: Vec<[F; NUM_MEMORY_INIT_COLS]> = memory_events
            .par_iter()
            .map(|event| {
                let MemoryInitializeFinalizeEvent { addr, value, shard, timestamp, used } =
                    event.to_owned();

                let mut blu = vec![];
                let mut row = [F::zero(); NUM_MEMORY_INIT_COLS];
                let cols: &mut MemoryInitCols<F> = row.as_mut_slice().borrow_mut();
                cols.addr = Word::from(addr);
                cols.addr_range_checker.populate(cols.addr, &mut blu);
                cols.shard = F::from_canonical_u32(shard);
                cols.timestamp = F::from_canonical_u32(timestamp);
                cols.value = Word::from(value);
                cols.is_real = F::from_canonical_u32(used);
                row
            })
            .collect::<Vec<_>>();

        let mut blu = vec![];
        for i in 0..memory_events.len() {
            let addr = memory_events[i].addr;
            let cols: &mut MemoryInitCols<F> = rows[i].as_mut_slice().borrow_mut();
            let prev_addr = if i == 0 { previous_addr } else { memory_events[i - 1].addr };
            if prev_addr == 0 && i != 0 {
                cols.prev_valid = F::zero();
            } else {
                cols.prev_valid = F::one();
            }
            cols.index = F::from_canonical_u32(i as u32);
            cols.prev_addr = Word::from(prev_addr);
            cols.prev_addr_range_checker.populate(cols.prev_addr, &mut blu);
            cols.is_prev_addr_zero.populate(prev_addr);
            cols.is_index_zero.populate(i as u32);
            if prev_addr != 0 || i != 0 {
                cols.is_comp = F::one();
                cols.lt_cols.populate_unsigned(&mut blu, 1, prev_addr, addr, true);
            }
        }

        // Pad the trace to a power of two depending on the proof shape in `input`.
        rows.resize(
            <MemoryGlobalChip as MachineAir<F>>::num_rows(self, input).unwrap(),
            [F::zero(); NUM_MEMORY_INIT_COLS],
        );

        RowMajorMatrix::new(rows.into_iter().flatten().collect::<Vec<_>>(), NUM_MEMORY_INIT_COLS)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        if let Some(shape) = shard.shape.as_ref() {
            shape.included::<F, _>(self)
        } else {
            match self.kind {
                MemoryChipType::Initialize => !shard.global_memory_initialize_events.is_empty(),
                MemoryChipType::Finalize => !shard.global_memory_finalize_events.is_empty(),
            }
        }
    }

    fn local_only(&self) -> bool {
        true
    }

    fn commit_scope(&self) -> InteractionScope {
        InteractionScope::Local
    }
}

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct MemoryInitCols<T: Copy> {
    /// The shard number of the memory access.
    pub shard: T,

    /// The timestamp of the memory access.
    pub timestamp: T,

    /// The index of the memory access.
    pub index: T,

    /// The address of the previous memory access.
    pub prev_addr: Word<T>,

    /// The address of the memory access.
    pub addr: Word<T>,

    /// Comparison assertions for address to be strictly increasing.
    pub lt_cols: LtOperationUnsigned<T>,

    /// The range checker for `prev_addr`.
    pub prev_addr_range_checker: BabyBearWordRangeChecker<T>,

    /// The range checker for `addr`.
    pub addr_range_checker: BabyBearWordRangeChecker<T>,

    /// The value of the memory access.
    pub value: Word<T>,

    /// Whether the memory access is a real access.
    pub is_real: T,

    /// Whether or not we are making the assertion `prev_addr < addr`.
    pub is_comp: T,

    /// The validity of previous state.
    /// The unique invalid state is when the chip only initializes address 0 once.
    pub prev_valid: T,

    /// A witness to assert whether or not `prev_addr` is zero.
    pub is_prev_addr_zero: IsZeroOperation<T>,

    /// A witness to assert whether or not the index is zero.
    pub is_index_zero: IsZeroOperation<T>,
}

pub(crate) const NUM_MEMORY_INIT_COLS: usize = size_of::<MemoryInitCols<u8>>();

impl<AB> Air<AB> for MemoryGlobalChip
where
    AB: SP1AirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &MemoryInitCols<AB::Var> = (*local).borrow();

        // Constrain that `local.is_real` is boolean.
        builder.assert_bool(local.is_real);
        // Constrain that the value is a valid `Word`.
        builder.slice_range_check_u16(&local.value.0, local.is_real);
        // Constrain that the previous address is a valid `Word`.
        builder.slice_range_check_u16(&local.prev_addr.0, local.is_real);
        // Constrain that the address is a valid `Word`.
        builder.slice_range_check_u16(&local.addr.0, local.is_real);

        let interaction_kind = match self.kind {
            MemoryChipType::Initialize => InteractionKind::MemoryGlobalInitControl,
            MemoryChipType::Finalize => InteractionKind::MemoryGlobalFinalizeControl,
        };

        // Receive the previous index, address, and validity state.
        builder.receive(
            AirInteraction::new(
                vec![local.index]
                    .into_iter()
                    .chain(local.prev_addr.0)
                    .chain(once(local.prev_valid))
                    .map(Into::into)
                    .collect(),
                local.is_real.into(),
                interaction_kind,
            ),
            InteractionScope::Local,
        );

        // Send the next index, address, and validity state.
        builder.send(
            AirInteraction::new(
                vec![local.index + AB::Expr::one()]
                    .into_iter()
                    .chain(local.addr.0.map(Into::into))
                    .chain(once(local.is_comp.into()))
                    .collect(),
                local.is_real.into(),
                interaction_kind,
            ),
            InteractionScope::Local,
        );

        if self.kind == MemoryChipType::Initialize {
            // Send the "send interaction" to the global table.
            builder.send(
                AirInteraction::new(
                    vec![
                        AB::Expr::zero(),
                        AB::Expr::zero(),
                        local.addr.reduce::<AB>(),
                        local.value.0[0].into(),
                        local.value.0[1].into(),
                        AB::Expr::zero(),
                        AB::Expr::zero(),
                        AB::Expr::one(),
                        AB::Expr::zero(),
                        AB::Expr::from_canonical_u8(InteractionKind::Memory as u8),
                    ],
                    local.is_real.into(),
                    InteractionKind::Global,
                ),
                InteractionScope::Local,
            );
        } else {
            // Send the "receive interaction" to the global table.
            builder.send(
                AirInteraction::new(
                    vec![
                        local.shard.into(),
                        local.timestamp.into(),
                        local.addr.reduce::<AB>(),
                        local.value.0[0].into(),
                        local.value.0[1].into(),
                        AB::Expr::zero(),
                        AB::Expr::zero(),
                        AB::Expr::zero(),
                        AB::Expr::one(),
                        AB::Expr::from_canonical_u8(InteractionKind::Memory as u8),
                    ],
                    local.is_real.into(),
                    InteractionKind::Global,
                ),
                InteractionScope::Local,
            );
        }

        // Check that the previous address is a valid BabyBear word.
        BabyBearWordRangeChecker::<AB::F>::range_check(
            builder,
            local.prev_addr,
            local.prev_addr_range_checker,
            local.is_real.into(),
        );

        // Check that the address is a valid BabyBear word.
        BabyBearWordRangeChecker::<AB::F>::range_check(
            builder,
            local.addr,
            local.addr_range_checker,
            local.is_real.into(),
        );

        // Assert that `prev_addr < addr` when `prev_addr != 0 or index != 0`.
        IsZeroOperation::<AB::F>::eval(
            builder,
            local.prev_addr.reduce::<AB>(),
            local.is_prev_addr_zero,
            local.is_real.into(),
        );
        IsZeroOperation::<AB::F>::eval(
            builder,
            local.index.into(),
            local.is_index_zero,
            local.is_real.into(),
        );

        // Comparison will be done unless `prev_addr == 0` and `index == 0`.
        // If `is_real = 0`, then `is_comp` will be zero.
        // If `is_real = 1`, then `is_comp` will be zero when `prev_addr == 0` and `index == 0`.
        // If `is_real = 1`, then `is_comp` will be one when `prev_addr != 0` or `index != 0`.
        builder.assert_eq(
            local.is_comp,
            local.is_real
                * (AB::Expr::one() - local.is_prev_addr_zero.result * local.is_index_zero.result),
        );
        builder.assert_bool(local.is_comp);
        // If `is_comp = 1`, then `prev_addr < addr` should hold.
        LtOperationUnsigned::<AB::F>::eval_lt_unsigned(
            builder,
            local.prev_addr.map(Into::into),
            local.addr.map(Into::into),
            local.lt_cols,
            local.is_comp.into(),
        );
        builder.when(local.is_comp).assert_one(local.lt_cols.u16_compare_operation.bit);

        // If `prev_addr == 0` and `index == 0`, then `addr == 0`, and the `value` should be zero.
        // This forces the initialization of address 0 with value 0.
        // Constraints related to register %x0: Register %x0 should always be 0.
        // See 2.6 Load and Store Instruction on P.18 of the RISC-V spec.
        let is_not_comp = local.is_real - local.is_comp;
        builder.when(is_not_comp.clone()).assert_word_zero(local.addr);
        builder.when(is_not_comp.clone()).assert_word_zero(local.value);

        // Make assertions for specific types of memory chips.
        if self.kind == MemoryChipType::Initialize {
            builder.when(local.is_real).assert_eq(local.timestamp, AB::F::one());
        }
    }
}

// #[cfg(test)]
// mod tests {
//     #![allow(clippy::print_stdout)]

//     use super::*;
//     use crate::programs::tests::*;
//     use crate::{
//         riscv::RiscvAir, syscall::precompiles::sha256::extend_tests::sha_extend_program,
//         utils::setup_logger,
//     };
//     use p3_baby_bear::BabyBear;
//     use sp1_core_executor::{Executor, Trace};
//     use sp1_stark::InteractionKind;
//     use sp1_stark::{
//         baby_bear_poseidon2::BabyBearPoseidon2, debug_interactions_with_all_chips, SP1CoreOpts,
//         StarkMachine,
//     };

//     #[test]
//     fn test_memory_generate_trace() {
//         let program = simple_program();
//         let mut runtime = Executor::new(program, SP1CoreOpts::default());
//         runtime.run::<Trace>().unwrap();
//         let shard = runtime.record.clone();

//         let chip: MemoryGlobalChip = MemoryGlobalChip::new(MemoryChipType::Initialize);

//         let trace: RowMajorMatrix<BabyBear> =
//             chip.generate_trace(&shard, &mut ExecutionRecord::default());
//         println!("{:?}", trace.values);

//         let chip: MemoryGlobalChip = MemoryGlobalChip::new(MemoryChipType::Finalize);
//         let trace: RowMajorMatrix<BabyBear> =
//             chip.generate_trace(&shard, &mut ExecutionRecord::default());
//         println!("{:?}", trace.values);

//         for mem_event in shard.global_memory_finalize_events {
//             println!("{:?}", mem_event);
//         }
//     }

//     #[test]
//     fn test_memory_lookup_interactions() {
//         setup_logger();
//         let program = sha_extend_program();
//         let program_clone = program.clone();
//         let mut runtime = Executor::new(program, SP1CoreOpts::default());
//         runtime.run::<Trace>().unwrap();
//         let machine: StarkMachine<BabyBearPoseidon2, RiscvAir<BabyBear>> =
//             RiscvAir::machine(BabyBearPoseidon2::new());
//         let (pkey, _) = machine.setup(&program_clone);
//         let opts = SP1CoreOpts::default();
//         machine.generate_dependencies(
//             &mut runtime.records.clone().into_iter().map(|r| *r).collect::<Vec<_>>(),
//             &opts,
//             None,
//         );

//         let shards = runtime.records;
//         for shard in shards.clone() {
//             debug_interactions_with_all_chips::<BabyBearPoseidon2, RiscvAir<BabyBear>>(
//                 &machine,
//                 &pkey,
//                 &[*shard],
//                 vec![InteractionKind::Memory],
//                 InteractionScope::Local,
//             );
//         }
//         debug_interactions_with_all_chips::<BabyBearPoseidon2, RiscvAir<BabyBear>>(
//             &machine,
//             &pkey,
//             &shards.into_iter().map(|r| *r).collect::<Vec<_>>(),
//             vec![InteractionKind::Memory],
//             InteractionScope::Global,
//         );
//     }

//     #[test]
//     fn test_byte_lookup_interactions() {
//         setup_logger();
//         let program = sha_extend_program();
//         let program_clone = program.clone();
//         let mut runtime = Executor::new(program, SP1CoreOpts::default());
//         runtime.run::<Trace>().unwrap();
//         let machine = RiscvAir::machine(BabyBearPoseidon2::new());
//         let (pkey, _) = machine.setup(&program_clone);
//         let opts = SP1CoreOpts::default();
//         machine.generate_dependencies(
//             &mut runtime.records.clone().into_iter().map(|r| *r).collect::<Vec<_>>(),
//             &opts,
//             None,
//         );

//         let shards = runtime.records;
//         debug_interactions_with_all_chips::<BabyBearPoseidon2, RiscvAir<BabyBear>>(
//             &machine,
//             &pkey,
//             &shards.into_iter().map(|r| *r).collect::<Vec<_>>(),
//             vec![InteractionKind::Byte],
//             InteractionScope::Global,
//         );
//     }
// }
