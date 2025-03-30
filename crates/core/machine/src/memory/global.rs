use super::MemoryChipType;
use crate::{
    operations::{AssertLtColsBits, BabyBearBitDecomposition, IsZeroOperation},
    utils::next_power_of_two,
};
use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use sp1_core_executor::events::GlobalInteractionEvent;
use sp1_core_executor::{events::MemoryInitializeFinalizeEvent, ExecutionRecord, Program};
use sp1_derive::AlignedBorrow;
use sp1_stark::{
    air::{AirInteraction, InteractionScope, MachineAir, SP1AirBuilder},
    InteractionKind,
};
use std::{array, iter::once};

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

        memory_events.sort_by_key(|event| event.addr);

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

        let previous_addr_bits = match self.kind {
            MemoryChipType::Initialize => input.public_values.previous_init_addr_bits,
            MemoryChipType::Finalize => input.public_values.previous_finalize_addr_bits,
        };

        memory_events.sort_by_key(|event| event.addr);
        let mut rows: Vec<[F; NUM_MEMORY_INIT_COLS]> = memory_events
            .par_iter()
            .map(|event| {
                let MemoryInitializeFinalizeEvent { addr, value, shard, timestamp, used } =
                    event.to_owned();

                let mut row = [F::zero(); NUM_MEMORY_INIT_COLS];
                let cols: &mut MemoryInitCols<F> = row.as_mut_slice().borrow_mut();
                cols.addr = F::from_canonical_u32(addr);
                cols.addr_bits.populate(addr);
                cols.shard = F::from_canonical_u32(shard);
                cols.timestamp = F::from_canonical_u32(timestamp);
                cols.value = array::from_fn(|i| F::from_canonical_u32((value >> i) & 1));
                cols.is_real = F::from_canonical_u32(used);

                row
            })
            .collect::<Vec<_>>();

        for i in 0..memory_events.len() {
            let addr = memory_events[i].addr;
            let cols: &mut MemoryInitCols<F> = rows[i].as_mut_slice().borrow_mut();
            let prev_addr = if i == 0 {
                previous_addr_bits.iter().enumerate().map(|(j, bit)| bit * (1 << j)).sum::<u32>()
            } else {
                memory_events[i - 1].addr
            };
            if prev_addr == 0 && i != 0 {
                cols.prev_valid = F::zero();
            } else {
                cols.prev_valid = F::one();
            }
            cols.index = F::from_canonical_u32(i as u32);
            cols.prev_addr = F::from_canonical_u32(prev_addr);
            cols.prev_addr_bits.populate(prev_addr);
            cols.is_prev_addr_zero.populate(prev_addr);
            cols.is_index_zero.populate(i as u32);
            if prev_addr != 0 || i != 0 {
                cols.is_comp = F::one();
                let addr_bits: [_; 32] = array::from_fn(|i| (addr >> i) & 1);
                let prev_addr_bits: [_; 32] = array::from_fn(|i| (prev_addr >> i) & 1);
                cols.lt_cols.populate(&prev_addr_bits, &addr_bits);
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
    pub prev_addr: T,

    /// The address of the memory access.
    pub addr: T,

    /// Comparison assertions for address to be strictly increasing.
    pub lt_cols: AssertLtColsBits<T, 32>,

    /// A bit decomposition of `prev_addr`.
    pub prev_addr_bits: BabyBearBitDecomposition<T>,

    /// A bit decomposition of `addr`.
    pub addr_bits: BabyBearBitDecomposition<T>,

    /// The value of the memory access.
    pub value: [T; 32],

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

        // Constrain that `local.is_real` and the value bits are boolean.
        builder.assert_bool(local.is_real);
        for i in 0..32 {
            builder.assert_bool(local.value[i]);
        }

        // Combine the value bits into two u16 limbs.
        let mut limb1 = AB::Expr::zero();
        let mut limb2 = AB::Expr::zero();
        for i in 0..16 {
            limb1 = limb1.clone() + local.value[i].into() * AB::F::from_canonical_u16(1 << i);
            limb2 = limb2.clone() + local.value[i + 16].into() * AB::F::from_canonical_u16(1 << i);
        }
        let value = [limb1, limb2];

        let interaction_kind = match self.kind {
            MemoryChipType::Initialize => InteractionKind::MemoryGlobalInitControl,
            MemoryChipType::Finalize => InteractionKind::MemoryGlobalFinalizeControl,
        };

        // Receive the previous index, address, and validity state.
        builder.receive(
            AirInteraction::new(
                vec![local.index]
                    .into_iter()
                    .chain(local.prev_addr_bits.bits)
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
                    .chain(local.addr_bits.bits.map(Into::into))
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
                        local.addr.into(),
                        value[0].clone(),
                        value[1].clone(),
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
                        local.addr.into(),
                        value[0].clone(),
                        value[1].clone(),
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

        // Canonically decompose the previous address into bits so we can do comparisons.
        BabyBearBitDecomposition::<AB::F>::range_check(
            builder,
            local.prev_addr,
            local.prev_addr_bits,
            local.is_real.into(),
        );

        // Canonically decompose the address into bits so we can do comparisons.
        BabyBearBitDecomposition::<AB::F>::range_check(
            builder,
            local.addr,
            local.addr_bits,
            local.is_real.into(),
        );

        // We assert that `prev_addr < addr` when `prev_addr != 0 or index != 0`.
        IsZeroOperation::<AB::F>::eval(
            builder,
            local.prev_addr.into(),
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
        // If `is_comp = 1`, then `prev_addr < addr` should hold.
        local.lt_cols.eval(
            builder,
            &local.prev_addr_bits.bits,
            &local.addr_bits.bits,
            local.is_comp,
        );
        // If `prev_addr == 0` and `index == 0`, then `addr == 0`, and the `value` should be zero.
        // This forces the initialization of address 0 with value 0.
        // Constraints related to register %x0: Register %x0 should always be 0.
        // See 2.6 Load and Store Instruction on P.18 of the RISC-V spec.
        let is_not_comp = local.is_real - local.is_comp;
        for i in 0..32 {
            builder.when(is_not_comp.clone()).assert_zero(local.addr_bits.bits[i]);
            builder.when(is_not_comp.clone()).assert_zero(local.value[i]);
        }

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
