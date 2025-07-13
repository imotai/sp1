use core::borrow::Borrow;
use slop_air::{Air, BaseAir, PairBuilder};
use slop_algebra::{extension::BinomiallyExtendable, AbstractField, Field, PrimeField32};
use slop_baby_bear::BabyBear;
use slop_matrix::{dense::RowMajorMatrix, Matrix};
use slop_maybe_rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
use sp1_core_machine::{
    operations::poseidon2::air::{external_linear_layer_mut, internal_linear_layer_mut},
    utils::next_multiple_of_32,
};
use sp1_derive::AlignedBorrow;
use sp1_recursion_executor::{
    Address, Block, ExecutionRecord, Instruction, Poseidon2LinearLayerInstr,
    Poseidon2LinearLayerIo, RecursionProgram, D, PERMUTATION_WIDTH,
};
use sp1_stark::air::MachineAir;
use std::{borrow::BorrowMut, iter::zip};

use crate::builder::SP1RecursionAirBuilder;

pub const NUM_LINEAR_ENTRIES_PER_ROW: usize = 1;

#[derive(Default)]
pub struct Poseidon2LinearLayerChip;

pub const NUM_LINEAR_COLS: usize = core::mem::size_of::<Poseidon2LinearLayerCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Poseidon2LinearLayerCols<F: Copy> {
    pub values: [Poseidon2LinearLayerValueCols<F>; NUM_LINEAR_ENTRIES_PER_ROW],
}
const NUM_LINEAR_VALUE_COLS: usize = core::mem::size_of::<Poseidon2LinearLayerValueCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Poseidon2LinearLayerValueCols<F: Copy> {
    pub input: [Block<F>; 4],
}

pub const NUM_LINEAR_PREPROCESSED_COLS: usize =
    core::mem::size_of::<Poseidon2LinearLayerPreprocessedCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Poseidon2LinearLayerPreprocessedCols<F: Copy> {
    pub accesses: [Poseidon2LinearLayerAccessCols<F>; NUM_LINEAR_ENTRIES_PER_ROW],
}

pub const NUM_LINEAR_ACCESS_COLS: usize =
    core::mem::size_of::<Poseidon2LinearLayerAccessCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Poseidon2LinearLayerAccessCols<F: Copy> {
    pub addrs: Poseidon2LinearLayerIo<Address<F>>,
    pub external: F,
    pub internal: F,
}

impl<F: Field> BaseAir<F> for Poseidon2LinearLayerChip {
    fn width(&self) -> usize {
        NUM_LINEAR_COLS
    }
}

impl<F: PrimeField32 + BinomiallyExtendable<D>> MachineAir<F> for Poseidon2LinearLayerChip {
    type Record = ExecutionRecord<F>;

    type Program = RecursionProgram<F>;

    fn name(&self) -> String {
        "Poseidon2LinearLayer".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_LINEAR_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        let height = program.shape.as_ref().and_then(|shape| shape.height(self));

        let nb_rows = instrs_len.div_ceil(NUM_LINEAR_ENTRIES_PER_ROW);
        Some(next_multiple_of_32(nb_rows, height))
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<RowMajorMatrix<F>> {
        assert_eq!(
            std::any::TypeId::of::<F>(),
            std::any::TypeId::of::<BabyBear>(),
            "generate_preprocessed_trace only supports BabyBear field"
        );

        let instrs = unsafe {
            std::mem::transmute::<
                Vec<&Poseidon2LinearLayerInstr<F>>,
                Vec<&Poseidon2LinearLayerInstr<BabyBear>>,
            >(
                program
                    .inner
                    .iter()
                    .filter_map(|instruction| match instruction.inner() {
                        Instruction::Poseidon2LinearLayer(x) => Some(x.as_ref()),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
            )
        };
        let padded_nb_rows = self.preprocessed_num_rows(program, instrs.len()).unwrap();
        let mut values = vec![BabyBear::zero(); padded_nb_rows * NUM_LINEAR_PREPROCESSED_COLS];

        // Generate the trace rows & corresponding records for each chunk of events in parallel.
        let populate_len = instrs.len() * NUM_LINEAR_ACCESS_COLS;
        values[..populate_len].par_chunks_mut(NUM_LINEAR_ACCESS_COLS).zip_eq(instrs).for_each(
            |(row, instr)| {
                let Poseidon2LinearLayerInstr { addrs, mults, external } = instr;
                let access: &mut Poseidon2LinearLayerAccessCols<_> = row.borrow_mut();
                access.addrs = addrs.to_owned();
                #[allow(clippy::needless_range_loop)]
                for i in 0..PERMUTATION_WIDTH / D {
                    assert!(mults[i] == BabyBear::one());
                }
                if *external {
                    access.external = BabyBear::one();
                    access.internal = BabyBear::zero();
                } else {
                    access.external = BabyBear::zero();
                    access.internal = BabyBear::one();
                }
            },
        );

        // Convert the trace to a row major matrix.
        Some(RowMajorMatrix::new(
            unsafe { std::mem::transmute::<Vec<BabyBear>, Vec<F>>(values) },
            NUM_LINEAR_PREPROCESSED_COLS,
        ))
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {
        // This is a no-op.
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let height = input.program.shape.as_ref().and_then(|shape| shape.height(self));
        let events = &input.poseidon2_linear_layer_events;
        let nb_rows = events.len().div_ceil(NUM_LINEAR_ENTRIES_PER_ROW);
        Some(next_multiple_of_32(nb_rows, height))
    }

    fn generate_trace(&self, input: &Self::Record, _: &mut Self::Record) -> RowMajorMatrix<F> {
        assert_eq!(
            std::any::TypeId::of::<F>(),
            std::any::TypeId::of::<BabyBear>(),
            "generate_trace only supports BabyBear field"
        );

        let events = unsafe {
            std::mem::transmute::<
                &Vec<Poseidon2LinearLayerIo<Block<F>>>,
                &Vec<Poseidon2LinearLayerIo<Block<BabyBear>>>,
            >(&input.poseidon2_linear_layer_events)
        };
        let padded_nb_rows = self.num_rows(input).unwrap();
        let mut values = vec![BabyBear::zero(); padded_nb_rows * NUM_LINEAR_COLS];

        // Generate the trace rows & corresponding records for each chunk of events in parallel.
        let populate_len = events.len() * NUM_LINEAR_VALUE_COLS;
        values[..populate_len].par_chunks_mut(NUM_LINEAR_VALUE_COLS).zip_eq(events).for_each(
            |(row, &vals)| {
                let cols: &mut Poseidon2LinearLayerValueCols<_> = row.borrow_mut();
                cols.input = vals.input.to_owned();
            },
        );

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(
            unsafe { std::mem::transmute::<Vec<BabyBear>, Vec<F>>(values) },
            NUM_LINEAR_COLS,
        )
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl<AB> Air<AB> for Poseidon2LinearLayerChip
where
    AB: SP1RecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &Poseidon2LinearLayerCols<AB::Var> = (*local).borrow();
        let prep = builder.preprocessed();
        let prep_local = prep.row_slice(0);
        let prep_local: &Poseidon2LinearLayerPreprocessedCols<AB::Var> = (*prep_local).borrow();

        for (
            Poseidon2LinearLayerValueCols { input },
            Poseidon2LinearLayerAccessCols { addrs, external, internal },
        ) in zip(local.values, prep_local.accesses)
        {
            let is_real = external + internal;

            // Read the inputs from memory.
            #[allow(clippy::needless_range_loop)]
            for i in 0..PERMUTATION_WIDTH / D {
                builder.receive_block(addrs.input[i], input[i], is_real.clone());
            }

            let mut state_external: [_; PERMUTATION_WIDTH] =
                core::array::from_fn(|_| AB::Expr::zero());
            let mut state_internal: [_; PERMUTATION_WIDTH] =
                core::array::from_fn(|_| AB::Expr::zero());
            for i in 0..PERMUTATION_WIDTH / D {
                for j in 0..D {
                    state_external[i * D + j] = input[i].0[j].into();
                    state_internal[i * D + j] = input[i].0[j].into();
                }
            }
            external_linear_layer_mut(&mut state_external);
            internal_linear_layer_mut(&mut state_internal);

            // Write the output to memory, in the external linear layer case.
            for i in 0..PERMUTATION_WIDTH / D {
                builder.send_block(
                    Address(addrs.output[i].0.into()),
                    Block([
                        state_external[i * D].clone(),
                        state_external[i * D + 1].clone(),
                        state_external[i * D + 2].clone(),
                        state_external[i * D + 3].clone(),
                    ]),
                    external,
                );
                builder.send_block(
                    Address(addrs.output[i].0.into()),
                    Block([
                        state_internal[i * D].clone(),
                        state_internal[i * D + 1].clone(),
                        state_internal[i * D + 2].clone(),
                        state_internal[i * D + 3].clone(),
                    ]),
                    internal,
                );
            }
        }
    }
}
