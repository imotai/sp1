use core::borrow::Borrow;
use slop_air::{Air, BaseAir, PairBuilder};
use slop_algebra::{extension::BinomiallyExtendable, AbstractField, Field, PrimeField32};
use slop_baby_bear::BabyBear;
use slop_matrix::{dense::RowMajorMatrix, Matrix};
use slop_maybe_rayon::prelude::{IndexedParallelIterator, ParallelIterator, ParallelSliceMut};
use sp1_core_machine::utils::next_multiple_of_32;
use sp1_derive::AlignedBorrow;
use sp1_recursion_executor::{
    Address, Block, ExecutionRecord, Instruction, Poseidon2SBoxInstr, Poseidon2SBoxIo,
    RecursionProgram, D,
};
use sp1_stark::air::MachineAir;
use std::{borrow::BorrowMut, iter::zip};

use crate::builder::SP1RecursionAirBuilder;

pub const NUM_SBOX_ENTRIES_PER_ROW: usize = 1;

#[derive(Default)]
pub struct Poseidon2SBoxChip;

pub const NUM_SBOX_COLS: usize = core::mem::size_of::<Poseidon2SBoxCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Poseidon2SBoxCols<F: Copy> {
    pub values: [Poseidon2SBoxValueCols<F>; NUM_SBOX_ENTRIES_PER_ROW],
}
const NUM_SBOX_VALUE_COLS: usize = core::mem::size_of::<Poseidon2SBoxValueCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Poseidon2SBoxValueCols<F: Copy> {
    pub vals: Poseidon2SBoxIo<Block<F>>,
    pub intermediate: Block<F>,
}

pub const NUM_SBOX_PREPROCESSED_COLS: usize =
    core::mem::size_of::<Poseidon2SBoxPreprocessedCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Poseidon2SBoxPreprocessedCols<F: Copy> {
    pub accesses: [Poseidon2SBoxAccessCols<F>; NUM_SBOX_ENTRIES_PER_ROW],
}

pub const NUM_SBOX_ACCESS_COLS: usize = core::mem::size_of::<Poseidon2SBoxAccessCols<u8>>();

#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct Poseidon2SBoxAccessCols<F: Copy> {
    pub addrs: Poseidon2SBoxIo<Address<F>>,
    pub external: F,
    pub internal: F,
}

impl<F: Field> BaseAir<F> for Poseidon2SBoxChip {
    fn width(&self) -> usize {
        NUM_SBOX_COLS
    }
}

impl<F: PrimeField32 + BinomiallyExtendable<D>> MachineAir<F> for Poseidon2SBoxChip {
    type Record = ExecutionRecord<F>;

    type Program = RecursionProgram<F>;

    fn name(&self) -> String {
        "Poseidon2SBox".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        NUM_SBOX_PREPROCESSED_COLS
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, instrs_len: usize) -> Option<usize> {
        let height = program.shape.as_ref().and_then(|shape| shape.height(self));

        let nb_rows = instrs_len.div_ceil(NUM_SBOX_ENTRIES_PER_ROW);
        Some(next_multiple_of_32(nb_rows, height))
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<RowMajorMatrix<F>> {
        assert_eq!(
            std::any::TypeId::of::<F>(),
            std::any::TypeId::of::<BabyBear>(),
            "generate_preprocessed_trace only supports BabyBear field"
        );

        let instrs = unsafe {
            std::mem::transmute::<Vec<&Poseidon2SBoxInstr<F>>, Vec<&Poseidon2SBoxInstr<BabyBear>>>(
                program
                    .inner
                    .iter()
                    .filter_map(|instruction| match instruction.inner() {
                        Instruction::Poseidon2SBox(x) => Some(x),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
            )
        };
        let padded_nb_rows = self.preprocessed_num_rows(program, instrs.len()).unwrap();
        let mut values = vec![BabyBear::zero(); padded_nb_rows * NUM_SBOX_PREPROCESSED_COLS];

        // Generate the trace rows & corresponding records for each chunk of events in parallel.
        let populate_len = instrs.len() * NUM_SBOX_ACCESS_COLS;
        values[..populate_len].par_chunks_mut(NUM_SBOX_ACCESS_COLS).zip_eq(instrs).for_each(
            |(row, instr)| {
                let Poseidon2SBoxInstr { addrs, mults, external } = instr;
                let access: &mut Poseidon2SBoxAccessCols<_> = row.borrow_mut();
                access.addrs = addrs.to_owned();
                assert!(*mults == BabyBear::one());
                if *external {
                    access.external = mults.to_owned();
                    access.internal = BabyBear::zero();
                } else {
                    access.external = BabyBear::zero();
                    access.internal = mults.to_owned();
                }
            },
        );

        // Convert the trace to a row major matrix.
        Some(RowMajorMatrix::new(
            unsafe { std::mem::transmute::<Vec<BabyBear>, Vec<F>>(values) },
            NUM_SBOX_PREPROCESSED_COLS,
        ))
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {
        // This is a no-op.
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let height = input.program.shape.as_ref().and_then(|shape| shape.height(self));
        let events = &input.poseidon2_sbox_events;
        let nb_rows = events.len().div_ceil(NUM_SBOX_ENTRIES_PER_ROW);
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
                &Vec<Poseidon2SBoxIo<Block<F>>>,
                &Vec<Poseidon2SBoxIo<Block<BabyBear>>>,
            >(&input.poseidon2_sbox_events)
        };
        let padded_nb_rows = self.num_rows(input).unwrap();
        let mut values = vec![BabyBear::zero(); padded_nb_rows * NUM_SBOX_COLS];

        // Generate the trace rows & corresponding records for each chunk of events in parallel.
        let populate_len = events.len() * NUM_SBOX_VALUE_COLS;
        values[..populate_len].par_chunks_mut(NUM_SBOX_VALUE_COLS).zip_eq(events).for_each(
            |(row, &vals)| {
                let cols: &mut Poseidon2SBoxValueCols<_> = row.borrow_mut();
                cols.vals = vals.to_owned();
                for i in 0..D {
                    cols.intermediate.0[i] = vals.input.0[i] * vals.input.0[i] * vals.input.0[i];
                    cols.vals.output.0[i] =
                        vals.input.0[i] * cols.intermediate.0[i] * cols.intermediate.0[i];
                }
            },
        );

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(
            unsafe { std::mem::transmute::<Vec<BabyBear>, Vec<F>>(values) },
            NUM_SBOX_COLS,
        )
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl<AB> Air<AB> for Poseidon2SBoxChip
where
    AB: SP1RecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &Poseidon2SBoxCols<AB::Var> = (*local).borrow();
        let prep = builder.preprocessed();
        let prep_local = prep.row_slice(0);
        let prep_local: &Poseidon2SBoxPreprocessedCols<AB::Var> = (*prep_local).borrow();

        for (
            Poseidon2SBoxValueCols { vals, intermediate },
            Poseidon2SBoxAccessCols { addrs, external, internal },
        ) in zip(local.values, prep_local.accesses)
        {
            // First we constrain the x -> x^7 mapping.
            for i in 0..D {
                builder.assert_eq(
                    vals.input.0[i] * vals.input.0[i] * vals.input.0[i],
                    intermediate.0[i],
                );
                builder.assert_eq(
                    vals.input.0[i] * intermediate.0[i] * intermediate.0[i],
                    vals.output.0[i],
                );
            }

            let is_real = external + internal;

            // Read the inputs from memory.
            builder.receive_block(addrs.input, vals.input, is_real.clone());

            // Write the output to memory, in the external SBox case.
            builder.send_block(addrs.output, vals.output, external);

            // Write the output to memory, in the internal SBox case.
            builder.send_block(
                addrs.output,
                Block([vals.output.0[0], vals.input.0[1], vals.input.0[2], vals.input.0[3]]),
                internal,
            );
        }
    }
}
