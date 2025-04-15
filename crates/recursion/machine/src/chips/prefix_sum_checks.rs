use crate::builder::SP1RecursionAirBuilder;
use core::borrow::Borrow;
use itertools::Itertools;
use p3_air::{Air, BaseAir, PairBuilder};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use slop_algebra::{AbstractField, PrimeField32};
use slop_baby_bear::BabyBear;
use sp1_core_machine::utils::{next_power_of_two, pad_rows_fixed};
use sp1_derive::AlignedBorrow;
use sp1_recursion_executor::{
    Address, Block, ExecutionRecord, Instruction, PrefixSumChecksEvent, PrefixSumChecksInstr,
    RecursionProgram,
};
use sp1_stark::air::{BinomialExtension, MachineAir};

use std::borrow::BorrowMut;
use tracing::instrument;

pub const NUM_PREFIX_SUM_CHECKS_COLS: usize = core::mem::size_of::<PrefixSumChecksCols<u8>>();
pub const NUM_PREFIX_SUM_CHECKS_PREPROCESSED_COLS: usize =
    core::mem::size_of::<PrefixSumChecksPreprocessedCols<u8>>();

#[derive(Clone, Debug, Copy, Default)]
pub struct PrefixSumChecksChip;

/// The main columns for a prefix-sum-checks invocation.
#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct PrefixSumChecksCols<T: Copy> {
    pub x1: T,
    pub x2: Block<T>,
    pub prod: Block<T>,
    pub acc: Block<T>,
    pub new_acc: Block<T>,
    pub felt_acc: T,
    pub felt_new_acc: T,
}

#[derive(AlignedBorrow, Clone, Copy, Debug)]
#[repr(C)]
pub struct PrefixSumChecksPreprocessedCols<T: Copy> {
    pub x1_mem: Address<T>,
    pub x2_mem: Address<T>,
    pub acc_addr: Address<T>,
    pub next_acc_addr: Address<T>,
    pub next_acc_mult: T,
    pub felt_acc_addr: Address<T>,
    pub felt_next_acc_addr: Address<T>,
    pub felt_next_acc_mult: T,
    pub is_real: T,
}

impl<F> BaseAir<F> for PrefixSumChecksChip {
    fn width(&self) -> usize {
        NUM_PREFIX_SUM_CHECKS_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for PrefixSumChecksChip {
    type Record = ExecutionRecord<F>;

    type Program = RecursionProgram<F>;

    fn name(&self) -> String {
        "PrefixSumChecks".to_string()
    }

    fn generate_dependencies(&self, _: &Self::Record, _: &mut Self::Record) {
        // This is a no-op.
    }

    fn preprocessed_width(&self) -> usize {
        NUM_PREFIX_SUM_CHECKS_PREPROCESSED_COLS
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<RowMajorMatrix<F>> {
        assert_eq!(
            std::any::TypeId::of::<F>(),
            std::any::TypeId::of::<BabyBear>(),
            "generate_preprocessed_trace only supports BabyBear field"
        );

        let instrs = unsafe {
            std::mem::transmute::<
                Vec<&Box<PrefixSumChecksInstr<F>>>,
                Vec<&Box<PrefixSumChecksInstr<BabyBear>>>,
            >(
                program
                    .inner
                    .iter()
                    .filter_map(|instruction| match instruction {
                        Instruction::PrefixSumChecks(x) => Some(x),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
            )
        };

        let mut rows: Vec<[BabyBear; NUM_PREFIX_SUM_CHECKS_PREPROCESSED_COLS]> = Vec::new();

        instrs.iter().for_each(|instruction| {
            let PrefixSumChecksInstr { addrs, acc_mults, field_acc_mults } = instruction.as_ref();
            let len = addrs.x1.len();
            let mut row_add =
                vec![[BabyBear::zero(); NUM_PREFIX_SUM_CHECKS_PREPROCESSED_COLS]; len];
            row_add.iter_mut().enumerate().for_each(|(i, row)| {
                let cols: &mut PrefixSumChecksPreprocessedCols<BabyBear> =
                    row.as_mut_slice().borrow_mut();
                if i == 0 {
                    cols.acc_addr = addrs.one;
                    cols.felt_acc_addr = addrs.zero;
                } else {
                    cols.acc_addr = addrs.accs[i - 1];
                    cols.felt_acc_addr = addrs.field_accs[i - 1];
                }
                cols.x1_mem = addrs.x1[i];
                cols.x2_mem = addrs.x2[i];
                cols.next_acc_addr = addrs.accs[i];
                cols.next_acc_mult = acc_mults[i];
                cols.felt_next_acc_addr = addrs.field_accs[i];
                cols.felt_next_acc_mult = field_acc_mults[i];
                cols.is_real = BabyBear::one();
            });
            rows.extend(row_add);
        });

        // Pad the trace to a power of two.
        pad_rows_fixed(
            &mut rows,
            || [BabyBear::zero(); NUM_PREFIX_SUM_CHECKS_PREPROCESSED_COLS],
            None,
        );

        let trace = RowMajorMatrix::new(
            unsafe {
                std::mem::transmute::<Vec<BabyBear>, Vec<F>>(
                    rows.into_iter().flatten().collect::<Vec<BabyBear>>(),
                )
            },
            NUM_PREFIX_SUM_CHECKS_PREPROCESSED_COLS,
        );
        Some(trace)
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let events = &input.prefix_sum_checks_events;
        Some(next_power_of_two(events.len(), None))
    }

    #[instrument(name = "generate prefix sum checks trace", level = "debug", skip_all, fields(rows = input.prefix_sum_checks_events.len()))]
    fn generate_trace(
        &self,
        input: &ExecutionRecord<F>,
        _: &mut ExecutionRecord<F>,
    ) -> RowMajorMatrix<F> {
        assert!(
            std::any::TypeId::of::<F>() == std::any::TypeId::of::<BabyBear>(),
            "generate_trace only supports BabyBear field"
        );

        let mut rows: Vec<[BabyBear; NUM_PREFIX_SUM_CHECKS_COLS]> =
            input
                .prefix_sum_checks_events
                .iter()
                .map(|event| {
                    let bb_event = unsafe {
                        std::mem::transmute::<
                            &PrefixSumChecksEvent<F>,
                            &PrefixSumChecksEvent<BabyBear>,
                        >(event)
                    };
                    let mut row = [BabyBear::zero(); NUM_PREFIX_SUM_CHECKS_COLS];
                    let cols: &mut PrefixSumChecksCols<BabyBear> = row.as_mut_slice().borrow_mut();
                    cols.x1 = bb_event.x1;
                    cols.x2 = bb_event.x2;
                    cols.prod = bb_event.prod;
                    cols.acc = bb_event.acc;
                    cols.new_acc = bb_event.new_acc;
                    cols.felt_acc = bb_event.field_acc;
                    cols.felt_new_acc = bb_event.new_field_acc;
                    row
                })
                .collect_vec();

        // Pad the trace to a power of two.
        pad_rows_fixed(&mut rows, || [BabyBear::zero(); NUM_PREFIX_SUM_CHECKS_COLS], None);

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(
            unsafe {
                std::mem::transmute::<Vec<BabyBear>, Vec<F>>(
                    rows.into_iter().flatten().collect::<Vec<BabyBear>>(),
                )
            },
            NUM_PREFIX_SUM_CHECKS_COLS,
        )
    }

    fn included(&self, _: &Self::Record) -> bool {
        true
    }

    fn local_only(&self) -> bool {
        true
    }
}

impl<AB> Air<AB> for PrefixSumChecksChip
where
    AB: SP1RecursionAirBuilder + PairBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &PrefixSumChecksCols<AB::Var> = (*local).borrow();
        let prep = builder.preprocessed();
        let prep_local = prep.row_slice(0);
        let prep_local: &PrefixSumChecksPreprocessedCols<_> = (*prep_local).borrow();

        let x2 = local.x2.as_extension::<AB>();
        let prod = local.prod.as_extension::<AB>();
        let one: BinomialExtension<AB::Expr> = BinomialExtension::from_base(AB::Expr::one());
        let two = AB::Expr::from_canonical_u32(2);

        builder.assert_ext_eq(
            BinomialExtension::from_base(local.x1.into()) * x2.clone(),
            local.prod.as_extension::<AB>(),
        );

        let sum_x_y = BinomialExtension::from_base(local.x1.into()) + x2;

        builder.assert_bool(prep_local.is_real);

        // Booleanity check for x1.
        builder.assert_bool(local.x1);

        // Constrain the memory access for inputs.
        builder.receive_single(prep_local.x1_mem, local.x1, prep_local.is_real);
        builder.receive_block(prep_local.x2_mem, local.x2, prep_local.is_real);

        // Constrain the memory read for the current accumulator.
        builder.receive_block(prep_local.acc_addr, local.acc, prep_local.is_real);
        builder.receive_single(prep_local.felt_acc_addr, local.felt_acc, prep_local.is_real);

        // Constrain the memory write for the next accumulator for lagrange eval and bit2felt
        // (Horner's method).
        builder.assert_ext_eq(
            local.new_acc.as_extension::<AB>(),
            local.acc.as_extension::<AB>() * (one - sum_x_y + prod.clone() + prod),
        );
        builder.assert_eq(local.felt_new_acc, local.x1 + two * local.felt_acc);
        builder.send_block(prep_local.next_acc_addr, local.new_acc, prep_local.next_acc_mult);
        builder.send_single(
            prep_local.felt_next_acc_addr,
            local.felt_new_acc,
            prep_local.felt_next_acc_mult,
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::test::test_recursion_linear_program;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use slop_algebra::{extension::BinomialExtensionField, AbstractExtensionField};
    use sp1_recursion_executor::{instruction as instr, MemAccessKind};

    use super::*;

    #[tokio::test]
    async fn test_prefix_sum_checks() {
        type F = BabyBear;
        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let mut random_extfelt = move || {
            let inner: [F; 4] = core::array::from_fn(|_| rng.sample(rand::distributions::Standard));
            BinomialExtensionField::<F, 4>::from_base_slice(&inner)
        };
        let mut felt_rng = StdRng::seed_from_u64(0xDEADBEEF);
        let mut random_felt = move || -> BabyBear {
            if felt_rng.gen_bool(0.5) {
                BabyBear::one()
            } else {
                BabyBear::zero()
            }
        };
        let mut addr = 0;

        let instructions = (0..10)
            .flat_map(|_| {
                let x1 = [random_felt(), random_felt()];
                let one = BinomialExtensionField::<F, 4>::from_base(BabyBear::one());
                let x2 = [random_extfelt(), random_extfelt()];

                let mut result = one;
                for i in 0..2 {
                    let prod = BinomialExtensionField::<F, 4>::from_base(x1[i]) * x2[i];
                    result *= one - (BinomialExtensionField::<F, 4>::from_base(x1[i]) + x2[i])
                        + prod
                        + prod;
                }

                let mut felt = BabyBear::zero();
                let two = BabyBear::from_canonical_u32(2);
                for &x1 in &x1 {
                    felt = x1 + two * felt;
                }

                let alloc_size = 10;
                let a = (0..alloc_size).map(|x| x + addr).collect::<Vec<_>>();
                addr += alloc_size;
                [
                    instr::mem_single(MemAccessKind::Write, 1, a[0], x1[0]),
                    instr::mem_single(MemAccessKind::Write, 1, a[1], x1[1]),
                    instr::mem_ext(MemAccessKind::Write, 1, a[2], x2[0]),
                    instr::mem_ext(MemAccessKind::Write, 1, a[3], x2[1]),
                    instr::mem_ext(MemAccessKind::Write, 1, a[4], one),
                    instr::mem_single(MemAccessKind::Write, 1, a[5], BabyBear::zero()),
                    instr::prefix_sum_checks(
                        vec![1, 1],
                        vec![1, 1],
                        vec![F::from_canonical_u32(a[0]), F::from_canonical_u32(a[1])],
                        vec![F::from_canonical_u32(a[2]), F::from_canonical_u32(a[3])],
                        F::from_canonical_u32(a[5]),
                        F::from_canonical_u32(a[4]),
                        vec![F::from_canonical_u32(a[6]), F::from_canonical_u32(a[7])],
                        vec![F::from_canonical_u32(a[8]), F::from_canonical_u32(a[9])],
                    ),
                    instr::mem_ext(MemAccessKind::Read, 1, a[7], result),
                    instr::mem_single(MemAccessKind::Read, 1, a[9], felt),
                ]
            })
            .collect::<Vec<Instruction<BabyBear>>>();

        test_recursion_linear_program(instructions).await;
    }
}
