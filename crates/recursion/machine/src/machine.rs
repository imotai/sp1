use std::fmt;

use slop_algebra::{extension::BinomiallyExtendable, PrimeField32};
use sp1_recursion_executor::{ExecutionRecord, RecursionAirEventCount, RecursionProgram, D};
use sp1_stark::{
    air::{InteractionScope, MachineAir},
    Chip, Machine, PROOF_MAX_NUM_PVS,
};

use strum::EnumIter;
use strum_macros::EnumDiscriminants;

use crate::chips::{
    alu_base::{BaseAluChip, NUM_BASE_ALU_ENTRIES_PER_ROW},
    alu_ext::{ExtAluChip, NUM_EXT_ALU_ENTRIES_PER_ROW},
    batch_fri::BatchFRIChip,
    exp_reverse_bits::ExpReverseBitsLenChip,
    fri_fold::FriFoldChip,
    mem::{
        constant::NUM_CONST_MEM_ENTRIES_PER_ROW, variable::NUM_VAR_MEM_ENTRIES_PER_ROW,
        MemoryConstChip, MemoryVarChip,
    },
    poseidon2_skinny::Poseidon2SkinnyChip,
    poseidon2_wide::Poseidon2WideChip,
    public_values::{PublicValuesChip, PUB_VALUES_LOG_HEIGHT},
    select::SelectChip,
};

#[derive(sp1_derive::MachineAir, EnumDiscriminants)]
#[sp1_core_path = "sp1_core_machine"]
#[execution_record_path = "ExecutionRecord<F>"]
#[program_path = "RecursionProgram<F>"]
#[builder_path = "crate::builder::SP1RecursionAirBuilder<F = F>"]
#[eval_trait_bound = "AB::Var: 'static"]
#[strum_discriminants(derive(Hash, EnumIter))]
#[allow(dead_code)]
pub enum RecursionAir<F: PrimeField32 + BinomiallyExtendable<D>, const DEGREE: usize> {
    MemoryConst(MemoryConstChip<F>),
    MemoryVar(MemoryVarChip<F>),
    BaseAlu(BaseAluChip),
    ExtAlu(ExtAluChip),
    Poseidon2Skinny(Poseidon2SkinnyChip<DEGREE>),
    Poseidon2Wide(Poseidon2WideChip<DEGREE>),
    Select(SelectChip),
    FriFold(FriFoldChip<DEGREE>),
    BatchFRI(BatchFRIChip<DEGREE>),
    ExpReverseBitsLen(ExpReverseBitsLenChip<DEGREE>),
    PublicValues(PublicValuesChip),
}

impl<F: PrimeField32 + BinomiallyExtendable<D>, const DEGREE: usize> fmt::Debug
    for RecursionAir<F, DEGREE>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[allow(dead_code)]
impl<F: PrimeField32 + BinomiallyExtendable<D>, const DEGREE: usize> RecursionAir<F, DEGREE> {
    /// Get a machine with all chips, except the dummy chip.
    pub fn machine_wide_with_all_chips() -> Machine<F, Self> {
        let chips = [
            RecursionAir::MemoryConst(MemoryConstChip::default()),
            RecursionAir::MemoryVar(MemoryVarChip::default()),
            RecursionAir::BaseAlu(BaseAluChip),
            RecursionAir::ExtAlu(ExtAluChip),
            RecursionAir::Poseidon2Wide(Poseidon2WideChip::<DEGREE>),
            RecursionAir::FriFold(FriFoldChip::<DEGREE>::default()),
            RecursionAir::BatchFRI(BatchFRIChip::<DEGREE>),
            RecursionAir::Select(SelectChip),
            RecursionAir::ExpReverseBitsLen(ExpReverseBitsLenChip::<DEGREE>),
            RecursionAir::PublicValues(PublicValuesChip),
        ]
        .map(Chip::new)
        .into_iter()
        .collect::<Vec<_>>();
        Machine::new(chips, PROOF_MAX_NUM_PVS, false)
    }

    /// Get a machine with all chips, except the dummy chip.
    pub fn machine_skinny_with_all_chips() -> Machine<F, Self> {
        let chips = [
            RecursionAir::MemoryConst(MemoryConstChip::default()),
            RecursionAir::MemoryVar(MemoryVarChip::default()),
            RecursionAir::BaseAlu(BaseAluChip),
            RecursionAir::ExtAlu(ExtAluChip),
            RecursionAir::Poseidon2Skinny(Poseidon2SkinnyChip::<DEGREE>::default()),
            RecursionAir::FriFold(FriFoldChip::<DEGREE>::default()),
            RecursionAir::BatchFRI(BatchFRIChip::<DEGREE>),
            RecursionAir::Select(SelectChip),
            RecursionAir::ExpReverseBitsLen(ExpReverseBitsLenChip::<DEGREE>),
            RecursionAir::PublicValues(PublicValuesChip),
        ]
        .map(Chip::new)
        .into_iter()
        .collect::<Vec<_>>();
        Machine::new(chips, PROOF_MAX_NUM_PVS, false)
    }

    /// A machine with dyunamic chip sizes that includes the wide variant of the Poseidon2 chip.
    pub fn compress_machine() -> Machine<F, Self> {
        let chips = [
            RecursionAir::MemoryConst(MemoryConstChip::default()),
            RecursionAir::MemoryVar(MemoryVarChip::default()),
            RecursionAir::BaseAlu(BaseAluChip),
            RecursionAir::ExtAlu(ExtAluChip),
            RecursionAir::Poseidon2Wide(Poseidon2WideChip::<DEGREE>),
            RecursionAir::BatchFRI(BatchFRIChip::<DEGREE>),
            RecursionAir::Select(SelectChip),
            RecursionAir::ExpReverseBitsLen(ExpReverseBitsLenChip::<DEGREE>),
            RecursionAir::PublicValues(PublicValuesChip),
        ]
        .map(Chip::new)
        .into_iter()
        .collect::<Vec<_>>();
        Machine::new(chips, PROOF_MAX_NUM_PVS, false)
    }

    pub fn shrink_machine() -> Machine<F, Self> {
        Self::compress_machine()
    }

    /// A machine with dynamic chip sizes that includes the skinny variant of the Poseidon2 chip.
    ///
    /// This machine assumes that the `shrink` stage has a fixed shape, so there is no need to
    /// fix the trace sizes.
    pub fn wrap_machine() -> Machine<F, Self> {
        let chips = [
            RecursionAir::MemoryConst(MemoryConstChip::default()),
            RecursionAir::MemoryVar(MemoryVarChip::default()),
            RecursionAir::BaseAlu(BaseAluChip),
            RecursionAir::ExtAlu(ExtAluChip),
            RecursionAir::Poseidon2Skinny(Poseidon2SkinnyChip::<DEGREE>::default()),
            // RecursionAir::BatchFRI(BatchFRIChip::<DEGREE>),
            RecursionAir::Select(SelectChip),
            RecursionAir::PublicValues(PublicValuesChip),
        ]
        .map(Chip::new)
        .into_iter()
        .collect::<Vec<_>>();
        Machine::new(chips, PROOF_MAX_NUM_PVS, false)
    }

    pub fn heights(program: &RecursionProgram<F>) -> Vec<(String, usize)> {
        let heights = program
            .inner
            .iter()
            .fold(RecursionAirEventCount::default(), |heights, instruction| heights + instruction);

        [
            (
                Self::MemoryConst(MemoryConstChip::default()),
                heights.mem_const_events.div_ceil(NUM_CONST_MEM_ENTRIES_PER_ROW),
            ),
            (
                Self::MemoryVar(MemoryVarChip::default()),
                heights.mem_var_events.div_ceil(NUM_VAR_MEM_ENTRIES_PER_ROW),
            ),
            (
                Self::BaseAlu(BaseAluChip),
                heights.base_alu_events.div_ceil(NUM_BASE_ALU_ENTRIES_PER_ROW),
            ),
            (
                Self::ExtAlu(ExtAluChip),
                heights.ext_alu_events.div_ceil(NUM_EXT_ALU_ENTRIES_PER_ROW),
            ),
            (Self::Poseidon2Wide(Poseidon2WideChip::<DEGREE>), heights.poseidon2_wide_events),
            (Self::BatchFRI(BatchFRIChip::<DEGREE>), heights.batch_fri_events),
            (Self::Select(SelectChip), heights.select_events),
            (
                Self::ExpReverseBitsLen(ExpReverseBitsLenChip::<DEGREE>),
                heights.exp_reverse_bits_len_events,
            ),
            (Self::PublicValues(PublicValuesChip), PUB_VALUES_LOG_HEIGHT),
        ]
        .map(|(chip, log_height)| (chip.name(), log_height))
        .to_vec()
    }
}

#[cfg(test)]
pub mod tests {

    use std::{iter::once, sync::Arc};

    use rand::{rngs::StdRng, Rng, SeedableRng};
    use slop_algebra::{
        extension::{BinomialExtensionField, HasFrobenius},
        AbstractExtensionField, AbstractField, Field,
    };
    use slop_baby_bear::DiffusionMatrixBabyBear;
    use slop_jagged::JaggedConfig;
    use slop_merkle_tree::my_bb_16_perm;
    use sp1_recursion_executor::{
        instruction as instr, linear_program, BaseAluOpcode, Block, ExtAluOpcode, Instruction,
        MemAccessKind, RecursionProgram, Runtime, D,
    };
    use sp1_stark::BabyBearPoseidon2;

    use crate::test::tests::run_test_recursion;

    use super::RecursionAir;

    type SC = BabyBearPoseidon2;
    type F = <SC as JaggedConfig>::F;
    type EF = <SC as JaggedConfig>::EF;
    type A = RecursionAir<F, 3>;
    // type B = RecursionAir<F, 9>;

    /// Runs the given program on machines that use the wide and skinny Poseidon2 chips.
    pub async fn run_recursion_test_machines(program: RecursionProgram<F>, witness: Vec<Block<F>>) {
        let mut runtime = Runtime::<F, EF, DiffusionMatrixBabyBear>::new(
            Arc::new(program.clone()),
            my_bb_16_perm(),
        );
        runtime.witness_stream = witness.into();
        runtime.run().unwrap();

        // Run with the poseidon2 wide chip.
        let machine = A::machine_wide_with_all_chips();
        run_test_recursion(vec![runtime.record.clone()], machine, program.clone()).await.unwrap();

        // // Run with the poseidon2 skinny chip.
        // let skinny_machine = B::machine_skinny_with_all_chips();
        // run_test_recursion(vec![runtime.record], skinny_machine, program).await.unwrap();
        // println!("ran proof gen with machine skinny");
    }

    /// Constructs a linear program and runs it on machines that use the wide and skinny Poseidon2 chips.
    pub async fn test_recursion_linear_program(instrs: Vec<Instruction<F>>) {
        run_recursion_test_machines(linear_program(instrs).unwrap(), Vec::new()).await;
    }

    #[tokio::test]
    pub async fn fibonacci() {
        let n = 10;

        let instructions = once(instr::mem(MemAccessKind::Write, 1, 0, 0))
            .chain(once(instr::mem(MemAccessKind::Write, 2, 1, 1)))
            .chain((2..=n).map(|i| instr::base_alu(BaseAluOpcode::AddF, 2, i, i - 2, i - 1)))
            .chain(once(instr::mem(MemAccessKind::Read, 1, n - 1, 34)))
            .chain(once(instr::mem(MemAccessKind::Read, 2, n, 55)))
            .collect::<Vec<_>>();

        test_recursion_linear_program(instructions).await;
    }

    #[tokio::test]
    #[should_panic]
    pub async fn div_nonzero_by_zero() {
        let instructions = vec![
            instr::mem(MemAccessKind::Write, 1, 0, 0),
            instr::mem(MemAccessKind::Write, 1, 1, 1),
            instr::base_alu(BaseAluOpcode::DivF, 1, 2, 1, 0),
            instr::mem(MemAccessKind::Read, 1, 2, 1),
        ];

        test_recursion_linear_program(instructions).await;
    }

    #[tokio::test]
    pub async fn div_zero_by_zero() {
        let instructions = vec![
            instr::mem(MemAccessKind::Write, 1, 0, 0),
            instr::mem(MemAccessKind::Write, 1, 1, 0),
            instr::base_alu(BaseAluOpcode::DivF, 1, 2, 1, 0),
            instr::mem(MemAccessKind::Read, 1, 2, 1),
        ];

        test_recursion_linear_program(instructions).await;
    }

    #[tokio::test]
    pub async fn field_norm() {
        let mut instructions = Vec::new();

        let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
        let mut addr = 0;
        for _ in 0..100 {
            let inner: [F; 4] = std::iter::repeat_with(|| {
                core::array::from_fn(|_| rng.sample(rand::distributions::Standard))
            })
            .find(|xs| !xs.iter().all(F::is_zero))
            .unwrap();
            let x = BinomialExtensionField::<F, D>::from_base_slice(&inner);
            let gal = x.galois_group();

            let mut acc = BinomialExtensionField::one();

            instructions.push(instr::mem_ext(MemAccessKind::Write, 1, addr, acc));
            for conj in gal {
                instructions.push(instr::mem_ext(MemAccessKind::Write, 1, addr + 1, conj));
                instructions.push(instr::ext_alu(ExtAluOpcode::MulE, 1, addr + 2, addr, addr + 1));

                addr += 2;
                acc *= conj;
            }
            let base_cmp: F = acc.as_base_slice()[0];
            instructions.push(instr::mem_single(MemAccessKind::Read, 1, addr, base_cmp));
            addr += 1;
        }

        test_recursion_linear_program(instructions).await;
    }
}
