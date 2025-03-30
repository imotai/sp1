use std::{
    array,
    ops::{Add, AddAssign},
    sync::Arc,
};

use p3_field::{AbstractField, Field, PrimeField32};
use sp1_stark::{
    air::SP1AirBuilder, septic_digest::SepticDigest, MachineRecord, SP1CoreOpts, PROOF_MAX_NUM_PVS,
};

use crate::{
    instruction::{HintBitsInstr, HintExt2FeltsInstr, HintInstr},
    public_values::RecursionPublicValues,
    ExpReverseBitsInstr, Instruction, PrefixSumChecksEvent,
};

use super::{
    BaseAluEvent, BatchFRIEvent, CommitPublicValuesEvent, ExpReverseBitsEvent, ExtAluEvent,
    FriFoldEvent, MemEvent, Poseidon2Event, RecursionProgram, SelectEvent,
};

#[derive(Clone, Default, Debug)]
pub struct ExecutionRecord<F> {
    pub program: Arc<RecursionProgram<F>>,
    /// The index of the shard.
    pub index: u32,

    pub base_alu_events: Vec<BaseAluEvent<F>>,
    pub ext_alu_events: Vec<ExtAluEvent<F>>,
    pub mem_const_count: usize,
    pub mem_var_events: Vec<MemEvent<F>>,
    /// The public values.
    pub public_values: RecursionPublicValues<F>,

    pub poseidon2_events: Vec<Poseidon2Event<F>>,
    pub select_events: Vec<SelectEvent<F>>,
    pub exp_reverse_bits_len_events: Vec<ExpReverseBitsEvent<F>>,
    pub fri_fold_events: Vec<FriFoldEvent<F>>,
    pub batch_fri_events: Vec<BatchFRIEvent<F>>,
    pub prefix_sum_checks_events: Vec<PrefixSumChecksEvent<F>>,
    pub commit_pv_hash_events: Vec<CommitPublicValuesEvent<F>>,
}

impl<F: PrimeField32> MachineRecord for ExecutionRecord<F> {
    type Config = SP1CoreOpts;

    fn stats(&self) -> hashbrown::HashMap<String, usize> {
        [
            ("base_alu_events", self.base_alu_events.len()),
            ("ext_alu_events", self.ext_alu_events.len()),
            ("mem_const_count", self.mem_const_count),
            ("mem_var_events", self.mem_var_events.len()),
            ("poseidon2_events", self.poseidon2_events.len()),
            ("select_events", self.select_events.len()),
            ("exp_reverse_bits_len_events", self.exp_reverse_bits_len_events.len()),
            ("fri_fold_events", self.fri_fold_events.len()),
            ("batch_fri_events", self.batch_fri_events.len()),
            ("prefix_sum_checks_events", self.prefix_sum_checks_events.len()),
            ("commit_pv_hash_events", self.commit_pv_hash_events.len()),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect()
    }

    fn append(&mut self, other: &mut Self) {
        // Exhaustive destructuring for refactoring purposes.
        let Self {
            program: _,
            index: _,
            base_alu_events,
            ext_alu_events,
            mem_const_count,
            mem_var_events,
            public_values: _,
            poseidon2_events,
            select_events,
            exp_reverse_bits_len_events,
            fri_fold_events,
            batch_fri_events,
            prefix_sum_checks_events,
            commit_pv_hash_events,
        } = self;
        base_alu_events.append(&mut other.base_alu_events);
        ext_alu_events.append(&mut other.ext_alu_events);
        *mem_const_count += other.mem_const_count;
        mem_var_events.append(&mut other.mem_var_events);
        poseidon2_events.append(&mut other.poseidon2_events);
        select_events.append(&mut other.select_events);
        exp_reverse_bits_len_events.append(&mut other.exp_reverse_bits_len_events);
        fri_fold_events.append(&mut other.fri_fold_events);
        batch_fri_events.append(&mut other.batch_fri_events);
        prefix_sum_checks_events.append(&mut other.prefix_sum_checks_events);
        commit_pv_hash_events.append(&mut other.commit_pv_hash_events);
    }

    fn public_values<T: AbstractField>(&self) -> Vec<T> {
        let pv_elms = self.public_values.as_array();

        let ret: [T; PROOF_MAX_NUM_PVS] = array::from_fn(|i| {
            if i < pv_elms.len() {
                T::from_canonical_u32(pv_elms[i].as_canonical_u32())
            } else {
                T::zero()
            }
        });

        ret.to_vec()
    }

    fn update_global_cumulative_sum<T: PrimeField32>(
        &mut self,
        _global_cumulative_sum: SepticDigest<T>,
    ) {
        panic!("Recursion does not have global chip");
    }

    fn global_cumulative_sum<T: PrimeField32>(_public_values: &[T]) -> SepticDigest<T> {
        SepticDigest::<T>::zero()
    }

    // No public value constraints for recursion public values.
    fn eval_public_values<AB: SP1AirBuilder>(_builder: &mut AB) {}
}

impl<F: Field> ExecutionRecord<F> {
    pub fn preallocate(&mut self) {
        let event_counts =
            self.program.inner.iter().fold(RecursionAirEventCount::default(), Add::add);
        self.poseidon2_events.reserve(event_counts.poseidon2_wide_events);
        self.mem_var_events.reserve(event_counts.mem_var_events);
        self.base_alu_events.reserve(event_counts.base_alu_events);
        self.ext_alu_events.reserve(event_counts.ext_alu_events);
        self.exp_reverse_bits_len_events.reserve(event_counts.exp_reverse_bits_len_events);
        self.select_events.reserve(event_counts.select_events);
        self.prefix_sum_checks_events.reserve(event_counts.prefix_sum_checks_events);
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RecursionAirEventCount {
    pub mem_const_events: usize,
    pub mem_var_events: usize,
    pub base_alu_events: usize,
    pub ext_alu_events: usize,
    pub poseidon2_wide_events: usize,
    pub fri_fold_events: usize,
    pub batch_fri_events: usize,
    pub select_events: usize,
    pub exp_reverse_bits_len_events: usize,
    pub prefix_sum_checks_events: usize,
}

impl<F> AddAssign<&Instruction<F>> for RecursionAirEventCount {
    #[inline]
    fn add_assign(&mut self, rhs: &Instruction<F>) {
        match rhs {
            Instruction::BaseAlu(_) => self.base_alu_events += 1,
            Instruction::ExtAlu(_) => self.ext_alu_events += 1,
            Instruction::Mem(_) => self.mem_const_events += 1,
            Instruction::Poseidon2(_) => self.poseidon2_wide_events += 1,
            Instruction::Select(_) => self.select_events += 1,
            Instruction::ExpReverseBitsLen(ExpReverseBitsInstr { addrs, .. }) => {
                self.exp_reverse_bits_len_events += addrs.exp.len()
            }
            Instruction::Hint(HintInstr { output_addrs_mults })
            | Instruction::HintBits(HintBitsInstr {
                output_addrs_mults,
                input_addr: _, // No receive interaction for the hint operation
            }) => self.mem_var_events += output_addrs_mults.len(),
            Instruction::HintExt2Felts(HintExt2FeltsInstr {
                output_addrs_mults,
                input_addr: _, // No receive interaction for the hint operation
            }) => self.mem_var_events += output_addrs_mults.len(),
            Instruction::FriFold(_) => self.fri_fold_events += 1,
            Instruction::BatchFRI(instr) => {
                self.batch_fri_events += instr.base_vec_addrs.p_at_x.len()
            }
            Instruction::PrefixSumChecks(instr) => {
                self.prefix_sum_checks_events += instr.addrs.x1.len()
            }
            Instruction::HintAddCurve(instr) => {
                self.mem_var_events += instr.output_x_addrs_mults.len();
                self.mem_var_events += instr.output_y_addrs_mults.len();
            }
            Instruction::CommitPublicValues(_)
            | Instruction::Print(_)
            | Instruction::DebugBacktrace(_) => {}
        }
    }
}

impl<F> Add<&Instruction<F>> for RecursionAirEventCount {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: &Instruction<F>) -> Self::Output {
        self += rhs;
        self
    }
}
