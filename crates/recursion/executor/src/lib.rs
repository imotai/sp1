mod block;
pub mod instruction;
mod memory;
mod opcode;
mod program;
mod public_values;
mod record;
pub mod shape;

pub use public_values::PV_DIGEST_NUM_WORDS;

// Avoid triggering annoying branch of thiserror derive macro.
use backtrace::Backtrace as Trace;
pub use block::Block;
use cfg_if::cfg_if;
pub use instruction::Instruction;
use instruction::{
    FieldEltType, HintAddCurveInstr, HintBitsInstr, HintExt2FeltsInstr, HintInstr, PrintInstr,
};
use itertools::Itertools;
use memory::*;
pub use opcode::*;
use p3_field::{AbstractExtensionField, AbstractField, ExtensionField, PrimeField32, PrimeField64};
use p3_maybe_rayon::prelude::*;
use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use p3_symmetric::{CryptographicPermutation, Permutation};
use p3_util::reverse_bits_len;
pub use program::*;
pub use public_values::{RecursionPublicValues, NUM_PV_ELMS_TO_HASH, RECURSIVE_PROOF_NUM_PV_ELTS};
pub use record::*;
use serde::{Deserialize, Serialize};
use sp1_derive::AlignedBorrow;
use sp1_stark::{septic_curve::SepticCurve, septic_extension::SepticExtension, MachineRecord};
use std::{
    array,
    borrow::Borrow,
    collections::VecDeque,
    fmt::Debug,
    io::{stdout, Write},
    iter::zip,
    marker::PhantomData,
    sync::{Arc, Mutex},
};
use thiserror::Error;

/// The heap pointer address.
pub const HEAP_PTR: i32 = -4;
pub const STACK_SIZE: usize = 1 << 24;
pub const HEAP_START_ADDRESS: usize = STACK_SIZE + 4;
pub const MEMORY_SIZE: usize = 1 << 28;

/// The width of the Poseidon2 permutation.
pub const PERMUTATION_WIDTH: usize = 16;
pub const POSEIDON2_SBOX_DEGREE: u64 = 7;
pub const HASH_RATE: usize = 8;

/// The current verifier implementation assumes that we are using a 256-bit hash with 32-bit
/// elements.
pub const DIGEST_SIZE: usize = 8;

pub const NUM_BITS: usize = 31;

pub const D: usize = 4;

#[derive(
    AlignedBorrow, Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Default,
)]
#[repr(transparent)]
pub struct Address<F>(pub F);

impl<F: PrimeField64> Address<F> {
    #[inline]
    pub fn as_usize(&self) -> usize {
        self.0.as_canonical_u64() as usize
    }
}

// -------------------------------------------------------------------------------------------------

/// The inputs and outputs to an operation of the base field ALU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct BaseAluIo<V> {
    pub out: V,
    pub in1: V,
    pub in2: V,
}

pub type BaseAluEvent<F> = BaseAluIo<F>;

/// An instruction invoking the extension field ALU.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct BaseAluInstr<F> {
    pub opcode: BaseAluOpcode,
    pub mult: F,
    pub addrs: BaseAluIo<Address<F>>,
}

// -------------------------------------------------------------------------------------------------

/// The inputs and outputs to an operation of the extension field ALU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct ExtAluIo<V> {
    pub out: V,
    pub in1: V,
    pub in2: V,
}

pub type ExtAluEvent<F> = ExtAluIo<Block<F>>;

/// An instruction invoking the extension field ALU.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct ExtAluInstr<F> {
    pub opcode: ExtAluOpcode,
    pub mult: F,
    pub addrs: ExtAluIo<Address<F>>,
}

// -------------------------------------------------------------------------------------------------

/// The inputs and outputs to the manual memory management/memory initialization table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct MemIo<V> {
    pub inner: V,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemInstr<F> {
    pub addrs: MemIo<Address<F>>,
    pub vals: MemIo<Block<F>>,
    pub mult: F,
    pub kind: MemAccessKind,
}

pub type MemEvent<F> = MemIo<Block<F>>;

// -------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemAccessKind {
    Read,
    Write,
}

/// The inputs and outputs to a Poseidon2 permutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct Poseidon2Io<V> {
    pub input: [V; PERMUTATION_WIDTH],
    pub output: [V; PERMUTATION_WIDTH],
}

/// An instruction invoking the Poseidon2 permutation.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct Poseidon2SkinnyInstr<F> {
    pub addrs: Poseidon2Io<Address<F>>,
    pub mults: [F; PERMUTATION_WIDTH],
}

pub type Poseidon2Event<F> = Poseidon2Io<F>;

/// The inputs and outputs to a select operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct SelectIo<V> {
    pub bit: V,
    pub out1: V,
    pub out2: V,
    pub in1: V,
    pub in2: V,
}

/// An instruction invoking the select operation.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct SelectInstr<F> {
    pub addrs: SelectIo<Address<F>>,
    pub mult1: F,
    pub mult2: F,
}

/// The event encoding the inputs and outputs of a select operation.
pub type SelectEvent<F> = SelectIo<F>;

/// The inputs and outputs to the operations for prefix sum checks.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefixSumChecksIo<V> {
    pub zero: V,
    pub one: V,
    pub x1: Vec<V>,
    pub x2: Vec<V>,
    pub accs: Vec<V>,
    pub field_accs: Vec<V>,
}

/// An instruction invoking the PrefixSumChecks operation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrefixSumChecksInstr<F> {
    pub addrs: PrefixSumChecksIo<Address<F>>,
    pub acc_mults: Vec<F>,
    pub field_acc_mults: Vec<F>,
}

/// The event encoding the inputs and outputs of an PrefixSumChecks operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct PrefixSumChecksEvent<F> {
    pub x1: F,
    pub x2: Block<F>,
    pub zero: F,
    pub one: Block<F>,
    pub prod: Block<F>,
    pub acc: Block<F>,
    pub new_acc: Block<F>,
    pub field_acc: F,
    pub new_field_acc: F,
}

/// The inputs and outputs to an exp-reverse-bits operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpReverseBitsIo<V> {
    pub base: V,
    // The bits of the exponent in little-endian order in a vec.
    pub exp: Vec<V>,
    pub result: V,
}

pub type Poseidon2WideEvent<F> = Poseidon2Io<F>;
pub type Poseidon2Instr<F> = Poseidon2SkinnyInstr<F>;

/// An instruction invoking the exp-reverse-bits operation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExpReverseBitsInstr<F> {
    pub addrs: ExpReverseBitsIo<Address<F>>,
    pub mult: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct ExpReverseBitsInstrFFI<'a, F> {
    pub base: &'a Address<F>,
    pub exp_ptr: *const Address<F>,
    pub exp_len: usize,
    pub result: &'a Address<F>,

    pub mult: &'a F,
}

impl<'a, F> From<&'a ExpReverseBitsInstr<F>> for ExpReverseBitsInstrFFI<'a, F> {
    fn from(instr: &'a ExpReverseBitsInstr<F>) -> Self {
        Self {
            base: &instr.addrs.base,
            exp_ptr: instr.addrs.exp.as_ptr(),
            exp_len: instr.addrs.exp.len(),
            result: &instr.addrs.result,

            mult: &instr.mult,
        }
    }
}

/// The event encoding the inputs and outputs of an exp-reverse-bits operation. The `len` operand is
/// now stored as the length of the `exp` field.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpReverseBitsEvent<F> {
    pub base: F,
    pub exp: Vec<F>,
    pub result: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct ExpReverseBitsEventFFI<'a, F> {
    pub base: &'a F,
    pub exp_ptr: *const F,
    pub exp_len: usize,
    pub result: &'a F,
}

impl<'a, F> From<&'a ExpReverseBitsEvent<F>> for ExpReverseBitsEventFFI<'a, F> {
    fn from(event: &'a ExpReverseBitsEvent<F>) -> Self {
        Self {
            base: &event.base,
            exp_ptr: event.exp.as_ptr(),
            exp_len: event.exp.len(),
            result: &event.result,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FriFoldIo<V> {
    pub ext_single: FriFoldExtSingleIo<Block<V>>,
    pub ext_vec: FriFoldExtVecIo<Vec<Block<V>>>,
    pub base_single: FriFoldBaseIo<V>,
}

/// The extension-field-valued single inputs to the FRI fold operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct FriFoldExtSingleIo<V> {
    pub z: V,
    pub alpha: V,
}

/// The extension-field-valued vector inputs to the FRI fold operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct FriFoldExtVecIo<V> {
    pub mat_opening: V,
    pub ps_at_z: V,
    pub alpha_pow_input: V,
    pub ro_input: V,
    pub alpha_pow_output: V,
    pub ro_output: V,
}

/// The base-field-valued inputs to the FRI fold operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct FriFoldBaseIo<V> {
    pub x: V,
}

/// An instruction invoking the FRI fold operation. Addresses for extension field elements are of
/// the same type as for base field elements.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FriFoldInstr<F> {
    pub base_single_addrs: FriFoldBaseIo<Address<F>>,
    pub ext_single_addrs: FriFoldExtSingleIo<Address<F>>,
    pub ext_vec_addrs: FriFoldExtVecIo<Vec<Address<F>>>,
    pub alpha_pow_mults: Vec<F>,
    pub ro_mults: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct FriFoldInstrFFI<'a, F> {
    pub base_single_addrs: &'a FriFoldBaseIo<Address<F>>,
    pub ext_single_addrs: &'a FriFoldExtSingleIo<Address<F>>,

    pub ext_vec_addrs_mat_opening_ptr: *const Address<F>,
    pub ext_vec_addrs_mat_opening_len: usize,
    pub ext_vec_addrs_ps_at_z_ptr: *const Address<F>,
    pub ext_vec_addrs_ps_at_z_len: usize,
    pub ext_vec_addrs_alpha_pow_input_ptr: *const Address<F>,
    pub ext_vec_addrs_alpha_pow_input_len: usize,
    pub ext_vec_addrs_ro_input_ptr: *const Address<F>,
    pub ext_vec_addrs_ro_input_len: usize,
    pub ext_vec_addrs_alpha_pow_output_ptr: *const Address<F>,
    pub ext_vec_addrs_alpha_pow_output_len: usize,
    pub ext_vec_addrs_ro_output_ptr: *const Address<F>,
    pub ext_vec_addrs_ro_output_len: usize,

    pub alpha_pow_mults_ptr: *const F,
    pub alpha_pow_mults_len: usize,

    pub ro_mults_ptr: *const F,
    pub ro_mults_len: usize,
}

impl<'a, F> From<&'a FriFoldInstr<F>> for FriFoldInstrFFI<'a, F> {
    fn from(instr: &'a FriFoldInstr<F>) -> Self {
        Self {
            base_single_addrs: &instr.base_single_addrs,
            ext_single_addrs: &instr.ext_single_addrs,

            ext_vec_addrs_mat_opening_ptr: instr.ext_vec_addrs.mat_opening.as_ptr(),
            ext_vec_addrs_mat_opening_len: instr.ext_vec_addrs.mat_opening.len(),
            ext_vec_addrs_ps_at_z_ptr: instr.ext_vec_addrs.ps_at_z.as_ptr(),
            ext_vec_addrs_ps_at_z_len: instr.ext_vec_addrs.ps_at_z.len(),
            ext_vec_addrs_alpha_pow_input_ptr: instr.ext_vec_addrs.alpha_pow_input.as_ptr(),
            ext_vec_addrs_alpha_pow_input_len: instr.ext_vec_addrs.alpha_pow_input.len(),
            ext_vec_addrs_ro_input_ptr: instr.ext_vec_addrs.ro_input.as_ptr(),
            ext_vec_addrs_ro_input_len: instr.ext_vec_addrs.ro_input.len(),
            ext_vec_addrs_alpha_pow_output_ptr: instr.ext_vec_addrs.alpha_pow_output.as_ptr(),
            ext_vec_addrs_alpha_pow_output_len: instr.ext_vec_addrs.alpha_pow_output.len(),
            ext_vec_addrs_ro_output_ptr: instr.ext_vec_addrs.ro_output.as_ptr(),
            ext_vec_addrs_ro_output_len: instr.ext_vec_addrs.ro_output.len(),

            alpha_pow_mults_ptr: instr.alpha_pow_mults.as_ptr(),
            alpha_pow_mults_len: instr.alpha_pow_mults.len(),

            ro_mults_ptr: instr.ro_mults.as_ptr(),
            ro_mults_len: instr.ro_mults.len(),
        }
    }
}

impl<'a, F> From<&'a Box<FriFoldInstr<F>>> for FriFoldInstrFFI<'a, F> {
    fn from(instr: &'a Box<FriFoldInstr<F>>) -> Self {
        Self::from(instr.as_ref())
    }
}

/// The event encoding the data of a single iteration within the FRI fold operation.
/// For any given event, we are accessing a single element of the `Vec` inputs, so that the event
/// is not a type alias for `FriFoldIo` like many of the other events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct FriFoldEvent<F> {
    pub base_single: FriFoldBaseIo<F>,
    pub ext_single: FriFoldExtSingleIo<Block<F>>,
    pub ext_vec: FriFoldExtVecIo<Block<F>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchFRIIo<V> {
    pub ext_single: BatchFRIExtSingleIo<Block<V>>,
    pub ext_vec: BatchFRIExtVecIo<Vec<Block<V>>>,
    pub base_vec: BatchFRIBaseVecIo<V>,
}

/// The extension-field-valued single inputs to the batch FRI operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct BatchFRIExtSingleIo<V> {
    pub acc: V,
}

/// The extension-field-valued vector inputs to the batch FRI operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct BatchFRIExtVecIo<V> {
    pub p_at_z: V,
    pub alpha_pow: V,
}

/// The base-field-valued vector inputs to the batch FRI operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct BatchFRIBaseVecIo<V> {
    pub p_at_x: V,
}

/// An instruction invoking the batch FRI operation. Addresses for extension field elements are of
/// the same type as for base field elements.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchFRIInstr<F> {
    pub base_vec_addrs: BatchFRIBaseVecIo<Vec<Address<F>>>,
    pub ext_single_addrs: BatchFRIExtSingleIo<Address<F>>,
    pub ext_vec_addrs: BatchFRIExtVecIo<Vec<Address<F>>>,
    pub acc_mult: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct BatchFRIInstrFFI<'a, F> {
    pub base_vec_addrs_p_at_x_ptr: *const Address<F>,
    pub base_vec_addrs_p_at_x_len: usize,

    pub ext_single_addrs: &'a BatchFRIExtSingleIo<Address<F>>,

    pub ext_vec_addrs_p_at_z_ptr: *const Address<F>,
    pub ext_vec_addrs_p_at_z_len: usize,
    pub ext_vec_addrs_alpha_pow_ptr: *const Address<F>,
    pub ext_vec_addrs_alpha_pow_len: usize,

    pub acc_mult: &'a F,
}

impl<'a, F> From<&'a BatchFRIInstr<F>> for BatchFRIInstrFFI<'a, F> {
    fn from(instr: &'a BatchFRIInstr<F>) -> Self {
        Self {
            base_vec_addrs_p_at_x_ptr: instr.base_vec_addrs.p_at_x.as_ptr(),
            base_vec_addrs_p_at_x_len: instr.base_vec_addrs.p_at_x.len(),

            ext_single_addrs: &instr.ext_single_addrs,

            ext_vec_addrs_p_at_z_ptr: instr.ext_vec_addrs.p_at_z.as_ptr(),
            ext_vec_addrs_p_at_z_len: instr.ext_vec_addrs.p_at_z.len(),
            ext_vec_addrs_alpha_pow_ptr: instr.ext_vec_addrs.alpha_pow.as_ptr(),
            ext_vec_addrs_alpha_pow_len: instr.ext_vec_addrs.alpha_pow.len(),

            acc_mult: &instr.acc_mult,
        }
    }
}

impl<'a, 'b: 'a, F> From<&'b &'b Box<BatchFRIInstr<F>>> for BatchFRIInstrFFI<'a, F> {
    fn from(instr: &'b &'b Box<BatchFRIInstr<F>>) -> Self {
        Self::from(instr.as_ref())
    }
}

/// The event encoding the data of a single iteration within the batch FRI operation.
/// For any given event, we are accessing a single element of the `Vec` inputs, so that the event
/// is not a type alias for `BatchFRIIo` like many of the other events.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct BatchFRIEvent<F> {
    pub base_vec: BatchFRIBaseVecIo<F>,
    pub ext_single: BatchFRIExtSingleIo<Block<F>>,
    pub ext_vec: BatchFRIExtVecIo<Block<F>>,
}

/// An instruction that will save the public values to the execution record and will commit to
/// it's digest.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct CommitPublicValuesInstr<F> {
    pub pv_addrs: RecursionPublicValues<Address<F>>,
}

/// The event for committing to the public values.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[repr(C)]
pub struct CommitPublicValuesEvent<F> {
    pub public_values: RecursionPublicValues<F>,
}

type Perm<F, Diffusion> = Poseidon2<
    F,
    Poseidon2ExternalMatrixGeneral,
    Diffusion,
    PERMUTATION_WIDTH,
    POSEIDON2_SBOX_DEGREE,
>;

/// TODO fully document.
/// Taken from [`sp1_recursion_executor::runtime::Runtime`].
/// Many missing things (compared to the old `Runtime`) will need to be implemented.
pub struct Runtime<'a, F: PrimeField32, EF: ExtensionField<F>, Diffusion> {
    pub timestamp: usize,

    pub nb_poseidons: usize,

    pub nb_wide_poseidons: usize,

    pub nb_bit_decompositions: usize,

    pub nb_ext_ops: usize,

    pub nb_base_ops: usize,

    pub nb_memory_ops: usize,

    pub nb_branch_ops: usize,

    pub nb_select: usize,

    pub nb_exp_reverse_bits: usize,

    pub nb_fri_fold: usize,

    pub nb_batch_fri: usize,

    pub nb_print_f: usize,

    pub nb_print_e: usize,

    /// The program.
    pub program: Arc<RecursionProgram<F>>,

    /// Memory. From canonical usize of an Address to a MemoryEntry.
    pub memory: MemVec<F>,

    /// The execution record.
    pub record: ExecutionRecord<F>,

    pub witness_stream: VecDeque<Block<F>>,

    /// The stream that print statements write to.
    pub debug_stdout: Box<dyn Write + Send + 'a>,

    /// Entries for dealing with the Poseidon2 hash state.
    perm: Option<Perm<F, Diffusion>>,

    _marker_ef: PhantomData<EF>,

    _marker_diffusion: PhantomData<Diffusion>,
}

#[derive(Error, Debug)]
pub enum RuntimeError<F: Debug, EF: Debug> {
    #[error(
        "attempted to perform base field division {in1:?}/{in2:?}\n\
        \tin instruction {instr:#?}\n\
        \tnearest backtrace:\n{trace:#?}"
    )]
    DivFOutOfDomain { in1: F, in2: F, instr: BaseAluInstr<F>, trace: Option<Trace> },
    #[error(
        "attempted to perform extension field division {in1:?}/{in2:?}\n\
        \tin instruction {instr:#?}\n\
        \tnearest backtrace:\n{trace:#?}"
    )]
    DivEOutOfDomain { in1: EF, in2: EF, instr: ExtAluInstr<F>, trace: Option<Trace> },
    #[error("failed to print to `debug_stdout`: {0}")]
    DebugPrint(#[from] std::io::Error),
    #[error("attempted to read from empty witness stream")]
    EmptyWitnessStream,
}

impl<F: PrimeField32, EF: ExtensionField<F>, Diffusion> Runtime<'_, F, EF, Diffusion>
where
    Poseidon2<
        F,
        Poseidon2ExternalMatrixGeneral,
        Diffusion,
        PERMUTATION_WIDTH,
        POSEIDON2_SBOX_DEGREE,
    >: CryptographicPermutation<[F; PERMUTATION_WIDTH]>,
{
    pub fn new(
        program: Arc<RecursionProgram<F>>,
        perm: Poseidon2<
            F,
            Poseidon2ExternalMatrixGeneral,
            Diffusion,
            PERMUTATION_WIDTH,
            POSEIDON2_SBOX_DEGREE,
        >,
    ) -> Self {
        let record = ExecutionRecord::<F> { program: program.clone(), ..Default::default() };
        let memory = MemVec::with_capacity(program.total_memory);
        Self {
            timestamp: 0,
            nb_poseidons: 0,
            nb_wide_poseidons: 0,
            nb_bit_decompositions: 0,
            nb_select: 0,
            nb_exp_reverse_bits: 0,
            nb_ext_ops: 0,
            nb_base_ops: 0,
            nb_memory_ops: 0,
            nb_branch_ops: 0,
            nb_fri_fold: 0,
            nb_batch_fri: 0,
            nb_print_f: 0,
            nb_print_e: 0,
            program,
            memory,
            record,
            witness_stream: VecDeque::new(),
            debug_stdout: Box::new(stdout()),
            perm: Some(perm),
            _marker_ef: PhantomData,
            _marker_diffusion: PhantomData,
        }
    }

    pub fn print_stats(&self) {
        if tracing::event_enabled!(tracing::Level::DEBUG) {
            let mut stats = self.record.stats().into_iter().collect::<Vec<_>>();
            stats.sort_unstable();
            tracing::debug!("total events: {}", stats.iter().map(|(_, v)| *v).sum::<usize>());
            for (k, v) in stats {
                tracing::debug!("  {k}: {v}");
            }
        }
    }

    /// # Safety
    ///
    /// Safety is guaranteed if calls to this function (with the given `env` argument) obey the
    /// happens-before relation defined in the documentation of [`RecursionProgram::new_unchecked`].
    ///
    /// This function makes use of interior mutability of `env` via `UnsafeCell`.
    /// All of this function's unsafety stems from the `instruction` argument that indicates'
    /// whether/how to read and write from the memory in `env`. There must be a strict
    /// happens-before relation where reads happen before writes, and memory read from must be
    /// initialized.
    unsafe fn execute_one(
        state: &mut ExecState<F, Diffusion>,
        witness_stream: Option<&mut VecDeque<Block<F>>>,
        instruction: &Instruction<F>,
    ) -> Result<(), RuntimeError<F, EF>> {
        let ExecEnv { memory, perm, debug_stdout } = state.env;
        let record = &mut state.record;
        match *instruction {
            Instruction::BaseAlu(ref instr @ BaseAluInstr { opcode, mult: _, addrs }) => {
                let in1 = memory.mr_unchecked(addrs.in1).val[0];
                let in2 = memory.mr_unchecked(addrs.in2).val[0];
                // Do the computation.
                let out = match opcode {
                    BaseAluOpcode::AddF => in1 + in2,
                    BaseAluOpcode::SubF => in1 - in2,
                    BaseAluOpcode::MulF => in1 * in2,
                    BaseAluOpcode::DivF => match in1.try_div(in2) {
                        Some(x) => x,
                        None => {
                            // Check for division exceptions and error. Note that 0/0 is defined
                            // to be 1.
                            if in1.is_zero() {
                                AbstractField::one()
                            } else {
                                return Err(RuntimeError::DivFOutOfDomain {
                                    in1,
                                    in2,
                                    instr: *instr,
                                    trace: state.resolve_trace().cloned(),
                                });
                            }
                        }
                    },
                };
                memory.mw_unchecked(addrs.out, Block::from(out));
                record.base_alu_events.push(BaseAluEvent { out, in1, in2 });
            }
            Instruction::ExtAlu(ref instr @ ExtAluInstr { opcode, mult: _, addrs }) => {
                let in1 = memory.mr_unchecked(addrs.in1).val;
                let in2 = memory.mr_unchecked(addrs.in2).val;
                // Do the computation.
                let in1_ef = EF::from_base_fn(|i| in1.0[i]);
                let in2_ef = EF::from_base_fn(|i| in2.0[i]);
                let out_ef = match opcode {
                    ExtAluOpcode::AddE => in1_ef + in2_ef,
                    ExtAluOpcode::SubE => in1_ef - in2_ef,
                    ExtAluOpcode::MulE => in1_ef * in2_ef,
                    ExtAluOpcode::DivE => match in1_ef.try_div(in2_ef) {
                        Some(x) => x,
                        None => {
                            // Check for division exceptions and error. Note that 0/0 is defined
                            // to be 1.
                            if in1_ef.is_zero() {
                                AbstractField::one()
                            } else {
                                return Err(RuntimeError::DivEOutOfDomain {
                                    in1: in1_ef,
                                    in2: in2_ef,
                                    instr: *instr,
                                    trace: state.resolve_trace().cloned(),
                                });
                            }
                        }
                    },
                };
                let out = Block::from(out_ef.as_base_slice());
                memory.mw_unchecked(addrs.out, out);
                record.ext_alu_events.push(ExtAluEvent { out, in1, in2 });
            }
            Instruction::Mem(MemInstr {
                addrs: MemIo { inner: addr },
                vals: MemIo { inner: val },
                mult: _,
                kind,
            }) => {
                match kind {
                    MemAccessKind::Read => {
                        let mem_entry = memory.mr_unchecked(addr);
                        assert_eq!(
                            mem_entry.val, val,
                            "stored memory value should be the specified value"
                        );
                    }
                    MemAccessKind::Write => memory.mw_unchecked(addr, val),
                }
                record.mem_const_count += 1;
            }
            Instruction::Poseidon2(ref instr) => {
                let Poseidon2Instr { addrs: Poseidon2Io { input, output }, mults: _ } =
                    instr.as_ref();
                let in_vals = std::array::from_fn(|i| memory.mr_unchecked(input[i]).val[0]);
                let perm_output = perm.permute(in_vals);

                perm_output.iter().zip(output).for_each(|(&val, addr)| {
                    memory.mw_unchecked(*addr, Block::from(val));
                });
                record
                    .poseidon2_events
                    .push(Poseidon2Event { input: in_vals, output: perm_output });
            }
            Instruction::Select(SelectInstr {
                addrs: SelectIo { bit, out1, out2, in1, in2 },
                mult1: _,
                mult2: _,
            }) => {
                let bit = memory.mr_unchecked(bit).val[0];
                let in1 = memory.mr_unchecked(in1).val[0];
                let in2 = memory.mr_unchecked(in2).val[0];
                let out1_val = bit * in2 + (F::one() - bit) * in1;
                let out2_val = bit * in1 + (F::one() - bit) * in2;
                memory.mw_unchecked(out1, Block::from(out1_val));
                memory.mw_unchecked(out2, Block::from(out2_val));
                record.select_events.push(SelectEvent {
                    bit,
                    out1: out1_val,
                    out2: out2_val,
                    in1,
                    in2,
                })
            }
            Instruction::ExpReverseBitsLen(ExpReverseBitsInstr {
                addrs: ExpReverseBitsIo { base, ref exp, result },
                mult: _,
            }) => {
                let base_val = memory.mr_unchecked(base).val[0];
                let exp_bits: Vec<_> =
                    exp.iter().map(|bit| memory.mr_unchecked(*bit).val[0]).collect();
                let exp_val = exp_bits
                    .iter()
                    .enumerate()
                    .fold(0, |acc, (i, &val)| acc + val.as_canonical_u32() * (1 << i));
                let out =
                    base_val.exp_u64(reverse_bits_len(exp_val as usize, exp_bits.len()) as u64);
                memory.mw_unchecked(result, Block::from(out));
                record.exp_reverse_bits_len_events.push(ExpReverseBitsEvent {
                    result: out,
                    base: base_val,
                    exp: exp_bits,
                });
            }
            Instruction::HintBits(HintBitsInstr { ref output_addrs_mults, input_addr }) => {
                let num = memory.mr_unchecked(input_addr).val[0].as_canonical_u32();
                // Decompose the num into LE bits.
                let bits = (0..output_addrs_mults.len())
                    .map(|i| Block::from(F::from_canonical_u32((num >> i) & 1)))
                    .collect::<Vec<_>>();
                // Write the bits to the array at dst.
                for (bit, &(addr, _mult)) in bits.into_iter().zip(output_addrs_mults) {
                    memory.mw_unchecked(addr, bit);
                    record.mem_var_events.push(MemEvent { inner: bit });
                }
            }
            Instruction::HintAddCurve(ref instr) => {
                let HintAddCurveInstr {
                    output_x_addrs_mults,
                    output_y_addrs_mults,
                    input1_x_addrs,
                    input1_y_addrs,
                    input2_x_addrs,
                    input2_y_addrs,
                } = instr.as_ref();
                let input1_x = SepticExtension::<F>::from_base_fn(|i| {
                    memory.mr_unchecked(input1_x_addrs[i]).val[0]
                });
                let input1_y = SepticExtension::<F>::from_base_fn(|i| {
                    memory.mr_unchecked(input1_y_addrs[i]).val[0]
                });
                let input2_x = SepticExtension::<F>::from_base_fn(|i| {
                    memory.mr_unchecked(input2_x_addrs[i]).val[0]
                });
                let input2_y = SepticExtension::<F>::from_base_fn(|i| {
                    memory.mr_unchecked(input2_y_addrs[i]).val[0]
                });
                let point1 = SepticCurve { x: input1_x, y: input1_y };
                let point2 = SepticCurve { x: input2_x, y: input2_y };
                let output = point1.add_incomplete(point2);

                for (val, &(addr, _mult)) in output.x.0.into_iter().zip(output_x_addrs_mults.iter())
                {
                    memory.mw_unchecked(addr, Block::from(val));
                    record.mem_var_events.push(MemEvent { inner: Block::from(val) });
                }
                for (val, &(addr, _mult)) in output.y.0.into_iter().zip(output_y_addrs_mults.iter())
                {
                    memory.mw_unchecked(addr, Block::from(val));
                    record.mem_var_events.push(MemEvent { inner: Block::from(val) });
                }
            }
            Instruction::FriFold(ref instr) => {
                let FriFoldInstr {
                    base_single_addrs,
                    ext_single_addrs,
                    ext_vec_addrs,
                    alpha_pow_mults: _,
                    ro_mults: _,
                } = instr.as_ref();
                let x = memory.mr_unchecked(base_single_addrs.x).val[0];
                let z = memory.mr_unchecked(ext_single_addrs.z).val;
                let z: EF = z.ext();
                let alpha = memory.mr_unchecked(ext_single_addrs.alpha).val;
                let alpha: EF = alpha.ext();
                let mat_opening = ext_vec_addrs
                    .mat_opening
                    .iter()
                    .map(|addr| memory.mr_unchecked(*addr).val)
                    .collect_vec();
                let ps_at_z = ext_vec_addrs
                    .ps_at_z
                    .iter()
                    .map(|addr| memory.mr_unchecked(*addr).val)
                    .collect_vec();

                for m in 0..ps_at_z.len() {
                    // let m = F::from_canonical_u32(m);
                    // Get the opening values.
                    let p_at_x = mat_opening[m];
                    let p_at_x: EF = p_at_x.ext();
                    let p_at_z = ps_at_z[m];
                    let p_at_z: EF = p_at_z.ext();

                    // Calculate the quotient and update the values
                    let quotient = (-p_at_z + p_at_x) / (-z + x);

                    // First we peek to get the current value.
                    let alpha_pow: EF =
                        memory.mr_unchecked(ext_vec_addrs.alpha_pow_input[m]).val.ext();

                    let ro: EF = memory.mr_unchecked(ext_vec_addrs.ro_input[m]).val.ext();

                    let new_ro = ro + alpha_pow * quotient;
                    let new_alpha_pow = alpha_pow * alpha;

                    memory.mw_unchecked(
                        ext_vec_addrs.ro_output[m],
                        Block::from(new_ro.as_base_slice()),
                    );

                    memory.mw_unchecked(
                        ext_vec_addrs.alpha_pow_output[m],
                        Block::from(new_alpha_pow.as_base_slice()),
                    );

                    record.fri_fold_events.push(FriFoldEvent {
                        base_single: FriFoldBaseIo { x },
                        ext_single: FriFoldExtSingleIo {
                            z: Block::from(z.as_base_slice()),
                            alpha: Block::from(alpha.as_base_slice()),
                        },
                        ext_vec: FriFoldExtVecIo {
                            mat_opening: Block::from(p_at_x.as_base_slice()),
                            ps_at_z: Block::from(p_at_z.as_base_slice()),
                            alpha_pow_input: Block::from(alpha_pow.as_base_slice()),
                            ro_input: Block::from(ro.as_base_slice()),
                            alpha_pow_output: Block::from(new_alpha_pow.as_base_slice()),
                            ro_output: Block::from(new_ro.as_base_slice()),
                        },
                    });
                }
            }
            Instruction::BatchFRI(ref instr) => {
                let BatchFRIInstr { base_vec_addrs, ext_single_addrs, ext_vec_addrs, acc_mult: _ } =
                    instr.as_ref();

                let mut acc = EF::zero();
                let p_at_xs = base_vec_addrs
                    .p_at_x
                    .iter()
                    .map(|addr| memory.mr_unchecked(*addr).val[0])
                    .collect_vec();
                let p_at_zs = ext_vec_addrs
                    .p_at_z
                    .iter()
                    .map(|addr| memory.mr_unchecked(*addr).val.ext::<EF>())
                    .collect_vec();
                let alpha_pows: Vec<_> = ext_vec_addrs
                    .alpha_pow
                    .iter()
                    .map(|addr| memory.mr_unchecked(*addr).val.ext::<EF>())
                    .collect_vec();

                for m in 0..p_at_zs.len() {
                    acc += alpha_pows[m] * (p_at_zs[m] - EF::from_base(p_at_xs[m]));
                    record.batch_fri_events.push(BatchFRIEvent {
                        base_vec: BatchFRIBaseVecIo { p_at_x: p_at_xs[m] },
                        ext_single: BatchFRIExtSingleIo { acc: Block::from(acc.as_base_slice()) },
                        ext_vec: BatchFRIExtVecIo {
                            p_at_z: Block::from(p_at_zs[m].as_base_slice()),
                            alpha_pow: Block::from(alpha_pows[m].as_base_slice()),
                        },
                    });
                }

                memory.mw_unchecked(ext_single_addrs.acc, Block::from(acc.as_base_slice()));
            }
            Instruction::PrefixSumChecks(ref instr) => {
                let PrefixSumChecksInstr {
                    addrs: PrefixSumChecksIo { zero, one, x1, x2, accs, field_accs },
                    acc_mults: _,
                    field_acc_mults: _,
                } = instr.as_ref();
                let zero = memory.mr_unchecked(*zero).val[0];
                let one = memory.mr_unchecked(*one).val.ext::<EF>();
                let x1_f = x1.iter().map(|addr| memory.mr_unchecked(*addr).val[0]).collect_vec();
                let x2_ef =
                    x2.iter().map(|addr| memory.mr_unchecked(*addr).val.ext::<EF>()).collect_vec();

                let mut acc = EF::one();
                let mut field_acc = F::zero();
                for m in 0..x1_f.len() {
                    let product = EF::from_base(x1_f[m]) * x2_ef[m];
                    let lagrange_term = EF::one() - x1_f[m] - x2_ef[m] + product + product;
                    let new_field_acc = x1_f[m] + field_acc * F::from_canonical_u32(2);
                    let new_acc = acc * lagrange_term;
                    record.prefix_sum_checks_events.push(PrefixSumChecksEvent {
                        zero,
                        one: Block::from(one.as_base_slice()),
                        x1: x1_f[m],
                        x2: Block::from(x2_ef[m].as_base_slice()),
                        prod: Block::from(product.as_base_slice()),
                        acc: Block::from(acc.as_base_slice()),
                        new_acc: Block::from(new_acc.as_base_slice()),
                        field_acc,
                        new_field_acc,
                    });
                    acc = new_acc;
                    field_acc = new_field_acc;
                    memory.mw_unchecked(accs[m], Block::from(acc.as_base_slice()));
                    memory.mw_unchecked(field_accs[m], Block::from(field_acc));
                }
            }
            Instruction::CommitPublicValues(ref instr) => {
                let pv_addrs = instr.pv_addrs.as_array();
                let pv_values: [F; RECURSIVE_PROOF_NUM_PV_ELTS] =
                    array::from_fn(|i| memory.mr_unchecked(pv_addrs[i]).val[0]);
                record.public_values = *pv_values.as_slice().borrow();
                record
                    .commit_pv_hash_events
                    .push(CommitPublicValuesEvent { public_values: record.public_values });
            }

            Instruction::Print(PrintInstr { ref field_elt_type, addr }) => match field_elt_type {
                FieldEltType::Base => {
                    let f = memory.mr_unchecked(addr).val[0];
                    writeln!(debug_stdout.lock().unwrap(), "PRINTF={f}")
                }
                FieldEltType::Extension => {
                    let ef = memory.mr_unchecked(addr).val;
                    writeln!(debug_stdout.lock().unwrap(), "PRINTEF={ef:?}")
                }
            }
            .map_err(RuntimeError::DebugPrint)?,
            Instruction::HintExt2Felts(HintExt2FeltsInstr { output_addrs_mults, input_addr }) => {
                let fs = memory.mr_unchecked(input_addr).val;
                // Write the bits to the array at dst.
                for (f, (addr, _mult)) in fs.into_iter().zip(output_addrs_mults) {
                    let felt = Block::from(f);
                    memory.mw_unchecked(addr, felt);
                    record.mem_var_events.push(MemEvent { inner: felt });
                }
            }
            Instruction::Hint(HintInstr { ref output_addrs_mults }) => {
                let witness_stream =
                    witness_stream.expect("hint should be called outside parallel contexts");
                // Check that enough Blocks can be read, so `drain` does not panic.
                if witness_stream.len() < output_addrs_mults.len() {
                    return Err(RuntimeError::EmptyWitnessStream);
                }
                let witness = witness_stream.drain(0..output_addrs_mults.len());
                for (&(addr, _mult), val) in zip(output_addrs_mults, witness) {
                    // Inline [`Self::mw`] to mutably borrow multiple fields of `self`.
                    memory.mw_unchecked(addr, val);
                    record.mem_var_events.push(MemEvent { inner: val });
                }
            }
            Instruction::DebugBacktrace(ref backtrace) => {
                cfg_if! {
                    if #[cfg(feature = "debug")] {
                        state.last_trace = Some(backtrace.clone());
                    } else {
                        // Ignore.
                        let _ = backtrace;
                    }
                }
            }
        }

        Ok(())
    }

    /// # Safety
    ///
    /// This function makes the same safety assumptions as [`RecursionProgram::new_unchecked`].
    unsafe fn execute_raw(
        env: &ExecEnv<F, Diffusion>,
        program: &RawProgram<Instruction<F>>,
        root_program: &Arc<RecursionProgram<F>>,
        mut witness_stream: Option<&mut VecDeque<Block<F>>>,
        is_root: bool,
    ) -> Result<ExecutionRecord<F>, RuntimeError<F, EF>> {
        let fresh_record =
            || ExecutionRecord { program: Arc::clone(root_program), ..Default::default() };

        let mut state = ExecState {
            env: env.clone(),
            record: fresh_record(),
            #[cfg(feature = "debug")]
            last_trace: None,
        };

        if is_root {
            if let Some(event_counts) = root_program.event_counts {
                state.record.preallocate(event_counts)
            }
        }

        for block in &program.seq_blocks {
            match block {
                SeqBlock::Basic(basic_block) => {
                    for instruction in &basic_block.instrs {
                        unsafe {
                            Self::execute_one(
                                &mut state,
                                witness_stream.as_deref_mut(),
                                instruction,
                            )
                        }?;
                    }
                }
                SeqBlock::Parallel(vec) => {
                    tracing::debug_span!("parallel", len = vec.len()).in_scope(|| {
                        let span = tracing::Span::current();
                        let mut result = vec
                            .par_iter()
                            .map(|subprogram| {
                                tracing::debug_span!(parent: &span, "block").in_scope(|| {
                                    // Witness stream may not be called inside parallel contexts to
                                    // avoid nondeterminism.
                                    Self::execute_raw(env, subprogram, root_program, None, false)
                                })
                            })
                            .try_reduce(fresh_record, |mut record, mut res| {
                                tracing::debug_span!(parent: &span, "append").in_scope(|| {
                                    record.append(&mut res);
                                });
                                Ok(record)
                            })?;
                        tracing::debug_span!(parent: &span, "append_result").in_scope(|| {
                            state.record.append(&mut result);
                        });
                        Ok::<_, RuntimeError<F, EF>>(())
                    })?;
                }
            }
        }
        Ok(state.record)
    }

    /// Run the program.
    pub fn run(&mut self) -> Result<(), RuntimeError<F, EF>> {
        let record = unsafe {
            Self::execute_raw(
                &ExecEnv {
                    memory: &self.memory,
                    perm: self.perm.as_ref().unwrap(),
                    debug_stdout: &Mutex::new(&mut self.debug_stdout),
                },
                &self.program.inner,
                &self.program,
                Some(&mut self.witness_stream),
                true,
            )
        }?;

        self.record = record;

        Ok(())
    }
}

struct ExecState<'a, 'b, F, Diffusion> {
    pub env: ExecEnv<'a, 'b, F, Diffusion>,
    pub record: ExecutionRecord<F>,
    #[cfg(feature = "debug")]
    pub last_trace: Option<Trace>,
}

impl<F, Diffusion> ExecState<'_, '_, F, Diffusion> {
    fn resolve_trace(&mut self) -> Option<&mut Trace> {
        cfg_if::cfg_if! {
            if #[cfg(feature = "debug")] {
                // False positive.
                #[allow(clippy::manual_inspect)]
                self.last_trace.as_mut().map(|trace| {
                    trace.resolve();
                    trace
                })
            } else {
                None
            }
        }
    }
}

impl<'a, 'b, F, Diffusion> Clone for ExecState<'a, 'b, F, Diffusion>
where
    ExecEnv<'a, 'b, F, Diffusion>: Clone,
    ExecutionRecord<F>: Clone,
{
    fn clone(&self) -> Self {
        let Self {
            env,
            record,
            #[cfg(feature = "debug")]
            last_trace,
        } = self;
        Self {
            env: env.clone(),
            record: record.clone(),
            #[cfg(feature = "debug")]
            last_trace: last_trace.clone(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        let Self {
            env,
            record,
            #[cfg(feature = "debug")]
            last_trace,
        } = self;
        env.clone_from(&source.env);
        record.clone_from(&source.record);
        #[cfg(feature = "debug")]
        last_trace.clone_from(&source.last_trace);
    }
}

struct ExecEnv<'a, 'b, F, Diffusion> {
    pub memory: &'a MemVec<F>,
    pub perm: &'a Perm<F, Diffusion>,
    pub debug_stdout: &'a Mutex<dyn Write + Send + 'b>,
}

impl<F, Diffusion> Clone for ExecEnv<'_, '_, F, Diffusion> {
    fn clone(&self) -> Self {
        let Self { memory, perm, debug_stdout } = self;
        Self { memory, perm, debug_stdout }
    }

    fn clone_from(&mut self, source: &Self) {
        let Self { memory, perm, debug_stdout } = self;
        memory.clone_from(&source.memory);
        perm.clone_from(&source.perm);
        debug_stdout.clone_from(&source.debug_stdout);
    }
}
