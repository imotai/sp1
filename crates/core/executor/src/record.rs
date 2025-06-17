use enum_map::EnumMap;
use hashbrown::HashMap;
use itertools::{EitherOrBoth, Itertools};
use p3_air::AirBuilder;
use p3_field::{AbstractField, PrimeField};
use sp1_stark::{
    air::{
        AirInteraction, InteractionScope, MachineAir, PublicValues, SP1AirBuilder,
        SP1_PROOF_NUM_PV_ELTS,
    },
    septic_digest::SepticDigest,
    shape::Shape,
    InteractionKind, MachineRecord, Word,
};
use std::{
    borrow::Borrow,
    iter::once,
    mem::take,
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};

use crate::{
    events::{
        AUIPCEvent, AluEvent, BranchEvent, ByteLookupEvent, ByteRecord, GlobalInteractionEvent,
        JumpEvent, MemInstrEvent, MemoryInitializeFinalizeEvent, MemoryLocalEvent,
        MemoryRecordEnum, PrecompileEvent, PrecompileEvents, SyscallEvent,
    },
    program::Program,
    syscalls::SyscallCode,
    ByteOpcode, RetainedEventsPreset, RiscvAirId, SplitOpts,
};

/// A record of the execution of a program.
///
/// The trace of the execution is represented as a list of "events" that occur every cycle.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ExecutionRecord {
    /// The program.
    pub program: Arc<Program>,
    /// The number of CPU related events.
    pub cpu_event_count: u32,
    /// A trace of the ADD, and ADDI events.
    pub add_events: Vec<(AluEvent, RTypeRecord)>,
    /// A trace of the ADDI events.
    pub addi_events: Vec<(AluEvent, ITypeRecord)>,
    /// A trace of the MUL events.
    pub mul_events: Vec<(AluEvent, ALUTypeRecord)>,
    /// A trace of the SUB events.
    pub sub_events: Vec<(AluEvent, RTypeRecord)>,
    /// A trace of the XOR, XORI, OR, ORI, AND, and ANDI events.
    pub bitwise_events: Vec<(AluEvent, ALUTypeRecord)>,
    /// A trace of the SLL and SLLI events.
    pub shift_left_events: Vec<(AluEvent, ALUTypeRecord)>,
    /// A trace of the SRL, SRLI, SRA, and SRAI events.
    pub shift_right_events: Vec<(AluEvent, ALUTypeRecord)>,
    /// A trace of the DIV, DIVU, REM, and REMU events.
    pub divrem_events: Vec<(AluEvent, ALUTypeRecord)>,
    /// A trace of the SLT, SLTI, SLTU, and SLTIU events.
    pub lt_events: Vec<(AluEvent, ALUTypeRecord)>,
    /// A trace of load byte instructions.
    pub memory_load_byte_events: Vec<(MemInstrEvent, ITypeRecord)>,
    /// A trace of load half instructions.
    pub memory_load_half_events: Vec<(MemInstrEvent, ITypeRecord)>,
    /// A trace of load word instructions.
    pub memory_load_word_events: Vec<(MemInstrEvent, ITypeRecord)>,
    /// A trace of load instructions with `op_a = x0`.
    pub memory_load_x0_events: Vec<(MemInstrEvent, ITypeRecord)>,
    /// A trace of store byte instructions.
    pub memory_store_byte_events: Vec<(MemInstrEvent, ITypeRecord)>,
    /// A trace of store half instructions.
    pub memory_store_half_events: Vec<(MemInstrEvent, ITypeRecord)>,
    /// A trace of store word instructions.
    pub memory_store_word_events: Vec<(MemInstrEvent, ITypeRecord)>,
    /// A trace of the AUIPC events.
    pub auipc_events: Vec<(AUIPCEvent, JTypeRecord)>,
    /// A trace of the branch events.
    pub branch_events: Vec<(BranchEvent, ITypeRecord)>,
    /// A trace of the JAL events.
    pub jal_events: Vec<(JumpEvent, JTypeRecord)>,
    /// A trace of the JALR events.
    pub jalr_events: Vec<(JumpEvent, ITypeRecord)>,
    /// A trace of the byte lookups that are needed.
    pub byte_lookups: HashMap<ByteLookupEvent, usize>,
    /// A trace of the precompile events.
    pub precompile_events: PrecompileEvents,
    /// A trace of the global memory initialize events.
    pub global_memory_initialize_events: Vec<MemoryInitializeFinalizeEvent>,
    /// A trace of the global memory finalize events.
    pub global_memory_finalize_events: Vec<MemoryInitializeFinalizeEvent>,
    /// A trace of all the shard's local memory events.
    pub cpu_local_memory_access: Vec<MemoryLocalEvent>,
    /// A trace of all the syscall events.
    pub syscall_events: Vec<(SyscallEvent, RTypeRecord)>,
    /// A trace of all the global interaction events.
    pub global_interaction_events: Vec<GlobalInteractionEvent>,
    /// The global culmulative sum.
    pub global_cumulative_sum: Arc<Mutex<SepticDigest<u32>>>,
    /// The global interaction event count.
    pub global_interaction_event_count: u32,
    /// Memory records with `prev_clk >> 24` different from `clk >> 24`.
    pub bump_memory_events: Vec<(MemoryRecordEnum, u32)>,
    /// Record where the `clk >> 24` has incremented.
    pub bump_clk_high_events: Vec<(u64, u64, u32)>,
    /// The public values.
    pub public_values: PublicValues<u32, u32, u64, u32>,
    /// The next nonce to use for a new lookup.
    pub next_nonce: u64,
    /// The shape of the proof.
    pub shape: Option<Shape<RiscvAirId>>,
    /// The predicted counts of the proof.
    pub counts: Option<EnumMap<RiscvAirId, u64>>,
    /// The estimated total trace area of the proof.
    pub estimated_trace_area: u64,
    /// The initial timestamp of the shard.
    pub initial_timestamp: u64,
    /// The final timestamp of the shard.
    pub last_timestamp: u64,
    /// The start program counter.
    pub start_pc: Option<u32>,
    /// The final program counter.
    pub next_pc: u32,
    /// The exit code.
    pub exit_code: u32,
}

impl ExecutionRecord {
    /// Create a new [`ExecutionRecord`].
    #[must_use]
    pub fn new(program: Arc<Program>) -> Self {
        Self { program, ..Default::default() }
    }

    /// Take out events from the [`ExecutionRecord`] that should be deferred to a separate shard.
    ///
    /// Note: we usually defer events that would increase the recursion cost significantly if
    /// included in every shard.
    #[must_use]
    pub fn defer<'a>(
        &mut self,
        retain_presets: impl IntoIterator<Item = &'a RetainedEventsPreset>,
    ) -> ExecutionRecord {
        let mut execution_record = ExecutionRecord::new(self.program.clone());
        execution_record.precompile_events = std::mem::take(&mut self.precompile_events);

        // Take back the events that should be retained.
        self.precompile_events.events.extend(
            retain_presets.into_iter().flat_map(RetainedEventsPreset::syscall_codes).filter_map(
                |code| execution_record.precompile_events.events.remove(code).map(|x| (*code, x)),
            ),
        );

        execution_record.global_memory_initialize_events =
            std::mem::take(&mut self.global_memory_initialize_events);
        execution_record.global_memory_finalize_events =
            std::mem::take(&mut self.global_memory_finalize_events);
        execution_record
    }

    /// Splits the deferred [`ExecutionRecord`] into multiple [`ExecutionRecord`]s, each which
    /// contain a "reasonable" number of deferred events.
    ///
    /// The optional `last_record` will be provided if there are few enough deferred events that
    /// they can all be packed into the already existing last record.
    pub fn split(
        &mut self,
        done: bool,
        last_record: Option<&mut ExecutionRecord>,
        opts: SplitOpts,
    ) -> Vec<ExecutionRecord> {
        let mut shards = Vec::new();

        let precompile_events = take(&mut self.precompile_events);

        for (syscall_code, events) in precompile_events.into_iter() {
            let threshold = match syscall_code {
                SyscallCode::KECCAK_PERMUTE => opts.keccak,
                SyscallCode::SHA_EXTEND => opts.sha_extend,
                SyscallCode::SHA_COMPRESS => opts.sha_compress,
                SyscallCode::SECP256K1_ADD
                | SyscallCode::SECP256R1_ADD
                | SyscallCode::BN254_ADD
                | SyscallCode::ED_ADD
                | SyscallCode::SECP256K1_DECOMPRESS
                | SyscallCode::SECP256R1_DECOMPRESS
                | SyscallCode::ED_DECOMPRESS => opts.ec_add_256bit,
                SyscallCode::SECP256K1_DOUBLE
                | SyscallCode::SECP256R1_DOUBLE
                | SyscallCode::BN254_DOUBLE => opts.ec_double_256bit,
                SyscallCode::BN254_FP_ADD
                | SyscallCode::BN254_FP_SUB
                | SyscallCode::BN254_FP_MUL => opts.fp_operation_256bit,
                SyscallCode::BN254_FP2_ADD
                | SyscallCode::BN254_FP2_SUB
                | SyscallCode::BN254_FP2_MUL => opts.fp2_operation_256bit,
                SyscallCode::BLS12381_FP_ADD
                | SyscallCode::BLS12381_FP_SUB
                | SyscallCode::BLS12381_FP_MUL => opts.fp_operation_384bit,
                SyscallCode::BLS12381_FP2_ADD
                | SyscallCode::BLS12381_FP2_SUB
                | SyscallCode::BLS12381_FP2_MUL => opts.fp2_operation_384bit,
                _ => opts.deferred,
            };

            let chunks = events.chunks_exact(threshold);
            if done {
                let remainder = chunks.remainder().to_vec();
                if !remainder.is_empty() {
                    let mut execution_record = ExecutionRecord::new(self.program.clone());
                    execution_record.precompile_events.insert(syscall_code, remainder);
                    shards.push(execution_record);
                }
            } else {
                self.precompile_events.insert(syscall_code, chunks.remainder().to_vec());
            }
            let mut event_shards = chunks
                .map(|chunk| {
                    let mut execution_record = ExecutionRecord::new(self.program.clone());
                    execution_record.precompile_events.insert(syscall_code, chunk.to_vec());
                    execution_record
                })
                .collect::<Vec<_>>();
            shards.append(&mut event_shards);
        }

        if done {
            self.global_memory_initialize_events.sort_by_key(|event| event.addr);
            self.global_memory_finalize_events.sort_by_key(|event| event.addr);

            // If there are no precompile shards, and `last_record` is Some, pack the memory events
            // into the last record.
            let pack_memory_events_into_last_record = last_record.is_some() && shards.is_empty();
            let mut blank_record = ExecutionRecord::new(self.program.clone());

            // If `last_record` is None, use a blank record to store the memory events.
            let last_record_ref = if pack_memory_events_into_last_record {
                last_record.unwrap()
            } else {
                &mut blank_record
            };

            let mut init_addr_word = 0;
            let mut finalize_addr_word = 0;
            for mem_chunks in self
                .global_memory_initialize_events
                .chunks(opts.memory)
                .zip_longest(self.global_memory_finalize_events.chunks(opts.memory))
            {
                let (mem_init_chunk, mem_finalize_chunk) = match mem_chunks {
                    EitherOrBoth::Both(mem_init_chunk, mem_finalize_chunk) => {
                        (mem_init_chunk, mem_finalize_chunk)
                    }
                    EitherOrBoth::Left(mem_init_chunk) => (mem_init_chunk, [].as_slice()),
                    EitherOrBoth::Right(mem_finalize_chunk) => ([].as_slice(), mem_finalize_chunk),
                };
                last_record_ref.global_memory_initialize_events.extend_from_slice(mem_init_chunk);
                last_record_ref.public_values.previous_init_addr_word = init_addr_word;
                if let Some(last_event) = mem_init_chunk.last() {
                    init_addr_word = last_event.addr;
                }
                last_record_ref.public_values.last_init_addr_word = init_addr_word;

                last_record_ref.global_memory_finalize_events.extend_from_slice(mem_finalize_chunk);
                last_record_ref.public_values.previous_finalize_addr_word = finalize_addr_word;
                if let Some(last_event) = mem_finalize_chunk.last() {
                    finalize_addr_word = last_event.addr;
                }
                last_record_ref.public_values.last_finalize_addr_word = finalize_addr_word;

                if !pack_memory_events_into_last_record {
                    // If not packing memory events into the last record, add 'last_record_ref'
                    // to the returned records. `take` replaces `blank_program` with the default.
                    shards.push(take(last_record_ref));

                    // Reset the last record so its program is the correct one. (The default program
                    // provided by `take` contains no instructions.)
                    last_record_ref.program = self.program.clone();
                }
            }
        }
        shards
    }

    /// Return the number of rows needed for a chip, according to the proof shape specified in the
    /// struct.
    ///
    /// **deprecated**: TODO: remove this method.
    pub fn fixed_log2_rows<F: PrimeField, A: MachineAir<F>>(&self, _air: &A) -> Option<usize> {
        None
    }

    /// Determines whether the execution record contains CPU events.
    #[must_use]
    pub fn contains_cpu(&self) -> bool {
        self.cpu_event_count > 0
    }

    #[inline]
    /// Add a precompile event to the execution record.
    pub fn add_precompile_event(
        &mut self,
        syscall_code: SyscallCode,
        syscall_event: SyscallEvent,
        event: PrecompileEvent,
    ) {
        self.precompile_events.add_event(syscall_code, syscall_event, event);
    }

    /// Get all the precompile events for a syscall code.
    #[inline]
    #[must_use]
    pub fn get_precompile_events(
        &self,
        syscall_code: SyscallCode,
    ) -> &Vec<(SyscallEvent, PrecompileEvent)> {
        self.precompile_events.get_events(syscall_code).expect("Precompile events not found")
    }

    /// Get all the local memory events.
    #[inline]
    pub fn get_local_mem_events(&self) -> impl Iterator<Item = &MemoryLocalEvent> {
        let precompile_local_mem_events = self.precompile_events.get_local_mem_events();
        precompile_local_mem_events.chain(self.cpu_local_memory_access.iter())
    }
}

/// A memory access record.
#[derive(Debug, Copy, Clone, Default)]
pub struct MemoryAccessRecord {
    /// The memory access of the `a` register.
    pub a: Option<MemoryRecordEnum>,
    /// The memory access of the `b` register.
    pub b: Option<MemoryRecordEnum>,
    /// The memory access of the `c` register.
    pub c: Option<MemoryRecordEnum>,
    /// The memory access of the `memory` register.
    pub memory: Option<MemoryRecordEnum>,
}

/// Memory record where all three operands are registers.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RTypeRecord {
    /// The register `op_a` record.
    pub a: MemoryRecordEnum,
    /// The register `op_b` record.
    pub b: MemoryRecordEnum,
    /// The register `op_c` record.
    pub c: MemoryRecordEnum,
}

impl From<MemoryAccessRecord> for RTypeRecord {
    fn from(value: MemoryAccessRecord) -> Self {
        Self {
            a: value.a.expect("expected MemoryRecord for op_a in RTypeRecord"),
            b: value.b.expect("expected MemoryRecord for op_b in RTypeRecord"),
            c: value.c.expect("expected MemoryRecord for op_c in RTypeRecord"),
        }
    }
}

/// Memory record where the first two operands are registers.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ITypeRecord {
    /// The register `op_a` record.
    pub a: MemoryRecordEnum,
    /// The register `op_b` record.
    pub b: MemoryRecordEnum,
}

impl From<MemoryAccessRecord> for ITypeRecord {
    fn from(value: MemoryAccessRecord) -> Self {
        debug_assert!(value.c.is_none());
        Self {
            a: value.a.expect("expected MemoryRecord for op_a in ITypeRecord"),
            b: value.b.expect("expected MemoryRecord for op_b in ITypeRecord"),
        }
    }
}

/// Memory record where only one operand is a register.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct JTypeRecord {
    /// The register `op_a` record.
    pub a: MemoryRecordEnum,
}

impl From<MemoryAccessRecord> for JTypeRecord {
    fn from(value: MemoryAccessRecord) -> Self {
        debug_assert!(value.b.is_none());
        debug_assert!(value.c.is_none());
        Self { a: value.a.expect("expected MemoryRecord for op_a in JTypeRecord") }
    }
}

/// Memory record where only the first two operands are known to be registers, but the third isn't.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ALUTypeRecord {
    /// The register `op_a` record.
    pub a: MemoryRecordEnum,
    /// The register `op_b` record.
    pub b: MemoryRecordEnum,
    /// The register `op_c` record.
    pub c: Option<MemoryRecordEnum>,
}

impl From<MemoryAccessRecord> for ALUTypeRecord {
    fn from(value: MemoryAccessRecord) -> Self {
        Self {
            a: value.a.expect("expected MemoryRecord for op_a in ALUTypeRecord"),
            b: value.b.expect("expected MemoryRecord for op_b in ALUTypeRecord"),
            c: value.c,
        }
    }
}

impl MachineRecord for ExecutionRecord {
    fn stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("cpu_events".to_string(), self.cpu_event_count as usize);
        stats.insert("add_events".to_string(), self.add_events.len());
        stats.insert("mul_events".to_string(), self.mul_events.len());
        stats.insert("sub_events".to_string(), self.sub_events.len());
        stats.insert("bitwise_events".to_string(), self.bitwise_events.len());
        stats.insert("shift_left_events".to_string(), self.shift_left_events.len());
        stats.insert("shift_right_events".to_string(), self.shift_right_events.len());
        stats.insert("divrem_events".to_string(), self.divrem_events.len());
        stats.insert("lt_events".to_string(), self.lt_events.len());
        stats.insert("load_byte_events".to_string(), self.memory_load_byte_events.len());
        stats.insert("load_half_events".to_string(), self.memory_load_half_events.len());
        stats.insert("load_word_events".to_string(), self.memory_load_word_events.len());
        stats.insert("load_x0_events".to_string(), self.memory_load_x0_events.len());
        stats.insert("store_byte_events".to_string(), self.memory_store_byte_events.len());
        stats.insert("store_half_events".to_string(), self.memory_store_half_events.len());
        stats.insert("store_word_events".to_string(), self.memory_store_word_events.len());
        stats.insert("branch_events".to_string(), self.branch_events.len());
        stats.insert("jal_events".to_string(), self.jal_events.len());
        stats.insert("jalr_events".to_string(), self.jalr_events.len());
        stats.insert("auipc_events".to_string(), self.auipc_events.len());

        for (syscall_code, events) in self.precompile_events.iter() {
            stats.insert(format!("syscall {syscall_code:?}"), events.len());
        }

        stats.insert(
            "global_memory_initialize_events".to_string(),
            self.global_memory_initialize_events.len(),
        );
        stats.insert(
            "global_memory_finalize_events".to_string(),
            self.global_memory_finalize_events.len(),
        );
        stats.insert("local_memory_access_events".to_string(), self.cpu_local_memory_access.len());
        if self.contains_cpu() {
            stats.insert("byte_lookups".to_string(), self.byte_lookups.len());
        }
        // Filter out the empty events.
        stats.retain(|_, v| *v != 0);
        stats
    }

    fn append(&mut self, other: &mut ExecutionRecord) {
        self.cpu_event_count += other.cpu_event_count;
        other.cpu_event_count = 0;
        self.public_values.global_count += other.public_values.global_count;
        other.public_values.global_count = 0;
        self.public_values.global_init_count += other.public_values.global_init_count;
        other.public_values.global_init_count = 0;
        self.public_values.global_finalize_count += other.public_values.global_finalize_count;
        other.public_values.global_finalize_count = 0;
        self.estimated_trace_area += other.estimated_trace_area;
        other.estimated_trace_area = 0;
        self.add_events.append(&mut other.add_events);
        self.sub_events.append(&mut other.sub_events);
        self.mul_events.append(&mut other.mul_events);
        self.bitwise_events.append(&mut other.bitwise_events);
        self.shift_left_events.append(&mut other.shift_left_events);
        self.shift_right_events.append(&mut other.shift_right_events);
        self.divrem_events.append(&mut other.divrem_events);
        self.lt_events.append(&mut other.lt_events);
        self.memory_load_byte_events.append(&mut other.memory_load_byte_events);
        self.memory_load_half_events.append(&mut other.memory_load_half_events);
        self.memory_load_word_events.append(&mut other.memory_load_word_events);
        self.memory_load_x0_events.append(&mut other.memory_load_x0_events);
        self.memory_store_byte_events.append(&mut other.memory_store_byte_events);
        self.memory_store_half_events.append(&mut other.memory_store_half_events);
        self.memory_store_word_events.append(&mut other.memory_store_word_events);
        self.branch_events.append(&mut other.branch_events);
        self.jal_events.append(&mut other.jal_events);
        self.jalr_events.append(&mut other.jalr_events);
        self.auipc_events.append(&mut other.auipc_events);
        self.syscall_events.append(&mut other.syscall_events);
        self.bump_memory_events.append(&mut other.bump_memory_events);
        self.bump_clk_high_events.append(&mut other.bump_clk_high_events);
        self.precompile_events.append(&mut other.precompile_events);

        if self.byte_lookups.is_empty() {
            self.byte_lookups = std::mem::take(&mut other.byte_lookups);
        } else {
            self.add_byte_lookup_events_from_maps(vec![&other.byte_lookups]);
        }

        self.global_memory_initialize_events.append(&mut other.global_memory_initialize_events);
        self.global_memory_finalize_events.append(&mut other.global_memory_finalize_events);
        self.cpu_local_memory_access.append(&mut other.cpu_local_memory_access);
        self.global_interaction_events.append(&mut other.global_interaction_events);
    }

    /// Retrieves the public values.  This method is needed for the `MachineRecord` trait, since
    fn public_values<F: AbstractField>(&self) -> Vec<F> {
        let mut public_values = self.public_values;
        public_values.global_cumulative_sum = *self.global_cumulative_sum.lock().unwrap();
        public_values.to_vec()
    }

    /// Constrains the public values.
    #[allow(clippy::type_complexity)]
    fn eval_public_values<AB: SP1AirBuilder>(builder: &mut AB) {
        let public_values_slice: [AB::PublicVar; SP1_PROOF_NUM_PV_ELTS] =
            core::array::from_fn(|i| builder.public_values()[i]);
        let public_values: &PublicValues<
            [AB::PublicVar; 4],
            Word<AB::PublicVar>,
            [AB::PublicVar; 4],
            AB::PublicVar,
        > = public_values_slice.as_slice().borrow();

        Self::eval_state(public_values, builder);
        Self::eval_global_sum(public_values, builder);
        Self::eval_global_memory_init(public_values, builder);
        Self::eval_global_memory_finalize(public_values, builder);
    }
}

impl ByteRecord for ExecutionRecord {
    fn add_byte_lookup_event(&mut self, blu_event: ByteLookupEvent) {
        *self.byte_lookups.entry(blu_event).or_insert(0) += 1;
    }

    #[inline]
    fn add_byte_lookup_events_from_maps(
        &mut self,
        new_events: Vec<&HashMap<ByteLookupEvent, usize>>,
    ) {
        for new_blu_map in new_events {
            for (blu_event, count) in new_blu_map.iter() {
                *self.byte_lookups.entry(*blu_event).or_insert(0) += count;
            }
        }
    }
}

impl ExecutionRecord {
    #[allow(clippy::type_complexity)]
    fn eval_state<AB: SP1AirBuilder>(
        public_values: &PublicValues<
            [AB::PublicVar; 4],
            Word<AB::PublicVar>,
            [AB::PublicVar; 4],
            AB::PublicVar,
        >,
        builder: &mut AB,
    ) {
        let initial_timestamp_high = public_values.initial_timestamp[1].into()
            + public_values.initial_timestamp[0].into() * AB::Expr::from_canonical_u32(1 << 8);
        let initial_timestamp_low = public_values.initial_timestamp[3].into()
            + public_values.initial_timestamp[2].into() * AB::Expr::from_canonical_u32(1 << 16);
        let last_timestamp_high = public_values.last_timestamp[1].into()
            + public_values.last_timestamp[0].into() * AB::Expr::from_canonical_u32(1 << 8);
        let last_timestamp_low = public_values.last_timestamp[3].into()
            + public_values.last_timestamp[2].into() * AB::Expr::from_canonical_u32(1 << 16);

        // Range check all the timestamp limbs.
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::Range as u32),
            public_values.initial_timestamp[0].into(),
            AB::Expr::from_canonical_u32(16),
            AB::Expr::zero(),
            AB::Expr::one(),
        );
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::Range as u32),
            public_values.initial_timestamp[3].into(),
            AB::Expr::from_canonical_u32(16),
            AB::Expr::zero(),
            AB::Expr::one(),
        );
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::Range as u32),
            public_values.last_timestamp[0].into(),
            AB::Expr::from_canonical_u32(16),
            AB::Expr::zero(),
            AB::Expr::one(),
        );
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::Range as u32),
            public_values.last_timestamp[3].into(),
            AB::Expr::from_canonical_u32(16),
            AB::Expr::zero(),
            AB::Expr::one(),
        );
        builder.send_byte(
            AB::Expr::from_canonical_u8(ByteOpcode::U8Range as u8),
            AB::Expr::zero(),
            public_values.initial_timestamp[1],
            public_values.initial_timestamp[2],
            AB::Expr::one(),
        );
        builder.send_byte(
            AB::Expr::from_canonical_u8(ByteOpcode::U8Range as u8),
            AB::Expr::zero(),
            public_values.last_timestamp[1],
            public_values.last_timestamp[2],
            AB::Expr::one(),
        );

        // Send and receive the initial and last state.
        builder.send_state(
            initial_timestamp_high.clone(),
            initial_timestamp_low.clone(),
            public_values.start_pc,
            AB::Expr::one(),
        );
        builder.receive_state(
            last_timestamp_high.clone(),
            last_timestamp_low.clone(),
            public_values.next_pc,
            AB::Expr::one(),
        );

        // If execution shard is not incremented, assert that timestamp and pc remains equal.
        let increment_execution_shard =
            public_values.next_execution_shard.into() - public_values.execution_shard.into();
        builder.assert_bool(increment_execution_shard.clone());
        builder
            .when_not(increment_execution_shard.clone())
            .assert_eq(initial_timestamp_low.clone(), last_timestamp_low.clone());
        builder
            .when_not(increment_execution_shard.clone())
            .assert_eq(initial_timestamp_high.clone(), last_timestamp_high.clone());
        builder
            .when_not(increment_execution_shard.clone())
            .assert_eq(public_values.start_pc, public_values.next_pc);

        // IsZeroOperation on the high bits of the timestamp.
        builder.assert_bool(public_values.is_timestamp_high_eq);
        builder.assert_eq(
            (last_timestamp_high.clone() - initial_timestamp_high.clone())
                * public_values.inv_timestamp_high.into(),
            AB::Expr::one() - public_values.is_timestamp_high_eq.into(),
        );
        builder.assert_zero(
            (last_timestamp_high.clone() - initial_timestamp_high.clone())
                * public_values.is_timestamp_high_eq.into(),
        );

        // IsZeroOperation on the low bits of the timestamp.
        builder.assert_bool(public_values.is_timestamp_low_eq);
        builder.assert_eq(
            (last_timestamp_low.clone() - initial_timestamp_low.clone())
                * public_values.inv_timestamp_low.into(),
            AB::Expr::one() - public_values.is_timestamp_low_eq.into(),
        );
        builder.assert_zero(
            (last_timestamp_low.clone() - initial_timestamp_low.clone())
                * public_values.is_timestamp_low_eq.into(),
        );

        // If the execution shard is incremented, then the timestamp is different.
        builder.assert_eq(
            AB::Expr::one() - increment_execution_shard.clone(),
            public_values.is_timestamp_high_eq.into() * public_values.is_timestamp_low_eq.into(),
        );
    }

    #[allow(clippy::type_complexity)]
    fn eval_global_sum<AB: SP1AirBuilder>(
        public_values: &PublicValues<
            [AB::PublicVar; 4],
            Word<AB::PublicVar>,
            [AB::PublicVar; 4],
            AB::PublicVar,
        >,
        builder: &mut AB,
    ) {
        let initial_sum = SepticDigest::<AB::F>::zero().0;
        builder.send(
            AirInteraction::new(
                once(AB::Expr::zero())
                    .chain(initial_sum.x.0.into_iter().map(Into::into))
                    .chain(initial_sum.y.0.into_iter().map(Into::into))
                    .collect(),
                AB::Expr::one(),
                InteractionKind::GlobalAccumulation,
            ),
            InteractionScope::Local,
        );
        builder.receive(
            AirInteraction::new(
                once(public_values.global_count.into())
                    .chain(public_values.global_cumulative_sum.0.x.0.map(Into::into))
                    .chain(public_values.global_cumulative_sum.0.y.0.map(Into::into))
                    .collect(),
                AB::Expr::one(),
                InteractionKind::GlobalAccumulation,
            ),
            InteractionScope::Local,
        );
    }

    #[allow(clippy::type_complexity)]
    fn eval_global_memory_init<AB: SP1AirBuilder>(
        public_values: &PublicValues<
            [AB::PublicVar; 4],
            Word<AB::PublicVar>,
            [AB::PublicVar; 4],
            AB::PublicVar,
        >,
        builder: &mut AB,
    ) {
        builder.send(
            AirInteraction::new(
                once(AB::Expr::zero())
                    .chain(public_values.previous_init_addr_word.into_iter().map(Into::into))
                    .chain(once(AB::Expr::one()))
                    .collect(),
                AB::Expr::one(),
                InteractionKind::MemoryGlobalInitControl,
            ),
            InteractionScope::Local,
        );
        builder.receive(
            AirInteraction::new(
                once(public_values.global_init_count.into())
                    .chain(public_values.last_init_addr_word.into_iter().map(Into::into))
                    .chain(once(AB::Expr::one()))
                    .collect(),
                AB::Expr::one(),
                InteractionKind::MemoryGlobalInitControl,
            ),
            InteractionScope::Local,
        );
    }

    #[allow(clippy::type_complexity)]
    fn eval_global_memory_finalize<AB: SP1AirBuilder>(
        public_values: &PublicValues<
            [AB::PublicVar; 4],
            Word<AB::PublicVar>,
            [AB::PublicVar; 4],
            AB::PublicVar,
        >,
        builder: &mut AB,
    ) {
        builder.send(
            AirInteraction::new(
                once(AB::Expr::zero())
                    .chain(public_values.previous_finalize_addr_word.into_iter().map(Into::into))
                    .chain(once(AB::Expr::one()))
                    .collect(),
                AB::Expr::one(),
                InteractionKind::MemoryGlobalFinalizeControl,
            ),
            InteractionScope::Local,
        );
        builder.receive(
            AirInteraction::new(
                once(public_values.global_finalize_count.into())
                    .chain(public_values.last_finalize_addr_word.into_iter().map(Into::into))
                    .chain(once(AB::Expr::one()))
                    .collect(),
                AB::Expr::one(),
                InteractionKind::MemoryGlobalFinalizeControl,
            ),
            InteractionScope::Local,
        );
    }
}
