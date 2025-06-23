#[cfg(feature = "profiling")]
use std::{fs::File, io::BufWriter};
use std::{num::Wrapping, str::FromStr, sync::Arc};

use crate::estimator::RecordEstimator;
#[cfg(feature = "profiling")]
use crate::profiler::Profiler;

use clap::ValueEnum;
use enum_map::EnumMap;
use hashbrown::HashMap;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
// use sp1_primitives::consts::BABYBEAR_PRIME;
use sp1_stark::air::PublicValues;
use strum::IntoEnumIterator;
use thiserror::Error;

use crate::{
    context::{IoOptions, SP1Context},
    estimate_trace_elements,
    events::{
        AUIPCEvent, AluEvent, BranchEvent, JumpEvent, LogicalShard, MemInstrEvent,
        MemoryAccessPosition, MemoryEntry, MemoryInitializeFinalizeEvent, MemoryLocalEvent,
        MemoryReadRecord, MemoryWriteRecord, Shard, SyscallEvent,
        NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC,
    },
    hook::{HookEnv, HookRegistry},
    memory::{Entry, Memory},
    pad_rv32im_event_counts,
    record::{ExecutionRecord, MemoryAccessRecord},
    report::ExecutionReport,
    state::{ExecutionState, ForkState},
    subproof::SubproofVerifier,
    syscalls::{get_syscall, SyscallCode, SyscallContext},
    ALUTypeRecord, ITypeRecord, Instruction, JTypeRecord, Opcode, Program, RTypeRecord, Register,
    RetainedEventsPreset, RiscvAirId, SP1CoreOpts, ShardingThreshold,
};

/// Max u64 value.
pub const M64: u64 = 0xFFFFFFFFFFFFFFFF;

/// The increment for the program counter.  Is used for all instructions except
/// for branches and jumps.
pub const PC_INC: u32 = 4;
/// The default increment for the timestamp.
pub const CLK_INC: u32 = 8;
/// The executor uses this PC to determine if the program has halted.
/// As a PC, it is invalid since it is not a multiple of [`PC_INC`].
pub const HALT_PC_REL: u32 = 1;

/// The maximum number of instructions in a program.
pub const MAX_PROGRAM_SIZE: usize = 1 << 22;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Whether to verify deferred proofs during execution.
pub enum DeferredProofVerification {
    /// Verify deferred proofs during execution.
    Enabled,
    /// Skip verification of deferred proofs
    Disabled,
}

impl From<bool> for DeferredProofVerification {
    fn from(value: bool) -> Self {
        if value {
            DeferredProofVerification::Enabled
        } else {
            DeferredProofVerification::Disabled
        }
    }
}

/// An executor for the SP1 RISC-V zkVM.
///
/// The exeuctor is responsible for executing a user program and tracing important events which
/// occur during execution (i.e., memory reads, alu operations, etc).
pub struct Executor<'a> {
    /// The program.
    pub program: Arc<Program>,

    /// The length of the program.
    pub program_len: u64,

    /// The state of the execution.
    pub state: ExecutionState,

    /// Memory addresses that were touched in this batch of shards. Used to minimize the size of
    /// checkpoints.
    pub memory_checkpoint: Memory<Option<MemoryEntry>>,

    /// Memory addresses that were initialized in this batch of shards. Used to minimize the size
    /// of checkpoints. The value stored is whether or not it had a value at the beginning of
    /// the batch.
    pub uninitialized_memory_checkpoint: Memory<bool>,

    /// Statistics for event counts.
    pub local_counts: LocalCounts,

    /// Report of the program execution.
    pub report: ExecutionReport,

    /// The memory accesses for the current cycle.
    pub memory_accesses: MemoryAccessRecord,

    /// Whether we should write to the report.
    pub print_report: bool,

    /// Data used to estimate total trace area.
    pub record_estimator: Option<Box<RecordEstimator>>,

    /// Whether we should emit global memory init and finalize events. This can be enabled in
    /// Checkpoint mode and disabled in Trace mode.
    pub emit_global_memory_events: bool,

    /// The maximum size of each shard.
    pub shard_size: u32,

    /// The maximum number of shards to execute at once.
    pub shard_batch_size: u32,

    /// The maximum number of cycles for a syscall.
    pub max_syscall_cycles: u32,

    /// The options for the runtime.
    pub opts: SP1CoreOpts,

    /// The maximum number of cpu cycles to use for execution.
    pub max_cycles: Option<u64>,

    /// The current trace of the execution that is being collected.
    pub record: Box<ExecutionRecord>,

    /// The collected records, split by cpu cycles.
    pub records: Vec<Box<ExecutionRecord>>,

    /// Local memory access events.
    pub local_memory_access: HashMap<u64, MemoryLocalEvent>,

    /// A counter for the number of cycles that have been executed in certain functions.
    pub cycle_tracker: HashMap<String, (u64, u32)>,

    /// A buffer for stdout and stderr IO.
    pub io_buf: HashMap<u32, String>,

    /// The ZKVM program profiler.
    ///
    /// Keeps track of the number of cycles spent in each function.
    #[cfg(feature = "profiling")]
    pub profiler: Option<(Profiler, BufWriter<File>)>,

    /// The state of the runtime when in unconstrained mode.
    pub unconstrained_state: Box<ForkState>,

    /// Verifier used to sanity check `verify_sp1_proof` during runtime.
    pub subproof_verifier: Option<Arc<dyn SubproofVerifier>>,

    /// Registry of hooks, to be invoked by writing to certain file descriptors.
    pub hook_registry: HookRegistry<'a>,

    /// The costs of the program.
    pub costs: EnumMap<RiscvAirId, u64>,

    /// Skip deferred proof verification. This check is informational only, not related to circuit
    /// correctness.
    pub deferred_proof_verification: DeferredProofVerification,

    /// The frequency to check the stopping condition.
    pub size_check_frequency: u64,

    /// The maximum trace size and table height to allow.
    pub sharding_threshold: Option<ShardingThreshold>,

    /// Syscalls that have been overridden to be internal instead of external.
    pub internal_syscalls_override: Vec<SyscallCode>,

    /// The options for the IO.
    pub io_options: IoOptions<'a>,

    /// The total number of unconstrained cycles.
    pub total_unconstrained_cycles: u64,

    /// Temporary event counts for the current shard. This is a field to reuse memory.
    event_counts: EnumMap<RiscvAirId, u64>,
}

/// The configuration of the executor.
pub trait ExecutorConfig {
    /// The mode of the executor.
    const MODE: ExecutorMode;
    /// Whether the executor is in unconstrained mode.
    const UNCONSTRAINED: bool;
}

/// The simple mode of the executor.
pub struct Simple;
impl ExecutorConfig for Simple {
    const MODE: ExecutorMode = ExecutorMode::Simple;
    const UNCONSTRAINED: bool = false;
}

/// The checkpoint mode of the executor.
pub struct Checkpoint;
impl ExecutorConfig for Checkpoint {
    const MODE: ExecutorMode = ExecutorMode::Checkpoint;
    const UNCONSTRAINED: bool = false;
}

/// The trace mode of the executor.
pub struct Trace;
impl ExecutorConfig for Trace {
    const MODE: ExecutorMode = ExecutorMode::Trace;
    const UNCONSTRAINED: bool = false;
}

/// The unconstrained mode of the executor.
pub struct Unconstrained;
impl ExecutorConfig for Unconstrained {
    const MODE: ExecutorMode = ExecutorMode::Simple;
    const UNCONSTRAINED: bool = true;
}

/// The different modes the executor can run in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
pub enum ExecutorMode {
    /// Run the execution with no tracing or checkpointing.
    Simple,
    /// Run the execution with checkpoints for memory.
    Checkpoint,
    /// Run the execution with full tracing of events.
    Trace,
    /// Run the execution with full tracing of events and size bounds for shape collection.
    ShapeCollection,
}

/// Information about event counts which are relevant for shape fixing.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct LocalCounts {
    /// The event counts.
    pub event_counts: Box<EnumMap<Opcode, u64>>,
    /// The retained precompile counts.
    pub retained_precompile_counts: Box<EnumMap<RiscvAirId, u64>>,
    /// The load x0 counts.
    pub load_x0_counts: u64,
    /// The number of syscalls sent globally in the current shard.
    pub syscalls_sent: usize,
    /// The number of addresses touched in this shard.
    ///
    /// We increment the local memory event counter precisely when the main shard touches an
    /// address that was last touched by another shard. (Main shards, sometimes referred to as
    /// core shards, are the shards that are directly produced by the executor and correspond to
    /// usual RISC-V interpretation.)
    ///
    /// We now describe the logic used to increment the counter (which is replicated in several
    /// places). Let `lshard` refer to the new/current [`LogicalShard`] and `record.lshard` refer
    /// to the `LogicalShard` of the last memory operation associated with the address being
    /// modified. To check for the above situation, we require two conditions:
    ///
    /// - `!lshard.external_flag()`: checks that the current shard is not an external shard, i.e.
    ///   it is a main shard.
    /// - `record.lshard != lshard`: checks that the address was last touched by a shard other than
    ///   the current one. The (packed) fields of `LogicalShard` check for two cases:
    ///   - If [`LogicalShard::external_flag`] is different, then one shard is external and one is
    ///     not. If this field is the only difference, then (currently) one shard is the main shard
    ///     and the other is a precompile invoked while interpreting in the main shard. This is
    ///     because we set `external_flag` only when creating a [`SyscallContext`].
    ///   - If [`LogicalShard::shard`] is different, then the shards are different in the sense of
    ///     the SP1 proof system and memory argument. If this field is the only difference, then
    ///     one shard is the current shard and the other is a previous shard in the execution or
    ///     the memory initialization shard.
    ///
    /// Therefore, comparing equality of consecutive `LogicalShard`s in an address's memory
    /// operation sequence enables detection of moments when its memory entry is transferred
    /// between shards via the global (inter-shard) memory argument. By constantly testing
    /// equality, we can detect these interruptions and calculate the endpoints of an
    /// uninterrupted telescoping series of memory operations in a single SP1 shard --
    /// that is, the data of a local memory event.
    pub local_mem: usize,
}

/// Errors that the [``Executor``] can throw.
#[derive(Error, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecutionError {
    /// The execution failed with a non-zero exit code.
    #[error("execution failed with exit code {0}")]
    HaltWithNonZeroExitCode(u32),

    /// The execution failed with an invalid memory access.
    #[error("invalid memory access for opcode {0} and address {1}")]
    InvalidMemoryAccess(Opcode, u64),

    /// The execution failed with an unimplemented syscall.
    #[error("unimplemented syscall {0}")]
    UnsupportedSyscall(u32),

    /// The execution failed with a breakpoint.
    #[error("breakpoint encountered")]
    Breakpoint(),

    /// The execution failed with an exceeded cycle limit.
    #[error("exceeded cycle limit of {0}")]
    ExceededCycleLimit(u64),

    /// The execution failed because the syscall was called in unconstrained mode.
    #[error("syscall called in unconstrained mode")]
    InvalidSyscallUsage(u64),

    /// The execution failed with an unimplemented feature.
    #[error("got unimplemented as opcode")]
    Unimplemented(),

    /// The program ended in unconstrained mode.
    #[error("program ended in unconstrained mode")]
    EndInUnconstrained(),

    /// The unconstrained cycle limit was exceeded.
    #[error("unconstrained cycle limit exceeded")]
    UnconstrainedCycleLimitExceeded(u64),
}

impl<'a> Executor<'a> {
    /// Create a new [``Executor``] from a program and options.
    #[must_use]
    pub fn new(program: Arc<Program>, opts: SP1CoreOpts) -> Self {
        Self::with_context(program, opts, SP1Context::default())
    }

    /// WARNING: This function's API is subject to change without a major version bump.
    ///
    /// If the feature `"profiling"` is enabled, this sets up the profiler. Otherwise, it does
    /// nothing. The argument `elf_bytes` must describe the same program as `self.program`.
    ///
    /// The profiler is configured by the following environment variables:
    ///
    /// - `TRACE_FILE`: writes Gecko traces to this path. If unspecified, the profiler is disabled.
    /// - `TRACE_SAMPLE_RATE`: The period between clock cycles where samples are taken. Defaults to
    ///   1.
    #[inline]
    #[allow(unused_variables)]
    pub fn maybe_setup_profiler(&mut self, elf_bytes: &[u8]) {
        #[cfg(feature = "profiling")]
        {
            let trace_buf = std::env::var("TRACE_FILE").ok().map(|file| {
                let file = File::create(file).unwrap();
                BufWriter::new(file)
            });

            if let Some(trace_buf) = trace_buf {
                eprintln!("Profiling enabled");

                let sample_rate = std::env::var("TRACE_SAMPLE_RATE")
                    .ok()
                    .and_then(|rate| {
                        eprintln!("Profiling sample rate: {rate}");
                        rate.parse::<u32>().ok()
                    })
                    .unwrap_or(1);

                self.profiler = Some((
                    Profiler::new(elf_bytes, sample_rate as u64)
                        .expect("Failed to create profiler"),
                    trace_buf,
                ));
            }
        }
    }

    /// Create a new runtime from a program, options, and a context.
    #[must_use]
    pub fn with_context(program: Arc<Program>, opts: SP1CoreOpts, context: SP1Context<'a>) -> Self {
        // Create a default record with the program.
        let record = ExecutionRecord::new(program.clone());

        // Determine the maximum number of cycles for any syscall.
        let max_syscall_cycles = SyscallCode::iter()
            .map(|code| get_syscall::<Simple>(code).unwrap().num_extra_cycles)
            .max()
            .expect("No syscalls found");

        let hook_registry = context.hook_registry.unwrap_or_default();

        let costs: HashMap<String, usize> =
            serde_json::from_str(include_str!("./artifacts/rv32im_costs.json")).unwrap();
        let costs: EnumMap<RiscvAirId, usize> =
            costs.into_iter().map(|(k, v)| (RiscvAirId::from_str(&k).unwrap(), v)).collect();

        let program_len = program.instructions.len() as u64;

        let internal_syscalls_override = opts
            .retained_events_presets
            .iter()
            .flat_map(RetainedEventsPreset::syscall_codes)
            .copied()
            .unique()
            .sorted()
            .collect();

        Self {
            record: Box::new(record),
            records: vec![],
            state: ExecutionState::new(program.pc_start_rel_u32()),
            program,
            program_len,
            memory_accesses: MemoryAccessRecord::default(),
            shard_size: (opts.shard_size as u32) * 4,
            shard_batch_size: opts.shard_batch_size as u32,
            cycle_tracker: HashMap::new(),
            io_buf: HashMap::new(),
            #[cfg(feature = "profiling")]
            profiler: None,
            unconstrained_state: Box::new(ForkState::default()),
            emit_global_memory_events: true,
            max_syscall_cycles,
            report: ExecutionReport::default(),
            local_counts: LocalCounts::default(),
            print_report: false,
            record_estimator: None,
            subproof_verifier: context.subproof_verifier,
            hook_registry,
            max_cycles: context.max_cycles,
            deferred_proof_verification: context.deferred_proof_verification.into(),
            memory_checkpoint: Memory::default(),
            uninitialized_memory_checkpoint: Memory::default(),
            local_memory_access: HashMap::new(),
            costs: costs.into_iter().map(|(k, v)| (k, v as u64)).collect(),
            size_check_frequency: 16,
            sharding_threshold: Some(opts.sharding_threshold),
            event_counts: EnumMap::default(),
            internal_syscalls_override,
            io_options: context.io_options,
            total_unconstrained_cycles: 0,
            opts,
        }
    }

    /// Invokes a hook with the given file descriptor `fd` with the data `buf`.
    ///
    /// # Errors
    ///
    /// If the file descriptor is not found in the [``HookRegistry``], this function will return an
    /// error.
    pub fn hook(&self, fd: u32, buf: &[u8]) -> eyre::Result<Vec<Vec<u8>>> {
        Ok(self
            .hook_registry
            .get(fd)
            .ok_or(eyre::eyre!("no hook found for file descriptor {}", fd))?
            .invoke_hook(self.hook_env(), buf))
    }

    /// Prepare a `HookEnv` for use by hooks.
    #[must_use]
    pub fn hook_env<'b>(&'b self) -> HookEnv<'b, 'a> {
        HookEnv { runtime: self }
    }

    /// Recover runtime state from a program and existing execution state.
    #[must_use]
    pub fn recover(program: Arc<Program>, state: ExecutionState, opts: SP1CoreOpts) -> Self {
        let mut runtime = Self::new(program, opts);
        runtime.state = state;
        // Disable deferred proof verification since we're recovering from a checkpoint, and the
        // checkpoint creator already had a chance to check the proofs.
        runtime.deferred_proof_verification = DeferredProofVerification::Disabled;
        runtime
    }

    /// Get the current values of the registers.
    #[allow(clippy::single_match_else)]
    #[must_use]
    pub fn registers<E: ExecutorConfig>(&mut self) -> [u64; 32] {
        let mut registers = [0; 32];
        for i in 0..32 {
            let record = self.state.memory.registers.get(i);

            // Only add the previous memory state to checkpoint map if we're in checkpoint mode,
            // or if we're in unconstrained mode. In unconstrained mode, the mode is always
            // Simple.
            if E::MODE == ExecutorMode::Checkpoint || E::UNCONSTRAINED {
                match record {
                    Some(record) => {
                        self.memory_checkpoint.registers.entry(i).or_insert_with(|| Some(*record));
                    }
                    None => {
                        self.memory_checkpoint.registers.entry(i).or_insert(None);
                    }
                }
            }

            registers[i as usize] = match record {
                Some(record) => record.value,
                None => 0,
            };
        }
        registers
    }

    /// Get the current value of a register.
    #[must_use]
    pub fn register<E: ExecutorConfig>(&mut self, register: Register) -> u64 {
        let addr = register as u64;
        let record = self.state.memory.registers.get(addr);

        if E::MODE == ExecutorMode::Checkpoint || E::UNCONSTRAINED {
            match record {
                Some(record) => {
                    self.memory_checkpoint.registers.entry(addr).or_insert_with(|| Some(*record));
                }
                None => {
                    self.memory_checkpoint.registers.entry(addr).or_insert(None);
                }
            }
        }
        match record {
            Some(record) => record.value,
            None => 0,
        }
    }

    /// Get the current value of a word.
    ///
    /// Assumes `addr` is a valid memory address, not a register.
    #[must_use]
    pub fn word<E: ExecutorConfig>(&mut self, addr: u64) -> u64 {
        #[allow(clippy::single_match_else)]
        let record = self.state.memory.page_table.get(addr);

        if E::MODE == ExecutorMode::Checkpoint || E::UNCONSTRAINED {
            match record {
                Some(record) => {
                    self.memory_checkpoint.page_table.entry(addr).or_insert_with(|| Some(*record));
                }
                None => {
                    self.memory_checkpoint.page_table.entry(addr).or_insert(None);
                }
            }
        }

        match record {
            Some(record) => record.value,
            None => 0,
        }
    }

    /// Get the current value of a byte.
    ///
    /// Assumes `addr` is a valid memory address, not a register.
    #[must_use]
    pub fn byte<E: ExecutorConfig>(&mut self, addr: u64) -> u8 {
        let word = self.word::<E>(addr - addr % 8);
        (word >> ((addr % 8) * 8)) as u8
    }

    /// Get the current timestamp for a given memory access position.
    #[must_use]
    pub const fn timestamp(&self, position: &MemoryAccessPosition) -> u64 {
        self.state.clk + *position as u64
    }

    /// Get the current shard.
    #[must_use]
    #[inline]
    pub fn shard(&self) -> Shard {
        self.state.current_shard
    }

    /// Get the current logical shard.
    #[must_use]
    #[inline]
    pub fn lshard(&self) -> LogicalShard {
        LogicalShard::new(self.shard(), false)
    }

    /// Read a word from memory and create an access record.
    pub fn mr<E: ExecutorConfig>(
        &mut self,
        addr: u64,
        lshard: LogicalShard,
        timestamp: u64,
        local_memory_access: Option<&mut HashMap<u64, MemoryLocalEvent>>,
    ) -> MemoryReadRecord {
        // Check that the memory address is within the babybear field and not within the registers'
        // address space.  Also check that the address is aligned.
        // if addr % 4 != 0 || addr <= Register::X31 as u32 || addr >= BABYBEAR_PRIME {
        //     panic!("Invalid memory access: addr={addr}");
        // }

        // Get the memory record entry.
        let entry = self.state.memory.page_table.entry(addr);
        if E::MODE == ExecutorMode::Checkpoint || E::UNCONSTRAINED {
            match entry {
                Entry::Occupied(ref entry) => {
                    let record = entry.get();
                    self.memory_checkpoint.page_table.entry(addr).or_insert_with(|| Some(*record));
                }
                Entry::Vacant(_) => {
                    self.memory_checkpoint.page_table.entry(addr).or_insert(None);
                }
            }
        }

        // If we're in unconstrained mode, we don't want to modify state, so we'll save the
        // original state if it's the first time modifying it.
        if E::UNCONSTRAINED {
            let record = match entry {
                Entry::Occupied(ref entry) => Some(entry.get()),
                Entry::Vacant(_) => None,
            };
            self.unconstrained_state.memory_diff.entry(addr).or_insert(record.copied());
        }

        // If it's the first time accessing this address, initialize previous values.
        let record: &mut MemoryEntry = match entry {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                // If addr has a specific value to be initialized with, use that, otherwise 0.
                let value = self.state.uninitialized_memory.page_table.get(addr).unwrap_or(&0);
                self.uninitialized_memory_checkpoint
                    .page_table
                    .entry(addr)
                    .or_insert_with(|| *value != 0);
                entry.insert(MemoryEntry::init(*value))
            }
        };

        // For an explanation of this logic, see the documentation of `LocalCounts::local_mem`.
        if !E::UNCONSTRAINED && (record.lshard != lshard && !lshard.external_flag()) {
            self.local_counts.local_mem += 1;
        }

        let prev_record = *record;
        record.lshard = lshard;
        record.timestamp = timestamp;

        if !E::UNCONSTRAINED && E::MODE == ExecutorMode::Trace {
            let local_memory_access = if let Some(local_memory_access) = local_memory_access {
                local_memory_access
            } else {
                &mut self.local_memory_access
            };

            local_memory_access
                .entry(addr)
                .and_modify(|e| {
                    e.final_mem_access = (*record).into();
                })
                .or_insert(MemoryLocalEvent {
                    addr,
                    initial_mem_access: prev_record.into(),
                    final_mem_access: (*record).into(),
                });
        }

        // Construct the memory read record.
        MemoryReadRecord::new(record, &prev_record)
    }

    /// Read a register and return its value.
    ///
    /// Assumes that the executor mode IS NOT [`ExecutorMode::Trace`]
    pub fn rr<E: ExecutorConfig>(
        &mut self,
        register: Register,
        lshard: LogicalShard,
        timestamp: u64,
    ) -> u64 {
        // Get the memory record entry.
        let addr = register as u64;
        let entry = self.state.memory.registers.entry(addr);
        if E::MODE == ExecutorMode::Checkpoint || E::UNCONSTRAINED {
            match entry {
                Entry::Occupied(ref entry) => {
                    let record = entry.get();
                    self.memory_checkpoint.registers.entry(addr).or_insert_with(|| Some(*record));
                }
                Entry::Vacant(_) => {
                    self.memory_checkpoint.registers.entry(addr).or_insert(None);
                }
            }
        }

        // If we're in unconstrained mode, we don't want to modify state, so we'll save the
        // original state if it's the first time modifying it.
        if E::UNCONSTRAINED {
            let record = match entry {
                Entry::Occupied(ref entry) => Some(entry.get()),
                Entry::Vacant(_) => None,
            };
            self.unconstrained_state.memory_diff.entry(addr).or_insert(record.copied());
        }

        // If it's the first time accessing this address, initialize previous values.
        let record: &mut MemoryEntry = match entry {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                // If addr has a specific value to be initialized with, use that, otherwise 0.
                let value = self.state.uninitialized_memory.registers.get(addr).unwrap_or(&0);
                self.uninitialized_memory_checkpoint
                    .registers
                    .entry(addr)
                    .or_insert_with(|| *value != 0);
                entry.insert(MemoryEntry::init(*value))
            }
        };

        // For an explanation of this logic, see the documentation of `LocalCounts::local_mem`.
        if !E::UNCONSTRAINED && (record.lshard != lshard && !lshard.external_flag()) {
            self.local_counts.local_mem += 1;
        }

        record.lshard = lshard;
        record.timestamp = timestamp;
        record.value
    }

    /// Read a register and create an access record.
    ///
    /// Assumes that self.mode IS [`ExecutorMode::Trace`].
    pub fn rr_traced<E: ExecutorConfig>(
        &mut self,
        register: Register,
        lshard: LogicalShard,
        timestamp: u64,
        local_memory_access: Option<&mut HashMap<u64, MemoryLocalEvent>>,
    ) -> MemoryReadRecord {
        // Get the memory record entry.
        let addr = register as u64;
        let entry = self.state.memory.registers.entry(addr);
        if E::MODE == ExecutorMode::Checkpoint || E::UNCONSTRAINED {
            match entry {
                Entry::Occupied(ref entry) => {
                    let record = entry.get();
                    self.memory_checkpoint.registers.entry(addr).or_insert_with(|| Some(*record));
                }
                Entry::Vacant(_) => {
                    self.memory_checkpoint.registers.entry(addr).or_insert(None);
                }
            }
        }
        // If we're in unconstrained mode, we don't want to modify state, so we'll save the
        // original state if it's the first time modifying it.
        if E::UNCONSTRAINED {
            let record = match entry {
                Entry::Occupied(ref entry) => Some(entry.get()),
                Entry::Vacant(_) => None,
            };
            self.unconstrained_state.memory_diff.entry(addr).or_insert(record.copied());
        }
        // If it's the first time accessing this address, initialize previous values.
        let record: &mut MemoryEntry = match entry {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                // If addr has a specific value to be initialized with, use that, otherwise 0.
                let value = self.state.uninitialized_memory.registers.get(addr).unwrap_or(&0);
                self.uninitialized_memory_checkpoint
                    .registers
                    .entry(addr)
                    .or_insert_with(|| *value != 0);
                entry.insert(MemoryEntry::init(*value))
            }
        };

        // For an explanation of this logic, see the documentation of `LocalCounts::local_mem`.
        if !E::UNCONSTRAINED && (record.lshard != lshard && !lshard.external_flag()) {
            self.local_counts.local_mem += 1;
        }

        let prev_record = *record;
        record.lshard = lshard;
        record.timestamp = timestamp;
        if !E::UNCONSTRAINED && E::MODE == ExecutorMode::Trace {
            let local_memory_access = if let Some(local_memory_access) = local_memory_access {
                local_memory_access
            } else {
                &mut self.local_memory_access
            };
            local_memory_access
                .entry(addr)
                .and_modify(|e| {
                    e.final_mem_access = (*record).into();
                })
                .or_insert(MemoryLocalEvent {
                    addr,
                    initial_mem_access: prev_record.into(),
                    final_mem_access: (*record).into(),
                });
        }
        // Construct the memory read record.
        MemoryReadRecord::new(record, &prev_record)
    }
    /// Write a word to memory and create an access record.
    pub fn mw<E: ExecutorConfig>(
        &mut self,
        addr: u64,
        value: u64,
        lshard: LogicalShard,
        timestamp: u64,
        local_memory_access: Option<&mut HashMap<u64, MemoryLocalEvent>>,
    ) -> MemoryWriteRecord {
        // Check that the memory address is within the babybear field and not within the registers'
        // address space.  Also check that the address is aligned.
        // if addr % 4 != 0 || addr <= Register::X31 as u32 || addr >= BABYBEAR_PRIME {
        //     panic!("Invalid memory access: addr={addr}");
        // }

        // Get the memory record entry.
        let entry = self.state.memory.page_table.entry(addr);
        if E::MODE == ExecutorMode::Checkpoint || E::UNCONSTRAINED {
            match entry {
                Entry::Occupied(ref entry) => {
                    let record = entry.get();
                    self.memory_checkpoint.page_table.entry(addr).or_insert_with(|| Some(*record));
                }
                Entry::Vacant(_) => {
                    self.memory_checkpoint.page_table.entry(addr).or_insert(None);
                }
            }
        }
        // If we're in unconstrained mode, we don't want to modify state, so we'll save the
        // original state if it's the first time modifying it.
        if E::UNCONSTRAINED {
            let record = match entry {
                Entry::Occupied(ref entry) => Some(entry.get()),
                Entry::Vacant(_) => None,
            };
            self.unconstrained_state.memory_diff.entry(addr).or_insert(record.copied());
        }
        // If it's the first time accessing this address, initialize previous values.
        let record: &mut MemoryEntry = match entry {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                // If addr has a specific value to be initialized with, use that, otherwise 0.
                let value = self.state.uninitialized_memory.page_table.get(addr).unwrap_or(&0);
                self.uninitialized_memory_checkpoint
                    .page_table
                    .entry(addr)
                    .or_insert_with(|| *value != 0);

                entry.insert(MemoryEntry::init(*value))
            }
        };

        // For an explanation of this logic, see the documentation of `LocalCounts::local_mem`.
        if !E::UNCONSTRAINED && (record.lshard != lshard && !lshard.external_flag()) {
            self.local_counts.local_mem += 1;
        }

        let prev_record = *record;
        record.value = value;
        record.lshard = lshard;
        record.timestamp = timestamp;
        if !E::UNCONSTRAINED && E::MODE == ExecutorMode::Trace {
            let local_memory_access = if let Some(local_memory_access) = local_memory_access {
                local_memory_access
            } else {
                &mut self.local_memory_access
            };

            local_memory_access
                .entry(addr)
                .and_modify(|e| {
                    e.final_mem_access = (*record).into();
                })
                .or_insert(MemoryLocalEvent {
                    addr,
                    initial_mem_access: prev_record.into(),
                    final_mem_access: (*record).into(),
                });
        }

        // Construct the memory write record.
        MemoryWriteRecord::new(record, &prev_record)
    }

    /// Write a word to a register and create an access record.
    ///
    /// Assumes that `E::MODE` IS [`ExecutorMode::Trace`].
    pub fn rw_traced<E: ExecutorConfig>(
        &mut self,
        register: Register,
        value: u64,
        lshard: LogicalShard,
        timestamp: u64,
        local_memory_access: Option<&mut HashMap<u64, MemoryLocalEvent>>,
    ) -> MemoryWriteRecord {
        let addr = register as u64;

        // Get the memory record entry.
        let entry = self.state.memory.registers.entry(addr);
        if E::UNCONSTRAINED {
            match entry {
                Entry::Occupied(ref entry) => {
                    let record = entry.get();
                    self.memory_checkpoint.registers.entry(addr).or_insert_with(|| Some(*record));
                }
                Entry::Vacant(_) => {
                    self.memory_checkpoint.registers.entry(addr).or_insert(None);
                }
            }
        }

        // If we're in unconstrained mode, we don't want to modify state, so we'll save the
        // original state if it's the first time modifying it.
        if E::UNCONSTRAINED {
            let record = match entry {
                Entry::Occupied(ref entry) => Some(entry.get()),
                Entry::Vacant(_) => None,
            };
            self.unconstrained_state.memory_diff.entry(addr).or_insert(record.copied());
        }

        // If it's the first time accessing this register, initialize previous values.
        let record: &mut MemoryEntry = match entry {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                // If addr has a specific value to be initialized with, use that, otherwise 0.
                let value = self.state.uninitialized_memory.registers.get(addr).unwrap_or(&0);
                self.uninitialized_memory_checkpoint
                    .registers
                    .entry(addr)
                    .or_insert_with(|| *value != 0);

                entry.insert(MemoryEntry::init(*value))
            }
        };

        // For an explanation of this logic, see the documentation of `LocalCounts::local_mem`.
        if !E::UNCONSTRAINED && (record.lshard != lshard && !lshard.external_flag()) {
            self.local_counts.local_mem += 1;
        }

        let prev_record = *record;
        record.value = value;
        record.lshard = lshard;
        record.timestamp = timestamp;

        if !E::UNCONSTRAINED {
            let local_memory_access = if let Some(local_memory_access) = local_memory_access {
                local_memory_access
            } else {
                &mut self.local_memory_access
            };

            local_memory_access
                .entry(addr)
                .and_modify(|e| {
                    e.final_mem_access = (*record).into();
                })
                .or_insert(MemoryLocalEvent {
                    addr,
                    initial_mem_access: prev_record.into(),
                    final_mem_access: (*record).into(),
                });
        }

        // Construct the memory write record.
        MemoryWriteRecord::new(record, &prev_record)
    }

    /// Write a word to a register and create an access record.
    ///
    /// Assumes that the executor mode IS NOT [`ExecutorMode::Trace`].
    #[inline]
    pub fn rw<E: ExecutorConfig>(
        &mut self,
        register: Register,
        value: u64,
        lshard: LogicalShard,
        timestamp: u64,
    ) {
        let addr = register as u64;
        // Get the memory record entry.
        let entry = self.state.memory.registers.entry(addr);
        if E::MODE == ExecutorMode::Checkpoint || E::UNCONSTRAINED {
            match entry {
                Entry::Occupied(ref entry) => {
                    let record = entry.get();
                    self.memory_checkpoint.registers.entry(addr).or_insert_with(|| Some(*record));
                }
                Entry::Vacant(_) => {
                    self.memory_checkpoint.registers.entry(addr).or_insert(None);
                }
            }
        }

        // If we're in unconstrained mode, we don't want to modify state, so we'll save the
        // original state if it's the first time modifying it.
        if E::UNCONSTRAINED {
            let record = match entry {
                Entry::Occupied(ref entry) => Some(entry.get()),
                Entry::Vacant(_) => None,
            };
            self.unconstrained_state.memory_diff.entry(addr).or_insert(record.copied());
        }

        // If it's the first time accessing this register, initialize previous values.
        let record: &mut MemoryEntry = match entry {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                // If addr has a specific value to be initialized with, use that, otherwise 0.
                let value = self.state.uninitialized_memory.registers.get(addr).unwrap_or(&0);
                self.uninitialized_memory_checkpoint
                    .registers
                    .entry(addr)
                    .or_insert_with(|| *value != 0);

                entry.insert(MemoryEntry::init(*value))
            }
        };

        // For an explanation of this logic, see the documentation of `LocalCounts::local_mem`.
        if !E::UNCONSTRAINED && (record.lshard != lshard && !lshard.external_flag()) {
            self.local_counts.local_mem += 1;
        }

        record.value = value;
        record.lshard = lshard;
        record.timestamp = timestamp;
    }

    /// Read from memory, assuming that all addresses are aligned.
    #[inline]
    pub fn mr_cpu<E: ExecutorConfig>(&mut self, addr: u64) -> u64 {
        // Read the address from memory and create a memory read record.
        let record =
            self.mr::<E>(addr, self.lshard(), self.timestamp(&MemoryAccessPosition::Memory), None);
        // If we're not in unconstrained mode, record the access for the current cycle.
        if E::MODE == ExecutorMode::Trace {
            self.memory_accesses.memory = Some(record.into());
        }
        record.value
    }

    /// Read a register.
    #[inline]
    pub fn rr_cpu<E: ExecutorConfig>(
        &mut self,
        register: Register,
        position: MemoryAccessPosition,
    ) -> u64 {
        // Read the address from memory and create a memory read record if in trace mode.
        if E::MODE == ExecutorMode::Trace {
            let record =
                self.rr_traced::<E>(register, self.lshard(), self.timestamp(&position), None);
            if !E::UNCONSTRAINED {
                match position {
                    MemoryAccessPosition::A => self.memory_accesses.a = Some(record.into()),
                    MemoryAccessPosition::B => self.memory_accesses.b = Some(record.into()),
                    MemoryAccessPosition::C => self.memory_accesses.c = Some(record.into()),
                    MemoryAccessPosition::Memory => {
                        self.memory_accesses.memory = Some(record.into());
                    }
                }
            }
            record.value
        } else {
            self.rr::<E>(register, self.lshard(), self.timestamp(&position))
        }
    }

    /// Write to memory.
    ///
    /// # Panics
    ///
    /// This function will panic if the address is not aligned or if the memory accesses are already
    /// initialized.
    pub fn mw_cpu<E: ExecutorConfig>(&mut self, addr: u64, value: u64) {
        // Read the address from memory and create a memory read record.
        let record = self.mw::<E>(
            addr,
            value,
            self.lshard(),
            self.timestamp(&MemoryAccessPosition::Memory),
            None,
        );
        // If we're not in unconstrained mode, record the access for the current cycle.
        if E::MODE == ExecutorMode::Trace {
            debug_assert!(self.memory_accesses.memory.is_none());
            self.memory_accesses.memory = Some(record.into());
        }
    }

    /// Write to a register.
    pub fn rw_cpu<E: ExecutorConfig>(&mut self, register: Register, value: u64) {
        // The only time we are writing to a register is when it is in operand A.
        let position = MemoryAccessPosition::A;

        // Register %x0 should always be 0. See 2.6 Load and Store Instruction on
        // P.18 of the RISC-V spec. We always write 0 to %x0.
        let value = if register == Register::X0 { 0 } else { value };

        // Read the address from memory and create a memory read record.
        if E::MODE == ExecutorMode::Trace {
            let record = self.rw_traced::<E>(
                register,
                value,
                self.lshard(),
                self.timestamp(&position),
                None,
            );
            if !E::UNCONSTRAINED {
                // The only time we are writing to a register is when it is in operand A.
                debug_assert!(self.memory_accesses.a.is_none());
                self.memory_accesses.a = Some(record.into());
            }
        } else {
            self.rw::<E>(register, value, self.lshard(), self.timestamp(&position));
        }
    }

    /// Emit events for this cycle.
    #[allow(clippy::too_many_arguments)]
    fn emit_events(
        &mut self,
        clk: u64,
        next_pc_rel: u32,
        instruction: &Instruction,
        syscall_code: SyscallCode,
        a: u64,
        b: u64,
        c: u64,
        op_a_0: bool,
        record: MemoryAccessRecord,
        exit_code: u32,
    ) {
        self.record.pc_start_rel.get_or_insert(self.state.pc_rel);
        self.record.next_pc_rel = next_pc_rel;
        self.record.exit_code = exit_code;
        self.record.cpu_event_count += 1;

        let increment = self.state.clk + 8 - clk;

        if clk % (1 << 24) + increment >= (1 << 24) {
            self.record.bump_clk_high_events.push((clk, increment, next_pc_rel));
        }

        if let Some(x) = self.memory_accesses.a {
            if x.current_record().timestamp >> 24 != x.previous_record().timestamp >> 24 {
                self.record.bump_memory_events.push((x, instruction.op_a as u64));
            }
        }
        if let Some(x) = self.memory_accesses.b {
            if x.current_record().timestamp >> 24 != x.previous_record().timestamp >> 24 {
                self.record.bump_memory_events.push((x, instruction.op_b));
            }
        }
        if let Some(x) = self.memory_accesses.c {
            if x.current_record().timestamp >> 24 != x.previous_record().timestamp >> 24 {
                self.record.bump_memory_events.push((x, instruction.op_c));
            }
        }

        if instruction.is_alu_instruction() {
            self.emit_alu_event(instruction.opcode, a, b, c, record, op_a_0);
        } else if instruction.is_memory_load_instruction()
            || instruction.is_memory_store_instruction()
        {
            self.emit_mem_instr_event(instruction.opcode, a, b, c, record, op_a_0);
        } else if instruction.is_branch_instruction() {
            self.emit_branch_event(instruction.opcode, a, b, c, record, op_a_0, next_pc_rel);
        } else if instruction.is_jal_instruction() {
            self.emit_jal_event(instruction.opcode, a, b, c, record, op_a_0, next_pc_rel);
        } else if instruction.is_jalr_instruction() {
            self.emit_jalr_event(instruction.opcode, a, b, c, record, op_a_0, next_pc_rel);
        } else if instruction.is_auipc_instruction() {
            self.emit_auipc_event(instruction.opcode, a, b, c, record, op_a_0);
        } else if instruction.is_ecall_instruction() {
            self.emit_syscall_event(
                clk,
                syscall_code,
                b,
                c,
                record,
                op_a_0,
                next_pc_rel,
                exit_code,
            );
        } else {
            unreachable!()
        }
    }

    /// Emit an ALU event.
    fn emit_alu_event(
        &mut self,
        opcode: Opcode,
        a: u64,
        b: u64,
        c: u64,
        record: MemoryAccessRecord,
        op_a_0: bool,
    ) {
        let event =
            AluEvent { clk: self.state.clk, pc_rel: self.state.pc_rel, opcode, a, b, c, op_a_0 };
        match opcode {
            Opcode::ADD => {
                let record = RTypeRecord::from(record);
                self.record.add_events.push((event, record));
            }
            Opcode::ADDW => {
                let record = RTypeRecord::from(record);
                self.record.addw_events.push((event, record));
            }
            Opcode::ADDI => {
                let record = ITypeRecord::from(record);
                self.record.addi_events.push((event, record));
            }
            Opcode::ADDIW => {
                let record = ITypeRecord::from(record);
                self.record.addiw_events.push((event, record));
            }
            Opcode::SUB => {
                let record = RTypeRecord::from(record);
                self.record.sub_events.push((event, record));
            }
            Opcode::SUBW => {
                let record = RTypeRecord::from(record);
                self.record.subw_events.push((event, record));
            }
            Opcode::XOR | Opcode::OR | Opcode::AND => {
                let record = ALUTypeRecord::from(record);
                self.record.bitwise_events.push((event, record));
            }
            Opcode::SLL | Opcode::SLLW | Opcode::SLLIW => {
                let record = ALUTypeRecord::from(record);
                self.record.shift_left_events.push((event, record));
            }
            Opcode::SRL
            | Opcode::SRA
            | Opcode::SRLW
            | Opcode::SRAW
            | Opcode::SRLIW
            | Opcode::SRAIW => {
                let record = ALUTypeRecord::from(record);
                self.record.shift_right_events.push((event, record));
            }
            Opcode::SLT | Opcode::SLTU => {
                let record = ALUTypeRecord::from(record);
                self.record.lt_events.push((event, record));
            }
            Opcode::MUL | Opcode::MULHU | Opcode::MULHSU | Opcode::MULH | Opcode::MULW => {
                let record = ALUTypeRecord::from(record);
                self.record.mul_events.push((event, record));
            }
            Opcode::DIVU
            | Opcode::REMU
            | Opcode::DIV
            | Opcode::REM
            | Opcode::DIVW
            | Opcode::DIVUW
            | Opcode::REMUW
            | Opcode::REMW => {
                let record = ALUTypeRecord::from(record);
                self.record.divrem_events.push((event, record));
            }
            _ => unreachable!(),
        }
    }

    /// Emit a memory instruction event.
    #[inline]
    fn emit_mem_instr_event(
        &mut self,
        opcode: Opcode,
        a: u64,
        b: u64,
        c: u64,
        record: MemoryAccessRecord,
        op_a_0: bool,
    ) {
        let event = MemInstrEvent {
            shard: self.shard().get(),
            clk: self.state.clk,
            pc_rel: self.state.pc_rel,
            opcode,
            a,
            b,
            c,
            op_a_0,
            mem_access: self.memory_accesses.memory.expect("Must have memory access"),
        };
        let record = ITypeRecord::from(record);
        if matches!(opcode, Opcode::LB | Opcode::LBU | Opcode::LH | Opcode::LHU | Opcode::LW)
            && op_a_0
        {
            self.record.memory_load_x0_events.push((event, record));
        } else if matches!(opcode, Opcode::LB | Opcode::LBU) {
            self.record.memory_load_byte_events.push((event, record));
        } else if matches!(opcode, Opcode::LH | Opcode::LHU) {
            self.record.memory_load_half_events.push((event, record));
        } else if matches!(opcode, Opcode::LW | Opcode::LWU) {
            self.record.memory_load_word_events.push((event, record));
        } else if opcode == Opcode::LD {
            self.record.memory_load_double_events.push((event, record));
        } else if opcode == Opcode::SB {
            self.record.memory_store_byte_events.push((event, record));
        } else if opcode == Opcode::SH {
            self.record.memory_store_half_events.push((event, record));
        } else if opcode == Opcode::SW {
            self.record.memory_store_word_events.push((event, record));
        } else if opcode == Opcode::SD {
            self.record.memory_store_double_events.push((event, record));
        }
    }

    /// Emit a branch event.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn emit_branch_event(
        &mut self,
        opcode: Opcode,
        a: u64,
        b: u64,
        c: u64,
        record: MemoryAccessRecord,
        op_a_0: bool,
        next_pc_rel: u32,
    ) {
        let event = BranchEvent {
            clk: self.state.clk,
            pc_rel: self.state.pc_rel,
            next_pc_rel,
            opcode,
            a,
            b,
            c,
            op_a_0,
        };
        let record = ITypeRecord::from(record);
        self.record.branch_events.push((event, record));
    }

    /// Emit a jal event.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn emit_jal_event(
        &mut self,
        opcode: Opcode,
        a: u64,
        b: u64,
        c: u64,
        record: MemoryAccessRecord,
        op_a_0: bool,
        next_pc_rel: u32,
    ) {
        let event = JumpEvent {
            clk: self.state.clk,
            pc_rel: self.state.pc_rel,
            next_pc_rel,
            opcode,
            a,
            b,
            c,
            op_a_0,
        };
        let record = JTypeRecord::from(record);
        self.record.jal_events.push((event, record));
    }

    /// Emit a jalr event.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn emit_jalr_event(
        &mut self,
        opcode: Opcode,
        a: u64,
        b: u64,
        c: u64,
        record: MemoryAccessRecord,
        op_a_0: bool,
        next_pc_rel: u32,
    ) {
        let event = JumpEvent {
            clk: self.state.clk,
            pc_rel: self.state.pc_rel,
            next_pc_rel,
            opcode,
            a,
            b,
            c,
            op_a_0,
        };
        let record = ITypeRecord::from(record);
        self.record.jalr_events.push((event, record));
    }

    /// Emit an AUIPC event.
    #[inline]
    fn emit_auipc_event(
        &mut self,
        opcode: Opcode,
        a: u64,
        b: u64,
        c: u64,
        record: MemoryAccessRecord,
        op_a_0: bool,
    ) {
        let event =
            AUIPCEvent { clk: self.state.clk, pc_rel: self.state.pc_rel, opcode, a, b, c, op_a_0 };
        let record = JTypeRecord::from(record);
        self.record.auipc_events.push((event, record));
    }

    /// Create a syscall event.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub(crate) fn syscall_event(
        &self,
        clk: u64,
        syscall_code: SyscallCode,
        arg1: u64,
        arg2: u64,
        op_a_0: bool,
        next_pc_rel: u32,
        exit_code: u32,
    ) -> SyscallEvent {
        // should_send: if the syscall is usually sent and it is not manually set as internal.
        let should_send = (syscall_code.should_send() != 0)
            && !self.internal_syscalls_override.contains(&syscall_code);
        SyscallEvent {
            pc_rel: self.state.pc_rel,
            next_pc_rel,
            shard: self.shard().get(),
            clk,
            op_a_0,
            should_send,
            syscall_code,
            syscall_id: syscall_code.syscall_id(),
            arg1,
            arg2,
            exit_code,
        }
    }

    /// Emit a syscall event.
    #[allow(clippy::too_many_arguments)]
    fn emit_syscall_event(
        &mut self,
        clk: u64,
        syscall_code: SyscallCode,
        arg1: u64,
        arg2: u64,
        record: MemoryAccessRecord,
        op_a_0: bool,
        next_pc_rel: u32,
        exit_code: u32,
    ) {
        let syscall_event =
            self.syscall_event(clk, syscall_code, arg1, arg2, op_a_0, next_pc_rel, exit_code);
        let record = RTypeRecord::from(record);
        self.record.syscall_events.push((syscall_event, record));
    }

    /// Fetch the destination register and input operand values for an ALU instruction.
    fn alu_rr<E: ExecutorConfig>(&mut self, instruction: &Instruction) -> (Register, u64, u64) {
        if !instruction.imm_c {
            let (rd, rs1, rs2) = instruction.r_type();
            let c = self.rr_cpu::<E>(rs2, MemoryAccessPosition::C);
            let b = self.rr_cpu::<E>(rs1, MemoryAccessPosition::B);
            (rd, b, c)
        } else if !instruction.imm_b && instruction.imm_c {
            let (rd, rs1, imm) = instruction.i_type();
            let (rd, b, c) = (rd, self.rr_cpu::<E>(rs1, MemoryAccessPosition::B), imm);
            (rd, b, c)
        } else {
            debug_assert!(instruction.imm_b && instruction.imm_c);
            let (rd, b, c) =
                (Register::from_u8(instruction.op_a), instruction.op_b, instruction.op_c);
            (rd, b, c)
        }
    }

    /// Set the destination register with the result.
    #[inline]
    fn alu_rw<E: ExecutorConfig>(&mut self, rd: Register, a: u64) {
        self.rw_cpu::<E>(rd, a);
    }

    /// Fetch the input operand values for a load instruction.
    fn load_rr<E: ExecutorConfig>(
        &mut self,
        instruction: &Instruction,
    ) -> (Register, u64, u64, u64, u64) {
        let (rd, rs1, imm) = instruction.i_type();
        let (b, c) = (self.rr_cpu::<E>(rs1, MemoryAccessPosition::B), imm);
        let addr = b.wrapping_add(c);
        let memory_value = self.mr_cpu::<E>(align(addr));
        (rd, b, c, addr, memory_value)
    }

    /// Fetch the input operand values for a store instruction.
    fn store_rr<E: ExecutorConfig>(
        &mut self,
        instruction: &Instruction,
    ) -> (u64, u64, u64, u64, u64) {
        let (rs1, rs2, imm) = instruction.s_type();
        let c = imm;
        let b = self.rr_cpu::<E>(rs2, MemoryAccessPosition::B);
        let a = self.rr_cpu::<E>(rs1, MemoryAccessPosition::A);
        let addr = b.wrapping_add(c);
        let memory_value = self.word::<E>(align(addr));
        (a, b, c, addr, memory_value)
    }

    /// Fetch the input operand values for a branch instruction.
    fn branch_rr<E: ExecutorConfig>(&mut self, instruction: &Instruction) -> (u64, u64, u64) {
        let (rs1, rs2, imm) = instruction.b_type();
        let c = imm;
        let b = self.rr_cpu::<E>(rs2, MemoryAccessPosition::B);
        let a = self.rr_cpu::<E>(rs1, MemoryAccessPosition::A);
        (a, b, c)
    }

    /// Fetch the instruction at the current program counter.
    #[inline]
    fn fetch(&self) -> Instruction {
        // if 2500 <= self.state.global_clk && self.state.global_clk <= 2600 {
        //     tracing::info!(
        //         "global_clk: {}, pc: {}, instruction: {:?}",
        //         self.state.global_clk,
        //         self.state.pc,
        //         self.program.fetch(self.state.pc)
        //     );
        // }
        *self.program.fetch(self.state.pc_rel)
    }

    /// Execute the given instruction over the current state of the runtime.
    #[allow(clippy::too_many_lines)]
    fn execute_instruction<E: ExecutorConfig>(
        &mut self,
        instruction: &Instruction,
    ) -> Result<(), ExecutionError> {
        // The `clk` variable contains the cycle before the current instruction is executed.  The
        // `state.clk` can be updated before the end of this function by precompiles' execution.
        let mut clk = self.state.clk;
        let mut exit_code = 0u32;
        let mut next_pc_rel = self.state.pc_rel.wrapping_add(4);
        // Will be set to a non-default value if the instruction is a syscall.

        let (mut a, b, c): (u64, u64, u64);

        if E::MODE == ExecutorMode::Trace {
            self.memory_accesses = MemoryAccessRecord::default();
        }

        // The syscall id for precompiles.  This is only used/set when opcode == ECALL.
        let mut syscall = SyscallCode::default();

        if !E::UNCONSTRAINED {
            if self.print_report {
                self.report.opcode_counts[instruction.opcode] += 1;
            }
            self.local_counts.event_counts[instruction.opcode] += 1;
            if instruction.is_memory_load_instruction() && instruction.op_a == Register::X0 as u8 {
                self.local_counts.event_counts[instruction.opcode] -= 1;
                self.local_counts.load_x0_counts += 1;
            }
        }

        if instruction.is_alu_instruction() {
            (a, b, c) = self.execute_alu::<E>(instruction);
        } else if instruction.is_memory_load_instruction() {
            (a, b, c) = self.execute_load::<E>(instruction)?;
        } else if instruction.is_memory_store_instruction() {
            (a, b, c) = self.execute_store::<E>(instruction)?;
        } else if instruction.is_branch_instruction() {
            (a, b, c, next_pc_rel) = self.execute_branch::<E>(instruction, next_pc_rel);
        } else if instruction.is_jump_instruction() {
            (a, b, c, next_pc_rel) = self.execute_jump::<E>(instruction);
        } else if instruction.is_auipc_instruction() {
            let (rd, imm) = instruction.u_type();
            (b, c) = (imm, imm);
            a = (self.state.pc_rel as u64).wrapping_add(self.program.pc_base).wrapping_add(b);
            self.rw_cpu::<E>(rd, a);
        } else if instruction.is_ecall_instruction() {
            (a, b, c, clk, next_pc_rel, syscall, exit_code) = self.execute_ecall::<E>()?;
        } else if instruction.is_ebreak_instruction() {
            return Err(ExecutionError::Breakpoint());
        } else if instruction.is_unimp_instruction() {
            // See https://github.com/riscv-non-isa/riscv-asm-manual/blob/master/riscv-asm.md#instruction-aliases
            return Err(ExecutionError::Unimplemented());
        } else {
            eprintln!("unreachable: {:?}", instruction.opcode);
            unreachable!()
        }

        // If the destination register is x0, then we need to make sure that a's value is 0.
        let op_a_0 = instruction.op_a == Register::X0 as u8;
        if op_a_0 {
            a = 0;
        }

        // Emit the events for this cycle.
        if E::MODE == ExecutorMode::Trace {
            self.emit_events(
                clk,
                next_pc_rel,
                instruction,
                syscall,
                a,
                b,
                c,
                op_a_0,
                self.memory_accesses,
                exit_code,
            );
        }

        // Update the program counter.
        self.state.pc_rel = next_pc_rel;

        // Update the clk to the next cycle.
        self.state.clk += 8;

        Ok(())
    }

    /// Execute an ALU instruction.
    fn execute_alu<E: ExecutorConfig>(&mut self, instruction: &Instruction) -> (u64, u64, u64) {
        let (rd, b, c) = self.alu_rr::<E>(instruction);
        let a = match instruction.opcode {
            Opcode::ADD | Opcode::ADDI => (Wrapping(b) + Wrapping(c)).0,
            Opcode::SUB => (Wrapping(b) - Wrapping(c)).0,
            Opcode::XOR => b ^ c,
            Opcode::OR => b | c,
            Opcode::AND => b & c,
            Opcode::SLL => b << (c & 0x3f),
            Opcode::SRL => b >> (c & 0x3f),
            Opcode::SRA => ((b as i64) >> (c & 0x3f)) as u64,
            Opcode::SLT => {
                if (b as i64) < (c as i64) {
                    1
                } else {
                    0
                }
            }
            Opcode::SLTU => {
                if b < c {
                    1
                } else {
                    0
                }
            }
            Opcode::MUL => (Wrapping(b as i64) * Wrapping(c as i64)).0 as u64,
            Opcode::MULH => (((b as i64) as i128).wrapping_mul((c as i64) as i128) >> 64) as u64,
            Opcode::MULHU => ((b as u128 * c as u128) >> 64) as u64,
            Opcode::MULHSU => ((((b as i64) as i128) * (c as i128)) >> 64) as u64,
            Opcode::DIV => {
                if c == 0 {
                    M64
                } else {
                    (b as i64).wrapping_div(c as i64) as u64
                }
            }
            Opcode::DIVU => {
                if c == 0 {
                    M64
                } else {
                    b / c
                }
            }
            Opcode::REM => {
                if c == 0 {
                    b
                } else {
                    (b as i64).wrapping_rem(c as i64) as u64
                }
            }
            Opcode::REMU => {
                if c == 0 {
                    b
                } else {
                    b % c
                }
            }
            // RISCV-64
            Opcode::ADDW => (Wrapping(b as i32) + Wrapping(c as i32)).0 as i64 as u64,
            Opcode::SUBW => (Wrapping(b as i32) - Wrapping(c as i32)).0 as i64 as u64,
            Opcode::MULW => (Wrapping(b as i32) * Wrapping(c as i32)).0 as i64 as u64,
            Opcode::DIVW => {
                if c as i32 == 0 {
                    M64
                } else {
                    (b as i32).wrapping_div(c as i32) as i64 as u64
                }
            }
            Opcode::DIVUW => {
                if c as i32 == 0 {
                    M64
                } else {
                    ((b as u32 / c as u32) as i32) as i64 as u64
                }
            }
            Opcode::REMW => {
                if c as i32 == 0 {
                    (b as i32) as u64
                } else {
                    (b as i32).wrapping_rem(c as i32) as i64 as u64
                }
            }
            Opcode::REMUW => {
                if c as u32 == 0 {
                    (b as i32) as u64
                } else {
                    (((b as u32) % (c as u32)) as i32) as i64 as u64
                }
            }
            // RISC-V 64-bit operations
            Opcode::SLLW | Opcode::SLLIW => (((b as i64) << (c & 0x1f)) as i32) as i64 as u64,
            Opcode::SRLW | Opcode::SRLIW => (((b as u32) >> ((c & 0x1f) as u32)) as i32) as u64,
            Opcode::SRAW | Opcode::SRAIW => {
                (b as i32).wrapping_shr(((c as i64 & 0x1f) as i32) as u32) as i64 as u64
            }
            Opcode::ADDIW => (Wrapping(b as i32) + Wrapping(c as i32)).0 as u64,
            _ => unreachable!(),
        };
        self.alu_rw::<E>(rd, a);
        (a, b, c)
    }

    /// Execute a load instruction.
    fn execute_load<E: ExecutorConfig>(
        &mut self,
        instruction: &Instruction,
    ) -> Result<(u64, u64, u64), ExecutionError> {
        let (rd, b, c, addr, memory_read_value) = self.load_rr::<E>(instruction);

        let a = match instruction.opcode {
            Opcode::LB => ((memory_read_value >> ((addr % 8) * 8)) & 0xFF) as i8 as i64 as u64,
            Opcode::LH => {
                if addr % 2 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LH, addr));
                }
                ((memory_read_value >> (((addr / 2) % 4) * 16)) & 0xFFFF) as i16 as i64 as u64
            }
            Opcode::LW => {
                if addr % 4 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LW, addr));
                }
                ((memory_read_value >> (((addr / 4) % 2) * 32)) & 0xFFFFFFFF) as i32 as u64
            }
            Opcode::LBU => ((memory_read_value >> ((addr % 8) * 8)) & 0xFF) as u8 as u64,
            Opcode::LHU => {
                if addr % 2 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LHU, addr));
                }
                ((memory_read_value >> (((addr / 2) % 4) * 16)) & 0xFFFF) as u16 as u64
            }
            // RISCV-64
            Opcode::LWU => {
                if addr % 4 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LWU, addr));
                }
                (memory_read_value >> (((addr / 4) % 2) * 32)) & 0xFFFFFFFF
            }
            Opcode::LD => {
                if addr % 8 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::LD, addr));
                }
                memory_read_value
            }
            _ => unreachable!(),
        };
        self.rw_cpu::<E>(rd, a);
        Ok((a, b, c))
    }

    /// Execute a store instruction.
    fn execute_store<E: ExecutorConfig>(
        &mut self,
        instruction: &Instruction,
    ) -> Result<(u64, u64, u64), ExecutionError> {
        let (a, b, c, addr, memory_read_value) = self.store_rr::<E>(instruction);

        let memory_store_value = match instruction.opcode {
            Opcode::SB => {
                let shift = (addr % 8) * 8;
                ((a & 0xFF) << shift) | (memory_read_value & !(0xFF << shift))
            }
            Opcode::SH => {
                if addr % 2 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::SH, addr));
                }
                let shift = ((addr / 2) % 4) * 16;
                ((a & 0xFFFF) << shift) | (memory_read_value & !(0xFFFF << shift))
            }
            Opcode::SW => {
                if addr % 4 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::SW, addr));
                }
                let shift = ((addr / 4) % 2) * 32;
                ((a & 0xFFFFFFFF) << shift) | (memory_read_value & !(0xFFFFFFFF << shift))
            }
            // RISCV-64
            Opcode::SD => {
                if addr % 8 != 0 {
                    return Err(ExecutionError::InvalidMemoryAccess(Opcode::SD, addr));
                }
                a
            }
            _ => unreachable!(),
        };
        self.mw_cpu::<E>(align(addr), memory_store_value);
        Ok((a, b, c))
    }

    /// Execute a branch instruction.
    fn execute_branch<E: ExecutorConfig>(
        &mut self,
        instruction: &Instruction,
        mut next_pc_rel: u32,
    ) -> (u64, u64, u64, u32) {
        let (a, b, c) = self.branch_rr::<E>(instruction);
        let branch = match instruction.opcode {
            Opcode::BEQ => a == b,
            Opcode::BNE => a != b,
            Opcode::BLT => (a as i64) < (b as i64),
            Opcode::BGE => (a as i64) >= (b as i64),
            Opcode::BLTU => a < b,
            Opcode::BGEU => a >= b,
            _ => {
                unreachable!()
            }
        };
        if branch {
            next_pc_rel = self.state.pc_rel.wrapping_add(c as u32);
        }
        (a, b, c, next_pc_rel)
    }

    /// Execute an ecall instruction.
    #[allow(clippy::type_complexity)]
    fn execute_ecall<E: ExecutorConfig>(
        &mut self,
    ) -> Result<(u64, u64, u64, u64, u32, SyscallCode, u32), ExecutionError> {
        // We peek at register x5 to get the syscall id. The reason we don't `self.rr` this
        // register is that we write to it later.
        let t0 = Register::X5;
        let syscall_id = self.register::<E>(t0);
        let c = self.rr_cpu::<E>(Register::X11, MemoryAccessPosition::C);
        let b = self.rr_cpu::<E>(Register::X10, MemoryAccessPosition::B);
        let syscall = SyscallCode::from_u32(syscall_id as u32);

        if self.print_report && !E::UNCONSTRAINED {
            self.report.syscall_counts[syscall] += 1;
        }

        // `hint_slice` is allowed in unconstrained mode since it is used to write the hint.
        // Other syscalls are not allowed because they can lead to non-deterministic
        // behavior, especially since many syscalls modify memory in place,
        // which is not permitted in unconstrained mode. This will result in
        // non-zero memory interactions when generating a proof.

        if E::UNCONSTRAINED
            && (syscall != SyscallCode::EXIT_UNCONSTRAINED && syscall != SyscallCode::WRITE)
        {
            return Err(ExecutionError::InvalidSyscallUsage(syscall_id));
        }

        // Update the syscall counts.
        let syscall_for_count = syscall.count_map();
        let syscall_count = self.state.syscall_counts.entry(syscall_for_count).or_insert(0);
        *syscall_count += 1;

        let syscall_impl = get_syscall(syscall)?;
        let external = !self.internal_syscalls_override.contains(&syscall);

        if !E::UNCONSTRAINED && syscall.should_send() == 1 {
            if external {
                self.local_counts.syscalls_sent += 1;
            } else {
                self.local_counts.retained_precompile_counts[syscall.as_air_id().unwrap()] += 1;
            }
        }

        let mut precompile_rt: SyscallContext<'_, '_, E> = SyscallContext::new(self, external);
        let (a, precompile_next_pc, _, returned_exit_code) = {
            // Executing a syscall optionally returns a value to write to the t0
            // register. If it returns None, we just keep the
            // syscall_id in t0.
            let res = (syscall_impl.handler)(&mut precompile_rt, syscall, b, c);
            let a = if let Some(val) = res { val } else { syscall_id };

            (a, precompile_rt.next_pc_rel, syscall_impl.num_extra_cycles, precompile_rt.exit_code)
        };

        // TODO(tqn) measure local memory events for the precompiles,
        // taking into account whether it should be sent
        // if let (Some(estimator), Some(syscall_id)) =
        //     (&mut self.record_estimator, syscall.as_air_id())
        // {
        //     let threshold = match syscall_id {
        //         RiscvAirId::ShaExtend => self.opts.split_opts.sha_extend,
        //         RiscvAirId::ShaCompress => self.opts.split_opts.sha_compress,
        //         RiscvAirId::KeccakPermute => self.opts.split_opts.keccak,
        //         _ => self.opts.split_opts.deferred,
        //     } as u64;
        //     let shards = &mut estimator.precompile_records[syscall_id];
        //     let local_memory_ct =
        //         estimator.current_precompile_touched_compressed_addresses.len() as u64;
        //     match shards.last_mut().filter(|shard| shard.0 < threshold) {
        //         Some((shard_precompile_event_ct, shard_local_memory_ct)) => {
        //             *shard_precompile_event_ct += 1;
        //             *shard_local_memory_ct += local_memory_ct;
        //         }
        //         None => shards.push((1, local_memory_ct)),
        //     }
        //     estimator.current_precompile_touched_compressed_addresses.clear();
        // }

        // If the syscall is `EXIT_UNCONSTRAINED`, the memory was restored to pre-unconstrained code
        // in the execute function, so we need to re-read from x10 and x11.  Just do a peek on the
        // registers.
        let (b, c) = if syscall == SyscallCode::EXIT_UNCONSTRAINED {
            (self.register::<E>(Register::X10), self.register::<E>(Register::X11))
        } else {
            (b, c)
        };

        // Allow the syscall impl to modify state.clk/pc (exit unconstrained does this)
        self.rw_cpu::<E>(t0, a);
        let clk = self.state.clk;
        self.state.clk += 256;

        Ok((a, b, c, clk, precompile_next_pc, syscall, returned_exit_code))
    }

    /// Execute a jump instruction.
    fn execute_jump<E: ExecutorConfig>(
        &mut self,
        instruction: &Instruction,
    ) -> (u64, u64, u64, u32) {
        let (a, b, c, next_pc_rel) = match instruction.opcode {
            Opcode::JAL => {
                let (rd, imm) = instruction.j_type();
                let imm_se = sign_extend_imm(imm, 21);
                let a = self.program.pc_base.wrapping_add(self.state.pc_rel as u64).wrapping_add(4);
                self.rw_cpu::<E>(rd, a);
                let next_pc_rel = ((self.state.pc_rel as i64).wrapping_add(imm_se)) as u32;
                let b = imm_se as u64;
                let c = 0;
                (a, b, c, next_pc_rel)
            }
            Opcode::JALR => {
                let (rd, rs1, c) = instruction.i_type();
                let imm_se = sign_extend_imm(c, 12);
                let b = self.rr_cpu::<E>(rs1, MemoryAccessPosition::B);
                let a = self.program.pc_base.wrapping_add(self.state.pc_rel as u64).wrapping_add(4);
                // Calculate next PC: (rs1 + imm) & ~1
                let next_pc_rel = (((b as i64).wrapping_add(imm_se) as u64) & !1_u64)
                    .wrapping_sub(self.program.pc_base) as u32;
                self.rw_cpu::<E>(rd, a);

                (a, b, c, next_pc_rel)
            }
            _ => unreachable!(),
        };
        (a, b, c, next_pc_rel)
    }

    /// Executes one cycle of the program, returning whether the program has finished.
    #[inline]
    #[allow(clippy::too_many_lines)]
    fn execute_cycle<E: ExecutorConfig>(&mut self) -> Result<bool, ExecutionError> {
        // Fetch the instruction at the current program counter.
        let instruction = self.fetch();

        // Log the current state of the runtime.
        self.log::<E>(&instruction);

        // Execute the instruction.
        self.execute_instruction::<E>(&instruction)?;

        // Increment the clock.
        self.state.global_clk += 1;

        if E::UNCONSTRAINED {
            self.total_unconstrained_cycles += 1;
        }

        if !E::UNCONSTRAINED {
            // Every N cycles, check if there exists at least one shape that fits.
            //
            // If we're close to not fitting, early stop the shard to ensure we don't OOM.
            let mut maximal_size_reached = true;
            if self.state.global_clk % self.size_check_frequency == 0 {
                // Estimate the number of events in the trace.
                Self::estimate_riscv_event_counts(
                    &mut self.event_counts,
                    &self.local_counts,
                    self.local_counts.load_x0_counts,
                    &self.internal_syscalls_override,
                );

                // Check if the main trace area or table height is too large.
                if let Some(ShardingThreshold { element_threshold, height_threshold }) =
                    self.sharding_threshold
                {
                    let padded_event_counts =
                        pad_rv32im_event_counts(self.event_counts, self.size_check_frequency);
                    let bump_clk_high =
                        (self.state.clk >> 24) + 1 - (self.record.initial_timestamp >> 24);
                    let (padded_element_count, max_height) = estimate_trace_elements(
                        padded_event_counts,
                        bump_clk_high,
                        &self.costs,
                        self.program_len,
                        &self.internal_syscalls_override,
                    );

                    if padded_element_count > element_threshold || max_height > height_threshold {
                        tracing::info!(
                            "stopping shard at clk {}, max height {}",
                            self.state.clk,
                            max_height
                        );
                        maximal_size_reached = false;
                    }
                }
            }

            if !maximal_size_reached {
                self.state.current_shard = Shard::new(self.state.current_shard.get() + 1)
                    .expect("Shard number should not overflow");
                self.record.last_timestamp = self.state.clk;
                self.bump_record::<E>();
            }

            // If the cycle limit is exceeded, return an error.
            if let Some(max_cycles) = self.max_cycles {
                if self.state.global_clk > max_cycles {
                    return Err(ExecutionError::ExceededCycleLimit(max_cycles));
                }
            }
        }

        let done = self.state.pc_rel == HALT_PC_REL
            || self.state.pc_rel as usize >= self.program.instructions.len() * 4;
        if done && E::UNCONSTRAINED {
            tracing::error!("program ended in unconstrained mode at clk {}", self.state.global_clk);
            return Err(ExecutionError::EndInUnconstrained());
        }
        Ok(done)
    }

    /// Bump the record.
    pub fn bump_record<E: ExecutorConfig>(&mut self) {
        if let Some(estimator) = &mut self.record_estimator {
            Self::estimate_riscv_event_counts(
                &mut self.event_counts,
                &self.local_counts,
                self.local_counts.load_x0_counts,
                &self.internal_syscalls_override,
            );
            // The above method estimates event counts only for core shards.
            estimator.core_records.push(self.event_counts);
        }
        let bump_clk_high = (self.state.clk >> 24) + 1 - (self.record.initial_timestamp >> 24);
        self.record.estimated_trace_area = estimate_trace_elements(
            self.event_counts,
            bump_clk_high,
            &self.costs,
            self.program_len,
            &self.internal_syscalls_override,
        )
        .0;
        self.local_counts = LocalCounts::default();
        // Copy all of the existing local memory accesses to the record's local_memory_access vec.
        if E::MODE == ExecutorMode::Trace {
            for (_, event) in self.local_memory_access.drain() {
                self.record.cpu_local_memory_access.push(event);
            }
        }
        if self.record.last_timestamp == 0 {
            self.record.last_timestamp = self.state.clk;
        }

        let removed_record = std::mem::replace(
            &mut self.record,
            Box::new(ExecutionRecord::new(self.program.clone())),
        );
        let public_values = removed_record.public_values;
        self.record.public_values = public_values;
        self.record.initial_timestamp = self.state.clk;
        self.records.push(removed_record);
    }

    /// Execute up to `self.shard_batch_size` cycles, returning the events emitted and whether the
    /// program ended.
    ///
    /// # Errors
    ///
    /// This function will return an error if the program execution fails.
    pub fn execute_record(
        &mut self,
        emit_global_memory_events: bool,
    ) -> Result<(Vec<Box<ExecutionRecord>>, bool), ExecutionError> {
        self.emit_global_memory_events = emit_global_memory_events;
        self.print_report = true;
        let done = self.execute::<Trace>()?;
        Ok((std::mem::take(&mut self.records), done))
    }

    /// Execute up to `self.shard_batch_size` cycles, returning the checkpoint from before execution
    /// and whether the program ended.
    ///
    /// # Errors
    ///
    /// This function will return an error if the program execution fails.
    #[allow(clippy::type_complexity)]
    pub fn execute_state(
        &mut self,
        emit_global_memory_events: bool,
    ) -> Result<(ExecutionState, PublicValues<u64, u64, u64, u32>, bool), ExecutionError> {
        self.memory_checkpoint.clear();
        self.emit_global_memory_events = emit_global_memory_events;

        // Clone self.state without memory, uninitialized_memory, proof_stream in it so it's faster.
        let memory = std::mem::take(&mut self.state.memory);
        let uninitialized_memory = std::mem::take(&mut self.state.uninitialized_memory);
        let proof_stream = std::mem::take(&mut self.state.proof_stream);
        let mut checkpoint = tracing::debug_span!("clone").in_scope(|| self.state.clone());
        self.state.memory = memory;
        self.state.uninitialized_memory = uninitialized_memory;
        self.state.proof_stream = proof_stream;

        let done = tracing::debug_span!("execute").in_scope(|| self.execute::<Checkpoint>())?;
        // Create a checkpoint using `memory_checkpoint`. Just include all memory if `done` since we
        // need it all for MemoryFinalize.
        let next_pc_rel = self.state.pc_rel;
        tracing::debug_span!("create memory checkpoint").in_scope(|| {
            let replacement_memory_checkpoint = Memory::<_>::new_preallocated();
            let replacement_uninitialized_memory_checkpoint = Memory::<_>::new_preallocated();
            let memory_checkpoint =
                std::mem::replace(&mut self.memory_checkpoint, replacement_memory_checkpoint);
            let uninitialized_memory_checkpoint = std::mem::replace(
                &mut self.uninitialized_memory_checkpoint,
                replacement_uninitialized_memory_checkpoint,
            );
            if done && !self.emit_global_memory_events {
                // If it's the last shard, and we're not emitting memory events, we need to include
                // all memory so that memory events can be emitted from the checkpoint. But we need
                // to first reset any modified memory to as it was before the execution.
                checkpoint.memory.clone_from(&self.state.memory);
                memory_checkpoint.into_iter().for_each(|(addr, record)| {
                    if let Some(record) = record {
                        checkpoint.memory.insert(addr, record);
                    } else {
                        checkpoint.memory.remove(addr);
                    }
                });
                checkpoint.uninitialized_memory = self.state.uninitialized_memory.clone();
                // Remove memory that was written to in this batch.
                for (addr, is_old) in uninitialized_memory_checkpoint {
                    if !is_old {
                        checkpoint.uninitialized_memory.remove(addr);
                    }
                }
            } else {
                checkpoint.memory = memory_checkpoint
                    .into_iter()
                    .filter_map(|(addr, record)| record.map(|record| (addr, record)))
                    .collect();
                checkpoint.uninitialized_memory = uninitialized_memory_checkpoint
                    .into_iter()
                    .filter(|&(_, has_value)| has_value)
                    .map(|(addr, _)| (addr, *self.state.uninitialized_memory.get(addr).unwrap()))
                    .collect();
            }
        });
        let mut public_values = self.records.last().as_ref().unwrap().public_values;
        public_values.pc_start_rel = next_pc_rel;
        public_values.next_pc_rel = next_pc_rel;
        if !done {
            self.records.clear();
        }
        Ok((checkpoint, public_values, done))
    }

    fn initialize(&mut self) {
        self.state.clk = 1;

        tracing::debug!("loading memory image");
        for (&addr, value) in &self.program.memory_image {
            self.state.memory.insert(addr, MemoryEntry::init(*value));
        }
        self.state.memory.insert(0, MemoryEntry::init(0));
    }

    /// Executes the program without tracing and without emitting events.
    ///
    /// # Errors
    ///
    /// This function will return an error if the program execution fails.
    pub fn run_fast(&mut self) -> Result<(), ExecutionError> {
        self.print_report = true;
        while !self.execute::<Simple>()? {}

        #[cfg(feature = "profiling")]
        if let Some((profiler, writer)) = self.profiler.take() {
            profiler.write(writer).expect("Failed to write profile to output file");
        }

        Ok(())
    }

    /// Executes the program in checkpoint mode, without emitting the checkpoints.
    ///
    /// # Errors
    ///
    /// This function will return an error if the program execution fails.
    pub fn run_checkpoint(
        &mut self,
        emit_global_memory_events: bool,
    ) -> Result<(), ExecutionError> {
        self.print_report = true;
        while !self.execute_state(emit_global_memory_events)?.2 {}
        Ok(())
    }

    /// Lightweight execution in unconstrained mode.
    pub fn run_unconstrained(&mut self) -> Result<(), ExecutionError> {
        let mut done = false;
        while !done {
            // Fetch the instruction at the current program counter.
            let instruction = self.fetch();

            // Execute the instruction.
            self.execute_instruction::<Unconstrained>(&instruction)?;

            done = self.state.pc_rel == HALT_PC_REL
                || self.state.pc_rel as usize >= self.program.instructions.len() * 4;
        }

        Ok(())
    }

    /// Executes the program and prints the execution report.
    ///
    /// # Errors
    ///
    /// This function will return an error if the program execution fails.
    pub fn run<E: ExecutorConfig>(&mut self) -> Result<(), ExecutionError> {
        self.print_report = true;
        while !self.execute::<E>()? {}

        #[cfg(feature = "profiling")]
        if let Some((profiler, writer)) = self.profiler.take() {
            profiler.write(writer).expect("Failed to write profile to output file");
        }

        Ok(())
    }

    /// Executes up to `self.shard_batch_size` cycles of the program, returning whether the program
    /// has finished.
    pub fn execute<E: ExecutorConfig>(&mut self) -> Result<bool, ExecutionError> {
        // Get the program.
        let program = self.program.clone();

        // Get the current shard.
        let start_shard = self.state.current_shard;

        // If it's the first cycle, initialize the program.
        if self.state.global_clk == 0 {
            self.initialize();
        }

        self.record.initial_timestamp = self.state.clk;

        let unconstrained_cycle_limit =
            std::env::var("UNCONSTRAINED_CYCLE_LIMIT").map(|v| v.parse::<u64>().unwrap()).ok();

        // Loop until we've executed `self.shard_batch_size` shards if `self.shard_batch_size` is
        // set.
        let mut done = false;
        let mut current_shard = self.state.current_shard;
        let mut num_shards_executed = 0;
        loop {
            if self.execute_cycle::<E>()? {
                done = true;
                break;
            }

            // Check if the unconstrained cycle limit was exceeded.
            if let Some(unconstrained_cycle_limit) = unconstrained_cycle_limit {
                if self.total_unconstrained_cycles > unconstrained_cycle_limit {
                    return Err(ExecutionError::UnconstrainedCycleLimitExceeded(
                        unconstrained_cycle_limit,
                    ));
                }
            }

            if self.shard_batch_size > 0 && current_shard != self.state.current_shard {
                num_shards_executed += 1;
                current_shard = self.state.current_shard;
                if num_shards_executed == self.shard_batch_size {
                    break;
                }
            }
        }

        // Get the final public values.
        let public_values = self.record.public_values;

        if done {
            self.postprocess::<E>();

            // Push the remaining execution record with memory initialize & finalize events.
            self.bump_record::<E>();

            // Flush stdout and stderr.
            if let Some(ref mut w) = self.io_options.stdout {
                if let Err(e) = w.flush() {
                    tracing::error!("failed to flush stdout override: {e}");
                }
            }

            if let Some(ref mut w) = self.io_options.stderr {
                if let Err(e) = w.flush() {
                    tracing::error!("failed to flush stderr override: {e}");
                }
            }
        }

        // Push the remaining execution record, if there are any CPU events.
        if self.record.contains_cpu() {
            self.bump_record::<E>();
        }

        // Set the global public values for all shards.
        let mut last_next_pc = 0;
        let mut last_exit_code = 0;
        for (i, record) in self.records.iter_mut().enumerate() {
            record.program = program.clone();
            record.public_values = public_values;
            record.public_values.committed_value_digest = public_values.committed_value_digest;
            record.public_values.deferred_proofs_digest = public_values.deferred_proofs_digest;
            record.public_values.execution_shard = start_shard.get() + i as u32;
            if record.contains_cpu() {
                record.public_values.pc_start_rel = record.pc_start_rel.unwrap();
                record.public_values.next_pc_rel = record.next_pc_rel;
                record.public_values.exit_code = record.exit_code;
                record.public_values.last_timestamp = record.last_timestamp;
                record.public_values.initial_timestamp = record.initial_timestamp;
                last_next_pc = record.public_values.next_pc_rel;
                last_exit_code = record.public_values.exit_code;
            } else {
                record.public_values.pc_start_rel = last_next_pc;
                record.public_values.next_pc_rel = last_next_pc;
                record.public_values.prev_exit_code = last_exit_code;
                record.public_values.exit_code = last_exit_code;
            }
        }

        Ok(done)
    }

    fn postprocess<E: ExecutorConfig>(&mut self) {
        // Flush remaining stdout/stderr
        for (fd, buf) in &self.io_buf {
            if !buf.is_empty() {
                match fd {
                    1 => {
                        eprintln!("stdout: {buf}");
                    }
                    2 => {
                        eprintln!("stderr: {buf}");
                    }
                    _ => {}
                }
            }
        }

        // Ensure that all proofs and input bytes were read, otherwise warn the user.
        if self.state.proof_stream_ptr != self.state.proof_stream.len() {
            tracing::warn!(
                "Not all proofs were read. Proving will fail during recursion. Did you pass too
        many proofs in or forget to call verify_sp1_proof?"
            );
        }

        if !self.state.input_stream.is_empty() {
            tracing::warn!("Not all input bytes were read.");
        }

        if let Some(estimator) = &mut self.record_estimator {
            // Mirror the logic below.
            // Register 0 is always init and finalized, so we add 1
            // registers 1..32
            let touched_reg_ct =
                1 + (1..32).filter(|&r| self.state.memory.registers.get(r).is_some()).count();
            let total_mem = touched_reg_ct + self.state.memory.page_table.exact_len();
            // The memory_image is already initialized in the MemoryProgram chip
            // so we subtract it off. It is initialized in the executor in the `initialize`
            // function.
            estimator.memory_global_init_events = total_mem
                .checked_sub(self.record.program.memory_image.len())
                .expect("program memory image should be accounted for in memory exact len")
                as u64;
            estimator.memory_global_finalize_events = total_mem as u64;
        }

        if self.emit_global_memory_events
            && (E::MODE == ExecutorMode::Trace || E::MODE == ExecutorMode::Checkpoint)
        {
            // SECTION: Set up all MemoryInitializeFinalizeEvents needed for memory argument.
            let memory_finalize_events = &mut self.record.global_memory_finalize_events;
            memory_finalize_events.reserve_exact(self.state.memory.page_table.estimate_len() + 32);

            // We handle the addr = 0 case separately, as we constrain it to be 0 in the first row
            // of the memory finalize table so it must be first in the array of events.
            let addr_0_record = self.state.memory.get(0);

            let addr_0_final_record = match addr_0_record {
                Some(record) => record,
                None => &MemoryEntry { lshard: LogicalShard::default(), timestamp: 0, value: 0 },
            };
            memory_finalize_events
                .push(MemoryInitializeFinalizeEvent::finalize_from_record(0, addr_0_final_record));

            let memory_initialize_events = &mut self.record.global_memory_initialize_events;
            memory_initialize_events
                .reserve_exact(self.state.memory.page_table.estimate_len() + 32);
            let addr_0_initialize_event = MemoryInitializeFinalizeEvent::initialize(0, 0);
            memory_initialize_events.push(addr_0_initialize_event);

            // Count the number of touched memory addresses manually, since `PagedMemory` doesn't
            // already know its length.
            if self.print_report {
                self.report.touched_memory_addresses = 0;
            }
            for addr in 1..32 {
                let record = self.state.memory.registers.get(addr);
                if record.is_some() {
                    if self.print_report {
                        self.report.touched_memory_addresses += 1;
                    }
                    // Program memory is initialized in the MemoryProgram chip and doesn't require
                    // any events, so we only send init events for other memory
                    // addresses.
                    if !self.record.program.memory_image.contains_key(&addr) {
                        let initial_value =
                            self.state.uninitialized_memory.registers.get(addr).unwrap_or(&0);
                        memory_initialize_events
                            .push(MemoryInitializeFinalizeEvent::initialize(addr, *initial_value));
                    }

                    let record = *record.unwrap();
                    memory_finalize_events
                        .push(MemoryInitializeFinalizeEvent::finalize_from_record(addr, &record));
                }
            }
            for addr in self.state.memory.page_table.keys() {
                if self.print_report {
                    self.report.touched_memory_addresses += 1;
                }

                // Program memory is initialized in the MemoryProgram chip and doesn't require any
                // events, so we only send init events for other memory addresses.
                if !self.record.program.memory_image.contains_key(&addr) {
                    let initial_value = self.state.uninitialized_memory.get(addr).unwrap_or(&0);
                    memory_initialize_events
                        .push(MemoryInitializeFinalizeEvent::initialize(addr, *initial_value));
                }

                let record = *self.state.memory.get(addr).unwrap();
                memory_finalize_events
                    .push(MemoryInitializeFinalizeEvent::finalize_from_record(addr, &record));
            }
        }
    }

    /// Maps the opcode counts to the number of events in each air.
    fn estimate_riscv_event_counts(
        event_counts: &mut EnumMap<RiscvAirId, u64>,
        local_counts: &LocalCounts,
        load_x0_counts: u64,
        internal_syscalls_override: &[SyscallCode],
    ) {
        let touched_addresses: u64 = local_counts.local_mem as u64;
        let syscalls_sent: u64 = local_counts.syscalls_sent as u64;
        let opcode_counts: &EnumMap<Opcode, u64> = &local_counts.event_counts;

        // Compute the number of events in the add chip.
        event_counts[RiscvAirId::Add] = opcode_counts[Opcode::ADD];

        // Compute the number of events in the addi chip.
        event_counts[RiscvAirId::Addi] = opcode_counts[Opcode::ADDI];

        // Compute the number of events in the sub chip.
        event_counts[RiscvAirId::Sub] = opcode_counts[Opcode::SUB];

        // Compute the number of events in the bitwise chip.
        event_counts[RiscvAirId::Bitwise] =
            opcode_counts[Opcode::XOR] + opcode_counts[Opcode::OR] + opcode_counts[Opcode::AND];

        // Compute the number of events in the divrem chip.
        event_counts[RiscvAirId::DivRem] = opcode_counts[Opcode::DIV]
            + opcode_counts[Opcode::DIVU]
            + opcode_counts[Opcode::REM]
            + opcode_counts[Opcode::REMU]
            + opcode_counts[Opcode::DIVW]
            + opcode_counts[Opcode::DIVUW]
            + opcode_counts[Opcode::REMW]
            + opcode_counts[Opcode::REMUW];

        // Compute the number of events in the lt chip.
        event_counts[RiscvAirId::Lt] = opcode_counts[Opcode::SLT] + opcode_counts[Opcode::SLTU];

        // Compute the number of events in the mul chip.
        event_counts[RiscvAirId::Mul] = opcode_counts[Opcode::MUL]
            + opcode_counts[Opcode::MULH]
            + opcode_counts[Opcode::MULHU]
            + opcode_counts[Opcode::MULHSU]
            + opcode_counts[Opcode::MULW];

        // Compute the number of events in the shift left chip.
        event_counts[RiscvAirId::ShiftLeft] =
            opcode_counts[Opcode::SLL] + opcode_counts[Opcode::SLLW] + opcode_counts[Opcode::SLLIW];

        // Compute the number of events in the shift right chip.
        event_counts[RiscvAirId::ShiftRight] = opcode_counts[Opcode::SRL]
            + opcode_counts[Opcode::SRA]
            + opcode_counts[Opcode::SRLW]
            + opcode_counts[Opcode::SRLIW]
            + opcode_counts[Opcode::SRAW]
            + opcode_counts[Opcode::SRAIW];

        // Compute the number of events in the memory local chip.
        event_counts[RiscvAirId::MemoryLocal] =
            touched_addresses.div_ceil(NUM_LOCAL_MEMORY_ENTRIES_PER_ROW_EXEC as u64);

        // Compute the number of events in the branch chip.
        event_counts[RiscvAirId::Branch] = opcode_counts[Opcode::BEQ]
            + opcode_counts[Opcode::BNE]
            + opcode_counts[Opcode::BLT]
            + opcode_counts[Opcode::BGE]
            + opcode_counts[Opcode::BLTU]
            + opcode_counts[Opcode::BGEU];

        // Compute the number of events in the jump chip.
        event_counts[RiscvAirId::Jal] = opcode_counts[Opcode::JAL];
        event_counts[RiscvAirId::Jalr] = opcode_counts[Opcode::JALR];

        // Compute the number of events in the auipc chip.
        event_counts[RiscvAirId::Auipc] = opcode_counts[Opcode::AUIPC]
            + opcode_counts[Opcode::UNIMP]
            + opcode_counts[Opcode::EBREAK];

        // Compute the number of events in the memory instruction chip.
        event_counts[RiscvAirId::LoadByte] = opcode_counts[Opcode::LB] + opcode_counts[Opcode::LBU];
        event_counts[RiscvAirId::LoadHalf] = opcode_counts[Opcode::LH] + opcode_counts[Opcode::LHU];
        event_counts[RiscvAirId::LoadWord] = opcode_counts[Opcode::LW] + opcode_counts[Opcode::LWU];
        event_counts[RiscvAirId::LoadDouble] = opcode_counts[Opcode::LD];
        event_counts[RiscvAirId::LoadX0] = load_x0_counts;

        event_counts[RiscvAirId::StoreByte] = opcode_counts[Opcode::SB];
        event_counts[RiscvAirId::StoreHalf] = opcode_counts[Opcode::SH];
        event_counts[RiscvAirId::StoreWord] = opcode_counts[Opcode::SW];
        event_counts[RiscvAirId::StoreDouble] = opcode_counts[Opcode::SD];

        // Compute the number of events in the syscall instruction chip.
        event_counts[RiscvAirId::SyscallInstrs] = opcode_counts[Opcode::ECALL];

        // Compute the number of events in the syscall core chip.
        event_counts[RiscvAirId::SyscallCore] = syscalls_sent;

        // Compute the number of events in the global chip.
        event_counts[RiscvAirId::Global] = 2 * touched_addresses + syscalls_sent;

        // Compute the number of events in the retained precompiles.
        for syscall in internal_syscalls_override {
            let air_id = SyscallCode::as_air_id(*syscall).unwrap();
            event_counts[air_id] = local_counts.retained_precompile_counts[air_id];
        }
    }

    #[inline]
    fn log<E: ExecutorConfig>(&mut self, _: &Instruction) {
        #[cfg(feature = "profiling")]
        if let Some((ref mut profiler, _)) = self.profiler {
            if !E::UNCONSTRAINED {
                profiler.record(
                    self.state.global_clk,
                    (self.state.pc_rel as u64).wrapping_add(self.program.pc_base),
                );
            }
        }

        if !E::UNCONSTRAINED && self.state.global_clk % 10_000_000 == 0 {
            tracing::info!("clk = {} pc = 0x{:x?}", self.state.global_clk, self.state.pc_rel);
        }
    }
}

impl Default for ExecutorMode {
    fn default() -> Self {
        Self::Simple
    }
}

/// Aligns an address to the nearest word below or equal to it.
#[must_use]
pub const fn align(addr: u64) -> u64 {
    addr - addr % 8
}

fn sign_extend_imm(value: u64, bits: u8) -> i64 {
    let shift = 64 - bits;
    ((value as i64) << shift) >> shift
}

#[cfg(test)]
mod tests {

    use std::sync::Arc;

    use sp1_zkvm::syscalls::SHA_COMPRESS;

    use crate::programs::tests::{
        fibonacci_program, panic_program, secp256r1_add_program, secp256r1_double_program,
        simple_memory_program, simple_program, ssz_withdrawals_program, u256xu2048_mul_program,
    };

    use crate::{Register, SP1Context, SP1CoreOpts, Simple};

    use super::{Executor, Instruction, Opcode, Program};

    fn _assert_send<T: Send>() {}

    /// Runtime needs to be Send so we can use it across async calls.
    #[allow(clippy::used_underscore_items)]
    fn _assert_runtime_is_send() {
        #[allow(clippy::used_underscore_items)]
        _assert_send::<Executor>();
    }

    #[test]
    fn test_simple_program_run() {
        let program = Arc::new(simple_program());
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run::<Simple>().unwrap();
        assert_eq!(runtime.register::<Simple>(Register::X31), 42);
    }

    #[test]
    fn test_fibonacci_program_run() {
        let program = Arc::new(fibonacci_program());
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run::<Simple>().unwrap();
    }

    #[test]
    fn test_fibonacci_program_run_with_max_cycles() {
        let program = Arc::new(fibonacci_program());
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run::<Simple>().unwrap();

        let max_cycles = runtime.state.global_clk;

        let program = Arc::new(fibonacci_program());
        let context = SP1Context::builder().max_cycles(max_cycles).build();
        let mut runtime = Executor::with_context(program, SP1CoreOpts::default(), context);
        runtime.run::<Simple>().unwrap();
    }

    #[test]
    fn test_secp256r1_add_program_run() {
        let program = Arc::new(secp256r1_add_program());
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run::<Simple>().unwrap();
    }

    #[test]
    fn test_secp256r1_double_program_run() {
        let program = Arc::new(secp256r1_double_program());
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run::<Simple>().unwrap();
    }

    #[test]
    fn test_u256xu2048_mul() {
        let program = Arc::new(u256xu2048_mul_program());
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run::<Simple>().unwrap();
    }

    #[test]
    fn test_ssz_withdrawals_program_run() {
        let program = Arc::new(ssz_withdrawals_program());
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run::<Simple>().unwrap();
    }

    #[test]
    fn test_panic() {
        let program = Arc::new(panic_program());
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run::<Simple>().unwrap();
    }

    #[test]
    fn test_add() {
        // main:
        //     addi x29, x0, 5
        //     addi x30, x0, 37
        //     add x31, x30, x29
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 37, false, true),
            Instruction::new(Opcode::ADD, 31, 30, 29, false, false),
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run::<Simple>().unwrap();
        assert_eq!(runtime.register::<Simple>(Register::X31), 42);
    }

    #[test]
    fn test_sub() {
        //     addi x29, x0, 5
        //     addi x30, x0, 37
        //     sub x31, x30, x29
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 37, false, true),
            Instruction::new(Opcode::SUB, 31, 30, 29, false, false),
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));

        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run::<Simple>().unwrap();
        assert_eq!(runtime.register::<Simple>(Register::X31), 32);
    }

    #[test]
    fn test_xor() {
        //     addi x29, x0, 5
        //     addi x30, x0, 37
        //     xor x31, x30, x29
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 37, false, true),
            Instruction::new(Opcode::XOR, 31, 30, 29, false, false),
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));

        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run::<Simple>().unwrap();
        assert_eq!(runtime.register::<Simple>(Register::X31), 32);
    }

    #[test]
    fn test_or() {
        //     addi x29, x0, 5
        //     addi x30, x0, 37
        //     or x31, x30, x29
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 37, false, true),
            Instruction::new(Opcode::OR, 31, 30, 29, false, false),
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));

        let mut runtime = Executor::new(program, SP1CoreOpts::default());

        runtime.run::<Simple>().unwrap();
        assert_eq!(runtime.register::<Simple>(Register::X31), 37);
    }

    #[test]
    fn test_and() {
        //     addi x29, x0, 5
        //     addi x30, x0, 37
        //     and x31, x30, x29
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 37, false, true),
            Instruction::new(Opcode::AND, 31, 30, 29, false, false),
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));

        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run::<Simple>().unwrap();
        assert_eq!(runtime.register::<Simple>(Register::X31), 5);
    }

    #[test]
    fn test_sll() {
        //     addi x29, x0, 5
        //     addi x30, x0, 37
        //     sll x31, x30, x29
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 37, false, true),
            Instruction::new(Opcode::SLL, 31, 30, 29, false, false),
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));

        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run_fast().unwrap();
        assert_eq!(runtime.register::<Simple>(Register::X31), 1184);
    }

    #[test]
    fn test_srl() {
        //     addi x29, x0, 5
        //     addi x30, x0, 37
        //     srl x31, x30, x29
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 37, false, true),
            Instruction::new(Opcode::SRL, 31, 30, 29, false, false),
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));

        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run_fast().unwrap();
        assert_eq!(runtime.register::<Simple>(Register::X31), 1);
    }

    #[test]
    fn test_sra() {
        //     addi x29, x0, 5
        //     addi x30, x0, 37
        //     sra x31, x30, x29
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 37, false, true),
            Instruction::new(Opcode::SRA, 31, 30, 29, false, false),
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));

        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run_fast().unwrap();
        assert_eq!(runtime.register::<Simple>(Register::X31), 1);
    }

    #[test]
    fn test_slt() {
        //     addi x29, x0, 5
        //     addi x30, x0, 37
        //     slt x31, x30, x29
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 37, false, true),
            Instruction::new(Opcode::SLT, 31, 30, 29, false, false),
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));

        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run_fast().unwrap();
        assert_eq!(runtime.register::<Simple>(Register::X31), 0);
    }

    #[test]
    fn test_sltu() {
        //     addi x29, x0, 5
        //     addi x30, x0, 37
        //     sltu x31, x30, x29
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 30, 0, 37, false, true),
            Instruction::new(Opcode::SLTU, 31, 30, 29, false, false),
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));

        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run_fast().unwrap();
        assert_eq!(runtime.register::<Simple>(Register::X31), 0);
    }

    #[test]
    fn test_addi() {
        //     addi x29, x0, 5
        //     addi x30, x29, 37
        //     addi x31, x30, 42
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 30, 29, 37, false, true),
            Instruction::new(Opcode::ADD, 31, 30, 42, false, true),
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));

        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run_fast().unwrap();
        assert_eq!(runtime.register::<Simple>(Register::X31), 84);
    }

    #[test]
    fn test_addi_negative() {
        //     addi x29, x0, 5
        //     addi x30, x29, -1
        //     addi x31, x30, 4
        // TODO: update negative in u64
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::ADD, 30, 29, 0xFFFF_FFFF, false, true),
            Instruction::new(Opcode::ADD, 31, 30, 4, false, true),
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run_fast().unwrap();
        assert_eq!(runtime.register::<Simple>(Register::X31), 5 - 1 + 4);
    }

    #[test]
    fn test_xori() {
        //     addi x29, x0, 5
        //     xori x30, x29, 37
        //     xori x31, x30, 42
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::XOR, 30, 29, 37, false, true),
            Instruction::new(Opcode::XOR, 31, 30, 42, false, true),
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run_fast().unwrap();
        assert_eq!(runtime.register::<Simple>(Register::X31), 10);
    }

    #[test]
    fn test_ori() {
        //     addi x29, x0, 5
        //     ori x30, x29, 37
        //     ori x31, x30, 42
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::OR, 30, 29, 37, false, true),
            Instruction::new(Opcode::OR, 31, 30, 42, false, true),
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run_fast().unwrap();
        assert_eq!(runtime.register::<Simple>(Register::X31), 47);
    }

    #[test]
    fn test_andi() {
        //     addi x29, x0, 5
        //     andi x30, x29, 37
        //     andi x31, x30, 42
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::AND, 30, 29, 37, false, true),
            Instruction::new(Opcode::AND, 31, 30, 42, false, true),
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run_fast().unwrap();
        assert_eq!(runtime.register::<Simple>(Register::X31), 0);
    }

    #[test]
    fn test_slli() {
        //     addi x29, x0, 5
        //     slli x31, x29, 37
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 5, false, true),
            Instruction::new(Opcode::SLL, 31, 29, 4, false, true),
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run_fast().unwrap();
        assert_eq!(runtime.register::<Simple>(Register::X31), 80);
    }

    #[test]
    fn test_srli() {
        //    addi x29, x0, 5
        //    srli x31, x29, 37
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 42, false, true),
            Instruction::new(Opcode::SRL, 31, 29, 4, false, true),
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run_fast().unwrap();
        assert_eq!(runtime.register::<Simple>(Register::X31), 2);
    }

    #[test]
    fn test_srai() {
        //   addi x29, x0, 5
        //   srai x31, x29, 37
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 42, false, true),
            Instruction::new(Opcode::SRA, 31, 29, 4, false, true),
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run_fast().unwrap();
        assert_eq!(runtime.register::<Simple>(Register::X31), 2);
    }

    #[test]
    fn test_slti() {
        //   addi x29, x0, 5
        //   slti x31, x29, 37
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 42, false, true),
            Instruction::new(Opcode::SLT, 31, 29, 37, false, true),
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run_fast().unwrap();
        assert_eq!(runtime.register::<Simple>(Register::X31), 0);
    }

    #[test]
    fn test_sltiu() {
        //   addi x29, x0, 5
        //   sltiu x31, x29, 37
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 42, false, true),
            Instruction::new(Opcode::SLTU, 31, 29, 37, false, true),
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run_fast().unwrap();
        assert_eq!(runtime.register::<Simple>(Register::X31), 0);
    }

    #[test]
    fn test_jalr() {
        //   addi x11, x11, 100
        //   jalr x5, x11, 8
        //
        // `JALR rd offset(rs)` reads the value at rs, adds offset to it and uses it as the
        // destination address. It then stores the address of the next instruction in rd in case
        // we'd want to come back here.

        let instructions = vec![
            Instruction::new(Opcode::ADD, 11, 11, 100, false, true),
            Instruction::new(Opcode::JALR, 5, 11, 8, false, true),
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run_fast().unwrap();
        assert_eq!(runtime.registers::<Simple>()[Register::X5 as usize], 8);
        assert_eq!(runtime.registers::<Simple>()[Register::X11 as usize], 100);
        assert_eq!(runtime.state.pc_rel, 108);
    }

    fn simple_op_code_test(opcode: Opcode, expected: u64, a: u64, b: u64) {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 10, 0, a, false, true),
            Instruction::new(Opcode::ADD, 11, 0, b, false, true),
            Instruction::new(opcode, 12, 10, 11, false, false),
        ];
        let program = Arc::new(Program::new(instructions, 0, 0));
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run_fast().unwrap();
        assert_eq!(runtime.registers::<Simple>()[Register::X12 as usize], expected);
    }

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn multiplication_tests() {
        simple_op_code_test(Opcode::MULHU, 0x00000000, 0x00000000, 0x00000000);
        simple_op_code_test(Opcode::MULHU, 0x00000000, 0x00000001, 0x00000001);
        simple_op_code_test(Opcode::MULHU, 0x00000000, 0x00000003, 0x00000007);
        simple_op_code_test(Opcode::MULHU, 0x00000000, 0x00000000, 0xffff8000);
        simple_op_code_test(Opcode::MULHU, 0x00000000, 0x80000000, 0x00000000);
        simple_op_code_test(Opcode::MULHU, 0x7fffc000, 0x80000000, 0xffff8000);
        simple_op_code_test(Opcode::MULHU, 0x0001fefe, 0xaaaaaaab, 0x0002fe7d);
        simple_op_code_test(Opcode::MULHU, 0x0001fefe, 0x0002fe7d, 0xaaaaaaab);
        simple_op_code_test(Opcode::MULHU, 0xfe010000, 0xff000000, 0xff000000);
        simple_op_code_test(Opcode::MULHU, 0xfffffffe, 0xffffffff, 0xffffffff);
        simple_op_code_test(Opcode::MULHU, 0x00000000, 0xffffffff, 0x00000001);
        simple_op_code_test(Opcode::MULHU, 0x00000000, 0x00000001, 0xffffffff);

        simple_op_code_test(Opcode::MULHSU, 0x00000000, 0x00000000, 0x00000000);
        simple_op_code_test(Opcode::MULHSU, 0x00000000, 0x00000001, 0x00000001);
        simple_op_code_test(Opcode::MULHSU, 0x00000000, 0x00000003, 0x00000007);
        simple_op_code_test(Opcode::MULHSU, 0x00000000, 0x00000000, 0xffff8000);
        simple_op_code_test(Opcode::MULHSU, 0x00000000, 0x80000000, 0x00000000);
        simple_op_code_test(Opcode::MULHSU, 0x80004000, 0x80000000, 0xffff8000);
        simple_op_code_test(Opcode::MULHSU, 0xffff0081, 0xaaaaaaab, 0x0002fe7d);
        simple_op_code_test(Opcode::MULHSU, 0x0001fefe, 0x0002fe7d, 0xaaaaaaab);
        simple_op_code_test(Opcode::MULHSU, 0xff010000, 0xff000000, 0xff000000);
        simple_op_code_test(Opcode::MULHSU, 0xffffffff, 0xffffffff, 0xffffffff);
        simple_op_code_test(Opcode::MULHSU, 0xffffffff, 0xffffffff, 0x00000001);
        simple_op_code_test(Opcode::MULHSU, 0x00000000, 0x00000001, 0xffffffff);

        simple_op_code_test(Opcode::MULH, 0x00000000, 0x00000000, 0x00000000);
        simple_op_code_test(Opcode::MULH, 0x00000000, 0x00000001, 0x00000001);
        simple_op_code_test(Opcode::MULH, 0x00000000, 0x00000003, 0x00000007);
        simple_op_code_test(Opcode::MULH, 0x00000000, 0x00000000, 0xffff8000);
        simple_op_code_test(Opcode::MULH, 0x00000000, 0x80000000, 0x00000000);
        simple_op_code_test(Opcode::MULH, 0x00000000, 0x80000000, 0x00000000);
        simple_op_code_test(Opcode::MULH, 0xffff0081, 0xaaaaaaab, 0x0002fe7d);
        simple_op_code_test(Opcode::MULH, 0xffff0081, 0x0002fe7d, 0xaaaaaaab);
        simple_op_code_test(Opcode::MULH, 0x00010000, 0xff000000, 0xff000000);
        simple_op_code_test(Opcode::MULH, 0x00000000, 0xffffffff, 0xffffffff);
        simple_op_code_test(Opcode::MULH, 0xffffffff, 0xffffffff, 0x00000001);
        simple_op_code_test(Opcode::MULH, 0xffffffff, 0x00000001, 0xffffffff);

        simple_op_code_test(Opcode::MUL, 0x00001200, 0x00007e00, 0xb6db6db7);
        simple_op_code_test(Opcode::MUL, 0x00001240, 0x00007fc0, 0xb6db6db7);
        simple_op_code_test(Opcode::MUL, 0x00000000, 0x00000000, 0x00000000);
        simple_op_code_test(Opcode::MUL, 0x00000001, 0x00000001, 0x00000001);
        simple_op_code_test(Opcode::MUL, 0x00000015, 0x00000003, 0x00000007);
        simple_op_code_test(Opcode::MUL, 0x00000000, 0x00000000, 0xffff8000);
        simple_op_code_test(Opcode::MUL, 0x00000000, 0x80000000, 0x00000000);
        simple_op_code_test(Opcode::MUL, 0x00000000, 0x80000000, 0xffff8000);
        simple_op_code_test(Opcode::MUL, 0x0000ff7f, 0xaaaaaaab, 0x0002fe7d);
        simple_op_code_test(Opcode::MUL, 0x0000ff7f, 0x0002fe7d, 0xaaaaaaab);
        simple_op_code_test(Opcode::MUL, 0x00000000, 0xff000000, 0xff000000);
        simple_op_code_test(Opcode::MUL, 0x00000001, 0xffffffff, 0xffffffff);
        simple_op_code_test(Opcode::MUL, 0xffffffff, 0xffffffff, 0x00000001);
        simple_op_code_test(Opcode::MUL, 0xffffffff, 0x00000001, 0xffffffff);
    }

    fn neg(a: u64) -> u64 {
        u64::MAX - a + 1
    }

    #[test]
    fn division_tests() {
        simple_op_code_test(Opcode::DIVU, 3, 20, 6);
        simple_op_code_test(Opcode::DIVU, 715_827_879, (u32::MAX - 20 + 1) as u64, 6);
        simple_op_code_test(Opcode::DIVU, 0, 20, (u32::MAX - 6 + 1) as u64);
        simple_op_code_test(Opcode::DIVU, 0, (u32::MAX - 20 + 1) as u64, (u32::MAX - 6 + 1) as u64);

        simple_op_code_test(Opcode::DIVU, 1 << 31, 1 << 31, 1);
        simple_op_code_test(Opcode::DIVU, 0, 1 << 31, (u32::MAX - 1 + 1) as u64);

        simple_op_code_test(Opcode::DIVU, u32::MAX as u64, 1 << 31, 0);
        simple_op_code_test(Opcode::DIVU, u32::MAX as u64, 1, 0);
        simple_op_code_test(Opcode::DIVU, u32::MAX as u64, 0, 0);

        simple_op_code_test(Opcode::DIV, 3, 18, 6);
        simple_op_code_test(Opcode::DIV, neg(6), neg(24), 4);
        simple_op_code_test(Opcode::DIV, neg(2), 16, neg(8));
        simple_op_code_test(Opcode::DIV, neg(1), 0, 0);

        // Overflow cases
        simple_op_code_test(Opcode::DIV, 1 << 31, 1 << 31, neg(1));
        simple_op_code_test(Opcode::REM, 0, 1 << 31, neg(1));
    }

    #[test]
    fn remainder_tests() {
        simple_op_code_test(Opcode::REM, 7, 16, 9);
        simple_op_code_test(Opcode::REM, neg(4), neg(22), 6);
        simple_op_code_test(Opcode::REM, 1, 25, neg(3));
        simple_op_code_test(Opcode::REM, neg(2), neg(22), neg(4));
        simple_op_code_test(Opcode::REM, 0, 873, 1);
        simple_op_code_test(Opcode::REM, 0, 873, neg(1));
        simple_op_code_test(Opcode::REM, 5, 5, 0);
        simple_op_code_test(Opcode::REM, neg(5), neg(5), 0);
        simple_op_code_test(Opcode::REM, 0, 0, 0);

        simple_op_code_test(Opcode::REMU, 4, 18, 7);
        simple_op_code_test(Opcode::REMU, 6, neg(20), 11);
        simple_op_code_test(Opcode::REMU, 23, 23, neg(6));
        simple_op_code_test(Opcode::REMU, neg(21), neg(21), neg(11));
        simple_op_code_test(Opcode::REMU, 5, 5, 0);
        simple_op_code_test(Opcode::REMU, neg(1), neg(1), 0);
        simple_op_code_test(Opcode::REMU, 0, 0, 0);
    }

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn shift_tests() {
        simple_op_code_test(Opcode::SLL, 0x00000001, 0x00000001, 0);
        simple_op_code_test(Opcode::SLL, 0x00000002, 0x00000001, 1);
        simple_op_code_test(Opcode::SLL, 0x00000080, 0x00000001, 7);
        simple_op_code_test(Opcode::SLL, 0x00004000, 0x00000001, 14);
        simple_op_code_test(Opcode::SLL, 0x80000000, 0x00000001, 31);
        simple_op_code_test(Opcode::SLL, 0xffffffff, 0xffffffff, 0);
        simple_op_code_test(Opcode::SLL, 0xfffffffe, 0xffffffff, 1);
        simple_op_code_test(Opcode::SLL, 0xffffff80, 0xffffffff, 7);
        simple_op_code_test(Opcode::SLL, 0xffffc000, 0xffffffff, 14);
        simple_op_code_test(Opcode::SLL, 0x80000000, 0xffffffff, 31);
        simple_op_code_test(Opcode::SLL, 0x21212121, 0x21212121, 0);
        simple_op_code_test(Opcode::SLL, 0x42424242, 0x21212121, 1);
        simple_op_code_test(Opcode::SLL, 0x90909080, 0x21212121, 7);
        simple_op_code_test(Opcode::SLL, 0x48484000, 0x21212121, 14);
        simple_op_code_test(Opcode::SLL, 0x80000000, 0x21212121, 31);
        simple_op_code_test(Opcode::SLL, 0x21212121, 0x21212121, 0xffffffe0);
        simple_op_code_test(Opcode::SLL, 0x42424242, 0x21212121, 0xffffffe1);
        simple_op_code_test(Opcode::SLL, 0x90909080, 0x21212121, 0xffffffe7);
        simple_op_code_test(Opcode::SLL, 0x48484000, 0x21212121, 0xffffffee);
        simple_op_code_test(Opcode::SLL, 0x00000000, 0x21212120, 0xffffffff);

        simple_op_code_test(Opcode::SRL, 0xffff8000, 0xffff8000, 0);
        simple_op_code_test(Opcode::SRL, 0x7fffc000, 0xffff8000, 1);
        simple_op_code_test(Opcode::SRL, 0x01ffff00, 0xffff8000, 7);
        simple_op_code_test(Opcode::SRL, 0x0003fffe, 0xffff8000, 14);
        simple_op_code_test(Opcode::SRL, 0x0001ffff, 0xffff8001, 15);
        simple_op_code_test(Opcode::SRL, 0xffffffff, 0xffffffff, 0);
        simple_op_code_test(Opcode::SRL, 0x7fffffff, 0xffffffff, 1);
        simple_op_code_test(Opcode::SRL, 0x01ffffff, 0xffffffff, 7);
        simple_op_code_test(Opcode::SRL, 0x0003ffff, 0xffffffff, 14);
        simple_op_code_test(Opcode::SRL, 0x00000001, 0xffffffff, 31);
        simple_op_code_test(Opcode::SRL, 0x21212121, 0x21212121, 0);
        simple_op_code_test(Opcode::SRL, 0x10909090, 0x21212121, 1);
        simple_op_code_test(Opcode::SRL, 0x00424242, 0x21212121, 7);
        simple_op_code_test(Opcode::SRL, 0x00008484, 0x21212121, 14);
        simple_op_code_test(Opcode::SRL, 0x00000000, 0x21212121, 31);
        simple_op_code_test(Opcode::SRL, 0x21212121, 0x21212121, 0xffffffe0);
        simple_op_code_test(Opcode::SRL, 0x10909090, 0x21212121, 0xffffffe1);
        simple_op_code_test(Opcode::SRL, 0x00424242, 0x21212121, 0xffffffe7);
        simple_op_code_test(Opcode::SRL, 0x00008484, 0x21212121, 0xffffffee);
        simple_op_code_test(Opcode::SRL, 0x00000000, 0x21212121, 0xffffffff);

        simple_op_code_test(Opcode::SRA, 0x00000000, 0x00000000, 0);
        simple_op_code_test(Opcode::SRA, 0xc0000000, 0x80000000, 1);
        simple_op_code_test(Opcode::SRA, 0xff000000, 0x80000000, 7);
        simple_op_code_test(Opcode::SRA, 0xfffe0000, 0x80000000, 14);
        simple_op_code_test(Opcode::SRA, 0xffffffff, 0x80000001, 31);
        simple_op_code_test(Opcode::SRA, 0x7fffffff, 0x7fffffff, 0);
        simple_op_code_test(Opcode::SRA, 0x3fffffff, 0x7fffffff, 1);
        simple_op_code_test(Opcode::SRA, 0x00ffffff, 0x7fffffff, 7);
        simple_op_code_test(Opcode::SRA, 0x0001ffff, 0x7fffffff, 14);
        simple_op_code_test(Opcode::SRA, 0x00000000, 0x7fffffff, 31);
        simple_op_code_test(Opcode::SRA, 0x81818181, 0x81818181, 0);
        simple_op_code_test(Opcode::SRA, 0xc0c0c0c0, 0x81818181, 1);
        simple_op_code_test(Opcode::SRA, 0xff030303, 0x81818181, 7);
        simple_op_code_test(Opcode::SRA, 0xfffe0606, 0x81818181, 14);
        simple_op_code_test(Opcode::SRA, 0xffffffff, 0x81818181, 31);
    }

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn test_simple_memory_program_run() {
        let program = Arc::new(simple_memory_program());
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run_fast().unwrap();

        // Assert SW & LW case
        assert_eq!(runtime.register::<Simple>(Register::X28), 0x12348765);

        // Assert LBU cases
        assert_eq!(runtime.register::<Simple>(Register::X27), 0x65);
        assert_eq!(runtime.register::<Simple>(Register::X26), 0x87);
        assert_eq!(runtime.register::<Simple>(Register::X25), 0x34);
        assert_eq!(runtime.register::<Simple>(Register::X24), 0x12);

        // Assert LB cases
        assert_eq!(runtime.register::<Simple>(Register::X23), 0x65);
        assert_eq!(runtime.register::<Simple>(Register::X22), 0xffffff87);

        // Assert LHU cases
        assert_eq!(runtime.register::<Simple>(Register::X21), 0x8765);
        assert_eq!(runtime.register::<Simple>(Register::X20), 0x1234);

        // Assert LH cases
        assert_eq!(runtime.register::<Simple>(Register::X19), 0xffff8765);
        assert_eq!(runtime.register::<Simple>(Register::X18), 0x1234);

        // Assert SB cases
        assert_eq!(runtime.register::<Simple>(Register::X16), 0x12348725);
        assert_eq!(runtime.register::<Simple>(Register::X15), 0x12342525);
        assert_eq!(runtime.register::<Simple>(Register::X14), 0x12252525);
        assert_eq!(runtime.register::<Simple>(Register::X13), 0x25252525);

        // Assert SH cases
        assert_eq!(runtime.register::<Simple>(Register::X12), 0x12346525);
        assert_eq!(runtime.register::<Simple>(Register::X11), 0x65256525);
    }

    #[test]
    #[should_panic]
    fn test_invalid_address_access_sw() {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 20, false, true),
            Instruction::new(Opcode::SW, 0, 29, 0, false, true),
        ];

        let program = Arc::new(Program::new(instructions, 0, 0));
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run::<Simple>().unwrap();
    }

    #[test]
    #[should_panic]
    fn test_invalid_address_access_lw() {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 29, 0, 20, false, true),
            Instruction::new(Opcode::LW, 29, 29, 0, false, true),
        ];

        let program = Arc::new(Program::new(instructions, 0, 0));
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run::<Simple>().unwrap();
    }

    #[test]
    #[should_panic]
    fn test_invalid_address_syscall() {
        let instructions = vec![
            Instruction::new(Opcode::ADD, 5, 0, SHA_COMPRESS as u64, false, true),
            Instruction::new(Opcode::ADD, 10, 0, 10, false, true),
            Instruction::new(Opcode::ADD, 11, 10, 20, false, true),
            Instruction::new(Opcode::ECALL, 5, 10, 11, false, false),
        ];

        let program = Arc::new(Program::new(instructions, 0, 0));
        let mut runtime = Executor::new(program, SP1CoreOpts::default());
        runtime.run::<Simple>().unwrap();
    }
}
