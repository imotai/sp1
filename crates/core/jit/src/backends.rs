use std::{
    collections::{HashMap, VecDeque},
    io,
    ops::BitAnd,
    os::fd::{AsRawFd, RawFd},
    ptr::NonNull,
};

use crate::{RiscRegister, TraceChunkRaw};

use super::ExternFn;
use dynasmrt::ExecutableBuffer;
use memmap2::{Mmap, MmapMut, MmapOptions};
use std::sync::Arc;

#[cfg(target_arch = "x86_64")]
pub mod x86;

#[cfg(target_arch = "x86_64")]
pub use x86::*;

mod debug;
pub use debug::*;

/// An ALU instruction backend for a specific target architecture.
///
/// This trait is implemented for each target architecture supported by the JIT transpiler.
pub trait ALUBackend: Sized {
    type ScratchRegister: ScratchRegister;

    /// Add the values of two registers togethor, using 32bit arithmetic.
    ///
    /// add: lhs = lhs + rhs    
    fn add(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister);

    /// Subtract the values of two registers from eachother, using 32bit arithmetic.
    ///
    /// sub: lhs = lhs - rhs
    fn sub(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister);

    /// Bitwise XOR the values of two registers together.
    ///
    /// xor: lhs = lhs ^ rhs
    fn xor(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister);

    /// Bitwise OR the values of two registers together.
    ///
    /// or: lhs = lhs | rhs
    fn or(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister);

    /// Bitwise AND the values of two registers together.
    ///
    /// and: lhs = lhs & rhs
    fn and(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister);

    /// Shift the values of two registers left by the amount specified by the second register.
    ///
    /// sll: lhs = lhs << rhs
    fn sll(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister);

    /// Shift the values of two registers right by the amount specified by the second register.
    ///
    /// srl: lhs = lhs >> rhs
    fn srl(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister);

    /// Shift the values of two registers right by the amount specified by the second register, using arithmetic right shift.
    ///
    /// sra: lhs = lhs >> rhs
    fn sra(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister);

    /// Compare the values of two registers, setting the zero flag if they are equal.
    ///
    /// slt: zero = (lhs < rhs)
    fn slt(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister);

    /// Compare the values of two registers, setting the zero flag if they are equal.
    ///
    /// sltu: zero = (lhs < rhs)
    fn sltu(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister);

    /// Multiply the values of two registers together, using 32bit arithmetic.
    ///
    /// mul: lhs = lhs * rhs
    fn mul(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister);

    /// Multiply the values of two registers together, using 32bit arithmetic, and return the high half.
    ///
    /// mulh: lhs = (lhs * rhs) >> 32 (signed)
    fn mulh(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister);

    /// Multiply the values of two registers together, using 32bit arithmetic, and return the high half.
    ///
    /// mulhu: lhs = (lhs * rhs) >> 32 (unsigned)
    fn mulhu(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister);

    /// Multiply the values of two registers together, using 32bit arithmetic, and return the high half.
    ///
    /// mulhsu: lhs = (lhs * rhs) >> 16 (signed & unsigned)
    fn mulhsu(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister);

    /// Divide the values of two registers together, using 32bit arithmetic.
    ///
    /// div: rhs == 0 ? 0 : lhs / rhs
    fn div(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister);

    /// Divide the values of two registers, unsigned.
    ///
    /// divu: rhs == 0 ? 0 : lhs / rhs
    fn divu(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister);

    /// Remainder the values of two registers together, using 32bit arithmetic.
    ///
    /// rem: rhs == 0 ? 0 : lhs % rhs
    fn rem(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister);

    /// Remainder the values of two registers, unsigned.
    ///
    /// remu: rhs == 0 ? 0 : lhs % rhs
    fn remu(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister);

    fn call_extern_fn(&mut self, handler: ExternFn);

    /// Returns the function pointer to the generated code.
    ///
    /// This function is expected to be of the form: `fn(*mut JitContext)`.
    fn finalize(self) -> io::Result<JitFunction>;
}

pub trait ScratchRegister: Copy + Sized + BitAnd<u8, Output = u8> {
    const A: Self;
    const B: Self;
}

/// A type representing a JIT compiled function.
///
/// The underlying function should be of the form [`fn(*mut JitContext)`].
pub struct JitFunction {
    jump_table: Vec<*const u8>,
    trace_buf_size: usize,
    code: ExecutableBuffer,

    /// The initial memory image.
    initial_memory_image: Arc<HashMap<u32, u32>>,
    pc_start: u32,
    input_buffer: VecDeque<Vec<u8>>,

    /// The unconstrained context, this is used to create the COW memory at runtime.
    /// We preserve it here in case the execution ends due to cycle count.
    maybe_unconstrained: Option<UnconstrainedCtx>,

    /// Keep around the memfd, and pass it to the JIT context,
    /// we can use this to create the COW memory at runtime.
    mem_fd: memfd::Memfd,

    /// The JIT funciton may stop "in the middle" of an program,
    /// we want to be able to resune it, so this is the information needed to do so.
    pub memory: MmapMut,
    pub pc: u32,
    pub registers: [u32; 32],
    pub clk: u64,
}

impl JitFunction {
    pub(crate) fn new(
        code: ExecutableBuffer,
        jump_table: Vec<usize>,
        memory_size: usize,
        trace_buf_size: usize,
        pc_start: u32,
    ) -> std::io::Result<Self> {
        // Adjust the jump table to be absolute addresses.
        let buf_ptr = code.as_ptr();
        let jump_table =
            jump_table.into_iter().map(|offset| unsafe { buf_ptr.add(offset) }).collect();

        let fd = memfd::MemfdOptions::default()
            .create(uuid::Uuid::new_v4().to_string())
            .expect("Failed to create jit memory");

        fd.as_file().set_len((memory_size + std::mem::align_of::<u32>()) as u64)?;

        Ok(Self {
            jump_table,
            code,
            memory: unsafe { MmapMut::map_mut(fd.as_file())? },
            mem_fd: fd,
            trace_buf_size,
            pc: pc_start,
            clk: 1,
            registers: [0; 32],
            initial_memory_image: Arc::new(HashMap::new()),
            pc_start,
            input_buffer: VecDeque::new(),
            maybe_unconstrained: None,
        })
    }

    /// Write the initial memory image to the JIT memory.
    pub fn with_initial_memory_image(&mut self, memory: Arc<HashMap<u32, u32>>) {
        for (addr, val) in memory.iter() {
            let bytes = val.to_le_bytes();

            self.memory[*addr as usize] = bytes[0];
            self.memory[*addr as usize + 1] = bytes[1];
            self.memory[*addr as usize + 2] = bytes[2];
            self.memory[*addr as usize + 3] = bytes[3];
        }

        self.initial_memory_image = memory;
    }

    pub fn push_input(&mut self, input: Vec<u8>) {
        self.input_buffer.push_back(input);
    }

    pub fn set_input_buffer(&mut self, input: VecDeque<Vec<u8>>) {
        self.input_buffer = input;
        tracing::debug!(
            "got inputs of size: {:?}",
            self.input_buffer.iter().map(|i| i.len()).collect::<Vec<_>>()
        );
    }

    /// Call the function, returning the trace buffer, starting at the starting PC of the program.
    ///
    /// If the PC is 0, then the program has completed and we return None.
    ///
    /// # SAFETY
    /// Relies on the builder to emit valid assembly
    /// and that the pointer is valid for the duration of the function call.
    pub unsafe fn call(&mut self) -> Option<Mmap> {
        if self.pc == 0 {
            return None;
        }

        let as_fn = std::mem::transmute::<*const u8, fn(*mut JitContext)>(self.code.as_ptr());

        let mut trace_buf =
            MmapMut::map_anon(self.trace_buf_size).expect("Failed to create trace buf mmap");

        let align_4_offset = self.memory.as_ptr().align_offset(std::mem::align_of::<u32>());
        let mem_ptr = self.memory.as_mut_ptr().add(align_4_offset);

        // SAFETY:
        // - The jump table is valid for the duration of the function call, its owned by self.
        // - The memory is valid for the duration of the function call, its owned by self.
        // - The trace buf is valid for the duration of the function call, we just allocated it
        // - The input buffer is valid for the duration of the function call, its owned by self.
        let mut ctx = JitContext {
            jump_table: NonNull::new_unchecked(self.jump_table.as_mut_ptr()),
            memory: NonNull::new_unchecked(mem_ptr),
            trace_buf: NonNull::new_unchecked(trace_buf.as_mut_ptr()),
            input_buffer: NonNull::new_unchecked(&mut self.input_buffer),
            maybe_unconstrained: std::mem::take(&mut self.maybe_unconstrained),
            memory_fd: self.mem_fd.as_raw_fd(),
            registers: self.registers,
            pc: self.pc,
            clk: self.clk,
        };

        as_fn(&mut ctx);

        // Update the values we want to preserve.
        self.pc = ctx.pc;
        self.registers = ctx.registers;
        self.clk = ctx.clk;
        self.maybe_unconstrained = std::mem::take(&mut ctx.maybe_unconstrained);

        Some(trace_buf.make_read_only().expect("Failed to make trace buf read only"))
    }

    /// Reset the JIT function to the initial state.
    ///
    /// This will clear the registers, the program counter, the clock, and the memory, restoring the initial memory image.
    pub fn reset(&mut self) {
        self.pc = self.pc_start;
        self.registers = [0; 32];
        self.clk = 1;
        self.input_buffer = VecDeque::new();

        // ---------- Memory Reset ------------

        // Store the original size of the memory.
        let memory_size = self.memory.len();

        // Create a new memfd for the backing memory.
        self.mem_fd = memfd::MemfdOptions::default()
            .create(uuid::Uuid::new_v4().to_string())
            .expect("Failed to create jit memory");

        self.mem_fd
            .as_file()
            .set_len(memory_size as u64)
            .expect("Failed to set memfd size for backing memory.");

        self.memory =
            unsafe { MmapMut::map_mut(self.mem_fd.as_file()).expect("Failed to map memory") };

        // Restore the initial memory image.
        for (addr, val) in self.initial_memory_image.iter() {
            let bytes = val.to_le_bytes();

            self.memory[*addr as usize] = bytes[0];
            self.memory[*addr as usize + 1] = bytes[1];
            self.memory[*addr as usize + 2] = bytes[2];
            self.memory[*addr as usize + 3] = bytes[3];
        }
    }
}

#[derive(Debug)]
pub struct JitContext {
    /// Mapping from (pc - pc_base) / 4 => absolute address of the instruction.
    jump_table: NonNull<*const u8>,
    /// The pointer to the program memory.
    memory: NonNull<u8>,
    /// The pointer to the trace buffer.
    trace_buf: NonNull<u8>,
    /// The registers to start the execution with,
    /// these are loaded into real native registers at the start of execution.
    registers: [u32; 32],
    /// The input buffer to the program.
    input_buffer: NonNull<VecDeque<Vec<u8>>>,
    /// The memory file descriptor, this is used to create the COW memory at runtime.
    memory_fd: RawFd,
    /// The unconstrained context, this is used to create the COW memory at runtime.
    maybe_unconstrained: Option<UnconstrainedCtx>,
    /// The current program counter
    pub pc: u32,
    /// The number of cycles executed.
    pub clk: u64,
}

impl JitContext {
    /// # Safety
    /// - todo
    pub unsafe fn trace_mem_access(&self, reads: &[u32]) {
        // QUESTIONABLE: I think as long as Self is not Sync youre mostly fine, but its unclear,
        // how to actually call this method safe.

        // Read the current num reads from the trace buf.
        let raw = self.trace_buf.as_ptr();
        let num_reads_offset = std::mem::offset_of!(TraceChunkRaw, num_mem_reads);
        let num_reads_ptr = raw.add(num_reads_offset);
        let num_reads = std::ptr::read_unaligned(num_reads_ptr as *mut u32);

        // Write the new num reads to the trace buf.
        let new_num_reads = num_reads + reads.len() as u32;
        std::ptr::write_unaligned(num_reads_ptr as *mut u32, new_num_reads);

        // Write the new reads to the trace buf.
        let reads_start = std::mem::size_of::<TraceChunkRaw>();
        let reads_ptr = raw.add(reads_start);
        for (i, read) in reads.iter().enumerate() {
            std::ptr::write_unaligned(reads_ptr.add(i * 4) as *mut u32, *read);
        }
    }

    /// Obtain a mutable view of the emulated memory.
    pub const fn memory(&mut self) -> ContextMemory {
        ContextMemory::new(self)
    }

    /// Enter the unconstrained context, this will create a COW memory map of the memory file descriptor.
    pub fn enter_unconstrained(&mut self) -> io::Result<()> {
        // SAFETY: The memory is allocated by the [JitFunction] and is valid, not alisiaed, and has enough
        // space for the alignment.
        let mut cow_memory = unsafe { MmapOptions::default().map_copy(self.memory_fd)? };
        let cow_memory_ptr = cow_memory.as_mut_ptr();

        // Align the ptr to u32.
        // SAFETY: u8 has the minimum alignment, so any larger alignment will be a multiple of this.
        let align_offset = cow_memory_ptr.align_offset(std::mem::align_of::<u32>());
        let cow_memory_ptr = unsafe { cow_memory_ptr.add(align_offset) };

        // To match the semantics of the interpreter, we need to subtract 8 from the clock.
        // In the minimal executor, we bump the clock at the start of each instruction.
        self.clk -= 8;

        // Preserve the current state of the JIT context.
        self.maybe_unconstrained = Some(UnconstrainedCtx {
            cow_memory,
            actual_memory_ptr: self.memory,
            pc: self.pc,
            clk: self.clk,
            registers: self.registers,
        });

        // Bump the PC to the next instruction.
        self.pc = self.pc.wrapping_add(4);

        // Set the memory pointer used by the JIT as the COW memory.
        //
        // SAFETY: [memmap2] does not return a null pointer.
        self.memory = unsafe { NonNull::new_unchecked(cow_memory_ptr) };

        Ok(())
    }

    /// Exit the unconstrained context, this will restore the original memory map.
    pub fn exit_unconstrained(&mut self) {
        let unconstrained = std::mem::take(&mut self.maybe_unconstrained)
            .expect("Exit unconstrained called but not context is present, this is a bug.");

        self.memory = unconstrained.actual_memory_ptr;
        self.pc = unconstrained.pc;
        self.registers = unconstrained.registers;

        // To match the semantics of the interpreter, we need to add 8 to the clock.
        self.clk = unconstrained.clk + 8;

        // On drop of [UnconstrainedCtx], the COW memory will be unmapped.
    }

    /// # Safety
    /// - The input buffer must be non null and valid to read from.
    pub const unsafe fn input_buffer(&mut self) -> &mut VecDeque<Vec<u8>> {
        self.input_buffer.as_mut()
    }

    /// Obtain a view of the registers.
    pub const fn registers(&self) -> &[u32; 32] {
        &self.registers
    }

    pub const fn rw(&mut self, reg: RiscRegister, val: u32) {
        self.registers[reg as usize] = val;
    }

    pub const fn rr(&self, reg: RiscRegister) -> u32 {
        self.registers[reg as usize]
    }
}

/// The saved context of the JIT runtime, when entering the unconstrained context.
#[derive(Debug)]
pub struct UnconstrainedCtx {
    // An COW version of the memory.
    pub cow_memory: MmapMut,
    // The pointer to the actual memory.
    pub actual_memory_ptr: NonNull<u8>,
    // The program counter.
    pub pc: u32,
    // The clock.
    pub clk: u64,
    // The registers.
    pub registers: [u32; 32],
}

/// A type representing the memory of the emulated program.
///
/// This is used to read and write to the memory in precompile impls.
pub struct ContextMemory<'a> {
    ctx: &'a mut JitContext,
}

impl<'a> ContextMemory<'a> {
    /// Create a new memory view.
    ///
    /// This type takes in a mutable refrence with a lifetime to avoid aliasing the underlying memory region.
    const fn new(ctx: &'a mut JitContext) -> Self {
        Self { ctx }
    }

    /// Read a u32 from the memory.
    pub fn mr(&self, addr: u32) -> u32 {
        let ptr = unsafe { self.ctx.memory.add(addr as usize) };

        // SAFETY: The pointer is valid to read from, as it was aligned by us during allocation.
        // See [JitFunction::new] for more details.
        let value = unsafe { std::ptr::read(ptr.as_ptr() as *const u32) };

        unsafe { self.ctx.trace_mem_access(&[value]) };

        value
    }

    /// Write a u32 to the memory.
    pub const fn mw(&mut self, addr: u32, val: u32) {
        let ptr = unsafe { self.ctx.memory.add(addr as usize) };

        // SAFETY: The pointer is valid to write to, as it was aligned by us during allocation.
        // See [JitFunction::new] for more details.
        unsafe { std::ptr::write(ptr.as_ptr() as *mut u32, val) };
    }

    /// Read a slice of u32 from the memory.
    pub fn mr_slice(&self, addr: u32, len: usize) -> &[u32] {
        let ptr = unsafe { self.ctx.memory.add(addr as usize) };
        let ptr = ptr.as_ptr() as *const u32;

        // SAFETY: The pointer is valid to write to, as it was aligned by us during allocation.
        // See [JitFunction::new] for more details.
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };

        unsafe { self.ctx.trace_mem_access(slice) };

        slice
    }

    /// Write a slice of u32 to the memory.
    pub fn mw_slice(&mut self, addr: u32, vals: &[u32]) {
        let ptr = unsafe { self.ctx.memory.add(addr as usize) };
        let ptr = ptr.as_ptr() as *mut u32;

        // SAFETY: The pointer is valid to write to, as it was aligned by us during allocation.
        // See [JitFunction::new] for more details
        //
        // Non overlapping is safe here since we have a mut self, so any refrences to this same memory allocation,
        // assuming it was accquried through this type, would cause a borrow checker error.
        unsafe { std::ptr::copy_nonoverlapping(vals.as_ptr(), ptr, vals.len()) };
    }

    /// Read a byte from the memory.
    pub fn byte(&self, addr: u32) -> u8 {
        let ptr = unsafe { self.ctx.memory.add(addr as usize) };
        let ptr = ptr.as_ptr() as *const u8;

        // SAFETY: The pointer is valid to write to, as it was aligned by us during allocation.
        // See [JitFunction::new] for more details
        //
        // All alignments are valid for u8, so we can read from it directly.
        unsafe { std::ptr::read(ptr) }
    }
}
