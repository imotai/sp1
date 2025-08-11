use crate::{
    ALUBackend, DebugFn, Debuggable, EcallHandler, ExternFn, JitContext, JitFunction, RiscOperand,
    RiscRegister, SP1RiscvTranspiler, TraceCollector,
};
use std::io;

use std::sync::{mpsc, OnceLock};

#[allow(clippy::type_complexity)]
static DEBUG_REGISTERS: OnceLock<mpsc::Sender<Option<(u32, u64, [u32; 32])>>> = OnceLock::new();

pub fn init_debug_registers() -> mpsc::Receiver<Option<(u32, u64, [u32; 32])>> {
    let (tx, rx) = mpsc::channel();
    DEBUG_REGISTERS.set(tx).expect("DEBUG_REGISTERS already initialized");

    rx
}

pub struct DebugBackend<B: SP1RiscvTranspiler> {
    backend: B,
}

impl<B: SP1RiscvTranspiler> DebugBackend<B> {
    pub const fn new(backend: B) -> Self {
        Self { backend }
    }
}

impl<B: SP1RiscvTranspiler + Debuggable> SP1RiscvTranspiler for DebugBackend<B> {
    fn new(
        program_size: usize,
        memory_size: usize,
        trace_buf_size: usize,
        pc_start: u32,
        pc_base: u32,
    ) -> Result<Self, std::io::Error> {
        let backend = B::new(program_size, memory_size, trace_buf_size, pc_start, pc_base)?;

        Ok(Self::new(backend))
    }

    fn register_ecall_handler(&mut self, handler: EcallHandler) {
        self.backend.register_ecall_handler(handler);
    }

    fn start_instr(&mut self) {
        extern "C" fn print_bar(_: *mut JitContext) {
            eprintln!("--------------------------------");
        }

        extern "C" fn collect_registers(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };

            if let Some(sender) = DEBUG_REGISTERS.get() {
                sender.send(Some((ctx.pc, ctx.clk, *ctx.registers()))).unwrap();
            }
        }

        self.backend.start_instr();

        self.backend.call_extern_fn(collect_registers);
        self.backend.call_extern_fn(print_bar);
        self.backend.print_ctx();
    }

    fn load_riscv_operand(&mut self, src: RiscOperand, dst: Self::ScratchRegister) {
        self.backend.load_riscv_operand(src, dst);
    }

    fn store_riscv_register(&mut self, src: Self::ScratchRegister, dst: RiscRegister) {
        self.backend.store_riscv_register(src, dst);
    }

    fn bump_clk(&mut self, amt: u32) {
        self.backend.bump_clk(amt);
    }

    fn set_pc(&mut self, target: u32) {
        self.backend.set_pc(target);
    }

    fn exit_if_clk_exceeds(&mut self, max_cycles: u64) {
        self.backend.exit_if_clk_exceeds(max_cycles);
    }

    fn inspect_register(&mut self, reg: RiscRegister, handler: DebugFn) {
        self.backend.inspect_register(reg, handler);
    }

    fn inspect_immediate(&mut self, imm: u32, handler: DebugFn) {
        self.backend.inspect_immediate(imm, handler);
    }

    fn lb(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u32) {
        extern "C" fn lb(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = lb: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(lb);
        self.backend.lb(rd, rs1, imm);
    }

    fn lh(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u32) {
        extern "C" fn lh(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = lh: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(lh);
        self.backend.lh(rd, rs1, imm);
    }

    fn lw(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u32) {
        extern "C" fn lw(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = lw: pc={}", ctx.pc);
        }

        extern "C" fn lw_imm_value(imm: u32) {
            eprintln!("-- lw_imm_value: value={}", imm);
        }

        extern "C" fn lw_rs1_value(rs1: u32) {
            eprintln!("-- lw_rs1_value={}", rs1);
        }

        extern "C" fn lw_rs1(rs1: u32) {
            eprintln!("-- lw_rs1={}", rs1);
        }

        extern "C" fn lw_rd(rd: u32) {
            eprintln!("-- lw_rd={}", rd);
        }

        self.inspect_immediate(imm, lw_imm_value);
        self.inspect_immediate(rd as u8 as u32, lw_rd);
        self.inspect_immediate(rs1 as u8 as u32, lw_rs1);
        self.inspect_register(rs1, lw_rs1_value);

        self.backend.call_extern_fn(lw);
        self.backend.lw(rd, rs1, imm);
    }

    fn lbu(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u32) {
        extern "C" fn lbu(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = lbu: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(lbu);
        self.backend.lbu(rd, rs1, imm);
    }

    fn lhu(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u32) {
        extern "C" fn lhu(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = lhu: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(lhu);
        self.backend.lhu(rd, rs1, imm);
    }

    fn sb(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32) {
        extern "C" fn sb(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = sb: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(sb);
        self.backend.sb(rs1, rs2, imm);
    }

    fn sh(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32) {
        extern "C" fn sh(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = sh: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(sh);
        self.backend.sh(rs1, rs2, imm);
    }

    fn sw(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32) {
        extern "C" fn sw(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = sw: pc={}", ctx.pc);
        }

        extern "C" fn sw_imm_value(imm: u32) {
            eprintln!("-- sw_imm_value: value={}", imm);
        }

        extern "C" fn sw_rs1_value(rs1: u32) {
            eprintln!("-- sw_rs1_value={}", rs1);
        }

        extern "C" fn sw_rs2_value(rs2: u32) {
            eprintln!("-- sw_rs2_value={}", rs2);
        }

        extern "C" fn sw_rs2(rs2: u32) {
            eprintln!("-- sw_rs2={}", rs2);
        }

        extern "C" fn sw_rs1(rs1: u32) {
            eprintln!("-- sw_rs1={}", rs1);
        }

        self.inspect_immediate(rs1 as u8 as u32, sw_rs1);
        self.inspect_immediate(rs2 as u8 as u32, sw_rs2);
        self.inspect_immediate(imm, sw_imm_value);

        self.inspect_register(rs1, sw_rs1_value);
        self.inspect_register(rs2, sw_rs2_value);

        self.backend.call_extern_fn(sw);
        self.backend.sw(rs1, rs2, imm);
    }

    fn beq(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32) {
        extern "C" fn beq(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = beq: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(beq);
        self.backend.beq(rs1, rs2, imm);
    }

    fn bne(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32) {
        extern "C" fn bne(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = bne: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(bne);
        self.backend.bne(rs1, rs2, imm);
    }

    fn blt(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32) {
        extern "C" fn blt(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = blt: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(blt);
        self.backend.blt(rs1, rs2, imm);
    }

    fn bge(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32) {
        extern "C" fn bge(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = bge: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(bge);
        self.backend.bge(rs1, rs2, imm);
    }

    fn bltu(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32) {
        extern "C" fn bltu(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = bltu: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(bltu);
        self.backend.bltu(rs1, rs2, imm);
    }

    fn bgeu(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u32) {
        extern "C" fn bgeu(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = bgeu: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(bgeu);
        self.backend.bgeu(rs1, rs2, imm);
    }

    fn jal(&mut self, rd: RiscRegister, imm: u32) {
        extern "C" fn jal(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = jal: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(jal);
        self.backend.jal(rd, imm);
    }

    fn jalr(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u32) {
        extern "C" fn jalr(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = jalr: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(jalr);

        extern "C" fn jalr_reg_value(register: u32) {
            eprintln!("-- jalr_reg_value: value={}", register);
        }

        extern "C" fn jalr_imm_value(imm: u32) {
            eprintln!("-- jalr_imm_value: value={}", imm);
        }

        extern "C" fn jalr_rs1(rs1: u32) {
            eprintln!("-- jalr_rs1={}", rs1);
        }

        extern "C" fn jalr_rd(rd: u32) {
            eprintln!("-- jalr_rd={}", rd);
        }

        self.inspect_immediate(rd as u8 as u32, jalr_rd);
        self.backend.inspect_register(rs1, jalr_reg_value);
        self.backend.inspect_immediate(rs1 as u8 as u32, jalr_rs1);
        self.backend.inspect_immediate(imm, jalr_imm_value);

        self.backend.jalr(rd, rs1, imm);
    }

    fn auipc(&mut self, rd: RiscRegister, imm: u32) {
        extern "C" fn auipc(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = auipc: pc={}", ctx.pc);
        }

        extern "C" fn auipc_rd(rd: u32) {
            eprintln!("-- auipc_rd={}", rd);
        }

        extern "C" fn auipc_imm(imm: u32) {
            eprintln!("-- auipc_imm={}", imm);
        }

        self.inspect_immediate(rd as u8 as u32, auipc_rd);
        self.inspect_immediate(imm, auipc_imm);

        self.backend.call_extern_fn(auipc);
        self.backend.auipc(rd, imm);
    }

    fn ecall(&mut self) {
        self.backend.ecall();
    }

    fn unimp(&mut self) {
        extern "C" fn unimp(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("Unimplemented instruction at pc: {}", ctx.pc);
        }

        self.backend.call_extern_fn(unimp);
    }

    fn risc_add(&mut self, rs1: RiscOperand, rs2: RiscOperand, rd: RiscRegister) {
        extern "C" fn risc_add(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = risc_add: pc={}", ctx.pc);
        }

        extern "C" fn risc_add_rs1(rs1: u32) {
            eprintln!("-- risc_add_rs1={}", rs1);
        }

        extern "C" fn risc_add_rs1_imm(imm: u32) {
            eprintln!("-- risc_add_rs1_imm={}", imm);
        }

        extern "C" fn risc_add_rs2(rs2: u32) {
            eprintln!("-- risc_add_rs2={}", rs2);
        }

        extern "C" fn risc_add_rs2_imm(imm: u32) {
            eprintln!("-- risc_add_rs2_imm={}", imm);
        }

        extern "C" fn risc_add_rd(rd: u32) {
            eprintln!("-- risc_add_rd={}", rd);
        }

        match rs1 {
            RiscOperand::Register(r) => self.inspect_register(r, risc_add_rs1),
            RiscOperand::Immediate(i) => self.inspect_immediate(i as u32, risc_add_rs1_imm),
        }

        match rs2 {
            RiscOperand::Register(r) => self.inspect_register(r, risc_add_rs2),
            RiscOperand::Immediate(i) => self.inspect_immediate(i as u32, risc_add_rs2_imm),
        }

        self.inspect_immediate(rd as u8 as u32, risc_add_rd);

        self.backend.call_extern_fn(risc_add);
        self.backend.risc_add(rs1, rs2, rd);
    }

    fn risc_sub(&mut self, rs1: RiscOperand, rs2: RiscOperand, rd: RiscRegister) {
        extern "C" fn risc_sub(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = risc_sub: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(risc_sub);
        self.backend.risc_sub(rs1, rs2, rd);
    }

    fn risc_mul(&mut self, rs1: RiscOperand, rs2: RiscOperand, rd: RiscRegister) {
        extern "C" fn risc_mul(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = risc_mul: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(risc_mul);
        self.backend.risc_mul(rs1, rs2, rd);
    }

    fn risc_div(&mut self, rs1: RiscOperand, rs2: RiscOperand, rd: RiscRegister) {
        extern "C" fn risc_div(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = risc_div: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(risc_div);
        self.backend.risc_div(rs1, rs2, rd);
    }

    fn risc_divu(&mut self, rs1: RiscOperand, rs2: RiscOperand, rd: RiscRegister) {
        extern "C" fn risc_divu(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = risc_divu: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(risc_divu);
        self.backend.risc_divu(rs1, rs2, rd);
    }

    fn risc_rem(&mut self, rs1: RiscOperand, rs2: RiscOperand, rd: RiscRegister) {
        extern "C" fn risc_rem(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = risc_rem: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(risc_rem);
        self.backend.risc_rem(rs1, rs2, rd);
    }

    fn risc_remu(&mut self, rs1: RiscOperand, rs2: RiscOperand, rd: RiscRegister) {
        extern "C" fn risc_remu(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = risc_remu: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(risc_remu);
        self.backend.risc_remu(rs1, rs2, rd);
    }

    fn risc_mulh(&mut self, rs1: RiscOperand, rs2: RiscOperand, rd: RiscRegister) {
        extern "C" fn risc_mulh(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = risc_mulh: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(risc_mulh);
        self.backend.risc_mulh(rs1, rs2, rd);
    }

    fn risc_mulhu(&mut self, rs1: RiscOperand, rs2: RiscOperand, rd: RiscRegister) {
        extern "C" fn risc_mulhu(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = risc_mulhu: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(risc_mulhu);
        self.backend.risc_mulhu(rs1, rs2, rd);
    }

    fn risc_sll(&mut self, rs1: RiscOperand, rs2: RiscOperand, rd: RiscRegister) {
        extern "C" fn risc_sll(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = risc_sll: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(risc_sll);
        self.backend.risc_sll(rs1, rs2, rd);
    }

    fn risc_srl(&mut self, rs1: RiscOperand, rs2: RiscOperand, rd: RiscRegister) {
        extern "C" fn risc_srl(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = risc_srl: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(risc_srl);
        self.backend.risc_srl(rs1, rs2, rd);
    }

    fn risc_and(&mut self, rs1: RiscOperand, rs2: RiscOperand, rd: RiscRegister) {
        extern "C" fn risc_and(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = risc_and: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(risc_and);
        self.backend.risc_and(rs1, rs2, rd);
    }

    fn risc_mulhsu(&mut self, rs1: RiscOperand, rs2: RiscOperand, rd: RiscRegister) {
        extern "C" fn risc_mulhsu(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = risc_mulhsu: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(risc_mulhsu);
        self.backend.risc_mulhsu(rs1, rs2, rd);
    }

    fn risc_xor(&mut self, rs1: RiscOperand, rs2: RiscOperand, rd: RiscRegister) {
        extern "C" fn risc_xor(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = risc_xor: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(risc_xor);
        self.backend.risc_xor(rs1, rs2, rd);
    }

    fn risc_or(&mut self, rs1: RiscOperand, rs2: RiscOperand, rd: RiscRegister) {
        extern "C" fn risc_or(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = risc_or: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(risc_or);
        self.backend.risc_or(rs1, rs2, rd);
    }

    fn risc_slt(&mut self, rs1: RiscOperand, rs2: RiscOperand, rd: RiscRegister) {
        extern "C" fn risc_slt(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = risc_slt: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(risc_slt);
        self.backend.risc_slt(rs1, rs2, rd);
    }

    fn risc_sltu(&mut self, rs1: RiscOperand, rs2: RiscOperand, rd: RiscRegister) {
        extern "C" fn risc_sltu(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = risc_sltu: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(risc_sltu);
        self.backend.risc_sltu(rs1, rs2, rd);
    }

    fn risc_sra(&mut self, rs1: RiscOperand, rs2: RiscOperand, rd: RiscRegister) {
        extern "C" fn risc_sra(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("-- opcode = risc_sra: pc={}", ctx.pc);
        }

        self.backend.call_extern_fn(risc_sra);
        self.backend.risc_sra(rs1, rs2, rd);
    }
}

impl<B: SP1RiscvTranspiler> ALUBackend for DebugBackend<B> {
    type ScratchRegister = B::ScratchRegister;

    fn add(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        self.backend.add(lhs, rhs);
    }

    fn sub(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        self.backend.sub(lhs, rhs);
    }

    fn xor(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        self.backend.xor(lhs, rhs);
    }

    fn or(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        self.backend.or(lhs, rhs);
    }

    fn and(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        self.backend.and(lhs, rhs);
    }

    fn sll(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        self.backend.sll(lhs, rhs);
    }

    fn srl(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        self.backend.srl(lhs, rhs);
    }

    fn sra(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        self.backend.sra(lhs, rhs);
    }

    fn slt(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        self.backend.slt(lhs, rhs);
    }

    fn sltu(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        self.backend.sltu(lhs, rhs);
    }

    fn mul(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        self.backend.mul(lhs, rhs);
    }

    fn mulh(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        self.backend.mulh(lhs, rhs);
    }

    fn mulhu(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        self.backend.mulhu(lhs, rhs);
    }

    fn mulhsu(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        self.backend.mulhsu(lhs, rhs);
    }

    fn div(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        self.backend.div(lhs, rhs);
    }

    fn divu(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        self.backend.divu(lhs, rhs);
    }

    fn rem(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        self.backend.rem(lhs, rhs);
    }

    fn remu(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        self.backend.remu(lhs, rhs);
    }

    fn call_extern_fn(&mut self, handler: ExternFn) {
        self.backend.call_extern_fn(handler);
    }

    fn finalize(self) -> io::Result<JitFunction> {
        // todo fix this

        // extern "C" fn finalize_registers(_: *mut JitContext) {
        //     let tx = DEBUG_REGISTERS.lock().unwrap();
        //     eprintln!("finalize_registers_jit");
        //     if let Some(sender) = tx.as_ref() {
        //         sender.send(None).unwrap();
        //     }
        // }

        // self.backend.call_extern_fn(finalize_registers);

        self.backend.finalize()
    }
}

impl<B: SP1RiscvTranspiler> TraceCollector for DebugBackend<B> {
    fn trace_clk_start(&mut self) {
        self.backend.trace_clk_start();
    }

    fn trace_mem_value(&mut self, src: RiscRegister) {
        self.backend.trace_mem_value(src);
    }

    fn trace_pc_start(&mut self) {
        self.backend.trace_pc_start();
    }

    fn trace_registers(&mut self) {
        self.backend.trace_registers();
    }
}
