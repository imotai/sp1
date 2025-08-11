use super::TranspilerBackend;
use crate::{
    ALUBackend, Debuggable, JitContext, RiscOperand, RiscRegister, SP1RiscvTranspiler,
    TraceChunkRaw,
};

macro_rules! assert_register_is {
    ($expected:expr) => {{
        extern "C" fn assert_register_is_expected(val: u32) {
            assert_eq!(val, $expected);
        }

        assert_register_is_expected
    }};
}

fn new_backend() -> TranspilerBackend {
    TranspilerBackend::new(0, 16, std::mem::size_of::<TraceChunkRaw>() * 100, 1, 1).unwrap()
}

/// Finalize the function and call it.
fn run_test(assembler: TranspilerBackend) {
    let mut func = assembler.finalize().expect("Failed to finalize function");

    unsafe {
        func.call();
    }
}

mod alu {
    use super::*;

    #[test]
    fn test_add_immediate_correct() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.risc_add(RiscOperand::Immediate(5), RiscOperand::Immediate(10), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(15));

        run_test(backend);
    }

    #[test]
    fn test_multiple_adds() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.risc_add(RiscOperand::Immediate(5), RiscOperand::Immediate(10), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(15));

        backend.risc_add(RiscOperand::Immediate(10), RiscOperand::Immediate(10), RiscRegister::X6);
        backend.inspect_register(RiscRegister::X6, assert_register_is!(20));

        backend.risc_add(RiscOperand::Immediate(20), RiscOperand::Immediate(5), RiscRegister::X7);
        backend.inspect_register(RiscRegister::X7, assert_register_is!(25));

        run_test(backend);
    }

    #[test]
    fn test_add_handles_overflow_32bit() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.risc_add(
            RiscOperand::Immediate((u32::MAX - 1) as i32),
            RiscOperand::Immediate((u32::MAX - 1) as i32),
            RiscRegister::X5,
        );

        backend.inspect_register(
            RiscRegister::X5,
            assert_register_is!((u32::MAX - 1).wrapping_add(u32::MAX - 1)),
        );

        run_test(backend);
    }

    #[test]
    fn test_mul_immediate_correct() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.risc_mul(RiscOperand::Immediate(5), RiscOperand::Immediate(4), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(20));

        run_test(backend);
    }

    #[test]
    fn test_mul_handles_overflow_32bit() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.risc_mul(
            RiscOperand::Immediate((u32::MAX - 1) as i32),
            RiscOperand::Immediate((u32::MAX - 1) as i32),
            RiscRegister::X5,
        );

        backend.inspect_register(
            RiscRegister::X5,
            assert_register_is!((u32::MAX - 1).wrapping_mul(u32::MAX - 1)),
        );

        run_test(backend);
    }

    #[test]
    fn test_div_immediate_correct() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.risc_div(RiscOperand::Immediate(10), RiscOperand::Immediate(2), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(5));

        backend.risc_div(RiscOperand::Immediate(10), RiscOperand::Immediate(0), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(0));

        backend.risc_div(RiscOperand::Immediate(-10), RiscOperand::Immediate(2), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(-5_i32 as u32));

        run_test(backend);
    }

    #[test]
    fn test_mulh_immediate_correct() {
        let mut backend = new_backend();
        backend.start_instr();

        // 0x7FFF_FFFF * 2  → 0x0000_0000_FFFF_FFFE  → high=0
        backend.risc_mulh(
            RiscOperand::Immediate(0x7FFF_FFFF), // +2 147 483 647
            RiscOperand::Immediate(2),
            RiscRegister::X5,
        );
        backend.inspect_register(
            RiscRegister::X5,
            assert_register_is!(((0x7FFF_FFFF_i64 * 2) >> 32) as u32),
        );

        // -1 * 3 → 0xFFFF_FFFF_FFFF_FFFD  → high=0xFFFF_FFFF
        backend.risc_mulh(
            RiscOperand::Immediate(-1), // 0xFFFF_FFFF
            RiscOperand::Immediate(3),
            RiscRegister::X5,
        );
        backend.inspect_register(RiscRegister::X5, assert_register_is!((-3_i64 >> 32) as u32));

        run_test(backend);
    }

    #[test]
    fn test_mulhu_immediate_correct() {
        let mut backend = new_backend();

        backend.start_instr();

        // 0xFFFF_FFFF * 0xFFFF_FFFF → 0xFFFF_FFFE_0000_0001 → high=0xFFFF_FFFE
        backend.risc_mulhu(
            RiscOperand::Immediate(-1), // 0xFFFF_FFFF as i32
            RiscOperand::Immediate(-1),
            RiscRegister::X5,
        );
        backend.inspect_register(RiscRegister::X5, assert_register_is!(0xFFFF_FFFE));

        // 0x8000_0000 * 2 → 0x1_0000_0000 → high=1
        backend.risc_mulhu(
            RiscOperand::Immediate(0x8000_0000u32 as i32), // 2 147 483 648
            RiscOperand::Immediate(2),
            RiscRegister::X5,
        );
        backend.inspect_register(RiscRegister::X5, assert_register_is!(0x0000_0001));

        run_test(backend);
    }

    #[test]
    fn test_mulhsu_immediate_correct() {
        let mut backend = new_backend();

        backend.start_instr();

        // -1 * 3 → 0xFFFF_FFFF_FFFF_FFFD → high = 0xFFFF_FFFF
        backend.risc_mulhsu(
            RiscOperand::Immediate(-1), // signed -1
            RiscOperand::Immediate(3),  // unsigned 3
            RiscRegister::X5,
        );
        backend.inspect_register(RiscRegister::X5, assert_register_is!(0xFFFF_FFFF));

        // 0x7FFF_FFFF * 0xFFFF_FFFF → 0x7FFF_FFFE_8000_0001 → high = 0x7FFF_FFFE
        backend.risc_mulhsu(
            RiscOperand::Immediate(0x7FFF_FFFF), // +2 147 483 647
            RiscOperand::Immediate(-1),          // 0xFFFF_FFFF (unsigned 4 294 967 295)
            RiscRegister::X5,
        );
        backend.inspect_register(
            RiscRegister::X5,
            assert_register_is!(((0x7FFF_FFFF_i64 * 0xFFFF_FFFF_i64) >> 32) as u32),
        );

        run_test(backend);
    }

    #[test]
    fn test_rem_immediate_correct() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.risc_rem(RiscOperand::Immediate(10), RiscOperand::Immediate(3), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(1));

        backend.risc_rem(RiscOperand::Immediate(10), RiscOperand::Immediate(0), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(0));

        backend.risc_rem(RiscOperand::Immediate(-10), RiscOperand::Immediate(3), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(-1_i32 as u32));

        run_test(backend);
    }

    #[test]
    fn test_remu_immediate_correct() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.risc_remu(RiscOperand::Immediate(10), RiscOperand::Immediate(3), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(1));

        backend.risc_remu(RiscOperand::Immediate(10), RiscOperand::Immediate(0), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(0));

        backend.risc_remu(RiscOperand::Immediate(-10), RiscOperand::Immediate(3), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!((-10_i32 as u32) % 3));

        run_test(backend);
    }

    #[test]
    fn test_sll_immediate_correct() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.risc_sll(RiscOperand::Immediate(10), RiscOperand::Immediate(3), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(10 << 3));

        backend.risc_sll(RiscOperand::Immediate(10), RiscOperand::Immediate(500), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(10_u32.wrapping_shl(500)));

        run_test(backend);
    }

    #[test]
    fn test_srl_immediate_correct() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.risc_srl(RiscOperand::Immediate(10), RiscOperand::Immediate(3), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(10 >> 3));

        backend.risc_srl(RiscOperand::Immediate(10), RiscOperand::Immediate(500), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(0));

        run_test(backend);
    }

    #[test]
    fn test_sra_immediate_correct() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.risc_sra(RiscOperand::Immediate(10), RiscOperand::Immediate(3), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(10 >> 3));

        backend.risc_sra(RiscOperand::Immediate(-10), RiscOperand::Immediate(3), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!((-10 >> 3) as u32));

        run_test(backend);
    }

    #[test]
    fn test_slt_immediate_correct() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.risc_slt(RiscOperand::Immediate(10), RiscOperand::Immediate(3), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(0));

        backend.risc_slt(RiscOperand::Immediate(-10), RiscOperand::Immediate(3), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(1));

        run_test(backend);
    }

    #[test]
    fn test_sltu_immediate_correct() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.risc_sltu(RiscOperand::Immediate(10), RiscOperand::Immediate(3), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(0));

        backend.risc_sltu(RiscOperand::Immediate(10), RiscOperand::Immediate(10), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(0));

        backend.risc_sltu(RiscOperand::Immediate(-10), RiscOperand::Immediate(3), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(0));

        backend.risc_sltu(RiscOperand::Immediate(3), RiscOperand::Immediate(10), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(1));

        run_test(backend);
    }

    #[test]
    fn test_sub_immediate_correct() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.risc_sub(RiscOperand::Immediate(10), RiscOperand::Immediate(3), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(10 - 3));

        backend.risc_sub(RiscOperand::Immediate(10), RiscOperand::Immediate(10), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!(0));

        backend.risc_sub(RiscOperand::Immediate(-10), RiscOperand::Immediate(3), RiscRegister::X5);
        backend.inspect_register(RiscRegister::X5, assert_register_is!((-10 - 3) as u32));

        run_test(backend);
    }
}

mod control_flow {
    use super::*;

    #[test]
    fn test_set_pc() {
        let mut backend = new_backend();

        extern "C" fn assert_pc_is_100(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };

            assert_eq!(ctx.pc, 100);
        }

        extern "C" fn assert_pc_is_500(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };

            assert_eq!(ctx.pc, 500);
        }

        backend.start_instr();
        backend.set_pc(100);
        backend.call_extern_fn(assert_pc_is_100);
        backend.set_pc(500);
        backend.call_extern_fn(assert_pc_is_500);

        run_test(backend);
    }

    #[test]
    fn test_bump_clk() {
        let mut backend = new_backend();

        // Note: The clk starts at 1.

        extern "C" fn assert_clk_is_100(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };

            assert_eq!(ctx.clk, 101);
        }

        extern "C" fn assert_clk_is_500(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };

            assert_eq!(ctx.clk, 501);
        }

        backend.start_instr();
        backend.bump_clk(100);
        backend.call_extern_fn(assert_clk_is_100);
        backend.bump_clk(400);
        backend.call_extern_fn(assert_clk_is_500);

        run_test(backend);
    }

    #[test]
    fn test_jump_skips_instruction() {
        let mut backend = new_backend();

        // PC = 0
        backend.start_instr();
        backend.risc_add(RiscOperand::Immediate(1), RiscOperand::Immediate(1), RiscRegister::X1); // 1 + 1 = 2

        // // PC = 1
        backend.start_instr();
        backend.risc_add(RiscOperand::Immediate(2), RiscOperand::Immediate(2), RiscRegister::X2); // 2 + 2 = 4

        backend.print_ctx();

        // PC = 2
        backend.start_instr();
        backend.jal(RiscRegister::X0, 4 * 4); // jump to PC = 4

        backend.print_ctx();

        // // PC = 3 (also skipped due to jump of 3)
        backend.start_instr();
        backend.risc_add(RiscOperand::Immediate(100), RiscOperand::Immediate(23), RiscRegister::X3);

        backend.print_ctx();

        // // // PC = 4
        backend.start_instr();
        backend.risc_add(RiscOperand::Immediate(42), RiscOperand::Immediate(1), RiscRegister::X4); // 42 + 1 = 43

        backend.print_ctx();

        backend.inspect_register(RiscRegister::X1, assert_register_is!(2));
        backend.inspect_register(RiscRegister::X2, assert_register_is!(4));
        backend.inspect_register(RiscRegister::X3, assert_register_is!(0)); // skipped
        backend.inspect_register(RiscRegister::X4, assert_register_is!(43));

        run_test(backend);
    }

    #[test]
    fn test_branch_neq() {
        let mut backend = new_backend();

        // PC = 0
        backend.start_instr();
        backend.risc_add(RiscOperand::Immediate(5), RiscOperand::Immediate(0), RiscRegister::X1);
        backend.set_pc(4);

        // PC = 1
        backend.start_instr();
        backend.risc_add(RiscRegister::X2.into(), RiscOperand::Immediate(1), RiscRegister::X2);
        backend.set_pc(8);

        // PC = 2
        // Branch to PC = 1 if X1 != X2
        backend.start_instr();
        backend.bne(RiscRegister::X1, RiscRegister::X2, u32::MAX);
        backend.inspect_register(RiscRegister::X2, assert_register_is!(5));

        run_test(backend);
    }
}

mod memory {
    use super::*;

    fn run_test_with_memory(assembler: TranspilerBackend, memory: &[(u32, u32)]) {
        let mut func = assembler.finalize().expect("Failed to finalize function");

        for (addr, val) in memory {
            assert!(*addr < func.memory.len() as u32, "Addr out of bounds");
            assert!(*addr % 4 == 0, "Addr must be 4 byte aligned");

            let bytes = val.to_le_bytes();
            func.memory[*addr as usize] = bytes[0];
            func.memory[*addr as usize + 1] = bytes[1];
            func.memory[*addr as usize + 2] = bytes[2];
            func.memory[*addr as usize + 3] = bytes[3];
        }

        unsafe {
            func.call();
        }
    }

    fn run_test_and_check_memory(assembler: TranspilerBackend, check: impl Fn(&[u8])) {
        let mut func = assembler.finalize().expect("Failed to finalize function");

        unsafe {
            func.call();
        }

        check(&func.memory);
    }

    #[test]
    fn test_load_word_correct() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.lw(RiscRegister::X1, RiscRegister::X1, 0);
        backend.inspect_register(RiscRegister::X1, assert_register_is!(5));

        run_test_with_memory(backend, &[(0, 5)]);
    }

    #[test]
    fn test_load_byte_signed_correct() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.lb(RiscRegister::X1, RiscRegister::X1, 0); // LB x1, 0(x1)
        backend.inspect_register(RiscRegister::X1, assert_register_is!(0xFFFFFF80)); // −128 sign-extended

        // memory[0] = 0x80  (remaining three bytes are 0)
        run_test_with_memory(backend, &[(0, 0x0000_0080)]);
    }

    #[test]
    fn test_load_byte_unsigned_correct() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.lbu(RiscRegister::X1, RiscRegister::X1, 0); // LBU x1, 0(x1)
        backend.inspect_register(RiscRegister::X1, assert_register_is!(0x00000080)); // 128 zero-extended

        run_test_with_memory(backend, &[(0, 0x0000_0080)]);
    }

    #[test]
    fn test_load_half_signed_correct() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.lh(RiscRegister::X1, RiscRegister::X1, 0); // LH x1, 0(x1)
        backend.inspect_register(RiscRegister::X1, assert_register_is!(0xFFFF8000)); // −32768 sign-extended

        // memory[0..2] = 0x00,0x80  (i.e., little-endian 0x8000)
        run_test_with_memory(backend, &[(0, 0x0000_8000)]);
    }

    #[test]
    fn test_load_half_unsigned_correct() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.lhu(RiscRegister::X1, RiscRegister::X1, 0); // LHU x1, 0(x1)
        backend.inspect_register(RiscRegister::X1, assert_register_is!(0x00008000)); // 32768 zero-extended

        run_test_with_memory(backend, &[(0, 0x0000_8000)]);
    }

    #[test]
    fn test_store_word_correct() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.risc_add(RiscOperand::Immediate(5), RiscOperand::Immediate(0), RiscRegister::X1);

        // Store 5 into memory[0]
        backend.start_instr();
        backend.sw(RiscRegister::X0, RiscRegister::X1, 0); // SW: m(rs1 + imm) = rs2

        run_test_and_check_memory(backend, |memory| {
            assert_eq!(memory[0], 5);
        });
    }

    #[test]
    fn test_store_halfword_correct() {
        let mut backend = new_backend();

        // Put 0x01F2 (little-endian bytes F2 01) into x1
        backend.start_instr();
        backend.risc_add(
            RiscOperand::Immediate(0x01F2),
            RiscOperand::Immediate(0),
            RiscRegister::X1,
        );

        // SH: store 16-bit value at address 0
        backend.start_instr();
        backend.sh(RiscRegister::X0, RiscRegister::X1, 0);

        run_test_and_check_memory(backend, |memory| {
            assert_eq!(memory[0], 0xF2); // low byte
            assert_eq!(memory[1], 0x01); // high byte

            // sanity check: ensure we didn't clobber the rest
            assert_eq!(memory[2], 0x00);
            assert_eq!(memory[3], 0x00);
        });
    }

    #[test]
    fn test_store_byte_correct() {
        let mut backend = new_backend();

        // Put 0xAB into x1
        backend.start_instr();
        backend.risc_add(RiscOperand::Immediate(0xAB), RiscOperand::Immediate(0), RiscRegister::X1);

        // SB: store 8-bit value at address 0
        backend.start_instr();
        backend.sb(RiscRegister::X0, RiscRegister::X1, 0);

        run_test_and_check_memory(backend, |memory| {
            assert_eq!(memory[0], 0xAB);

            // confirm surrounding bytes remain zero
            assert_eq!(memory[1], 0x00);
        });
    }

    /// 0x1234 lives in the *low* half-word (bytes 0–1)
    #[test]
    fn test_lhu_low_halfword() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.lhu(RiscRegister::X2, RiscRegister::X1, 0); // X1 == 0 by default
        backend.inspect_register(RiscRegister::X2, assert_register_is!(0x00001234));

        // Memory word: 0xABCD_1234  ->  [34 12 CD AB] (little-endian)
        run_test_with_memory(backend, &[(0, 0xABCD_1234)]);
    }

    /// 0xABCD lives in the *high* half-word (bytes 2–3) of that same word
    #[test]
    fn test_lhu_high_halfword() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.lhu(RiscRegister::X3, RiscRegister::X1, 2); // imm = 2 => second half-word
        backend.inspect_register(RiscRegister::X3, assert_register_is!(0x0000ABCD));

        run_test_with_memory(backend, &[(0, 0xABCD_1234)]);
    }

    /// Value with the top bit set (0x8000) to verify *zero* extension
    #[test]
    fn test_lhu_zero_extension() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.lhu(RiscRegister::X1, RiscRegister::X1, 0);
        backend.inspect_register(RiscRegister::X1, assert_register_is!(0x00008000));

        run_test_with_memory(backend, &[(0, 0x0000_8000)]);
    }

    /// Low half-word is 0xF234 → −3564 after sign-extension
    #[test]
    fn test_lh_sign_negative_low() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.lh(RiscRegister::X2, RiscRegister::X1, 0);
        backend.inspect_register(RiscRegister::X2, assert_register_is!(0xFFFFF234));

        run_test_with_memory(backend, &[(0, 0xABCD_F234)]);
    }

    /// High half-word is 0x8000 → −32768 after sign-extension
    #[test]
    fn test_lh_sign_negative_high() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.lh(RiscRegister::X3, RiscRegister::X1, 2);
        backend.inspect_register(RiscRegister::X3, assert_register_is!(0xFFFF8000));

        run_test_with_memory(backend, &[(0, 0x8000_1234)]);
    }
}

mod infra {
    use super::*;

    #[test]
    fn test_assert_base_registrs_are_loaded() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.inspect_register(RiscRegister::X5, assert_register_is!(5));

        let mut func = backend.finalize().expect("Failed to finalize function");
        func.registers[5] = 5;

        unsafe {
            func.call();
        }
    }

    #[test]
    fn test_assert_registers_are_persisted_on_exit() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.risc_add(RiscOperand::Immediate(5), RiscOperand::Immediate(0), RiscRegister::X1);

        let mut func = backend.finalize().expect("Failed to finalize function");
        unsafe {
            func.call();
        }

        assert_eq!(func.registers[1], 5);
    }
}

mod trace {
    use crate::{TraceChunk, TraceCollector};

    use super::*;

    #[test]
    fn test_basic_trace() {
        let mut backend = new_backend();

        backend.start_instr();
        backend.risc_add(RiscOperand::Immediate(5), RiscOperand::Immediate(0), RiscRegister::X1);
        backend.set_pc(4);

        backend.trace_registers();
        backend.trace_pc_start();
        backend.trace_mem_value(RiscRegister::X1);

        let mut func = backend.finalize().expect("Failed to finalize function");
        let trace = unsafe { func.call() }.expect("No trace returned");

        let trace = TraceChunk::copy_from_bytes(&trace);
        assert_eq!(trace.start_registers[1], 5);
        assert_eq!(trace.pc_start, 4);
        assert_eq!(trace.mem_reads.len(), 1);
        assert_eq!(trace.mem_reads[0], 5);
    }
}
