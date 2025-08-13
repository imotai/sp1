#![allow(clippy::fn_to_numeric_cast)]

use super::{TranspilerBackend, CONTEXT, JUMP_TABLE, PC_OFFSET, TEMP_A, TEMP_B};
use crate::{
    impl_risc_alu, ComputeInstructions, ControlFlowInstructions, JitContext, MemoryInstructions,
    RiscOperand, RiscRegister, SP1RiscvTranspiler, SystemInstructions,
};
use dynasmrt::{dynasm, x64::Rq, DynasmApi, DynasmLabelApi};

impl ComputeInstructions for TranspilerBackend {
    fn add(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        // lhs <- lhs + rhs (64-bit)
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;
                add Rq(TEMP_A), Rq(TEMP_B)
            }
        });
    }

    fn mul(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        // rd <- rs1 * rs2 (64-bit)
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;
                imul Rq(TEMP_A), Rq(TEMP_B)
            }
        });
    }

    fn and(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        // rd <- rs1 & rs2 (64-bit)
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;
                and Rq(TEMP_A), Rq(TEMP_B)
            }
        });
    }

    fn or(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        // rd <- rs1 | rs2 (64-bit)
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;
                or Rq(TEMP_A), Rq(TEMP_B)
            }
        });
    }

    fn xor(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        // rd <- rs1 ^ rs2 (64-bit)
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;
                xor Rq(TEMP_A), Rq(TEMP_B)
            }
        });
    }

    fn div(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        // X86 uses [RAX::RDX] for the 64-bit divide operation.
        // So we need to sign extend the lhs into RDX.
        //
        // The quotient is stored in RAX, and the remainder is stored in RDX.
        //
        // We can just write the quotient back into lhs, and the remainder is discarded.
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;

                // ------------------------------------
                // 1. Skip fault on div-by-zero
                // ------------------------------------
                test Rq(TEMP_B), Rq(TEMP_B);  // ZF=1 if rhs == 0
                jz   >div_by_zero;

                // ------------------------------------
                // 2. Perform signed divide
                // ------------------------------------
                mov  rax, Rq(TEMP_A);         // dividend → RAX
                cqo;                          // sign-extend RAX into RDX (64-bit)
                idiv Rq(TEMP_B);              // quotient → RAX, remainder → RDX
                mov  Rq(TEMP_A), rax;         // write quotient back into lhs
                jmp >done;

                // ------------------------------------
                // 3. if rhs == 0
                // ------------------------------------
                div_by_zero:;
                xor  Rq(TEMP_A), Rq(TEMP_A);  // lhs = 0

                // ------------------------------------
                // Merge branch
                // ------------------------------------
                done:
            }
        });
    }

    fn divu(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        // lhs <- lhs / rhs   (unsigned 64-bit; 0 if rhs == 0)
        // clobbers: RAX, RDX
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;

                // ----- skip fault on div-by-zero -----
                test Rq(TEMP_B), Rq(TEMP_B);   // ZF = 1 when rhs == 0
                jz   >div_by_zero;

                // ----- perform unsigned divide -----
                mov  rax, Rq(TEMP_A);          // dividend → RAX
                xor  rdx, rdx;                 // zero-extend: RDX = 0
                div  Rq(TEMP_B);               // unsigned divide: RDX:RAX / rhs
                mov  Rq(TEMP_A), rax;          // quotient → lhs
                jmp  >done;

                // ----- rhs == 0 -----
                div_by_zero:;
                xor  Rq(TEMP_A), Rq(TEMP_A);   // lhs = 0

                done:
            }
        });
    }

    fn mulh(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;

                mov  rax, Rq(TEMP_A);      // RAX = lhs (signed)
                imul Rq(TEMP_B);           // signed 64×64 → 128; high → RDX
                mov  Rq(TEMP_A), rdx       // lhs = high 64 bits
            }
        });
    }

    fn mulhu(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;

                mov  rax, Rq(TEMP_A);      // RAX = TEMP_A (unsigned)
                mul  Rq(TEMP_B);           // unsigned 64×64 → 128; high → RDX
                mov  Rq(TEMP_A), rdx       // TEMP_A = high 64 bits
            }
        });
    }

    fn mulhsu(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;

                // ──────────────────────────────────────────────────────────────
                // 1. Move the **signed** left-hand operand (`TEMP_A`) into RAX.
                //    ✦ The x86-64 `mul` instruction always uses RAX as its implicit
                //      64-bit source operand, so we must place `TEMP_A` there first.
                // ──────────────────────────────────────────────────────────────
                mov rax, Rq(TEMP_A);

                // ──────────────────────────────────────────────────────────────
                // 2. Preserve a second copy of `TEMP_A` in RCX.
                //    ✦ The upcoming `mul` clobbers both RAX and RDX, erasing any
                //      trace of the original sign.  We save `TEMP_A` in RCX so that
                //      we can later decide whether the fix-up for a *negative*
                //      multiplicand is required.
                // ──────────────────────────────────────────────────────────────
                mov rcx, rax;

                // ──────────────────────────────────────────────────────────────
                // 3. Unsigned 64×64-bit multiply:
                //    mul Rq(TEMP_B)
                //    ✦ Computes  RDX:RAX = (unsigned)RAX × (unsigned)TEMP_B.
                //      The high 64 bits of the 128-bit product land in RDX.
                // ──────────────────────────────────────────────────────────────
                mul Rq(TEMP_B);

                // ──────────────────────────────────────────────────────────────
                // 4. Determine whether the *original* `TEMP_A` was negative.
                //    ✦ `test rcx, rcx` sets the sign flag from RCX (the saved `TEMP_A`).
                //    ✦ If the sign flag is *clear* (`TEMP_A` ≥ 0), we can skip the
                //      correction step because the high half already matches the
                //      semantics of the RISC-V MULHSU instruction.
                // ──────────────────────────────────────────────────────────────
                test rcx, rcx;
                jns >store_high;          // Jump if `TEMP_A` was non-negative.

                // ──────────────────────────────────────────────────────────────
                // 5. Fix-up for negative `TEMP_A` (signed × unsigned semantics):
                //    ✦ For a negative multiplicand, the unsigned `mul` delivered a
                //      product that is *2⁶⁴* too large in the high word.  Subtracting
                //      `TEMP_B` from RDX removes that excess and yields the correct
                //      signed-high result.
                // ──────────────────────────────────────────────────────────────
                sub rdx, Rq(TEMP_B);

                // ──────────────────────────────────────────────────────────────
                // 6. Write the corrected high 64 bits back to the destination
                //    RISC register specified by `TEMP_A`.
                // ──────────────────────────────────────────────────────────────
                store_high:;
                mov Rq(TEMP_A), rdx
            }
        });
    }

    /// Signed remainder: `rd = rs1 % rs2`  
    /// *RISC-V rule*: if `rs2 == 0`, the result must be **0** (no fault).
    fn rem(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;

                // ──────────────────────────────────────────────────────────────
                // 0. Guard: if divisor is 0, skip the IDIV and zero the result
                // ──────────────────────────────────────────────────────────────
                test Rq(TEMP_B), Rq(TEMP_B);        // ZF = 1  ⇒  TEMP_B == 0
                jz   >by_zero;                // jump to fix-up path

                // ──────────────────────────────────────────────────────────────
                // 1. Prepare the **signed** 64-bit dividend in EDX:EAX
                //    -------------------------------------------------
                //    • EAX ← low 32 bits of TEMP_A
                //    • CDQ  sign-extends EAX into EDX
                //      → EDX:EAX now holds the two's-complement 64-bit value a
                // ──────────────────────────────────────────────────────────────
                mov  rax, Rq(TEMP_A);            // RAX = a  (signed 64-bit)
                cqo;                          // RDX = sign(a)

                // ──────────────────────────────────────────────────────────────
                // 2. Signed divide:          a  /  b
                //    -------------------------------------------------
                //    • idiv r/m32   performs  (EDX:EAX) ÷ TEMP_B
                //      – Quotient  → EAX   (ignored)
                //      – Remainder → EDX   (what RISC-V REM returns)
                // ──────────────────────────────────────────────────────────────
                idiv Rq(TEMP_B);                 // signed divide

                // ──────────────────────────────────────────────────────────────
                // 3. Write the remainder (EDX) back to the destination register
                // ──────────────────────────────────────────────────────────────
                mov  Rq(TEMP_A), rdx;            // TEMP_A = remainder
                jmp  >done;

                // ──────────────────────────────────────────────────────────────
                // Divisor == 0  →  result must be 0 (no fault)
                // ──────────────────────────────────────────────────────────────
                by_zero:;
                xor  Rq(TEMP_A), Rq(TEMP_A);        // TEMP_A = 0

                done:
            }
        });
    }

    fn remu(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;

                // ──────────────────────────────────────────────────────────────
                // 0. Guard against /0 → result = 0
                // ──────────────────────────────────────────────────────────────
                test Rq(TEMP_B), Rq(TEMP_B);
                jz   >by_zero;

                // ──────────────────────────────────────────────────────────────
                // 1. Prepare the **unsigned** 128-bit dividend in RDX:RAX
                //    -------------------------------------------------
                //    • Zero-extend TEMP_A into RDX:RAX.
                // ──────────────────────────────────────────────────────────────
                mov  rax, Rq(TEMP_A);
                xor  rdx, rdx;

                // ──────────────────────────────────────────────────────────────
                // 2. Unsigned divide:       a  /  b
                //    -------------------------------------------------
                //    • div r/m64   performs  (RDX:RAX) ÷ TEMP_B
                //      – Quotient  → RAX   (unused)
                //      – Remainder → RDX   (what RISC-V REMU wants)
                // ──────────────────────────────────────────────────────────────
                div  Rq(TEMP_B);

                // ──────────────────────────────────────────────────────────────
                // 3. Write the remainder back to the destination register.
                // ──────────────────────────────────────────────────────────────
                mov  Rq(TEMP_A), rdx;
                jmp  >done;

                // ──────────────────────────────────────────────────────────────
                // Divisor == 0  →  result must be 0 (no fault)
                // ──────────────────────────────────────────────────────────────
                by_zero:;
                xor  Rq(TEMP_A), Rq(TEMP_A);

                done:
            }
        });
    }

    fn sll(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        // We only can use the lower 6 bits for the shift count in 64-bit mode.
        // In RV64I, this is also true!
        //
        // CL is an alias for the lower byte of RCX.
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;

                // ──────────────────────────────────────────────────────────────
                // 1. Move the shift count (lower 6 bits, RV64I spec) into CL.
                //    x86 uses only the low 6 bits for 64-bit shifts as well, so
                //    the masking is implicit—no extra AND is necessary.
                // ──────────────────────────────────────────────────────────────
                mov rcx, Rq(TEMP_B);        // CL = TEMP_B

                // ──────────────────────────────────────────────────────────────
                // 2. Logical left shift: Rq(TEMP_A) ← Rq(TEMP_A) << (CL & 0b11_1111)
                //    `shl r/m64, cl` preserves the semantics required by
                //    RISC-V SLL because both architectures ignore bits ≥ 6.
                // ──────────────────────────────────────────────────────────────
                shl Rq(TEMP_A), cl         // variable-count shift
            }
        });
    }

    fn sra(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;

                // ──────────────────────────────────────────────────────────────
                // 1. Put the variable shift count into CL.
                //    • Only the low 6 bits are used for 64-bit operands,
                //      which matches the RISC-V spec for RV64.
                // ──────────────────────────────────────────────────────────────
                mov  rcx, Rq(TEMP_B);        // CL ← TEMP_B

                // ──────────────────────────────────────────────────────────────
                // 2. Arithmetic right shift:
                //      Rq(TEMP_A) ← (signed)Rq(TEMP_A) >> (CL & 0x3F)
                //    • `sar` replicates the sign bit as it shifts, so
                //      negative values stay negative after the operation.
                // ──────────────────────────────────────────────────────────────
                sar  Rq(TEMP_A), cl         // variable-count shift
            }
        });
    }

    fn srl(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;

                // ──────────────────────────────────────────────────────────────
                // 1. Load shift count into CL (same reasoning as above).
                // ──────────────────────────────────────────────────────────────
                mov  rcx, Rq(TEMP_B);        // CL ← TEMP_B

                // ──────────────────────────────────────────────────────────────
                // 2. Logical right shift:
                //      Rq(TEMP_A) ← (unsigned)Rq(TEMP_A) >> (CL & 0x3F)
                //    • `shr` always inserts zeros from the left, regardless
                //      of the operand's sign.
                // ──────────────────────────────────────────────────────────────
                shr  Rq(TEMP_A), cl
            }
        });
    }

    fn slt(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;

                cmp  Rq(TEMP_A), Rq(TEMP_B);

                // ──────────────────────────────────────────────────────────────
                // 2. setl  r/m8
                //    • Writes   1  to the target byte if  (SF ≠ OF)
                //      which is the signed "less than" condition.
                //    • We store straight into the low-byte of TEMP_A —
                //      dynasm's `Rb()` gives us that alias.
                // ──────────────────────────────────────────────────────────────
                setl Rb(TEMP_A);               // byte = 1 if TEMP_A < TEMP_B (signed)

                // ──────────────────────────────────────────────────────────────
                // 3. Zero-extend that byte back to a full 32-bit register so
                //    that the RISC register ends up with 0x0000_0000 or 0x0000_0001.
                // ──────────────────────────────────────────────────────────────
                movzx Rq(TEMP_A), Rb(TEMP_A)     // Rd(TEMP_A) = 0 or 1
            }
        });
    }

    fn sltu(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;

                cmp  Rq(TEMP_A), Rq(TEMP_B);

                // ------------------------------------
                // `setb` ("below") checks the Carry Flag (CF):
                //   CF = 1  iff  TEMP_A < TEMP_B  in an *unsigned* sense.
                // ------------------------------------
                setb Rb(TEMP_A);

                // ------------------------------------
                // Zero-extend to 32 bits (0 or 1).
                // ------------------------------------
                movzx Rq(TEMP_A), Rb(TEMP_A)
            }
        });
    }

    fn sub(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        // rd <- rs1 - rs2 (64-bit)
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;
                sub Rq(TEMP_A), Rq(TEMP_B)
            }
        });
    }

    fn addw(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        // addw performs 32-bit addition on lower 32 bits, then sign-extends result to 64 bits
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;

                // Perform 32-bit addition (automatically truncates to 32 bits)
                add Rd(TEMP_A), Rd(TEMP_B);

                // Sign-extend the 32-bit result to 64 bits
                movsxd Rq(TEMP_A), Rd(TEMP_A)
            }
        });
    }

    fn subw(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        // subw performs 32-bit subtraction on lower 32 bits, then sign-extends result to 64 bits
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;

                // Perform 32-bit subtraction (automatically truncates to 32 bits)
                sub Rd(TEMP_A), Rd(TEMP_B);

                // Sign-extend the 32-bit result to 64 bits
                movsxd Rq(TEMP_A), Rd(TEMP_A)
            }
        });
    }

    fn sllw(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        // sllw performs 32-bit shift left, then sign-extends result to 64 bits
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;

                // Move shift count to CL
                mov ecx, Rd(TEMP_B);

                // Perform 32-bit left shift
                shl Rd(TEMP_A), cl;

                // Sign-extend the 32-bit result to 64 bits
                movsxd Rq(TEMP_A), Rd(TEMP_A)
            }
        });
    }

    fn srlw(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        // srlw performs logical right shift on lower 32 bits, then sign-extends result to 64 bits
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;

                // Move shift count to CL
                mov ecx, Rd(TEMP_B);

                // Perform 32-bit logical right shift
                shr Rd(TEMP_A), cl;

                // Sign-extend the 32-bit result to 64 bits
                movsxd Rq(TEMP_A), Rd(TEMP_A)
            }
        });
    }

    fn sraw(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        // sraw performs arithmetic right shift on lower 32 bits, then sign-extends result to 64 bits
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;

                // Move shift count to CL
                mov ecx, Rd(TEMP_B);

                // Perform 32-bit arithmetic right shift
                sar Rd(TEMP_A), cl;

                // Sign-extend the 32-bit result to 64 bits
                movsxd Rq(TEMP_A), Rd(TEMP_A)
            }
        });
    }

    fn mulw(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        // mulw performs 32-bit multiplication, then sign-extends result to 64 bits
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;

                // Perform 32-bit multiplication
                imul Rd(TEMP_A), Rd(TEMP_B);

                // Sign-extend the 32-bit result to 64 bits
                movsxd Rq(TEMP_A), Rd(TEMP_A)
            }
        });
    }

    fn divw(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        // divw performs 32-bit signed division, then sign-extends result to 64 bits
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;

                // Check for division by zero
                test Rd(TEMP_B), Rd(TEMP_B);
                jz >div_by_zero;

                // Perform signed 32-bit divide
                mov eax, Rd(TEMP_A);           // dividend → EAX
                cdq;                        // sign-extend EAX into EDX
                idiv Rd(TEMP_B);               // quotient → EAX
                movsxd Rq(TEMP_A), eax;        // sign-extend result to 64 bits
                jmp >done;

                div_by_zero:;
                // For RV64I, divw by zero returns 0xFFFFFFFFFFFFFFFF (-1 sign-extended)
                mov Rq(TEMP_A), -1;

                done:
            }
        });
    }

    fn divuw(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        // divuw performs 32-bit unsigned division, then sign-extends result to 64 bits
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;

                // Check for division by zero
                test Rd(TEMP_B), Rd(TEMP_B);
                jz >div_by_zero;

                // Perform unsigned 32-bit divide
                mov eax, Rd(TEMP_A);           // dividend → EAX
                xor edx, edx;               // zero-extend
                div Rd(TEMP_B);                // quotient → EAX
                movsxd Rq(TEMP_A), eax;        // sign-extend result to 64 bits
                jmp >done;

                div_by_zero:;
                // For RV64I, divuw by zero returns 0xFFFFFFFFFFFFFFFF (-1 sign-extended)
                mov Rq(TEMP_A), -1;

                done:
            }
        });
    }

    fn remw(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        // remw performs 32-bit signed remainder, then sign-extends result to 64 bits
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;

                // Check for division by zero
                test Rd(TEMP_B), Rd(TEMP_B);
                jz >rem_by_zero;

                // Perform signed 32-bit remainder
                mov eax, Rd(TEMP_A);           // dividend → EAX
                cdq;                        // sign-extend EAX into EDX
                idiv Rd(TEMP_B);               // remainder → EDX
                movsxd Rq(TEMP_A), edx;        // sign-extend result to 64 bits
                jmp >done;

                rem_by_zero:;
                // For RV64I, remw by zero returns the dividend (TEMP_A) sign-extended
                movsxd Rq(TEMP_A), Rd(TEMP_A);

                done:
            }
        });
    }

    fn remuw(&mut self, rd: RiscRegister, rs1: RiscOperand, rs2: RiscOperand) {
        // remuw performs 32-bit unsigned remainder, then sign-extends result to 64 bits
        impl_risc_alu!(self, rd, rs1, rs2, TEMP_A, TEMP_B, {
            dynasm! {
                self;
                .arch x64;

                // Check for division by zero
                test Rd(TEMP_B), Rd(TEMP_B);
                jz >rem_by_zero;

                // Perform unsigned 32-bit remainder
                mov eax, Rd(TEMP_A);           // dividend → EAX
                xor edx, edx;               // zero-extend
                div Rd(TEMP_B);                // remainder → EDX
                movsxd Rq(TEMP_A), edx;        // sign-extend result to 64 bits
                jmp >done;

                rem_by_zero:;
                // For RV64I, remuw by zero returns the dividend (TEMP_A) sign-extended
                movsxd Rq(TEMP_A), Rd(TEMP_A);

                done:
            }
        });
    }

    fn auipc(&mut self, rd: RiscRegister, imm: u64) {
        // rd <- pc + imm
        // pc <- pc + 4
        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 1. Copy the current PC into TEMP_A
            // ------------------------------------
            mov Rq(TEMP_A), [Rq(CONTEXT) + PC_OFFSET];

            // ------------------------------------
            // 2. Increment the PC by the immediate.
            // ------------------------------------
            add Rq(TEMP_A), imm as i32;

            // ------------------------------------
            // 3. Increment the PC to the next instruction.
            // ------------------------------------
            add QWORD [Rq(CONTEXT) + PC_OFFSET], 4
        }

        // Store the result in the destination register.
        self.emit_risc_register_store(TEMP_A, rd);
    }

    fn lui(&mut self, rd: RiscRegister, imm: u64) {
        // rd <- imm << 12
        // LUI loads a 20-bit immediate shifted left by 12 bits into the destination register
        dynasm! {
            self;
            .arch x64;

            mov Rq(TEMP_A), imm as i32
        }

        // Store the result in the destination register.
        self.emit_risc_register_store(TEMP_A, rd);
    }
}

impl ControlFlowInstructions for TranspilerBackend {
    fn jal(&mut self, rd: RiscRegister, imm: u64) {
        // Store the current pc + 4 into
        self.load_pc_into_register(TEMP_A);

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 1. Add 4 to the current PC in temp_a
            // ------------------------------------
            add Rq(TEMP_A), 4
        }

        // Store the current PC + 4 into the destination register.
        self.emit_risc_register_store(TEMP_A, rd);

        // Adjust the PC store in the context by the immediate.
        self.bump_pc(imm as u32);

        // Jump the ASM code corresponding to the PC.
        self.jump_to_pc();
    }

    fn jalr(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u64) {
        // ------------------------------------
        // 1. Compute the PC to store into rd.
        // ------------------------------------
        self.load_pc_into_register(TEMP_B);

        dynasm! {
            self;
            .arch x64;

            add Rq(TEMP_B), 4
        }

        // ------------------------------------
        // 2. Load rs1, add imm, and store it as PC
        // ------------------------------------

        self.emit_risc_operand_load(rs1.into(), TEMP_A);

        dynasm! {
            self;
            .arch x64;

            add Rq(TEMP_A), imm as i32;
            mov QWORD [Rq(CONTEXT) + PC_OFFSET], Rq(TEMP_A)
        }

        // ------------------------------------
        // 3. Store the computed PC into rd
        // ------------------------------------
        self.emit_risc_register_store(TEMP_B, rd);

        // ------------------------------------
        // 3. Jump to the target pc.
        // ------------------------------------
        self.jump_to_pc();
    }

    fn beq(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u64) {
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.emit_risc_operand_load(rs2.into(), TEMP_B);

        let pc_base = self.pc_base as i32;
        // Compare the registers
        dynasm! {
            self;
            .arch x64;

            // Check if rs1 == rs2
            cmp Rq(TEMP_A), Rq(TEMP_B);
            // If rs1 != rs2, jump to not_branched, since that would imply !(rs1 == rs2)
            jne >not_branched;

            // ------------------------------------
            // Branched:
            // 0. Bump the pc by the immediate.
            // ------------------------------------
            add QWORD [Rq(CONTEXT) + PC_OFFSET], imm as i32;

            // ------------------------------------
            // 1. Load the current pc into TEMP_A
            // ------------------------------------
            mov Rq(TEMP_A), QWORD [Rq(CONTEXT) + PC_OFFSET];

            // ------------------------------------
            // 2. Lookup into the jump table and load the asm offset into TEMP_A
            // ------------------------------------
            sub Rq(TEMP_A), pc_base;
            shr Rq(TEMP_A), 2;
            mov Rq(TEMP_A), QWORD [Rq(JUMP_TABLE) + Rq(TEMP_A) * 8];

            // ------------------------------------
            // 3. Jump to the target pc.
            // ------------------------------------
            jmp Rq(TEMP_A);

            // ------------------------------------
            // Not branched:
            // ------------------------------------
            not_branched:;

            // ------------------------------------
            // 1. Bump the pc by 4
            // ------------------------------------
            add QWORD [Rq(CONTEXT) + PC_OFFSET], 4
        }
    }

    fn bge(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u64) {
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.emit_risc_operand_load(rs2.into(), TEMP_B);

        let pc_base = self.pc_base as i32;
        dynasm! {
            self;
            .arch x64;

            // Check if rs1 == rs2
            cmp Rq(TEMP_A), Rq(TEMP_B);
            // If rs1 < rs2, jump to not_branched, since that would imply !(rs1 >= rs2)
            jl >not_branched;

            // ------------------------------------
            // Branched:
            // 0. Bump the pc by the immediate.
            // ------------------------------------
            add QWORD [Rq(CONTEXT) + PC_OFFSET], imm as i32;

            // ------------------------------------
            // 1. Load the current pc into TEMP_A
            // ------------------------------------
            mov Rq(TEMP_A), QWORD [Rq(CONTEXT) + PC_OFFSET];

            // ------------------------------------
            // 2. Lookup into the jump table and load the asm offset into TEMP_A
            // ------------------------------------
            sub Rq(TEMP_A), pc_base;
            shr Rq(TEMP_A), 2; // Divide by 4 to get the index.
            mov Rq(TEMP_A), QWORD [Rq(JUMP_TABLE) + Rq(TEMP_A) * 8];

            // ------------------------------------
            // 3. Jump to the target pc.
            // ------------------------------------
            jmp Rq(TEMP_A);

            // ------------------------------------
            // Not branched:
            // ------------------------------------
            not_branched:;

            // ------------------------------------
            // 1. Bump the pc by 4
            // ------------------------------------
            add QWORD [Rq(CONTEXT) + PC_OFFSET], 4
        }
    }

    fn bgeu(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u64) {
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.emit_risc_operand_load(rs2.into(), TEMP_B);

        let pc_base = self.pc_base as i32;
        dynasm! {
            self;
            .arch x64;

            cmp Rq(TEMP_A), Rq(TEMP_B);
            // If rs1 < rs2, jump to not_branched, since that would imply !(rs1 >= rs2)
            jb >not_branched;

            // ------------------------------------
            // Branched:
            // 0. Bump the pc by the immediate.
            // ------------------------------------
            add QWORD [Rq(CONTEXT) + PC_OFFSET], imm as i32;

            // ------------------------------------
            // 1. Load the current pc into TEMP_A
            // ------------------------------------
            mov Rq(TEMP_A), QWORD [Rq(CONTEXT) + PC_OFFSET];

            // ------------------------------------
            // 2. Lookup into the jump table and load the asm offset into TEMP_A
            // ------------------------------------
            sub Rq(TEMP_A), pc_base;
            shr Rq(TEMP_A), 2;
            mov Rq(TEMP_A), QWORD [Rq(JUMP_TABLE) + Rq(TEMP_A) * 8];

            // ------------------------------------
            // 3. Jump to the target pc.
            // ------------------------------------
            jmp Rq(TEMP_A);

            // ------------------------------------
            // Not branched:
            // ------------------------------------
            not_branched:;

            // ------------------------------------
            // 1. Bump the pc by 4
            // ------------------------------------
            add QWORD [Rq(CONTEXT) + PC_OFFSET], 4
        }
    }

    fn blt(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u64) {
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.emit_risc_operand_load(rs2.into(), TEMP_B);

        let pc_base = self.pc_base as i32;
        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // Compare the two registers.
            //
            cmp Rq(TEMP_A), Rq(TEMP_B);   // signed compare
            jge >not_branched;            // rs1 ≥ rs2  →  skip

            // ------------------------------------
            // Branched:
            // 0. Bump the pc by the immediate.
            // ------------------------------------
            add QWORD [Rq(CONTEXT) + PC_OFFSET], imm as i32;

            // ------------------------------------
            // 1. Load the current pc into TEMP_A
            // ------------------------------------
            mov Rq(TEMP_A), QWORD [Rq(CONTEXT) + PC_OFFSET];

            // ------------------------------------
            // 2. Lookup into the jump table and load the asm offset into TEMP_A
            // ------------------------------------
            sub Rq(TEMP_A), pc_base;
            shr Rq(TEMP_A), 2; // Divide by 4 to get the index.
            mov Rq(TEMP_A), QWORD [Rq(JUMP_TABLE) + Rq(TEMP_A) * 8];

            // ------------------------------------
            // 3. Jump to the target pc.
            // ------------------------------------
            jmp Rq(TEMP_A);

            // ------------------------------------
            // Not branched:
            // ------------------------------------
            not_branched:;

            // ------------------------------------
            // 1. Bump the pc by 4
            // ------------------------------------
            add QWORD [Rq(CONTEXT) + PC_OFFSET], 4
        }
    }

    fn bltu(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u64) {
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.emit_risc_operand_load(rs2.into(), TEMP_B);

        let pc_base = self.pc_base as i32;
        dynasm! {
            self;
            .arch x64;
            cmp Rq(TEMP_A), Rq(TEMP_B);   // unsigned compare
            jae >not_branched;             // rs1 ≥ rs2 (unsigned) → skip

            // ------------------------------------
            // Branched:
            // 0. Bump the pc by the immediate.
            // ------------------------------------
            add QWORD [Rq(CONTEXT) + PC_OFFSET], imm as i32;

            // ------------------------------------
            // 1. Load the current pc into TEMP_A
            // ------------------------------------
            mov Rq(TEMP_A), QWORD [Rq(CONTEXT) + PC_OFFSET];

            // ------------------------------------
            // 2. Lookup into the jump table and load the asm offset into TEMP_A
            // ------------------------------------
            sub Rq(TEMP_A), pc_base;
            shr Rq(TEMP_A), 2; // Divide by 4 to get the index.
            mov Rq(TEMP_A), QWORD [Rq(JUMP_TABLE) + Rq(TEMP_A) * 8];

            // ------------------------------------
            // 3. Jump to the target pc.
            // ------------------------------------
            jmp Rq(TEMP_A);

            // ------------------------------------
            // Not branched:
            // ------------------------------------
            not_branched:;

            // 1. Bump the pc by 4
            add QWORD [Rq(CONTEXT) + PC_OFFSET], 4
        }
    }

    fn bne(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u64) {
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.emit_risc_operand_load(rs2.into(), TEMP_B);

        let pc_base = self.pc_base as i32;
        dynasm! {
            self;
            .arch x64;
            cmp Rq(TEMP_A), Rq(TEMP_B);   // sets ZF
            je  >not_branched;            // rs1 == rs2  →  skip

            // ------------------------------------
            // Branched:
            // 0. Bump the pc in the context by the immediate.
            // ------------------------------------
            add QWORD [Rq(CONTEXT) + PC_OFFSET], imm as i32;

            // ------------------------------------
            // 1. Load the current pc into TEMP_A
            // ------------------------------------
            mov Rq(TEMP_A), QWORD [Rq(CONTEXT) + PC_OFFSET];

            // ------------------------------------
            // 2. Lookup into the jump table and load the asm offset into TEMP_A
            // ------------------------------------
            sub Rq(TEMP_A), pc_base;
            shr Rq(TEMP_A), 2; // Divide by 4 to get the index.
            mov Rq(TEMP_A), QWORD [Rq(JUMP_TABLE) + Rq(TEMP_A) * 8];

            // ------------------------------------
            // 3. Jump to the target pc.
            // ------------------------------------
            jmp Rq(TEMP_A);

            // ------------------------------------
            // Not branched:
            // ------------------------------------
            not_branched:;

            // ------------------------------------
            // 1. Bump the pc by 4
            // ------------------------------------
            add QWORD [Rq(CONTEXT) + PC_OFFSET], 4
        }
    }
}

impl MemoryInstructions for TranspilerBackend {
    fn lb(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u64) {
        // ------------------------------------
        // 1. Load in the base address and the phy sical memory pointer.
        // ------------------------------------
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.load_memory_ptr(TEMP_B);

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 2. Add the immediate to the base address
            // ------------------------------------
            add Rq(TEMP_A), imm as i32;

            // ------------------------------------
            // 3. Add the risc32 byte offset to the physical memory pointer
            // ------------------------------------
            add Rq(TEMP_B), Rq(TEMP_A);

            // ------------------------------------
            // 4. Load byte → sign-extend to 32 bits
            // ------------------------------------
            movsx Rq(TEMP_B), BYTE [Rq(TEMP_B)]
        }

        // 4. Write back to destination register
        self.emit_risc_register_store(TEMP_B, rd);
    }

    fn lbu(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u64) {
        // ------------------------------------
        // 1. Load in the base address
        //    and the physical memory pointer.
        // ------------------------------------
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.load_memory_ptr(TEMP_B);

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 2. Add the immediate to the base address
            // ------------------------------------
            add Rq(TEMP_A), imm as i32;

            // ------------------------------------
            // 3. Add the risc32 byte offset to
            //    the physical memory pointer
            // ------------------------------------
            add Rq(TEMP_B), Rq(TEMP_A);

            // ------------------------------------
            // 4. Load byte → zero-extend to 32 bits
            // ------------------------------------
            movzx Rq(TEMP_B), BYTE [Rq(TEMP_B)]
        }

        self.emit_risc_register_store(TEMP_B, rd);
    }

    fn lh(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u64) {
        // ------------------------------------
        // 1. Load in the base address
        //    and the physical memory pointer.
        // ------------------------------------
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.load_memory_ptr(TEMP_B);

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 2. Add the immediate to the base address
            // ------------------------------------
            add Rq(TEMP_A), imm as i32;

            // ------------------------------------
            // 3. Add the risc32 byte offset to the physical memory pointer
            // ------------------------------------
            add Rq(TEMP_B), Rq(TEMP_A);

            // ------------------------------------
            // 4. Load half-word → sign-extend to 32 bits
            // ------------------------------------
            movsx Rq(TEMP_B), WORD [Rq(TEMP_B)]
        }

        self.emit_risc_register_store(TEMP_B, rd);
    }

    fn lhu(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u64) {
        // ------------------------------------
        // 1. Load in the base address
        //    and the physical memory pointer.
        // ------------------------------------
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.load_memory_ptr(TEMP_B);

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 2. Add the immediate to the base address
            // ------------------------------------
            add Rq(TEMP_A), imm as i32;

            // ------------------------------------
            // 3. Add the risc32 byte offset to the physical memory pointer
            // ------------------------------------
            add Rq(TEMP_B), Rq(TEMP_A);

            // ------------------------------------
            // 4. Load half-word → zero-extend to 32 bits
            // ------------------------------------
            movzx Rq(TEMP_B), WORD [Rq(TEMP_B)]
        }

        self.emit_risc_register_store(TEMP_B, rd);
    }

    fn lw(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u64) {
        // ------------------------------------
        // 1. Load the base address into TEMP_A
        // and physical memory pointer into TEMP_B
        // ------------------------------------
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.load_memory_ptr(TEMP_B);

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 2. Add the immediate to the base address
            // ------------------------------------
            add Rq(TEMP_A), imm as i32;

            // ------------------------------------
            // 3. Add the risc32 byte offset to the physical memory pointer
            //
            // We use 64 bit arithmetic since were dealing with native memory.
            // If there was an overflow it wouldve been handled correctly by the previous add.
            // ------------------------------------
            add Rq(TEMP_B), Rq(TEMP_A);

            // ------------------------------------
            // 3. Load the word from physical memory into TEMP_B (sign-extended to 64-bit)
            // ------------------------------------
            movsxd Rq(TEMP_B), DWORD [Rq(TEMP_B)]
        }

        // ------------------------------------
        // 4. Store the result in the destination register.
        // ------------------------------------
        self.emit_risc_register_store(TEMP_B, rd);
    }

    fn lwu(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u64) {
        // ------------------------------------
        // 1. Load the base address into TEMP_A
        // and physical memory pointer into TEMP_B
        // ------------------------------------
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.load_memory_ptr(TEMP_B);

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 2. Add the immediate to the base address
            // ------------------------------------
            add Rq(TEMP_A), imm as i32;

            // ------------------------------------
            // 3. Add the risc32 byte offset to the physical memory pointer
            //
            // We use 64 bit arithmetic since were dealing with native memory.
            // If there was an overflow it wouldve been handled correctly by the previous add.
            // ------------------------------------
            add Rq(TEMP_B), Rq(TEMP_A);

            // ------------------------------------
            // 3. Load the word from physical memory into TEMP_B (zero-extended to 64-bit)
            // ------------------------------------
            mov Rd(TEMP_B), DWORD [Rq(TEMP_B)]
        }

        // ------------------------------------
        // 4. Store the result in the destination register.
        // ------------------------------------
        self.emit_risc_register_store(TEMP_B, rd);
    }

    fn ld(&mut self, rd: RiscRegister, rs1: RiscRegister, imm: u64) {
        // ------------------------------------
        // 1. Load the base address into TEMP_A
        // and physical memory pointer into TEMP_B
        // ------------------------------------
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.load_memory_ptr(TEMP_B);

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 2. Add the immediate to the base address
            // ------------------------------------
            add Rq(TEMP_A), imm as i32;

            // ------------------------------------
            // 3. Add the risc32 byte offset to the physical memory pointer
            //
            // We use 64 bit arithmetic since were dealing with native memory.
            // If there was an overflow it wouldve been handled correctly by the previous add.
            // ------------------------------------
            add Rq(TEMP_B), Rq(TEMP_A);

            // ------------------------------------
            // 3. Load the word from physical memory into TEMP_B
            // ------------------------------------
            mov Rq(TEMP_B), QWORD [Rq(TEMP_B)]
        }

        // ------------------------------------
        // 4. Store the result in the destination register.
        // ------------------------------------
        self.emit_risc_register_store(TEMP_B, rd);
    }

    fn sb(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u64) {
        // ------------------------------------
        // 1. Load the base address into TEMP_A
        // and physical memory pointer into TEMP_B
        // ------------------------------------
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.load_memory_ptr(TEMP_B);

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 2. Add the immediate to the base address
            // ------------------------------------
            add Rq(TEMP_A), imm as i32;

            // ------------------------------------
            // 3. Add the risc32 byte offset to the physical memory pointer
            // ------------------------------------
            add Rq(TEMP_B), Rq(TEMP_A)
        }

        // ------------------------------------
        // 4. Load the word from the RISC register into TEMP_A
        // ------------------------------------
        self.emit_risc_operand_load(rs2.into(), TEMP_A);

        // ------------------------------------
        // 5. Store the word into physical memory
        // ------------------------------------
        dynasm! {
            self;
            .arch x64;

            mov BYTE [Rq(TEMP_B)], Rb(TEMP_A)
        }
    }

    fn sh(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u64) {
        // ------------------------------------
        // 1. Load the base address into TEMP_A
        // and physical memory pointer into TEMP_B
        // ------------------------------------
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.load_memory_ptr(TEMP_B);

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 2. Add the immediate to the base address
            // ------------------------------------
            add Rq(TEMP_A), imm as i32;

            // ------------------------------------
            // 3. Add the risc32 byte offset to the physical memory pointer
            // ------------------------------------
            add Rq(TEMP_B), Rq(TEMP_A)
        }

        // ------------------------------------
        // 4. Load the word from the RISC register into TEMP_A
        // ------------------------------------
        self.emit_risc_operand_load(rs2.into(), TEMP_A);

        // ------------------------------------
        // 5. Store the word into physical memory
        // ------------------------------------
        dynasm! {
            self;
            .arch x64;

            mov WORD [Rq(TEMP_B)], Rw(TEMP_A)
        }
    }

    fn sw(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u64) {
        // ------------------------------------
        // 1. Load the base address into TEMP_A
        // and physical memory pointer into TEMP_B
        // ------------------------------------
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.load_memory_ptr(TEMP_B);

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 2. Add the immediate to the base address
            // ------------------------------------
            add Rq(TEMP_A), imm as i32;

            // ------------------------------------
            // 3. Add the risc32 byte offset to the physical memory pointer
            // ------------------------------------
            add Rq(TEMP_B), Rq(TEMP_A)
        }

        // ------------------------------------
        // 4. Load the word from the RISC register into TEMP_A
        // ------------------------------------
        self.emit_risc_operand_load(rs2.into(), TEMP_A);

        // ------------------------------------
        // 5. Store the word into physical memory
        // ------------------------------------
        dynasm! {
            self;
            .arch x64;

            mov DWORD [Rq(TEMP_B)], Rd(TEMP_A)
        }
    }

    fn sd(&mut self, rs1: RiscRegister, rs2: RiscRegister, imm: u64) {
        // ------------------------------------
        // 1. Load the base address into TEMP_A
        // and physical memory pointer into TEMP_B
        // ------------------------------------
        self.emit_risc_operand_load(rs1.into(), TEMP_A);
        self.load_memory_ptr(TEMP_B);

        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 2. Add the immediate to the base address
            // ------------------------------------
            add Rq(TEMP_A), imm as i32;

            // ------------------------------------
            // 3. Add the risc32 byte offset to the physical memory pointer
            // ------------------------------------
            add Rq(TEMP_B), Rq(TEMP_A)
        }

        // ------------------------------------
        // 4. Load the word from the RISC register into TEMP_A
        // ------------------------------------
        self.emit_risc_operand_load(rs2.into(), TEMP_A);

        // ------------------------------------
        // 5. Store the word into physical memory
        // ------------------------------------
        dynasm! {
            self;
            .arch x64;

            mov QWORD [Rq(TEMP_B)], Rq(TEMP_A)
        }
    }
}

impl SystemInstructions for TranspilerBackend {
    fn ecall(&mut self) {
        // Load the JitContext pointer into the argument register.
        dynasm! {
            self;
            .arch x64;
            mov rdi, Rq(CONTEXT)
        };

        self.call_extern_fn_raw(self.ecall_handler as _);

        // The ecall returns a u64 in RAX.
        self.emit_risc_register_store(Rq::RAX as u8, RiscRegister::X5);

        // The ecall may have modified the PC, so we need to jump to the next instruction.
        self.jump_to_pc();
    }

    fn unimp(&mut self) {
        extern "C" fn unimp(ctx: *mut JitContext) {
            let ctx = unsafe { &mut *ctx };
            eprintln!("Unimplemented instruction at pc: {}", ctx.pc);
        }

        self.call_extern_fn(unimp);
    }
}
