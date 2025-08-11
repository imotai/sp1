#![allow(clippy::fn_to_numeric_cast)]

use super::{ScratchRegisterX86, TranspilerBackend, CONTEXT};
use crate::{ALUBackend, ExternFn, JitFunction};
use dynasmrt::{dynasm, DynasmApi, DynasmLabelApi};
use std::io;

impl ALUBackend for TranspilerBackend {
    type ScratchRegister = ScratchRegisterX86;

    fn finalize(mut self) -> io::Result<JitFunction> {
        self.epilogue();

        let code = self.inner.finalize().expect("failed to finalize x86 backend");

        debug_assert!(code.size() > 0, "Got empty x86 code buffer");

        JitFunction::new(
            code,
            self.jump_table,
            self.memory_size,
            self.trace_buf_size,
            self.pc_start,
        )
    }

    fn call_extern_fn(&mut self, fn_ptr: ExternFn) {
        // Load the JitContext pointer into the argument register.
        dynasm! {
            self;
            .arch x64;
            mov rdi, Rq(CONTEXT)
        };

        self.call_extern_fn_raw(fn_ptr as _);
    }

    fn add(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        // lhs <- lhs + rhs
        dynasm! {
            self;
            .arch x64;
            add Rd(lhs), Rd(rhs)
        }
    }

    fn mul(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        // lhs <- lhs * rhs
        dynasm! {
            self;
            .arch x64;
            imul Rd(lhs), Rd(rhs)
        }
    }

    fn and(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        // lhs <- lhs & rhs
        dynasm! {
            self;
            .arch x64;
            and Rd(lhs), Rd(rhs)
        }
    }

    fn or(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        // lhs <- lhs | rhs
        dynasm! {
            self;
            .arch x64;
            or Rd(lhs), Rd(rhs)
        }
    }

    fn xor(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        // lhs <- lhs ^ rhs
        dynasm! {
            self;
            .arch x64;
            xor Rd(lhs), Rd(rhs)
        }
    }

    fn div(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        // X86 uses [EAX::EDX] for the divide operation.
        // So we need to sign extend the lhs into EDX.
        //
        // The quotient is stored in EAX, and the remainder is stored in EDX.
        //
        // We can just write the quotient back into lhs, and the remainder is discarded.
        dynasm! {
            self;
            .arch x64;

            // ------------------------------------
            // 1. Skip fault on div-by-zero
            // ------------------------------------
            test Rd(rhs), Rd(rhs);        // ZF=1 if rhs == 0
            jz   >div_by_zero;

            // ------------------------------------
            // 2. Perform signed divide
            // ------------------------------------
            mov  eax, Rd(lhs);            // dividend → EAX
            cdq;                          // sign-extend EAX into EDX
            idiv Rd(rhs);                 // quotient → EAX, remainder → EDX
            mov  Rd(lhs), eax;            // write quotient back into lhs
            jmp >done;

            // ------------------------------------
            // 3. if rhs == 0
            // ------------------------------------
            div_by_zero:;
            xor  Rd(lhs), Rd(lhs);        // lhs = 0

            // ------------------------------------
            // Merge branch
            // ------------------------------------
            done:
        }
    }

    fn divu(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        // lhs <- lhs / rhs   (unsigned 32-bit; 0 if rhs == 0)
        // clobbers: EAX, EDX
        dynasm! {
            self;
            .arch x64;

            // ----- skip fault on div-by-zero -----
            test Rd(rhs), Rd(rhs);         // ZF = 1 when rhs == 0
            jz   >div_by_zero;

            // ----- perform unsigned divide -----
            mov  eax, Rd(lhs);             // dividend → EAX
            xor  edx, edx;                 // zero-extend: EDX = 0
            div  Rd(rhs);                  // unsigned divide: EDX:EAX / rhs
            mov  Rd(lhs), eax;             // quotient → lhs
            jmp  >done;

            // ----- rhs == 0 -----
            div_by_zero:;
            xor  Rd(lhs), Rd(lhs);         // lhs = 0

            done:
        }
    }

    fn mulh(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        dynasm! {
            self;
            .arch x64;

            mov  eax, Rd(lhs);      // EAX = lhs (signed)
            imul Rd(rhs);           // signed 32×32 → 64; high → EDX
            mov  Rd(lhs), edx       // lhs = high 32 bits
        }
    }

    fn mulhu(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        dynasm! {
            self;
            .arch x64;

            mov  eax, Rd(lhs);      // EAX = lhs (unsigned)
            mul  Rd(rhs);           // unsigned 32×32 → 64; high → EDX
            mov  Rd(lhs), edx       // lhs = high 32 bits
        }
    }

    fn mulhsu(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        dynasm! {
            self;
            .arch x64;

            // ──────────────────────────────────────────────────────────────
            // 1. Move the **signed** left-hand operand (`lhs`) into EAX.
            //    ✦ The x86-64 `mul` instruction always uses EAX as its implicit
            //      32-bit source operand, so we must place `lhs` there first.
            // ──────────────────────────────────────────────────────────────
            mov eax, Rd(lhs);

            // ──────────────────────────────────────────────────────────────
            // 2. Preserve a second copy of `lhs` in ECX.
            //    ✦ The upcoming `mul` clobbers both EAX and EDX, erasing any
            //      trace of the original sign.  We save `lhs` in ECX so that
            //      we can later decide whether the fix-up for a *negative*
            //      multiplicand is required.
            // ──────────────────────────────────────────────────────────────
            mov ecx, eax;

            // ──────────────────────────────────────────────────────────────
            // 3. Unsigned 32×32-bit multiply:
            //    mul Rd(rhs)
            //    ✦ Computes  EDX:EAX = (unsigned)EAX × (unsigned)rhs.
            //      The high 32 bits of the 64-bit product land in EDX.
            // ──────────────────────────────────────────────────────────────
            mul Rd(rhs);

            // ──────────────────────────────────────────────────────────────
            // 4. Determine whether the *original* `lhs` was negative.
            //    ✦ `test ecx, ecx` sets the sign flag from ECX (the saved `lhs`).
            //    ✦ If the sign flag is *clear* (`lhs` ≥ 0), we can skip the
            //      correction step because the high half already matches the
            //      semantics of the RISC-V MULHSU instruction.
            // ──────────────────────────────────────────────────────────────
            test ecx, ecx;
            jns >store_high;          // Jump if `lhs` was non-negative.

            // ──────────────────────────────────────────────────────────────
            // 5. Fix-up for negative `lhs` (signed × unsigned semantics):
            //    ✦ For a negative multiplicand, the unsigned `mul` delivered a
            //      product that is *2³²* too large in the high word.  Subtracting
            //      `rhs` from EDX removes that excess and yields the correct
            //      signed-high result.
            // ──────────────────────────────────────────────────────────────
            sub edx, Rd(rhs);

            // ──────────────────────────────────────────────────────────────
            // 6. Write the corrected high 32 bits back to the destination
            //    RISC register specified by `lhs`.
            // ──────────────────────────────────────────────────────────────
            store_high:;
            mov Rd(lhs), edx
        }
    }

    /// Signed remainder: `lhs = lhs % rhs`  
    /// *RISC-V rule*: if `rhs == 0`, the result must be **0** (no fault).
    fn rem(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        dynasm! {
            self;
            .arch x64;

            // ──────────────────────────────────────────────────────────────
            // 0. Guard: if divisor is 0, skip the IDIV and zero the result
            // ──────────────────────────────────────────────────────────────
            test Rd(rhs), Rd(rhs);        // ZF = 1  ⇒  rhs == 0
            jz   >by_zero;                // jump to fix-up path

            // ──────────────────────────────────────────────────────────────
            // 1. Prepare the **signed** 64-bit dividend in EDX:EAX
            //    -------------------------------------------------
            //    • EAX ← low 32 bits of lhs
            //    • CDQ  sign-extends EAX into EDX
            //      → EDX:EAX now holds the two’s-complement 64-bit value a
            // ──────────────────────────────────────────────────────────────
            mov  eax, Rd(lhs);            // EAX = a  (signed 32-bit)
            cdq;                          // EDX = sign(a)

            // ──────────────────────────────────────────────────────────────
            // 2. Signed divide:          a  /  b
            //    -------------------------------------------------
            //    • idiv r/m32   performs  (EDX:EAX) ÷ rhs
            //      – Quotient  → EAX   (ignored)
            //      – Remainder → EDX   (what RISC-V REM returns)
            // ──────────────────────────────────────────────────────────────
            idiv Rd(rhs);                 // signed divide

            // ──────────────────────────────────────────────────────────────
            // 3. Write the remainder (EDX) back to the destination register
            // ──────────────────────────────────────────────────────────────
            mov  Rd(lhs), edx;            // lhs = remainder
            jmp  >done;

            // ──────────────────────────────────────────────────────────────
            // Divisor == 0  →  result must be 0 (no fault)
            // ──────────────────────────────────────────────────────────────
            by_zero:;
            xor  Rd(lhs), Rd(lhs);        // lhs = 0

            done:
        }
    }

    fn remu(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        dynasm! {
            self;
            .arch x64;

            // ──────────────────────────────────────────────────────────────
            // 0. Guard against /0 → result = 0
            // ──────────────────────────────────────────────────────────────
            test Rd(rhs), Rd(rhs);
            jz   >by_zero;

            // ──────────────────────────────────────────────────────────────
            // 1. Prepare the **unsigned** 64-bit dividend in EDX:EAX
            //    -------------------------------------------------
            //    • Zero-extend lhs into EDX:EAX.
            // ──────────────────────────────────────────────────────────────
            mov  eax, Rd(lhs);
            xor  edx, edx;

            // ──────────────────────────────────────────────────────────────
            // 2. Unsigned divide:       a  /  b
            //    -------------------------------------------------
            //    • div r/m32   performs  (EDX:EAX) ÷ rhs
            //      – Quotient  → EAX   (unused)
            //      – Remainder → EDX   (what RISC-V REMU wants)
            // ──────────────────────────────────────────────────────────────
            div  Rd(rhs);

            // ──────────────────────────────────────────────────────────────
            // 3. Write the remainder back to the destination register.
            // ──────────────────────────────────────────────────────────────
            mov  Rd(lhs), edx;
            jmp  >done;

            // ──────────────────────────────────────────────────────────────
            // Divisor == 0  →  result must be 0 (no fault)
            // ──────────────────────────────────────────────────────────────
            by_zero:;
            xor  Rd(lhs), Rd(lhs);

            done:
        }
    }

    fn sll(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        // We only can use the lower 5 bits for the shift count.
        // In RISC-V, this is also true!
        //
        // CL is an alias for the lower byte of ECX.
        dynasm! {
            self;
            .arch x64;

            // ──────────────────────────────────────────────────────────────
            // 1. Move the shift count (lower 5 bits, RISC-V spec) into CL.
            //    x86 uses only the low 5 bits for 32-bit shifts as well, so
            //    the masking is implicit—no extra AND is necessary.
            // ──────────────────────────────────────────────────────────────
            mov ecx, Rd(rhs);        // CL = rhs

            // ──────────────────────────────────────────────────────────────
            // 2. Logical left shift: Rd(lhs) ← Rd(lhs) << (CL & 0b1_1111)
            //    `shl r/m32, cl` preserves the semantics required by
            //    RISC-V SLL because both architectures ignore bits ≥ 5.
            // ──────────────────────────────────────────────────────────────
            shl Rd(lhs), cl         // variable-count shift
        }
    }

    fn sra(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        dynasm! {
            self;
            .arch x64;

            // ──────────────────────────────────────────────────────────────
            // 1. Put the variable shift count into CL.
            //    • Only the low 5 bits are used for 32-bit operands,
            //      which matches the RISC-V spec for RV32.
            // ──────────────────────────────────────────────────────────────
            mov  ecx, Rd(rhs);        // CL ← rhs

            // ──────────────────────────────────────────────────────────────
            // 2. Arithmetic right shift:
            //      Rd(lhs) ← (signed)Rd(lhs) >> (CL & 0x1F)
            //    • `sar` replicates the sign bit as it shifts, so
            //      negative values stay negative after the operation.
            // ──────────────────────────────────────────────────────────────
            sar  Rd(lhs), cl         // variable-count shift
        }
    }

    fn srl(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        dynasm! {
            self;
            .arch x64;

            // ──────────────────────────────────────────────────────────────
            // 1. Load shift count into CL (same reasoning as above).
            // ──────────────────────────────────────────────────────────────
            mov  ecx, Rd(rhs);        // CL ← rhs

            // ──────────────────────────────────────────────────────────────
            // 2. Logical right shift:
            //      Rd(lhs) ← (unsigned)Rd(lhs) >> (CL & 0x1F)
            //    • `shr` always inserts zeros from the left, regardless
            //      of the operand's sign.
            // ──────────────────────────────────────────────────────────────
            shr  Rd(lhs), cl
        }
    }

    fn slt(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        dynasm! {
            self;
            .arch x64;

            cmp  Rd(lhs), Rd(rhs);

            // ──────────────────────────────────────────────────────────────
            // 2. setl  r/m8
            //    • Writes   1  to the target byte if  (SF ≠ OF)
            //      which is the signed “less than” condition.
            //    • We store straight into the low-byte of lhs —
            //      dynasm’s `Rb()` gives us that alias.
            // ──────────────────────────────────────────────────────────────
            setl Rb(lhs);               // byte = 1 if lhs < rhs (signed)

            // ──────────────────────────────────────────────────────────────
            // 3. Zero-extend that byte back to a full 32-bit register so
            //    that the RISC register ends up with 0x0000_0000 or 0x0000_0001.
            // ──────────────────────────────────────────────────────────────
            movzx Rd(lhs), Rb(lhs)     // Rd(lhs) = 0 or 1
        }
    }

    fn sltu(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        dynasm! {
            self;
            .arch x64;

            cmp  Rd(lhs), Rd(rhs);

            // ------------------------------------
            // `setb` (“below”) checks the Carry Flag (CF):
            //   CF = 1  iff  lhs < rhs  in an *unsigned* sense.
            // ------------------------------------
            setb Rb(lhs);

            // ------------------------------------
            // Zero-extend to 32 bits (0 or 1).
            // ------------------------------------
            movzx Rd(lhs), Rb(lhs)
        }
    }

    fn sub(&mut self, lhs: Self::ScratchRegister, rhs: Self::ScratchRegister) {
        dynasm! {
            self;
            .arch x64;

            sub Rd(lhs), Rd(rhs)
        }
    }
}
