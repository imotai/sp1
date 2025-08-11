/// A macro to implement ALU operations for the riscv transpiler.
///
/// All operations are binary and aceept two operands, possibly an immediate or register.
#[macro_export]
macro_rules! impl_risc_alu {
    ($($name:ident),*) => {
        paste::paste! {
             $(fn [<risc_ $name>](&mut self, rs1: RiscOperand, rs2: RiscOperand, rd: RiscRegister) {
                self.load_riscv_operand(rs1, Self::ScratchRegister::A);
                self.load_riscv_operand(rs2, Self::ScratchRegister::B);

                self.$name(Self::ScratchRegister::A, Self::ScratchRegister::B);
                self.store_riscv_register(Self::ScratchRegister::A, rd);
            })*
        }
    };
}
