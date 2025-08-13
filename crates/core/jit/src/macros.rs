/// A macro to implement ALU operations for the riscv transpiler.
///
/// All operations are binary and accept two operands, possibly an immediate or register.
#[macro_export]
macro_rules! impl_risc_alu {
    ($self:expr, $rd:expr, $rs1:expr, $rs2:expr, $temp_a:expr, $temp_b:expr, $code:block) => {{
        $self.emit_risc_operand_load($rs1, $temp_a);
        $self.emit_risc_operand_load($rs2, $temp_b);
        $code
        $self.emit_risc_register_store($temp_a, $rd);
    }};
}
