use std::{
    borrow::Borrow,
    fmt::{Debug, Display},
};

use serde::{Deserialize, Serialize};
use sp1_primitives::consts::{WORD_BYTE_SIZE, WORD_SIZE};
use sp1_stark::{
    air::{AirInteraction, InteractionScope},
    Word,
};

use slop_algebra::{ExtensionField, Field};

use sp1_core_machine::{
    adapter::{
        register::{alu_type::ALUTypeReader, r_type::RTypeReader},
        state::CPUState,
    },
    memory::{MemoryAccessInShardCols, MemoryAccessInShardTimestamp},
    operations::{
        AddOperation, AddressOperation, BitwiseOperation, BitwiseU16Operation,
        IsEqualWordOperation, IsZeroOperation, IsZeroWordOperation, U16toU8Operation,
    },
};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum IrVar<F> {
    Public(usize),
    Preprocessed(usize),
    Main(usize),
    Constant(F),
    InputArg(usize),
    OutputArg(usize),
}

impl<F: Field> Display for IrVar<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IrVar::Public(i) => write!(f, "Public({i})"),
            IrVar::Preprocessed(i) => write!(f, "Preprocessed({i})"),
            IrVar::Main(i) => write!(f, "Main({i})"),
            IrVar::Constant(c) => write!(f, "{c}"),
            IrVar::InputArg(i) => write!(f, "Input({i})"),
            IrVar::OutputArg(i) => write!(f, "Output({i})"),
        }
    }
}

pub struct FuncCtx {
    input_idx: usize,
    output_idx: usize,
}

impl FuncCtx {
    pub fn new() -> Self {
        Self { input_idx: 0, output_idx: 0 }
    }
}

impl Default for FuncCtx {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExprRef<F> {
    IrVar(IrVar<F>),
    Expr(usize),
}

impl<F: Field> ExprRef<F> {
    /// An expression representing a variable from public inputs.
    pub fn public(index: usize) -> Self {
        ExprRef::IrVar(IrVar::Public(index))
    }

    /// An expression representing a variable from preprocessed trace.
    pub fn preprocessed(index: usize) -> Self {
        ExprRef::IrVar(IrVar::Preprocessed(index))
    }

    /// An expression representing a variable from main trace.
    pub fn main(index: usize) -> Self {
        ExprRef::IrVar(IrVar::Main(index))
    }

    /// An expression representing a constant value.
    pub fn constant(value: F) -> Self {
        ExprRef::IrVar(IrVar::Constant(value))
    }

    /// An expression representing a variable from input arguments.
    pub fn input_arg(ctx: &mut FuncCtx) -> Self {
        let index = ctx.input_idx;
        ctx.input_idx += 1;
        ExprRef::IrVar(IrVar::InputArg(index))
    }

    /// Get a struct with input arguments.
    ///
    /// Given a sized struct that can be flattened to a slice of `Self`, produce a new struct of
    /// this type where all the fields are replaced with input arguments.
    pub fn input_from_struct<T>(ctx: &mut FuncCtx) -> T
    where
        T: Copy,
        [Self]: Borrow<T>,
    {
        let size = std::mem::size_of::<T>() / std::mem::size_of::<Self>();
        let values = (0..size).map(|_| Self::input_arg(ctx)).collect::<Vec<_>>();
        let value_ref: &T = values.as_slice().borrow();
        *value_ref
    }

    /// An expression representing a variable from output arguments.
    pub fn output_arg(ctx: &mut FuncCtx) -> Self {
        let index = ctx.output_idx;
        ctx.output_idx += 1;
        ExprRef::IrVar(IrVar::OutputArg(index))
    }

    /// Get a struct with output arguments.
    ///
    /// Given a sized struct that can be flattened to a slice of `Self`, produce a new struct of
    /// this type where all the fields are replaced with output arguments.
    pub fn output_from_struct<T>(ctx: &mut FuncCtx) -> T
    where
        T: Copy,
        [Self]: Borrow<T>,
    {
        let size = std::mem::size_of::<T>() / std::mem::size_of::<Self>();
        let values = (0..size).map(|_| Self::output_arg(ctx)).collect::<Vec<_>>();
        let value_ref: &T = values.as_slice().borrow();
        *value_ref
    }
}

impl<F: Field> Display for ExprRef<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExprRef::IrVar(ir_var) => write!(f, "{}", ir_var),
            ExprRef::Expr(expr) => write!(f, "Expr({})", expr),
        }
    }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ExprExtRef<EF> {
    ExtConstant(EF),
    Expr(usize),
}

impl<EF: Field> ExprExtRef<EF> {
    /// An expression representing a variable from input arguments.
    pub fn input_arg(ctx: &mut FuncCtx) -> Self {
        let index = ctx.input_idx;
        ctx.input_idx += 1;
        ExprExtRef::Expr(index)
    }

    /// An expression representing a variable from output arguments.
    pub fn output_arg(ctx: &mut FuncCtx) -> Self {
        let index = ctx.output_idx;
        ctx.output_idx += 1;
        ExprExtRef::Expr(index)
    }
}

impl<EF: Field> Display for ExprExtRef<EF> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExprExtRef::ExtConstant(ext_constant) => write!(f, "{ext_constant}"),
            ExprExtRef::Expr(expr) => write!(f, "ExprExt({})", expr),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuncDecl<Expr, ExprExt> {
    pub name: String,
    pub input: Vec<Ty<Expr, ExprExt>>,
    pub output: Vec<Ty<Expr, ExprExt>>,
}

impl<Expr, ExprExt> FuncDecl<Expr, ExprExt> {
    pub fn new(name: &str, input: Vec<Ty<Expr, ExprExt>>, output: Vec<Ty<Expr, ExprExt>>) -> Self {
        Self { name: name.to_string(), input, output }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Func<Expr, ExprExt> {
    pub decl: FuncDecl<Expr, ExprExt>,
    pub body: Ast<Expr, ExprExt>,
}

impl<F: Field, EF: ExtensionField<F>> Display for Func<ExprRef<F>, ExprExtRef<EF>> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "fn {}(", self.decl.name)?;
        for (i, inp) in self.decl.input.iter().enumerate() {
            write!(f, "    {inp}")?;
            if i < self.decl.input.len() - 1 {
                writeln!(f, ",")?;
            }
        }
        write!(f, ")")?;
        if !self.decl.output.is_empty() {
            write!(f, "->")?;
            for (i, out) in self.decl.output.iter().enumerate() {
                write!(f, "{out}")?;
                if i < self.decl.output.len() - 1 {
                    write!(f, ", ")?;
                }
            }
        }
        writeln!(f, " {{")?;
        write!(f, "{}", self.body.to_string_pretty("   "))?;
        writeln!(f, "}}")
    }
}

/// A type in the IR.
///
/// Types can appear in function arguments as inputs and outputs, and in function declarations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Ty<Expr, ExprExt> {
    /// An arithmetic expression.
    Expr(Expr),
    /// An arithmetic expression over the extension field.
    ExprExt(ExprExt),
    /// A word in the base field.
    Word(Word<Expr>),
    /// An addition operation.
    AddOperation(AddOperation<Expr>),
    /// An address operation.
    AddressOperation(AddressOperation<Expr>),
    /// A conversion from a word to an array of words of size `WORD_SIZE`.
    U16toU8Operation(U16toU8Operation<Expr>),
    /// An array of words of size `WORD_SIZE`.
    ArrWordSize([Expr; WORD_SIZE]),
    /// An array of words of size `WORD_BYTE_SIZE`.
    ArrWordByteSize([Expr; WORD_BYTE_SIZE]),
    /// An is zero operation.
    IsZeroOperation(IsZeroOperation<Expr>),
    /// An is zero word operation.
    IsZeroWordOperation(IsZeroWordOperation<Expr>),
    /// An is equal word operation.
    IsEqualWordOperation(IsEqualWordOperation<Expr>),
    /// A bitwise operation.
    BitwiseOperation(BitwiseOperation<Expr>),
    /// A bitwise u16 operation.
    BitwiseU16Operation(BitwiseU16Operation<Expr>),
    /// An R-type reader operation.
    RTypeReader(RTypeReader<Expr>),
    /// An ALU-type reader operation.
    ALUTypeReader(ALUTypeReader<Expr>),
    /// A CPU state operation.
    CPUState(CPUState<Expr>),
    MemoryAccessInShardTimestamp(MemoryAccessInShardTimestamp<Expr>),
    MemoryAccessInShardCols(MemoryAccessInShardCols<Expr>),
}

impl<Expr, ExprExt> Display for Ty<Expr, ExprExt>
where
    Expr: Debug + Display,
    ExprExt: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ty::Expr(expr) => write!(f, "{expr}"),
            Ty::ExprExt(expr_ext) => write!(f, "{expr_ext}"),
            Ty::Word(word) => write!(f, "{word}"),
            Ty::AddOperation(add_operation) => write!(f, "{add_operation:?}"),
            Ty::AddressOperation(address_operation) => write!(f, "{address_operation:?}"),
            Ty::U16toU8Operation(u16to_u8_operation) => write!(f, "{u16to_u8_operation:?}"),
            Ty::ArrWordSize(arr) => write!(f, "{arr:?}"),
            Ty::ArrWordByteSize(arr) => write!(f, "{arr:?}"),
            Ty::IsZeroOperation(is_zero_operation) => write!(f, "{is_zero_operation:?}"),
            Ty::IsZeroWordOperation(is_zero_word_operation) => {
                write!(f, "{is_zero_word_operation:?}")
            }
            Ty::IsEqualWordOperation(is_equal_word_operation) => {
                write!(f, "{is_equal_word_operation:?}")
            }
            Ty::BitwiseOperation(bitwise_operation) => write!(f, "{bitwise_operation:?}"),
            Ty::BitwiseU16Operation(bitwise_u16_operation) => {
                write!(f, "{bitwise_u16_operation:?}")
            }
            Ty::RTypeReader(r_type_reader) => write!(f, "{r_type_reader:?}"),
            Ty::ALUTypeReader(alu_type_reader) => write!(f, "{alu_type_reader:?}"),
            Ty::CPUState(cpu_state) => write!(f, "{cpu_state:?}"),
            Ty::MemoryAccessInShardTimestamp(timestamp) => write!(f, "{timestamp:?}"),
            Ty::MemoryAccessInShardCols(cols) => write!(f, "{cols:?}"),
        }
    }
}

/// An operation in the IR.
///
/// Operations can appear in the AST, and are used to represent the program.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpExpr<Expr, ExprExt> {
    /// An assertion that an expression is zero.
    AssertZero(Expr),
    /// A send operation.
    Send(AirInteraction<Expr>, InteractionScope),
    /// A receive operation.
    Receive(AirInteraction<Expr>, InteractionScope),
    /// A function call.
    Call(FuncDecl<Expr, ExprExt>),
    /// A binary operation.
    BinOp(BinOp, Expr, Expr, Expr),
    /// A binary operation over the extension field.
    BinOpExt(BinOp, ExprExt, ExprExt, ExprExt),
    /// A binary operation over the base field and the extension field.
    BinOpBaseExt(BinOp, ExprExt, ExprExt, Expr),
    /// A negation operation.
    Neg(Expr, Expr),
    /// A negation operation over the extension field.
    NegExt(ExprExt, ExprExt),
    /// A conversion from the base field to the extension field.
    ExtFromBase(ExprExt, Expr),
    /// An assertion that an expression over the extension field is zero.
    AssertExtZero(ExprExt),
    /// An assignment operation.
    Assign(Expr, Expr),
}

pub fn write_interaction<Expr>(
    f: &mut std::fmt::Formatter<'_>,
    interaction: &AirInteraction<Expr>,
    scope: &InteractionScope,
) -> std::fmt::Result
where
    Expr: Display,
{
    write!(
        f,
        "kind: {}, scope: {scope}, multiplicity: {}, values: [",
        interaction.kind, interaction.multiplicity
    )?;
    for (i, value) in interaction.values.iter().enumerate() {
        write!(f, "{value}")?;
        if i < interaction.values.len() - 1 {
            write!(f, ", ")?;
        }
    }
    write!(f, "]")?;
    Ok(())
}

impl<F, EF> Display for OpExpr<ExprRef<F>, ExprExtRef<EF>>
where
    F: Field,
    EF: ExtensionField<F>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpExpr::AssertZero(x) => write!(f, "Assert({x} == 0)"),
            OpExpr::Send(interaction, scope) => {
                write!(f, "Send(")?;
                write_interaction(f, interaction, scope)?;
                write!(f, ")")?;
                Ok(())
            }
            OpExpr::Receive(interaction, scope) => {
                write!(f, "Receive(")?;
                write_interaction(f, interaction, scope)?;
                write!(f, ")")?;
                Ok(())
            }
            OpExpr::Assign(a, b) => write!(f, "{a} = {b}"),
            OpExpr::Call(func) => {
                if !func.output.is_empty() {
                    if func.output.len() > 1 {
                        write!(f, "(")?;
                    }
                    for out in func.output.iter() {
                        write!(f, "{out}")?;
                    }
                    if func.output.len() > 1 {
                        write!(f, ")")?;
                    }
                    write!(f, " = ")?;
                }
                write!(f, "{}(", func.name)?;
                for (i, inp) in func.input.iter().enumerate() {
                    write!(f, "{inp}")?;
                    if i < func.input.len() - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, ")")?;
                Ok(())
            }
            OpExpr::BinOp(op, a, b, c) => match op {
                BinOp::Add => write!(f, "{a} = {b} + {c}"),
                BinOp::Sub => write!(f, "{a} = {b} - {c}"),
                BinOp::Mul => write!(f, "{a} = {b} * {c}"),
            },
            OpExpr::BinOpExt(op, a, b, c) => match op {
                BinOp::Add => write!(f, "{a} = {b} + {c}"),
                BinOp::Sub => write!(f, "{a} = {b} - {c}"),
                BinOp::Mul => write!(f, "{a} = {b} * {c}"),
            },
            OpExpr::BinOpBaseExt(op, a, b, c) => match op {
                BinOp::Add => write!(f, "{a} = {b} + {c}"),
                BinOp::Sub => write!(f, "{a} = {b} - {c}"),
                BinOp::Mul => write!(f, "{a} = {b} * {c}"),
            },
            OpExpr::Neg(a, b) => write!(f, "{a} = -{b}"),
            OpExpr::NegExt(a, b) => write!(f, "{a} = -{b}"),
            OpExpr::ExtFromBase(a, b) => write!(f, "{a} = {b}"),
            OpExpr::AssertExtZero(a) => write!(f, "Assert({a} == 0)"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ast<Expr, ExprExt> {
    assignments: Vec<usize>,
    ext_assignments: Vec<usize>,
    operations: Vec<OpExpr<Expr, ExprExt>>,
}

impl<F: Field, EF: ExtensionField<F>> Ast<ExprRef<F>, ExprExtRef<EF>> {
    pub fn new() -> Self {
        Self { assignments: vec![], ext_assignments: vec![], operations: vec![] }
    }

    pub fn alloc(&mut self) -> ExprRef<F> {
        let id = self.assignments.len();
        self.assignments.push(self.operations.len());
        ExprRef::Expr(id)
    }

    pub fn alloc_array<const N: usize>(&mut self) -> [ExprRef<F>; N] {
        core::array::from_fn(|_| self.alloc())
    }

    pub fn assign(&mut self, a: ExprRef<F>, b: ExprRef<F>) {
        let op = OpExpr::Assign(a, b);
        self.operations.push(op);
    }

    pub fn alloc_ext(&mut self) -> ExprExtRef<EF> {
        let id = self.ext_assignments.len();
        self.ext_assignments.push(self.operations.len());
        ExprExtRef::Expr(id)
    }

    pub fn assert_zero(&mut self, x: ExprRef<F>) {
        let op = OpExpr::AssertZero(x);
        self.operations.push(op);
    }

    pub fn assert_ext_zero(&mut self, x: ExprExtRef<EF>) {
        let op = OpExpr::AssertExtZero(x);
        self.operations.push(op);
    }

    pub fn bin_op(&mut self, op: BinOp, a: ExprRef<F>, b: ExprRef<F>) -> ExprRef<F> {
        let result = self.alloc();
        self.assignments.push(self.operations.len());
        let op = OpExpr::BinOp(op, result, a, b);
        self.operations.push(op);
        result
    }

    pub fn negate(&mut self, a: ExprRef<F>) -> ExprRef<F> {
        let result = self.alloc();
        let op = OpExpr::Neg(result, a);
        self.operations.push(op);
        result
    }

    pub fn bin_op_ext(
        &mut self,
        op: BinOp,
        a: ExprExtRef<EF>,
        b: ExprExtRef<EF>,
    ) -> ExprExtRef<EF> {
        let result = self.alloc_ext();
        self.ext_assignments.push(self.operations.len());
        let op = OpExpr::BinOpExt(op, result, a, b);
        self.operations.push(op);
        result
    }

    pub fn bin_op_base_ext(
        &mut self,
        op: BinOp,
        a: ExprExtRef<EF>,
        b: ExprRef<F>,
    ) -> ExprExtRef<EF> {
        let result = self.alloc_ext();
        self.ext_assignments.push(self.operations.len());
        let op = OpExpr::BinOpBaseExt(op, result, a, b);
        self.operations.push(op);
        result
    }

    pub fn neg_ext(&mut self, a: ExprExtRef<EF>) -> ExprExtRef<EF> {
        let result = self.alloc_ext();
        let op = OpExpr::NegExt(result, a);
        self.operations.push(op);
        result
    }

    pub fn ext_from_base(&mut self, a: ExprRef<F>) -> ExprExtRef<EF> {
        let result = self.alloc_ext();
        let op = OpExpr::ExtFromBase(result, a);
        self.operations.push(op);
        result
    }

    pub fn send(&mut self, message: AirInteraction<ExprRef<F>>, scope: InteractionScope) {
        let op = OpExpr::Send(message, scope);
        self.operations.push(op);
    }

    pub fn receive(&mut self, message: AirInteraction<ExprRef<F>>, scope: InteractionScope) {
        let op = OpExpr::Receive(message, scope);
        self.operations.push(op);
    }

    pub fn to_string_pretty(&self, prefix: &str) -> String {
        let mut s = String::new();
        for op in &self.operations {
            s.push_str(&format!("{prefix}{}\n", op));
        }
        s
    }

    pub fn add_operation(
        &mut self,
        a: Word<ExprRef<F>>,
        b: Word<ExprRef<F>>,
        cols: AddOperation<ExprRef<F>>,
        is_real: ExprRef<F>,
    ) {
        let func = FuncDecl {
            name: "AddOperation".to_string(),
            input: vec![Ty::Word(a), Ty::Word(b), Ty::AddOperation(cols), Ty::Expr(is_real)],
            output: vec![],
        };
        let op = OpExpr::Call(func);
        self.operations.push(op);
    }

    pub fn address_operation(
        &mut self,
        b: Word<ExprRef<F>>,
        c: Word<ExprRef<F>>,
        offset_bit0: ExprRef<F>,
        offset_bit1: ExprRef<F>,
        is_real: ExprRef<F>,
        cols: AddressOperation<ExprRef<F>>,
    ) -> ExprRef<F> {
        let output = self.alloc();
        let func = FuncDecl {
            name: "AddressOperation".to_string(),
            input: vec![
                Ty::Word(b),
                Ty::Word(c),
                Ty::Expr(offset_bit0),
                Ty::Expr(offset_bit1),
                Ty::Expr(is_real),
                Ty::AddressOperation(cols),
            ],
            output: vec![Ty::Expr(output)],
        };
        let op = OpExpr::Call(func);
        self.operations.push(op);
        output
    }

    pub fn u16_to_u8_operation_safe(
        &mut self,
        u16_values: [ExprRef<F>; WORD_SIZE],
        cols: U16toU8Operation<ExprRef<F>>,
        is_real: ExprRef<F>,
    ) -> [ExprRef<F>; WORD_BYTE_SIZE] {
        let result = self.alloc_array();
        let func = FuncDecl {
            name: "U16toU8OperationSafe".to_string(),
            input: vec![Ty::ArrWordSize(u16_values), Ty::Expr(is_real), Ty::U16toU8Operation(cols)],
            output: vec![Ty::ArrWordByteSize(result)],
        };
        let op = OpExpr::Call(func);
        self.operations.push(op);
        result
    }

    pub fn u16_to_u8_operation_unsafe(
        &mut self,
        u16_values: [ExprRef<F>; WORD_SIZE],
        cols: U16toU8Operation<ExprRef<F>>,
    ) -> [ExprRef<F>; WORD_BYTE_SIZE] {
        let result = self.alloc_array();
        let func = FuncDecl {
            name: "U16toU8OperationUnsafe".to_string(),
            input: vec![Ty::ArrWordSize(u16_values), Ty::U16toU8Operation(cols)],
            output: vec![Ty::ArrWordByteSize(result)],
        };
        let op = OpExpr::Call(func);
        self.operations.push(op);
        result
    }

    pub fn is_zero_operation(
        &mut self,
        a: ExprRef<F>,
        cols: IsZeroOperation<ExprRef<F>>,
        is_real: ExprRef<F>,
    ) {
        let func = FuncDecl {
            name: "IsZeroOperation".to_string(),
            input: vec![Ty::Expr(a), Ty::Expr(is_real), Ty::IsZeroOperation(cols)],
            output: vec![],
        };
        let op = OpExpr::Call(func);
        self.operations.push(op);
    }

    pub fn is_zero_word_operation(
        &mut self,
        a: Word<ExprRef<F>>,
        cols: IsZeroWordOperation<ExprRef<F>>,
        is_real: ExprRef<F>,
    ) {
        let func = FuncDecl {
            name: "IsZeroWordOperation".to_string(),
            input: vec![Ty::Word(a), Ty::IsZeroWordOperation(cols), Ty::Expr(is_real)],
            output: vec![],
        };
        let op = OpExpr::Call(func);
        self.operations.push(op);
    }

    pub fn is_equal_word_operation(
        &mut self,
        a: Word<ExprRef<F>>,
        b: Word<ExprRef<F>>,
        cols: IsEqualWordOperation<ExprRef<F>>,
        is_real: ExprRef<F>,
    ) {
        let func = FuncDecl {
            name: "IsEqualWordOperation".to_string(),
            input: vec![
                Ty::Word(a),
                Ty::Word(b),
                Ty::IsEqualWordOperation(cols),
                Ty::Expr(is_real),
            ],
            output: vec![],
        };
        let op = OpExpr::Call(func);
        self.operations.push(op);
    }

    pub fn bitwise_operation(
        &mut self,
        a: [ExprRef<F>; WORD_BYTE_SIZE],
        b: [ExprRef<F>; WORD_BYTE_SIZE],
        cols: BitwiseOperation<ExprRef<F>>,
        opcode: ExprRef<F>,
        is_real: ExprRef<F>,
    ) {
        let func = FuncDecl {
            name: "BitwiseOperation".to_string(),
            input: vec![
                Ty::ArrWordByteSize(a),
                Ty::ArrWordByteSize(b),
                Ty::BitwiseOperation(cols),
                Ty::Expr(opcode),
                Ty::Expr(is_real),
            ],
            output: vec![],
        };
        let op = OpExpr::Call(func);
        self.operations.push(op);
    }

    pub fn bitwise_u16_operation(
        &mut self,
        b: Word<ExprRef<F>>,
        c: Word<ExprRef<F>>,
        cols: BitwiseU16Operation<ExprRef<F>>,
        opcode: ExprRef<F>,
        is_real: ExprRef<F>,
    ) -> Word<ExprRef<F>> {
        let output = Word(core::array::from_fn(|_| self.alloc()));
        let func = FuncDecl {
            name: "BitwiseU16Operation".to_string(),
            input: vec![
                Ty::Word(b),
                Ty::Word(c),
                Ty::BitwiseU16Operation(cols),
                Ty::Expr(opcode),
                Ty::Expr(is_real),
            ],
            output: vec![Ty::Word(output)],
        };
        let op = OpExpr::Call(func);
        self.operations.push(op);
        output
    }

    #[allow(clippy::too_many_arguments)]
    pub fn r_type_reader(
        &mut self,
        clk_high: ExprRef<F>,
        clk_low: ExprRef<F>,
        pc: ExprRef<F>,
        opcode: ExprRef<F>,
        op_a_write_value: Word<ExprRef<F>>,
        cols: RTypeReader<ExprRef<F>>,
        is_real: ExprRef<F>,
    ) {
        let func = FuncDecl {
            name: "RTypeReader".to_string(),
            input: vec![
                Ty::Expr(clk_high),
                Ty::Expr(clk_low),
                Ty::Expr(pc),
                Ty::Expr(opcode),
                Ty::Word(op_a_write_value),
                Ty::RTypeReader(cols),
                Ty::Expr(is_real),
            ],
            output: vec![],
        };
        let op = OpExpr::Call(func);
        self.operations.push(op);
    }

    pub fn r_type_reader_immutable(
        &mut self,
        clk_high: ExprRef<F>,
        clk_low: ExprRef<F>,
        pc: ExprRef<F>,
        opcode: ExprRef<F>,
        cols: RTypeReader<ExprRef<F>>,
        is_real: ExprRef<F>,
    ) {
        let func = FuncDecl {
            name: "RTypeReaderImmutable".to_string(),
            input: vec![
                Ty::Expr(clk_high),
                Ty::Expr(clk_low),
                Ty::Expr(pc),
                Ty::Expr(opcode),
                Ty::RTypeReader(cols),
                Ty::Expr(is_real),
            ],
            output: vec![],
        };
        let op = OpExpr::Call(func);
        self.operations.push(op);
    }

    pub fn cpu_state(
        &mut self,
        cols: CPUState<ExprRef<F>>,
        next_pc: ExprRef<F>,
        clk_increment: ExprRef<F>,
        is_real: ExprRef<F>,
    ) {
        let func = FuncDecl {
            name: "CPUState".to_string(),
            input: vec![
                Ty::CPUState(cols),
                Ty::Expr(next_pc),
                Ty::Expr(clk_increment),
                Ty::Expr(is_real),
            ],
            output: vec![],
        };
        let op = OpExpr::Call(func);
        self.operations.push(op);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn alu_type_reader(
        &mut self,
        clk_high: ExprRef<F>,
        clk_low: ExprRef<F>,
        pc: ExprRef<F>,
        opcode: ExprRef<F>,
        op_a_write_value: Word<ExprRef<F>>,
        cols: ALUTypeReader<ExprRef<F>>,
        is_real: ExprRef<F>,
    ) {
        let func = FuncDecl {
            name: "ALUTypeReader".to_string(),
            input: vec![
                Ty::Expr(clk_high),
                Ty::Expr(clk_low),
                Ty::Expr(pc),
                Ty::Expr(opcode),
                Ty::Word(op_a_write_value),
                Ty::ALUTypeReader(cols),
                Ty::Expr(is_real),
            ],
            output: vec![],
        };
        let op = OpExpr::Call(func);
        self.operations.push(op);
    }
}

impl<F: Field, EF: ExtensionField<F>> Default for Ast<ExprRef<F>, ExprExtRef<EF>> {
    fn default() -> Self {
        Self::new()
    }
}
