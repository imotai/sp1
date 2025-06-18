use std::ops::{Add, Mul, MulAssign, Sub};

use p3_air::AirBuilder;
use p3_field::{AbstractField, ExtensionField, Field};
use p3_uni_stark::SymbolicAirBuilder;
use sp1_stark::{
    air::SP1AirBuilder, ConstraintSumcheckFolder, GenericVerifierConstraintFolder,
    InteractionBuilder,
};

use crate::{
    adapter::{
        register::{
            alu_type::ALUTypeReader,
            r_type::{RTypeReader, RTypeReaderImmutable},
        },
        state::CPUState,
    },
    operations::{
        AddOperation, BitwiseOperation, BitwiseU16Operation, IsEqualWordOperation, IsZeroOperation,
        IsZeroWordOperation, U16toU8OperationSafe, U16toU8OperationUnsafe,
    },
};

type F<AB> = <AB as AirBuilder>::F;

pub trait SP1CoreOperationBuilder:
    SP1AirBuilder
    + SP1OperationBuilder<AddOperation<F<Self>>>
    + SP1OperationBuilder<U16toU8OperationSafe>
    + SP1OperationBuilder<U16toU8OperationUnsafe>
    + SP1OperationBuilder<IsZeroOperation<F<Self>>>
    + SP1OperationBuilder<IsZeroWordOperation<F<Self>>>
    + SP1OperationBuilder<IsEqualWordOperation<F<Self>>>
    + SP1OperationBuilder<BitwiseOperation<F<Self>>>
    + SP1OperationBuilder<BitwiseU16Operation<F<Self>>>
    + SP1OperationBuilder<RTypeReader<F<Self>>>
    + SP1OperationBuilder<RTypeReaderImmutable>
    + SP1OperationBuilder<ALUTypeReader<F<Self>>>
    + SP1OperationBuilder<CPUState<F<Self>>>
{
}

impl<AB> SP1CoreOperationBuilder for AB where
    AB: SP1AirBuilder
        + SP1OperationBuilder<AddOperation<F<Self>>>
        + SP1OperationBuilder<U16toU8OperationSafe>
        + SP1OperationBuilder<U16toU8OperationUnsafe>
        + SP1OperationBuilder<IsZeroOperation<F<Self>>>
        + SP1OperationBuilder<IsZeroWordOperation<F<Self>>>
        + SP1OperationBuilder<IsEqualWordOperation<F<Self>>>
        + SP1OperationBuilder<BitwiseOperation<F<Self>>>
        + SP1OperationBuilder<BitwiseU16Operation<F<Self>>>
        + SP1OperationBuilder<RTypeReader<F<Self>>>
        + SP1OperationBuilder<RTypeReaderImmutable>
        + SP1OperationBuilder<ALUTypeReader<F<Self>>>
        + SP1OperationBuilder<CPUState<F<Self>>>
{
}

/// A trait for operations that can be lowered to constraints in an AIR.
pub trait SP1Operation<AB: SP1AirBuilder> {
    /// The input arguments to the operation.
    type Input;

    /// The output of the operation.
    type Output;

    /// The underlying constraints corresponding to the operation.
    fn lower(builder: &mut AB, input: Self::Input) -> Self::Output;

    /// Evaluate the operation on the given inputs.
    fn eval(builder: &mut AB, input: Self::Input) -> Self::Output
    where
        Self: Sized,
        AB: SP1OperationBuilder<Self>,
    {
        builder.eval_operation(input)
    }
}

/// A trait for handling an SP1 operation.
///
/// This trait enables an AIR builder to evaluate an operation in a specific way, such as emitting
/// code or changing state while keeping the semantic meaning of the operation being separated from
/// arbitrary constraints.
pub trait SP1OperationBuilder<O: SP1Operation<Self>>: SP1AirBuilder {
    fn eval_operation(&mut self, input: O::Input) -> O::Output;
}

/// A trait for demarking a builder that lowers every operation to it's underlying constraints.
pub trait TrivialOperationBuilder: SP1AirBuilder {}

impl<AB: TrivialOperationBuilder, O: SP1Operation<AB>> SP1OperationBuilder<O> for AB {
    fn eval_operation(&mut self, input: O::Input) -> O::Output {
        O::lower(self, input)
    }
}

impl<F: Field> TrivialOperationBuilder for InteractionBuilder<F> {}

impl<F: Field> TrivialOperationBuilder for SymbolicAirBuilder<F> {}

impl<
        'a,
        F: Field,
        K: Field + From<F> + Add<F, Output = K> + Sub<F, Output = K> + Mul<F, Output = K>,
        EF: Field + Mul<K, Output = EF>,
    > TrivialOperationBuilder for ConstraintSumcheckFolder<'a, F, K, EF>
{
}

impl<'a, F, EF, PubVar, Var, Expr> TrivialOperationBuilder
    for GenericVerifierConstraintFolder<'a, F, EF, PubVar, Var, Expr>
where
    F: Field,
    EF: ExtensionField<F>,
    Expr: AbstractField<F = EF>
        + From<F>
        + Add<Var, Output = Expr>
        + Add<F, Output = Expr>
        + Sub<Var, Output = Expr>
        + Sub<F, Output = Expr>
        + Mul<Var, Output = Expr>
        + Mul<F, Output = Expr>
        + MulAssign<EF>,
    Var: Into<Expr>
        + Copy
        + Add<F, Output = Expr>
        + Add<Var, Output = Expr>
        + Add<Expr, Output = Expr>
        + Sub<F, Output = Expr>
        + Sub<Var, Output = Expr>
        + Sub<Expr, Output = Expr>
        + Mul<F, Output = Expr>
        + Mul<Var, Output = Expr>
        + Mul<Expr, Output = Expr>
        + Send
        + Sync,
    PubVar: Into<Expr> + Copy,
{
}
