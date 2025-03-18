use slop_multilinear::Point;
use sp1_recursion_compiler::ir::{Ext, Felt, SymbolicExt, SymbolicFelt};

use crate::CircuitConfig;

pub mod sumcheck;
pub mod witness;

pub(crate) trait IntoSymbolic<C: CircuitConfig> {
    type Output;

    fn as_symbolic(&self) -> Self::Output;
}

impl<C: CircuitConfig> IntoSymbolic<C> for Felt<C::F> {
    type Output = SymbolicFelt<C::F>;

    fn as_symbolic(&self) -> Self::Output {
        SymbolicFelt::from(*self)
    }
}

impl<C: CircuitConfig> IntoSymbolic<C> for Ext<C::F, C::EF> {
    type Output = SymbolicExt<C::F, C::EF>;

    fn as_symbolic(&self) -> Self::Output {
        SymbolicExt::from(*self)
    }
}

impl<C: CircuitConfig, T: IntoSymbolic<C>> IntoSymbolic<C> for Point<T> {
    type Output = Point<T::Output>;

    fn as_symbolic(&self) -> Self::Output {
        Point::from(self.values().as_slice().iter().map(|x| x.as_symbolic()).collect::<Vec<_>>())
    }
}

impl<C: CircuitConfig, T: IntoSymbolic<C>> IntoSymbolic<C> for Vec<T> {
    type Output = Vec<T::Output>;

    fn as_symbolic(&self) -> Self::Output {
        let mut ret = Vec::with_capacity(self.len());
        for x in self.iter() {
            ret.push(x.as_symbolic());
        }
        ret
    }
}
