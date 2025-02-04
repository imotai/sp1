use rayon::prelude::*;
use std::ops::{Add, AddAssign};

use slop_alloc::{Backend, CpuBackend};

use crate::Tensor;

pub trait AddBackend<T, U>: Backend {
    type AddOutput;
    fn add_into(
        lhs: &Tensor<T, Self>,
        rhs: &Tensor<U, Self>,
        dst: &mut Tensor<Self::AddOutput, Self>,
    );

    fn add(lhs: &Tensor<T, Self>, rhs: &Tensor<U, Self>) -> Tensor<Self::AddOutput, Self>;
}

pub trait AddAssignBackend<T, U>: Backend {
    fn add_assign(lhs: &mut Tensor<T, Self>, rhs: &Tensor<U, Self>);
}

impl<'a, T, U, A: AddBackend<T, U>> Add<&'a Tensor<U, A>> for &'a Tensor<T, A> {
    type Output = Tensor<A::AddOutput, A>;

    #[inline]
    fn add(self, rhs: &Tensor<U, A>) -> Self::Output {
        A::add(self, rhs)
    }
}

impl<'a, T, U, A: AddBackend<T, U>> Add<Tensor<U, A>> for &'a Tensor<T, A> {
    type Output = Tensor<A::AddOutput, A>;

    #[inline]
    fn add(self, rhs: Tensor<U, A>) -> Self::Output {
        self + &rhs
    }
}

impl<'a, T, U, A: AddBackend<T, U>> Add<&'a Tensor<U, A>> for Tensor<T, A> {
    type Output = Tensor<A::AddOutput, A>;

    #[inline]
    fn add(self, rhs: &Tensor<U, A>) -> Self::Output {
        &self + rhs
    }
}

impl<T, U, A: AddBackend<T, U>> Add<Tensor<U, A>> for Tensor<T, A> {
    type Output = Tensor<A::AddOutput, A>;

    #[inline]
    fn add(self, rhs: Tensor<U, A>) -> Self::Output {
        self + &rhs
    }
}

impl<'a, T, U, A: AddAssignBackend<T, U>> AddAssign<&'a Tensor<U, A>> for Tensor<T, A> {
    #[inline]
    fn add_assign(&mut self, rhs: &Tensor<U, A>) {
        assert_eq!(self.sizes(), rhs.sizes());
        A::add_assign(self, rhs);
    }
}

impl<T, U, A: AddAssignBackend<T, U>> AddAssign<Tensor<U, A>> for Tensor<T, A> {
    #[inline]
    fn add_assign(&mut self, rhs: Tensor<U, A>) {
        *self += &rhs;
    }
}

impl<T, U> AddBackend<T, U> for CpuBackend
where
    T: Clone + Add<U> + Send + Sync,
    U: Clone + Send + Sync,
    <T as Add<U>>::Output: Send + Sync,
{
    type AddOutput = <T as Add<U>>::Output;

    fn add_into(
        lhs: &Tensor<T, Self>,
        rhs: &Tensor<U, Self>,
        dst: &mut Tensor<Self::AddOutput, Self>,
    ) {
        let dst = dst.as_mut_buffer();
        let lhs = lhs.as_buffer();
        let rhs = rhs.as_buffer();
        dst.spare_capacity_mut().par_iter_mut().zip(lhs.par_iter().zip(rhs.par_iter())).for_each(
            |(d, (l, r))| {
                d.write(l.clone() + r.clone());
            },
        );
        unsafe {
            dst.assume_init();
        }
    }

    fn add(lhs: &Tensor<T, Self>, rhs: &Tensor<U, Self>) -> Tensor<Self::AddOutput, Self> {
        let mut dst = Tensor::<Self::AddOutput, Self>::with_sizes(lhs.sizes());
        Self::add_into(lhs, rhs, &mut dst);
        dst
    }
}

impl<T, U> AddAssignBackend<T, U> for CpuBackend
where
    T: AddAssign<U> + Send + Sync,
    U: Clone + Send + Sync,
{
    fn add_assign(lhs: &mut Tensor<T, Self>, rhs: &Tensor<U, Self>) {
        let dst = lhs.as_mut_buffer();
        let rhs = rhs.as_buffer();
        dst.par_iter_mut().zip(rhs.par_iter()).for_each(|(d, r)| {
            *d += r.clone();
        });
    }
}

#[cfg(test)]
mod tests {
    use slop_baby_bear::BabyBear;

    use super::*;

    #[test]
    fn test_tensor_add() {
        let mut rng = rand::thread_rng();

        let lhs = Tensor::<BabyBear>::rand(&mut rng, [10, 10]);
        let rhs = Tensor::<BabyBear>::rand(&mut rng, [10, 10]);
        let sum = &lhs + &rhs;
        assert_eq!(sum.sizes(), [10, 10]);
    }
}
