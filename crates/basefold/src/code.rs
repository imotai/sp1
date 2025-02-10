use std::borrow::{Borrow, BorrowMut};

use derive_where::derive_where;
use serde::{Deserialize, Serialize};
use slop_algebra::{AbstractField, TwoAdicField};
use slop_alloc::{Backend, CpuBackend};
use slop_tensor::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct ReedSolomonCode<F: TwoAdicField> {
    pub log_degree: usize,
    pub log_blowup: usize,
    pub shift: F,
}

#[derive(Debug, Clone)]
pub struct FriConfig<F: TwoAdicField> {
    pub code: ReedSolomonCode<F>,
    pub num_queries: usize,
    pub proof_of_work_bits: usize,
}

impl<F: TwoAdicField> FriConfig<F> {
    #[inline]
    pub const fn new(
        log_degree: usize,
        log_blowup: usize,
        shift: F,
        num_queries: usize,
        proof_of_work_bits: usize,
    ) -> Self {
        Self {
            code: ReedSolomonCode { log_degree, log_blowup, shift },
            num_queries,
            proof_of_work_bits,
        }
    }

    #[inline]
    pub const fn log_degree(&self) -> usize {
        self.code.log_degree
    }

    #[inline]
    pub const fn log_max_degree(&self) -> usize {
        self.code.log_degree + self.code.log_blowup
    }

    #[inline]
    pub const fn log_blowup(&self) -> usize {
        self.code.log_blowup
    }

    #[inline]
    pub const fn shift(&self) -> F {
        self.code.shift
    }

    #[inline]
    pub const fn num_queries(&self) -> usize {
        self.num_queries
    }

    #[inline]
    pub const fn proof_of_work_bits(&self) -> usize {
        self.proof_of_work_bits
    }
}

#[derive(Debug, Clone)]
#[derive_where(PartialEq, Eq; Tensor<F, A>)]
pub struct RsCodeWord<F: AbstractField, A: Backend = CpuBackend> {
    pub data: Tensor<F, A>,
}

impl<F: AbstractField, A: Backend> Borrow<Tensor<F, A>> for RsCodeWord<F, A> {
    fn borrow(&self) -> &Tensor<F, A> {
        &self.data
    }
}

impl<F: AbstractField, A: Backend> BorrowMut<Tensor<F, A>> for RsCodeWord<F, A> {
    fn borrow_mut(&mut self) -> &mut Tensor<F, A> {
        &mut self.data
    }
}
