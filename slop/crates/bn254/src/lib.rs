#![allow(clippy::disallowed_types)]
use std::marker::PhantomData;

pub use p3_bn254_fr::*;
use serde::{Deserialize, Serialize};
use slop_algebra::{ExtensionField, PrimeField31};
use slop_challenger::{IopCtx, MultiField32Challenger};
use slop_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use slop_symmetric::Hash;

pub const OUTER_CHALLENGER_STATE_WIDTH: usize = 3;
pub const OUTER_DIGEST_SIZE: usize = 1;
pub const OUTER_CHALLENGER_RATE: usize = 2;

#[derive(Debug, Clone, Default, Copy, Serialize, Deserialize, Hash, PartialEq, Eq)]
pub struct Poseidon2Bn254GlobalConfig<F, EF>(PhantomData<(F, EF)>);

pub type OuterPerm = Poseidon2<
    Bn254Fr,
    Poseidon2ExternalMatrixGeneral,
    DiffusionMatrixBN254,
    OUTER_CHALLENGER_STATE_WIDTH,
    5,
>;

impl<F: PrimeField31, EF: ExtensionField<F>> IopCtx for Poseidon2Bn254GlobalConfig<F, EF> {
    type F = F;

    type EF = EF;

    type Digest = Hash<F, Bn254Fr, 1>;

    type Challenger = MultiField32Challenger<
        F,
        Bn254Fr,
        OuterPerm,
        OUTER_CHALLENGER_STATE_WIDTH,
        OUTER_CHALLENGER_RATE,
    >;
}

pub type BNGC<F, EF> = Poseidon2Bn254GlobalConfig<F, EF>;
