#![allow(clippy::disallowed_types)]
pub use p3_merkle_tree::*;

mod bn254fr_poseidon2;
mod p3;
mod tcs;

pub use bn254fr_poseidon2::*;
pub use p3::*;
use slop_baby_bear::baby_bear_poseidon2::{
    my_bb_16_perm, BabyBearDegree4Duplex, Perm, Poseidon2BabyBearConfig,
};
use slop_koala_bear::{my_kb_16_perm, KoalaBearDegree4Duplex, KoalaPerm, Poseidon2KoalaBearConfig};
use slop_symmetric::{PaddingFreeSponge, TruncatedPermutation};
pub use tcs::*;

impl MerkleTreeConfig<KoalaBearDegree4Duplex> for Poseidon2KoalaBearConfig {
    type Hasher = PaddingFreeSponge<KoalaPerm, 16, 8, 8>;
    type Compressor = TruncatedPermutation<KoalaPerm, 2, 8, 16>;
}

impl DefaultMerkleTreeConfig<KoalaBearDegree4Duplex> for Poseidon2KoalaBearConfig {
    #[allow(clippy::disallowed_methods)]
    fn default_hasher_and_compressor() -> (Self::Hasher, Self::Compressor) {
        let perm = my_kb_16_perm();
        let hasher = Self::Hasher::new(perm.clone());
        let compressor = Self::Compressor::new(perm.clone());
        (hasher, compressor)
    }
}

impl MerkleTreeConfig<BabyBearDegree4Duplex> for Poseidon2BabyBearConfig {
    type Hasher = PaddingFreeSponge<Perm, 16, 8, 8>;
    type Compressor = TruncatedPermutation<Perm, 2, 8, 16>;
}

impl DefaultMerkleTreeConfig<BabyBearDegree4Duplex> for Poseidon2BabyBearConfig {
    fn default_hasher_and_compressor() -> (Self::Hasher, Self::Compressor) {
        let perm = my_bb_16_perm();
        let hasher = Self::Hasher::new(perm.clone());
        let compressor = Self::Compressor::new(perm.clone());
        (hasher, compressor)
    }
}
