pub use p3_merkle_tree::*;

mod bn254fr_poseidon2;
mod p3;
mod tcs;

pub use bn254fr_poseidon2::*;
pub use p3::*;
use slop_baby_bear::{
    baby_bear_poseidon2::{my_bb_16_perm, Perm, Poseidon2BabyBearConfig},
    BabyBear,
};
use slop_koala_bear::{my_kb_16_perm, KoalaBear, KoalaPerm, Poseidon2KoalaBearConfig};
use slop_symmetric::{PaddingFreeSponge, TruncatedPermutation};
pub use tcs::*;

impl MerkleTreeConfig for Poseidon2KoalaBearConfig {
    type Data = KoalaBear;
    type Digest = [KoalaBear; 8];
    type Hasher = PaddingFreeSponge<KoalaPerm, 16, 8, 8>;
    type Compressor = TruncatedPermutation<KoalaPerm, 2, 8, 16>;
}

impl DefaultMerkleTreeConfig for Poseidon2KoalaBearConfig {
    fn default_hasher_and_compressor() -> (Self::Hasher, Self::Compressor) {
        let perm = my_kb_16_perm();
        let hasher = Self::Hasher::new(perm.clone());
        let compressor = Self::Compressor::new(perm.clone());
        (hasher, compressor)
    }
}

impl MerkleTreeConfig for Poseidon2BabyBearConfig {
    type Data = BabyBear;
    type Digest = [BabyBear; 8];
    type Hasher = PaddingFreeSponge<Perm, 16, 8, 8>;
    type Compressor = TruncatedPermutation<Perm, 2, 8, 16>;
}

impl DefaultMerkleTreeConfig for Poseidon2BabyBearConfig {
    fn default_hasher_and_compressor() -> (Self::Hasher, Self::Compressor) {
        let perm = my_bb_16_perm();
        let hasher = Self::Hasher::new(perm.clone());
        let compressor = Self::Compressor::new(perm.clone());
        (hasher, compressor)
    }
}
