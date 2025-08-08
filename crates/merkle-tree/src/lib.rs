pub use p3_merkle_tree::*;

mod baby_bear_poseidon2;
mod bn254fr_poseidon2;
mod p3;
mod tcs;

pub use baby_bear_poseidon2::*;
pub use bn254fr_poseidon2::*;
pub use p3::*;
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
