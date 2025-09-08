use std::fmt::Debug;
use std::marker::PhantomData;

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use slop_algebra::{extension::BinomialExtensionField, ExtensionField, PrimeField31, TwoAdicField};
use slop_baby_bear::{
    baby_bear_poseidon2::{my_bb_16_perm, BabyBearDegree4Duplex, Perm, Poseidon2BabyBearConfig},
    BabyBear,
};
use slop_bn254::{
    Bn254Fr, OuterPerm, Poseidon2Bn254GlobalConfig, OUTER_CHALLENGER_RATE,
    OUTER_CHALLENGER_STATE_WIDTH,
};
use slop_challenger::{DuplexChallenger, IopCtx, MultiField32Challenger};
use slop_koala_bear::{
    my_kb_16_perm, KoalaBear, KoalaBearDegree4Duplex, KoalaPerm, Poseidon2KoalaBearConfig,
};
use slop_merkle_tree::{outer_perm, MerkleTreeConfig, MerkleTreeTcs, Poseidon2Bn254Config};

use crate::{BasefoldVerifier, FriConfig};

/// The configuration required for a Reed-Solomon-based Basefold.
pub trait BasefoldConfig<GC: IopCtx>:
    'static + Clone + Debug + Send + Sync + Serialize + DeserializeOwned
{
    /// The tensor commitment scheme.
    ///
    /// The tensor commitment scheme is used to send long messages in the protocol by converting
    /// them to a tensor committment providing oracle acccess.
    type Tcs: MerkleTreeConfig<GC>;

    fn default_challenger(_verifier: &BasefoldVerifier<GC, Self>) -> GC::Challenger;
}

pub trait DefaultBasefoldConfig<GC: IopCtx>: BasefoldConfig<GC> + Sized {
    fn default_verifier(log_blowup: usize) -> BasefoldVerifier<GC, Self>;
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BasefoldConfigImpl<F, EF, Tcs, Challenger>(PhantomData<(F, EF, Tcs, Challenger)>);

impl<F, EF, Tcs, Challenger> std::fmt::Debug for BasefoldConfigImpl<F, EF, Tcs, Challenger> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BasefoldConfigImpl")
    }
}

impl<F, EF, Tcs, Challenger> Default for BasefoldConfigImpl<F, EF, Tcs, Challenger> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

// impl<F, EF, Tcs, Challenger> BasefoldConfig for BasefoldConfigImpl<F, EF, Tcs, Challenger>
// where
//     F: TwoAdicField,
//     EF: ExtensionField<F>,
//     Tcs: TensorCs<Data = F>,
//     Challenger: FieldChallenger<F>
//         + GrindingChallenger
//         + CanObserve<<Tcs as TensorCs>::Commitment>
//         + 'static
//         + Send
//         + Sync,
// {
//     type F = F;
//     type EF = EF;
//     type Tcs = Tcs;
//     type Commitment = <Tcs as TensorCs>::Commitment;
//     type Challenger = Challenger;
// }

pub type Poseidon2BabyBear16BasefoldConfig = BasefoldConfigImpl<
    BabyBear,
    BinomialExtensionField<BabyBear, 4>,
    MerkleTreeTcs<BabyBearDegree4Duplex, Poseidon2BabyBearConfig>,
    DuplexChallenger<BabyBear, Perm, 16, 8>,
>;

pub type Poseidon2KoalaBear16BasefoldConfig = BasefoldConfigImpl<
    KoalaBear,
    BinomialExtensionField<KoalaBear, 4>,
    MerkleTreeTcs<KoalaBearDegree4Duplex, Poseidon2KoalaBearConfig>,
    DuplexChallenger<KoalaBear, KoalaPerm, 16, 8>,
>;

pub type Poseidon2Bn254FrBasefoldConfig<F, EF> = BasefoldConfigImpl<
    F,
    BinomialExtensionField<F, 4>,
    MerkleTreeTcs<Poseidon2Bn254GlobalConfig<F, EF>, Poseidon2Bn254Config<F>>,
    MultiField32Challenger<
        F,
        Bn254Fr,
        OuterPerm,
        OUTER_CHALLENGER_STATE_WIDTH,
        OUTER_CHALLENGER_RATE,
    >,
>;

impl BasefoldConfig<BabyBearDegree4Duplex> for Poseidon2BabyBear16BasefoldConfig {
    type Tcs = Poseidon2BabyBearConfig;

    fn default_challenger(
        _verifier: &BasefoldVerifier<BabyBearDegree4Duplex, Self>,
    ) -> DuplexChallenger<BabyBear, Perm, 16, 8> {
        let default_perm = my_bb_16_perm();
        DuplexChallenger::<BabyBear, Perm, 16, 8>::new(default_perm)
    }
}

impl DefaultBasefoldConfig<BabyBearDegree4Duplex> for Poseidon2BabyBear16BasefoldConfig {
    fn default_verifier(log_blowup: usize) -> BasefoldVerifier<BabyBearDegree4Duplex, Self> {
        let fri_config = FriConfig::<BabyBear>::auto(log_blowup, 84);
        let tcs = MerkleTreeTcs::<BabyBearDegree4Duplex, Poseidon2BabyBearConfig>::default();
        BasefoldVerifier { fri_config, tcs }
    }
}

impl<F: PrimeField31 + TwoAdicField, EF: ExtensionField<F>>
    BasefoldConfig<Poseidon2Bn254GlobalConfig<F, EF>> for Poseidon2Bn254FrBasefoldConfig<F, EF>
{
    type Tcs = Poseidon2Bn254Config<F>;

    fn default_challenger(
        _verifier: &BasefoldVerifier<Poseidon2Bn254GlobalConfig<F, EF>, Self>,
    ) -> MultiField32Challenger<
        F,
        Bn254Fr,
        OuterPerm,
        OUTER_CHALLENGER_STATE_WIDTH,
        OUTER_CHALLENGER_RATE,
    > {
        let default_perm = outer_perm();
        MultiField32Challenger::<
            F,
            Bn254Fr,
            OuterPerm,
            OUTER_CHALLENGER_STATE_WIDTH,
            OUTER_CHALLENGER_RATE,
        >::new(default_perm)
        .unwrap()
    }
}

impl<F: PrimeField31 + TwoAdicField, EF: ExtensionField<F>>
    DefaultBasefoldConfig<Poseidon2Bn254GlobalConfig<F, EF>>
    for Poseidon2Bn254FrBasefoldConfig<F, EF>
{
    fn default_verifier(
        log_blowup: usize,
    ) -> BasefoldVerifier<Poseidon2Bn254GlobalConfig<F, EF>, Self> {
        let fri_config = FriConfig::<F>::auto(log_blowup, 84);
        let tcs =
            MerkleTreeTcs::<Poseidon2Bn254GlobalConfig<F, EF>, Poseidon2Bn254Config<F>>::default();
        BasefoldVerifier { fri_config, tcs }
    }
}

impl BasefoldConfig<KoalaBearDegree4Duplex> for Poseidon2KoalaBear16BasefoldConfig {
    type Tcs = Poseidon2KoalaBearConfig;

    fn default_challenger(
        _verifier: &BasefoldVerifier<KoalaBearDegree4Duplex, Self>,
    ) -> <KoalaBearDegree4Duplex as IopCtx>::Challenger {
        let default_perm = my_kb_16_perm();
        DuplexChallenger::new(default_perm)
    }
}

impl DefaultBasefoldConfig<KoalaBearDegree4Duplex> for Poseidon2KoalaBear16BasefoldConfig {
    fn default_verifier(log_blowup: usize) -> BasefoldVerifier<KoalaBearDegree4Duplex, Self> {
        let fri_config = FriConfig::<KoalaBear>::auto(log_blowup, 84);
        let tcs = MerkleTreeTcs::<KoalaBearDegree4Duplex, Poseidon2KoalaBearConfig>::default();
        BasefoldVerifier { fri_config, tcs }
    }
}
