use std::fmt::Debug;
use std::marker::PhantomData;

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use slop_algebra::{extension::BinomialExtensionField, ExtensionField, TwoAdicField};
use slop_bn254::{Bn254Fr, OuterPerm, BNGC};
use slop_challenger::{
    CanObserve, DuplexChallenger, FieldChallenger, IopCtx, MultiField32Challenger,
};
use slop_koala_bear::{KoalaBear, KoalaBearDegree4Duplex, KoalaPerm};
use slop_merkle_tree::outer_perm;
use sp1_hypercube::inner_perm;

use slop_basefold::{
    BasefoldConfig, BasefoldVerifier, Poseidon2Bn254FrBasefoldConfig,
    Poseidon2KoalaBear16BasefoldConfig,
};

use crate::DeviceGrindingChallenger;

/// The configuration required for a Reed-Solomon-based Basefold.
pub trait BasefoldCudaConfig<GC: IopCtx>:
    BasefoldConfig<GC> + 'static + Clone + Debug + Send + Sync + Serialize + DeserializeOwned
where
    GC::F: TwoAdicField,
    GC::EF: ExtensionField<GC::F>,
{
    type DeviceChallenger: FieldChallenger<GC::F>
        + DeviceGrindingChallenger
        + CanObserve<GC::Digest>
        + 'static
        + Send
        + Sync
        + Clone;

    fn default_challenger(_verifier: &BasefoldVerifier<GC, Self>) -> Self::DeviceChallenger;
}
pub trait DefaultBasefoldCudaConfig<GC: IopCtx>: BasefoldConfig<GC> + Sized {
    fn default_verifier(log_blowup: usize) -> BasefoldVerifier<GC, Self>;
}

#[derive(Clone, Serialize, Deserialize)]
pub struct BasefoldConfigCudaImpl<F, EF, Tcs, Challenger>(PhantomData<(F, EF, Tcs, Challenger)>);

impl<F, EF, Tcs, Challenger> std::fmt::Debug for BasefoldConfigCudaImpl<F, EF, Tcs, Challenger> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BasefoldConfigCudaImpl")
    }
}

impl<F, EF, Tcs, Challenger> Default for BasefoldConfigCudaImpl<F, EF, Tcs, Challenger> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl BasefoldCudaConfig<KoalaBearDegree4Duplex> for Poseidon2KoalaBear16BasefoldConfig {
    type DeviceChallenger = DuplexChallenger<KoalaBear, KoalaPerm, 16, 8>;
    fn default_challenger(
        _verifier: &BasefoldVerifier<KoalaBearDegree4Duplex, Self>,
    ) -> DuplexChallenger<KoalaBear, KoalaPerm, 16, 8> {
        let default_perm = inner_perm();
        DuplexChallenger::<KoalaBear, KoalaPerm, 16, 8>::new(default_perm)
    }
}

impl BasefoldCudaConfig<BNGC<KoalaBear, BinomialExtensionField<KoalaBear, 4>>>
    for Poseidon2Bn254FrBasefoldConfig<KoalaBear, BinomialExtensionField<KoalaBear, 4>>
{
    type DeviceChallenger = MultiField32Challenger<KoalaBear, Bn254Fr, OuterPerm, 3, 2>;

    fn default_challenger(
        _verifier: &BasefoldVerifier<BNGC<KoalaBear, BinomialExtensionField<KoalaBear, 4>>, Self>,
    ) -> Self::DeviceChallenger {
        let default_perm = outer_perm();

        MultiField32Challenger::new(default_perm).expect("MultiField32Challenger::new failed")
    }
}
