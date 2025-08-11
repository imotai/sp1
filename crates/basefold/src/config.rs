use std::fmt::Debug;
use std::marker::PhantomData;

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use slop_algebra::{ExtensionField, TwoAdicField};
use slop_baby_bear::BabyBear;
use slop_bn254::Bn254Fr;
use slop_challenger::{CanObserve, DuplexChallenger, FieldChallenger, MultiField32Challenger};
use slop_commit::TensorCs;
use slop_merkle_tree::{outer_perm, OuterPerm};
use sp1_hypercube::inner_perm;
use sp1_primitives::SP1Perm;

use slop_basefold::{
    BasefoldConfig, BasefoldVerifier, Poseidon2BabyBear16BasefoldConfig,
    Poseidon2Bn254FrBasefoldConfig,
};

use crate::DeviceGrindingChallenger;

/// The configuration required for a Reed-Solomon-based Basefold.
pub trait BasefoldCudaConfig:
    BasefoldConfig + 'static + Clone + Debug + Send + Sync + Serialize + DeserializeOwned
where
    Self::F: TwoAdicField,
    Self::EF: ExtensionField<Self::F>,
    Self::Commitment: 'static + Clone + Send + Sync + Serialize + DeserializeOwned,
    Self::Tcs: TensorCs<Data = Self::F, Commitment = Self::Commitment>,
{
    type DeviceChallenger: FieldChallenger<Self::F>
        + DeviceGrindingChallenger
        + CanObserve<Self::Commitment>
        + 'static
        + Send
        + Sync
        + Clone;

    fn default_challenger(_verifier: &BasefoldVerifier<Self>) -> Self::DeviceChallenger;
}
pub trait DefaultBasefoldCudaConfig: BasefoldConfig + Sized {
    fn default_verifier(log_blowup: usize) -> BasefoldVerifier<Self>;
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

impl BasefoldCudaConfig for Poseidon2BabyBear16BasefoldConfig {
    type DeviceChallenger = DuplexChallenger<BabyBear, SP1Perm, 16, 8>;
    fn default_challenger(
        _verifier: &BasefoldVerifier<Self>,
    ) -> DuplexChallenger<BabyBear, SP1Perm, 16, 8> {
        let default_perm = inner_perm();
        DuplexChallenger::<BabyBear, SP1Perm, 16, 8>::new(default_perm)
    }
}

impl BasefoldCudaConfig for Poseidon2Bn254FrBasefoldConfig {
    type DeviceChallenger = MultiField32Challenger<BabyBear, Bn254Fr, OuterPerm, 3, 2>;

    fn default_challenger(_verifier: &BasefoldVerifier<Self>) -> Self::DeviceChallenger {
        let default_perm = outer_perm();

        MultiField32Challenger::new(default_perm).expect("MultiField32Challenger::new failed")
    }
}
