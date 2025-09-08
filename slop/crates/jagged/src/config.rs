use serde::{de::DeserializeOwned, Serialize};
use slop_challenger::IopCtx;
use slop_multilinear::MultilinearPcsVerifier;

pub trait JaggedConfig<GC: IopCtx>:
    'static + Clone + Send + Clone + Serialize + DeserializeOwned
{
    type BatchPcsVerifier: MultilinearPcsVerifier<GC>;
}

pub type JaggedProof<GC, JC> =
    <<JC as JaggedConfig<GC>>::BatchPcsVerifier as MultilinearPcsVerifier<GC>>::Proof;

pub type JaggedError<GC, JC> =
    <<JC as JaggedConfig<GC>>::BatchPcsVerifier as MultilinearPcsVerifier<GC>>::VerifierError;
