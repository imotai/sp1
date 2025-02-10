use std::borrow::Borrow;

use slop_algebra::{ExtensionField, TwoAdicField};
use slop_alloc::Backend;
use slop_basefold::{BasefoldConfig, RsCodeWord};
use slop_commit::{TensorCs, TensorCsProver};
use slop_multilinear::Mle;
use thiserror::Error;

use crate::{FriIoppProver, MleBatcher, ReedSolomonEncoder};

pub trait BasefoldProverComponents {
    type F: TwoAdicField;
    type EF: ExtensionField<Self::F>;
    type A: Backend;
    type Tcs: TensorCs<Data = Self::F>;
    type Config: BasefoldConfig<F = Self::F, EF = Self::EF, Tcs = Self::Tcs>;

    type Encoder: ReedSolomonEncoder<Self::F, Self::A>;
    type MleBatcher: MleBatcher<Self::F, Self::EF, Self::A>;
    type FriProver: FriIoppProver<Self::F, Self::EF, Self::A>;
    type TcsProver: TensorCsProver<Self::A, Cs = Self::Tcs>;
}

pub struct BasefoldProverData<C: BasefoldProverComponents> {
    pub tcs_prover_data: <C::TcsProver as TensorCsProver<C::A>>::ProverData,
}

pub struct BasefoldCommitment<C: BasefoldProverComponents> {
    pub encoded_messages: Vec<RsCodeWord<C::F, C::A>>,
    pub commitment: <C::Tcs as TensorCs>::Commitment,
}

#[derive(Debug, Error)]
pub enum BasefoldProverError<C: BasefoldProverComponents> {
    #[error("Commit error: {0}")]
    TcsCommitError(<C::TcsProver as TensorCsProver<C::A>>::ProverError),
    #[error("Encoder error: {0}")]
    EncoderError(<C::Encoder as ReedSolomonEncoder<C::F, C::A>>::Error),
}

pub struct BasefoldProver<C: BasefoldProverComponents> {
    pub encoder: C::Encoder,
    pub fri_prover: C::FriProver,
    pub tcs_prover: C::TcsProver,
}

impl<C: BasefoldProverComponents> BasefoldProver<C> {
    #[inline]
    pub const fn new(
        encoder: C::Encoder,
        fri_prover: C::FriProver,
        tcs_prover: C::TcsProver,
    ) -> Self {
        Self { encoder, fri_prover, tcs_prover }
    }

    #[inline]
    pub fn commit_mles<M>(
        &self,
        mles: &[M],
    ) -> Result<(BasefoldCommitment<C>, BasefoldProverData<C>), BasefoldProverError<C>>
    where
        M: Borrow<Mle<C::F, C::A>>,
    {
        // Encode the guts of the mle via Reed-Solomon encoding.

        let encoded_messages = mles
            .iter()
            .map(|mle| self.encoder.encode(mle.borrow().guts()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(BasefoldProverError::<C>::EncoderError)?;

        // Commit to the encoded messages.
        let (commitment, tcs_prover_data) = self
            .tcs_prover
            .commit_tensors(&encoded_messages)
            .map_err(BasefoldProverError::<C>::TcsCommitError)?;

        Ok((
            BasefoldCommitment { encoded_messages, commitment },
            BasefoldProverData { tcs_prover_data },
        ))
    }

    #[inline]
    pub fn prove_mle_evaluations<M>(
        &self,
        _mles: &[M],
    ) -> Result<(BasefoldCommitment<C>, BasefoldProverData<C>), BasefoldProverError<C>>
    where
        M: Borrow<Mle<C::F, C::A>>,
    {
        todo!()
    }
}
