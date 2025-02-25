use csl_cuda::TaskScope;
use serde::{Deserialize, Serialize};
use slop_algebra::TwoAdicField;
use slop_basefold::{FriConfig, RsCodeWord};
use slop_basefold_prover::ReedSolomonEncoder;
use slop_commit::Message;
use slop_dft::{Dft, DftOrdering};
use slop_futures::OwnedBorrow;
use slop_multilinear::Mle;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaDftEncoder<F, D> {
    pub config: FriConfig<F>,
    pub dft: D,
}

impl<F: TwoAdicField, D: Dft<F, TaskScope>> ReedSolomonEncoder<F, TaskScope>
    for CudaDftEncoder<F, D>
{
    type Error = D::Error;

    fn config(&self) -> &slop_basefold::FriConfig<F> {
        &self.config
    }

    /// Encodes the input into a new codeword.
    async fn encode_batch<M>(
        &self,
        data: Message<M>,
    ) -> Result<Message<RsCodeWord<F, TaskScope>>, Self::Error>
    where
        M: OwnedBorrow<Mle<F, TaskScope>>,
    {
        data.into_iter()
            .map(|mle| -> Result<RsCodeWord<F, TaskScope>, Self::Error> {
                // Perform a DFT along the second axis of the tensor (assumed to be the long dimension).
                let data = mle.borrow().guts();
                let dft =
                    self.dft.dft(data, self.config.log_blowup(), DftOrdering::BitReversed, 1)?;
                Ok(RsCodeWord { data: dft })
            })
            .collect()
    }
}
