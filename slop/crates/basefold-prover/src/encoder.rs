use std::{convert::Infallible, sync::Arc};

use slop_algebra::TwoAdicField;
use slop_basefold::{FriConfig, RsCodeWord};
use slop_commit::Message;
use slop_dft::{p3::Radix2DitParallel, Dft, DftOrdering};
use slop_futures::OwnedBorrow;
use slop_multilinear::Mle;

#[derive(Debug, Clone)]
pub struct CpuDftEncoder<F> {
    pub config: FriConfig<F>,
    pub dft: Arc<Radix2DitParallel>,
}

impl<F: TwoAdicField> CpuDftEncoder<F> {
    #[inline]
    pub(crate) fn config(&self) -> FriConfig<F> {
        self.config
    }

    pub(crate) async fn encode_batch<M>(
        &self,
        data: Message<M>,
    ) -> Result<Message<RsCodeWord<F>>, Infallible>
    where
        M: OwnedBorrow<Mle<F>>,
    {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let dft = self.dft.clone();
        let log_blowup = self.config.log_blowup();
        let data = data.to_vec();
        slop_futures::rayon::spawn(move || {
            let mut results = Vec::with_capacity(data.len());
            for data in data {
                let data = data.borrow().guts();
                assert_eq!(data.sizes().len(), 2, "Expected a 2D tensor");
                // Perform a DFT along the first axis of the tensor (assumed to be the long
                // dimension).
                let dft = dft.dft(data, log_blowup, DftOrdering::BitReversed, 0).unwrap();
                results.push(Arc::new(RsCodeWord { data: dft }));
            }
            tx.send(Message::from(results)).unwrap();
        });
        Ok(rx.await.unwrap())
    }
}
