mod synchronize;

use futures::prelude::*;
pub use p3_challenger::*;
pub use synchronize::*;

pub trait FromChallenger<Challenger: Send + Sync, A: Send + Sync>: Send + Sync + Sized {
    fn from_challenger(
        challenger: &Challenger,
        backend: A,
    ) -> impl Future<Output = Self> + Send + Sync;
}

impl<Challenger: Clone + Send + Sync, A: Send + Sync> FromChallenger<Challenger, A> for Challenger {
    async fn from_challenger(challenger: &Challenger, _backend: A) -> Self {
        challenger.clone()
    }
}
