use csl_device::cuda::TaskScope;
use slop_algebra::Field;

use crate::Point;

impl<F: Field> Point<F, TaskScope> {
    #[inline]
    pub async fn into_host(self) -> Point<F> {
        let values = self.into_values().into_host().await.unwrap();
        Point::new(values)
    }
}
