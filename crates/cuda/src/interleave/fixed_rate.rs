use slop_algebra::Field;
use slop_alloc::{Buffer, HasBackend};
use slop_commit::Message;
use slop_multilinear::Mle;
use slop_stacked::{FixedRateInterleave, FixedRateInterleaveBackend};
use slop_tensor::Tensor;

use crate::TaskScope;

impl<F> FixedRateInterleaveBackend<F> for TaskScope
where
    F: Field,
{
    async fn interleave_multilinears_with_fixed_rate(
        stacker: &FixedRateInterleave<F, Self>,
        multilinears: Message<Mle<F, Self>>,
        log_stacking_height: u32,
    ) -> Message<Mle<F, Self>> {
        let mut batch_multilinears = vec![];
        let batch_size = stacker.batch_size;
        let scope = multilinears[0].backend().clone();
        let mut overflow_buffer =
            Buffer::with_capacity_in(batch_size << log_stacking_height, scope);
        for mle in multilinears {
            let data = mle.guts().as_buffer();
            let mut needed_length = (batch_size << log_stacking_height) - overflow_buffer.len();

            let scope = data.backend();
            let mut data_slice = &data[..];
            while data_slice.len() > needed_length {
                let mut elements =
                    Buffer::with_capacity_in(batch_size << log_stacking_height, scope.clone());
                if !overflow_buffer.is_empty() {
                    elements.extend_from_device_slice(&overflow_buffer).unwrap();
                    unsafe {
                        overflow_buffer.set_len(0);
                    }
                }
                elements.extend_from_device_slice(&data_slice[0..needed_length]).unwrap();

                data_slice = &data_slice[needed_length..];
                assert_eq!(elements.len(), batch_size << log_stacking_height);

                let guts = Tensor::from(elements).reshape([batch_size, 1 << log_stacking_height]);
                let mle = Mle::new(guts);
                batch_multilinears.push(mle);
                needed_length = batch_size << log_stacking_height;
            }
            // Insert the remaining elements into the overflow buffer
            overflow_buffer.extend_from_device_slice(data_slice).unwrap();
        }
        // Make an mle from the overflow buffer, buf first padding it with zeros to get to the
        // next multiple of 2^{log_stacking_height}.
        let new_overflow_len = overflow_buffer.len().next_multiple_of(1 << log_stacking_height);
        let len = (new_overflow_len - overflow_buffer.len()) * size_of::<F>();
        overflow_buffer.write_bytes(0, len).unwrap();
        let overflow_batch_size = overflow_buffer.len() / (1 << log_stacking_height);
        let overflow_guts =
            Tensor::from(overflow_buffer).reshape([overflow_batch_size, 1 << log_stacking_height]);
        let overflow_mle = Mle::new(overflow_guts);
        batch_multilinears.push(overflow_mle);

        batch_multilinears.into_iter().collect::<Message<Mle<F, TaskScope>>>()
    }
}
