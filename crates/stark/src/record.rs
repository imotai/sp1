use crate::{air::SP1AirBuilder, septic_digest::SepticDigest};
use hashbrown::HashMap;
use slop_algebra::{AbstractField, PrimeField32};

/// A record that can be proven by a machine.
pub trait MachineRecord: Default + Sized + Send + Sync + Clone {
    /// The configuration of the machine.
    type Config: 'static + Copy + Send + Sync;

    /// The statistics of the record.
    fn stats(&self) -> HashMap<String, usize>;

    /// Appends two records together.
    fn append(&mut self, other: &mut Self);

    /// Registers the nonces of the record.
    fn register_nonces(&mut self, _opts: &Self::Config) {}

    /// Returns the public values of the record.
    fn public_values<F: AbstractField>(&self) -> Vec<F>;

    /// Updates the global cumulative sum of the record.
    fn update_global_cumulative_sum<F: PrimeField32>(
        &mut self,
        global_cumulative_sum: SepticDigest<F>,
    );

    /// Extracts the global cumulative sum from the public values.
    fn global_cumulative_sum<F: PrimeField32>(public_values: &[F]) -> SepticDigest<F>;

    /// Constrains the public values of the record.
    fn eval_public_values<AB: SP1AirBuilder>(builder: &mut AB);
}
