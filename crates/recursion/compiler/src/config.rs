use slop_algebra::extension::BinomialExtensionField;
use slop_bn254::Bn254Fr;
use sp1_primitives::SP1Field;

use crate::{circuit::AsmConfig, prelude::Config};

pub type InnerConfig = AsmConfig<SP1Field, BinomialExtensionField<SP1Field, 4>>;

#[derive(Clone, Default, Debug)]
pub struct OuterConfig;

impl Config for OuterConfig {
    type N = Bn254Fr;
    type F = SP1Field;
    type EF = BinomialExtensionField<SP1Field, 4>;
}
