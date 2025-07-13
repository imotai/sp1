use slop_algebra::extension::BinomialExtensionField;
use slop_baby_bear::BabyBear;
use slop_bn254::Bn254Fr;

use crate::{circuit::AsmConfig, prelude::Config};

pub type InnerConfig = AsmConfig<BabyBear, BinomialExtensionField<BabyBear, 4>>;

#[derive(Clone, Default, Debug)]
pub struct OuterConfig;

impl Config for OuterConfig {
    type N = Bn254Fr;
    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;
}
