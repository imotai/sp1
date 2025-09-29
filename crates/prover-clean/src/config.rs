//! A global configuration of types for device code.
//!
//! We are defining the field element and extension element as type aliases rather than using
//! generics in order to avoid complicated trait bounds but remain flexible enough to support
//! different field and extension element types.

use slop_algebra::extension::BinomialExtensionField;
use slop_koala_bear::{KoalaBear, KoalaBearDegree4Duplex};

/// The base field element type.
pub type Felt = KoalaBear;

/// The extension field element type.
pub type Ext = BinomialExtensionField<KoalaBear, 4>;

pub type GC = KoalaBearDegree4Duplex;
