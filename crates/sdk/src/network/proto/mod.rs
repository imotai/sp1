#![allow(clippy::all)]
#![allow(missing_docs)]
#![allow(clippy::pedantic)]

#[rustfmt::skip]
pub mod artifact;

cfg_if::cfg_if! {
    if #[cfg(not(feature = "reserved-capacity"))] {
        mod sepolia {
            #[rustfmt::skip]
            pub mod network;
            #[rustfmt::skip]
            pub mod types;
        }

        #[rustfmt::skip]
        pub use self::sepolia::{network, types};
    } else {
        // Re-export types from sp1-prover-types at the proto level
        pub use sp1_prover_types::network_base_types as types;

        mod base {
            // Re-export types so network.rs can reference it via super::super::types
            pub use super::types;

            #[rustfmt::skip]
            pub mod network;
        }

        #[rustfmt::skip]
        pub use self::base::network;
    }
}
