//! Allocators for the SP1 zkVM.
//!
//! Currently, the only allocator available is the `"embedded"` allocator, which is enabled by
//! default.

pub mod embedded;
pub use embedded::init;
