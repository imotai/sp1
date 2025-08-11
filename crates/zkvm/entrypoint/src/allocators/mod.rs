//! Allocators for the SP1 zkVM.
//!
//! Currently, the only allocator available is the `"bump"` allocator, which is enabled by default.

#[cfg(all(feature = "bump"))]
mod bump;
