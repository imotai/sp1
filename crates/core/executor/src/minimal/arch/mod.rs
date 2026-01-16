cfg_if::cfg_if! {
    // When profiling is enabled, always use the portable executor for accurate instruction-level tracing.
    if #[cfg(all(target_arch = "x86_64", target_endian = "little", not(feature = "profiling")))] {
        mod x86_64;
        pub use x86_64::*;
    } else {
        mod portable;
        pub use portable::*;
    }
}
