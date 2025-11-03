cfg_if::cfg_if! {
    if #[cfg(all(target_arch = "x86_64", target_endian = "little"))] {
        mod x86_64;
        pub use x86_64::*;
    } else {
        mod portable;
        pub use portable::*;
    }
}
