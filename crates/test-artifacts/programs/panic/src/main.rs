#![no_main]
sp1_zkvm::entrypoint!(main);
use std::hint::black_box;
pub fn main() {
    let mut sum: u64 = 0;
    for i in 0..2000000 {
        sum += black_box(i);
    }
    assert_eq!(sum, 1);
}
