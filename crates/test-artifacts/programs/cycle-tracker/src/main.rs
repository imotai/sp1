#![no_main]
sp1_zkvm::entrypoint!(main);

use std::hint::black_box;
use tiny_keccak::{Hasher, Keccak};

/// Test function using the cycle_tracker derive macro (non-report variant).
/// This uses eprintln internally and should be parsed but NOT accumulated to report.
#[sp1_derive::cycle_tracker]
pub fn f(x: usize) -> usize {
    x + 1
}

/// Test function using manual println (non-report variant).
/// This should be parsed but NOT accumulated to report.
pub fn g(x: usize) -> usize {
    println!("cycle-tracker-start: g");
    println!("cycle-tracker-start: g2");
    let y = x + 3;
    println!("cycle-tracker-end: g2");
    println!("cycle-tracker-end: g");
    y
}

/// Test function using report variant.
/// This SHOULD be accumulated to ExecutionReport.cycle_tracker.
pub fn h(x: usize) -> usize {
    println!("cycle-tracker-report-start: h");
    let y = x + 1;
    println!("cycle-tracker-report-end: h");
    y
}

/// Test function using report variant with multiple invocations.
/// This tests that cycles accumulate correctly across invocations.
pub fn repeated(x: usize) -> usize {
    println!("cycle-tracker-report-start: repeated");
    let y = x * 2;
    println!("cycle-tracker-report-end: repeated");
    y
}

pub fn main() {
    // Test non-report variants (should parse but not populate report)
    black_box(f(black_box(1)));
    black_box(g(black_box(1)));

    // Test report variant (should populate ExecutionReport.cycle_tracker)
    black_box(h(black_box(1)));

    // Test multiple invocations of same label (should accumulate cycles)
    black_box(repeated(black_box(1)));
    black_box(repeated(black_box(2)));
    black_box(repeated(black_box(3)));
}
