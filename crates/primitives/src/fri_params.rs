use slop_primitives::FriConfig;

use crate::SP1Field;

pub const CORE_LOG_BLOWUP: usize = 2;
pub const RECURSION_LOG_BLOWUP: usize = 2;

pub fn core_fri_config() -> FriConfig<SP1Field> {
    FriConfig::new(
        CORE_LOG_BLOWUP,
        unique_decoding_queries(CORE_LOG_BLOWUP),
        SP1_PROOF_OF_WORK_BITS,
    )
}

pub const SHRINK_LOG_BLOWUP: usize = 4;
pub const WRAP_LOG_BLOWUP: usize = 4;

pub fn recursion_fri_config() -> FriConfig<SP1Field> {
    FriConfig::new(
        RECURSION_LOG_BLOWUP,
        unique_decoding_queries(RECURSION_LOG_BLOWUP),
        SP1_PROOF_OF_WORK_BITS,
    )
}

pub fn shrink_fri_config() -> FriConfig<SP1Field> {
    FriConfig::new(
        SHRINK_LOG_BLOWUP,
        conjectured_queries(SHRINK_LOG_BLOWUP),
        SP1_PROOF_OF_WORK_BITS,
    )
}

pub fn wrap_fri_config() -> FriConfig<SP1Field> {
    FriConfig::new(WRAP_LOG_BLOWUP, conjectured_queries(WRAP_LOG_BLOWUP), SP1_PROOF_OF_WORK_BITS)
}

pub const SP1_TARGET_BITS_OF_SECURITY: usize = 100;
pub const SP1_PROOF_OF_WORK_BITS: usize = 16;

pub fn unique_decoding_queries(log_blowup: usize) -> usize {
    let rate = 1.0 / (1 << log_blowup) as f64;
    let half_rate_plus_half = 0.5 + (rate / 2.0);
    (-((SP1_TARGET_BITS_OF_SECURITY - SP1_PROOF_OF_WORK_BITS) as f64) / half_rate_plus_half.log2())
        .ceil() as usize
}

/// The number of queries needed to achieve the target security level under the conjecture (adding
/// 10 extra queries for safety against the attack of Diamond-Gruen).
pub fn conjectured_queries(log_blowup: usize) -> usize {
    (SP1_TARGET_BITS_OF_SECURITY - SP1_PROOF_OF_WORK_BITS + 10).div_ceil(log_blowup)
}
