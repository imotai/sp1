/// Reverse the bits of an integer within a specified bit length.
///
/// Takes an integer `x` and reverses its bits within the least significant `bit_len` bits.
/// For example, reverse_bits_len(0b101, 3) = 0b101 (reversed) = 0b101.
/// reverse_bits_len(0b001, 3) = 0b100.
pub fn reverse_bits_len(x: usize, bit_len: usize) -> usize {
    let mut result = 0;
    let mut x = x;
    for _ in 0..bit_len {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}
