use std::{hash::Hash, str::FromStr};

use hashbrown::HashMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{Opcode, RiscvAirId};

/// Serialize a `HashMap<u32, V>` as a `Vec<(u32, V)>`.
pub fn serialize_hashmap_as_vec<K: Eq + Hash + Serialize, V: Serialize, S: Serializer>(
    map: &HashMap<K, V>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    Serialize::serialize(&map.iter().collect::<Vec<_>>(), serializer)
}

/// Deserialize a `Vec<(u32, V)>` as a `HashMap<u32, V>`.
pub fn deserialize_hashmap_as_vec<
    'de,
    K: Eq + Hash + Deserialize<'de>,
    V: Deserialize<'de>,
    D: Deserializer<'de>,
>(
    deserializer: D,
) -> Result<HashMap<K, V>, D::Error> {
    let seq: Vec<(K, V)> = Deserialize::deserialize(deserializer)?;
    Ok(seq.into_iter().collect())
}

/// Returns `true` if the given `opcode` is a signed operation.
#[must_use]
pub fn is_signed_operation(opcode: Opcode) -> bool {
    opcode == Opcode::DIV || opcode == Opcode::REM
}

/// Calculate the correct `quotient` and `remainder` for the given `b` and `c` per RISC-V spec.
#[must_use]
pub fn get_quotient_and_remainder(b: u64, c: u64, opcode: Opcode) -> (u64, u64) {
    if c == 0 {
        // When c is 0, the quotient is 2^32 - 1 and the remainder is b regardless of whether we
        // perform signed or unsigned division.
        (u64::MAX, b)
    } else if is_signed_operation(opcode) {
        ((b as i64).wrapping_div(c as i64) as u64, (b as i64).wrapping_rem(c as i64) as u64)
    } else {
        (b.wrapping_div(c), b.wrapping_rem(c))
    }
}

/// Calculate the most significant bit of the given 64-bit integer `a`, and returns it as a u8.
#[must_use]
pub const fn get_msb(a: u64) -> u8 {
    ((a >> 63) & 1) as u8
}

/// Load the cost of each air from the predefined JSON.
#[must_use]
pub fn rv32im_costs() -> HashMap<RiscvAirId, usize> {
    let costs: HashMap<String, usize> =
        serde_json::from_str(include_str!("./artifacts/rv32im_costs.json")).unwrap();
    costs.into_iter().map(|(k, v)| (RiscvAirId::from_str(&k).unwrap(), v)).collect()
}
