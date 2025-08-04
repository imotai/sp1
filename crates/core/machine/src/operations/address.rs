use serde::{Deserialize, Serialize};
use slop_algebra::{AbstractField, Field, PrimeField32};
use sp1_derive::AlignedBorrow;

use sp1_core_executor::{events::ByteRecord, ByteOpcode};
use sp1_primitives::consts::u64_to_u16_limbs;
use sp1_stark::Word;

use crate::air::SP1CoreAirBuilder;

use super::AddrAddOperation;

/// A set of columns needed to validate the address and return the aligned address.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct AddressOperation<T> {
    /// Instance of `AddOperation` for addr.
    pub addr_operation: AddrAddOperation<T>,

    /// This is used to check if the top two limbs of the address is not both zero.
    pub top_two_limb_inv: T,
}

impl<F: PrimeField32> AddressOperation<F> {
    pub fn populate(&mut self, record: &mut impl ByteRecord, b: u64, c: u64) -> u64 {
        let memory_addr = b.wrapping_add(c);
        let addr_limbs = u64_to_u16_limbs(memory_addr);
        self.addr_operation.populate(record, b, c);
        let sum_top_two_limb =
            F::from_canonical_u16(addr_limbs[1]) + F::from_canonical_u16(addr_limbs[2]);
        self.top_two_limb_inv = sum_top_two_limb.inverse();
        record.add_bit_range_check(addr_limbs[0] / 8, 13);
        memory_addr
    }
}

impl<F: Field> AddressOperation<F> {
    /// Given `op_b` and `op_c` in a memory opcode, derive the memory address.
    /// The memory address is constrained to be `>= 2^16` and less than 2^48.
    /// Both `is_real` and offset bits are constrained to be boolean and correct.
    /// The returned value is the aligned memory address used for memory access.
    #[allow(clippy::too_many_arguments)]
    pub fn eval<AB: SP1CoreAirBuilder>(
        builder: &mut AB,
        b: Word<AB::Expr>,
        c: Word<AB::Expr>,
        offset_bit0: AB::Expr,
        offset_bit1: AB::Expr,
        offset_bit2: AB::Expr,
        is_real: AB::Expr,
        cols: AddressOperation<AB::Var>,
    ) -> [AB::Expr; 3] {
        // Check that `is_real` and offset bits are boolean.
        builder.assert_bool(is_real.clone());
        builder.assert_bool(offset_bit0.clone());
        builder.assert_bool(offset_bit1.clone());
        builder.assert_bool(offset_bit2.clone());

        // `addr` is computed as `op_b + op_c`, and is range checked to be three u16 limbs.
        AddrAddOperation::<AB::F>::eval(builder, b, c, cols.addr_operation, is_real.clone());
        let addr = cols.addr_operation.value;

        let sum_top_two_limb = addr[1] + addr[2];

        // Check that `addr >= 2^16`, so it doesn't touch registers.
        // This implements a stack guard of size 2^16 bytes = 64KB.
        // If `is_real = 1`, then `addr[1] + addr[2] != 0`, so `addr >= 2^16`.
        builder.assert_eq(cols.top_two_limb_inv * sum_top_two_limb.clone(), is_real.clone());

        // Check `0 <= (addr[0] - 4 * bit2 - 2 * bit1 - bit0) / 8 < 2^13`.
        // This shows `addr[0] - 4 * bit2 - 2 * bit1 - bit0` is a multiple of `8` within `u16`.
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::Range as u32),
            (addr[0]
                - AB::Expr::from_canonical_u32(4) * offset_bit2.clone()
                - AB::Expr::from_canonical_u32(2) * offset_bit1.clone()
                - offset_bit0.clone())
                * AB::F::from_canonical_u32(8).inverse(),
            AB::Expr::from_canonical_u32(13),
            AB::Expr::zero(),
            is_real.clone(),
        );

        [
            addr[0].into()
                - AB::Expr::from_canonical_u32(4) * offset_bit2
                - AB::Expr::from_canonical_u32(2) * offset_bit1
                - offset_bit0,
            addr[1].into(),
            addr[2].into(),
        ]
    }
}
