use p3_field::{AbstractField, Field, PrimeField32};
use sp1_derive::AlignedBorrow;

use sp1_core_executor::{events::ByteRecord, ByteOpcode};
use sp1_primitives::consts::u32_to_u16_limbs;
use sp1_stark::{air::SP1AirBuilder, Word};

use super::{AddOperation, BabyBearWordRangeChecker, IsZeroOperation};

/// A set of columns needed to validate the address and return the aligned address.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct AddressOperation<T> {
    /// Instance of `AddOperation` for addr_word.
    pub addr_word_operation: AddOperation<T>,

    /// Gadget to verify that the address word is within the BabyBear field.
    pub addr_word_range_checker: BabyBearWordRangeChecker<T>,

    /// This is used to check if the most significant limb of the memory address is zero.
    pub most_sig_limb_zero: IsZeroOperation<T>,
}

impl<F: PrimeField32> AddressOperation<F> {
    pub fn populate(&mut self, record: &mut impl ByteRecord, b: u32, c: u32) -> u32 {
        let memory_addr = b.wrapping_add(c);
        self.addr_word_operation.populate(record, b, c);
        self.addr_word_range_checker.populate(Word::from(memory_addr), record);
        let addr_word_limbs = u32_to_u16_limbs(memory_addr);
        let top_limb_zero = self.most_sig_limb_zero.populate(addr_word_limbs[1] as u32) as u16;
        record.add_bit_range_check((addr_word_limbs[0] - 32 * top_limb_zero) / 4, 14);
        memory_addr
    }
}

impl<F: Field> AddressOperation<F> {
    /// Given `op_b` and `op_c` in a memory opcode, derive the memory address.
    /// The memory address is constrained to be `>= 0x20` and less than BabyBear.
    /// Both `is_real` and offset bits are constrained to be boolean and correct.
    /// The returned value is the aligned memory address used for memory access.
    #[allow(clippy::too_many_arguments)]
    pub fn eval<AB: SP1AirBuilder>(
        builder: &mut AB,
        b: Word<AB::Expr>,
        c: Word<AB::Expr>,
        offset_bit0: AB::Expr,
        offset_bit1: AB::Expr,
        is_real: AB::Expr,
        cols: AddressOperation<AB::Var>,
    ) -> AB::Expr {
        // Check that `is_real` and offset bits are boolean.
        builder.assert_bool(is_real.clone());
        builder.assert_bool(offset_bit0.clone());
        builder.assert_bool(offset_bit1.clone());

        // `addr` is computed as `op_b + op_c`, and is range checked to be two u16 limbs.
        AddOperation::<AB::F>::eval(builder, b, c, cols.addr_word_operation, is_real.clone());
        let addr = cols.addr_word_operation.value;

        // `addr` is checked to be less than the BabyBear prime.
        BabyBearWordRangeChecker::<AB::F>::range_check(
            builder,
            addr,
            cols.addr_word_range_checker,
            is_real.clone(),
        );

        // Check that `addr >= 32`, so it doesn't touch registers.
        // At the same time, check that `addr` is a multiple of `4`, to check alignment.
        // First, check if `addr`'s high limb is zero.
        IsZeroOperation::<AB::F>::eval(
            builder,
            addr.0[1].into(),
            cols.most_sig_limb_zero,
            is_real.clone(),
        );

        // Check `0 <= (addr[0] - 32 * is_zero - 2 * bit1 - bit0) / 4 < 2^14`, where `is_zero = (addr[1] == 0)`
        // This enforces that `addr[0] - 32 * is_zero - 2 * bit1 - bit0` is a multiple of `4` within `u16`.
        // This shows `addr[0] - 2 * bit1 - bit0` is a multiple of `4`.
        // If `is_zero = 1`, then this checks `addr[0] - 32 - 2 * bit1 - bit0` is nonnegative, showing the address bound.
        // If `is_zero = 0`, this checks `addr[0] - 2 * bit1 - bit0` is within `u16`, which is correct.
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::Range as u32),
            (addr.0[0]
                - AB::Expr::from_canonical_u32(32) * cols.most_sig_limb_zero.result
                - AB::Expr::from_canonical_u32(2) * offset_bit1.clone()
                - offset_bit0.clone())
                * AB::F::from_canonical_u32(4).inverse(),
            AB::Expr::from_canonical_u32(14),
            AB::Expr::zero(),
            is_real.clone(),
        );

        addr.reduce::<AB>() - AB::Expr::from_canonical_u32(2) * offset_bit1 - offset_bit0
    }
}
