use p3_air::AirBuilder;
use p3_field::{AbstractField, Field, PrimeField32};
use sp1_derive::AlignedBorrow;

use sp1_core_executor::{events::ByteRecord, ByteOpcode};
use sp1_primitives::consts::{u32_to_u16_limbs, BABYBEAR_PRIME};
use sp1_stark::{air::SP1AirBuilder, Word};

use super::{LtOperationUnsigned, LtOperationUnsignedInput, U16CompareOperation};
use crate::air::{SP1Operation, SP1OperationBuilder, WordAirBuilder};

/// A set of columns needed to validate the address and return the aligned address.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct SyscallAddrOperation<T> {
    /// The address in Word form.
    pub addr_word: Word<T>,

    /// This is used to check if the most significant limb of the memory address is zero.
    pub most_sig_limb_inv: T,

    /// The upper bound range check on the `addr_word`. Can be optimized if needed.
    pub range_check: LtOperationUnsigned<T>,
}

impl<F: PrimeField32> SyscallAddrOperation<F> {
    pub fn populate(&mut self, record: &mut impl ByteRecord, addr: u32, len: u32) {
        self.addr_word = Word::from(addr);
        let addr_word_limbs = u32_to_u16_limbs(addr);
        record.add_u16_range_checks(&addr_word_limbs);
        self.most_sig_limb_inv = F::from_canonical_u16(addr_word_limbs[1]).inverse();
        self.range_check.populate_unsigned(record, 1, addr, BABYBEAR_PRIME - len);
        record.add_bit_range_check(addr_word_limbs[0] / 4, 14);
    }
}

impl<F: Field> SyscallAddrOperation<F> {
    /// The memory address is constrained to be aligned, `>= 2^16` and less than `BabyBear - len`.
    #[allow(clippy::too_many_arguments)]
    pub fn eval<AB>(
        builder: &mut AB,
        len: u32,
        cols: SyscallAddrOperation<AB::Var>,
        is_real: AB::Expr,
    ) -> AB::Expr
    where
        AB: SP1AirBuilder
            + SP1OperationBuilder<LtOperationUnsigned<<AB as AirBuilder>::F>>
            + SP1OperationBuilder<U16CompareOperation<<AB as AirBuilder>::F>>,
    {
        // Check that `is_real` and offset bits are boolean.
        builder.assert_bool(is_real.clone());

        // Check that `addr >= 2^16`, so it doesn't touch registers.
        // This implements a stack guard of size 2^16 bytes = 64KB.
        // If `is_real = 1`, then `addr.0[1] != 0`, so `addr >= 2^16`.
        builder.assert_eq(cols.most_sig_limb_inv * cols.addr_word.0[1], is_real.clone());

        // Check `0 <= addr[0] / 4 < 2^14`, which shows `addr` is a multiple of `4` within `u16`.
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::Range as u32),
            cols.addr_word.0[0].into() * AB::F::from_canonical_u32(4).inverse(),
            AB::Expr::from_canonical_u32(14),
            AB::Expr::zero(),
            is_real.clone(),
        );

        builder.slice_range_check_u16(&cols.addr_word.0, is_real.clone());

        // Check that `addr < upper_bound`.
        <LtOperationUnsigned<AB::F> as SP1Operation<AB>>::eval(
            builder,
            LtOperationUnsignedInput::<AB>::new(
                cols.addr_word.map(Into::into),
                Word([
                    AB::Expr::from_canonical_u32((BABYBEAR_PRIME - len) & 0xFFFF),
                    AB::Expr::from_canonical_u32((BABYBEAR_PRIME - len) >> 16),
                ]),
                cols.range_check,
                is_real.clone(),
            ),
        );
        builder.assert_eq(cols.range_check.u16_compare_operation.bit, is_real.clone());

        cols.addr_word.reduce::<AB>()
    }
}
