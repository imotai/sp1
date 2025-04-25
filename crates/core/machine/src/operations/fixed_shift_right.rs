use p3_air::AirBuilder;
use p3_field::{AbstractField, Field};
use sp1_core_executor::{events::ByteRecord, ByteOpcode};
use sp1_derive::AlignedBorrow;
use sp1_primitives::consts::{u32_to_u16_limbs, WORD_SIZE};
use sp1_stark::{air::SP1AirBuilder, Word};

/// A set of columns needed to compute `>>` of a word with a fixed offset R.
///
/// Note that we decompose shifts into a limb shift and a bit shift.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct FixedShiftRightOperation<T> {
    /// The output value.
    pub value: Word<T>,

    /// The higher bits of each limb.
    pub higher_limb: Word<T>,
}

impl<F: Field> FixedShiftRightOperation<F> {
    pub const fn nb_limbs_to_shift(rotation: usize) -> usize {
        rotation / 16
    }

    pub const fn nb_bits_to_shift(rotation: usize) -> usize {
        rotation % 16
    }

    pub const fn carry_multiplier(rotation: usize) -> u32 {
        let nb_bits_to_shift = Self::nb_bits_to_shift(rotation);
        1 << (16 - nb_bits_to_shift)
    }

    pub fn populate(&mut self, record: &mut impl ByteRecord, input: u32, rotation: usize) -> u32 {
        let input_limbs = u32_to_u16_limbs(input);
        let expected = input >> rotation;
        self.value = Word::from(expected);

        // Compute some constants with respect to the rotation needed for the rotation.
        let nb_limbs_to_shift = Self::nb_limbs_to_shift(rotation);
        let nb_bits_to_shift = Self::nb_bits_to_shift(rotation);

        // Perform the limb shift.
        let mut word = [0u16; 2];
        for i in 0..WORD_SIZE {
            if i + nb_limbs_to_shift < WORD_SIZE {
                word[i] = input_limbs[i + nb_limbs_to_shift];
            }
        }

        for i in (0..WORD_SIZE).rev() {
            let limb = word[i];
            let lower_limb = (limb & ((1 << nb_bits_to_shift) - 1)) as u16;
            let higher_limb = (limb >> nb_bits_to_shift) as u16;
            self.higher_limb.0[i] = F::from_canonical_u16(higher_limb);
            record.add_bit_range_check(lower_limb, nb_bits_to_shift as u8);
            record.add_bit_range_check(higher_limb, (16 - nb_bits_to_shift) as u8);
        }

        expected
    }

    pub fn eval<AB: SP1AirBuilder>(
        builder: &mut AB,
        input: Word<AB::Var>,
        rotation: usize,
        cols: FixedShiftRightOperation<AB::Var>,
        is_real: AB::Var,
    ) {
        builder.assert_bool(is_real);

        // Compute some constants with respect to the rotation needed for the rotation.
        let nb_limbs_to_shift = Self::nb_limbs_to_shift(rotation);
        let nb_bits_to_shift = Self::nb_bits_to_shift(rotation);
        let carry_multiplier = AB::F::from_canonical_u32(Self::carry_multiplier(rotation));

        // Perform the limb shift.
        let input_limbs_rotated = Word(std::array::from_fn(|i| {
            if i + nb_limbs_to_shift < WORD_SIZE {
                input[i + nb_limbs_to_shift].into()
            } else {
                AB::Expr::zero()
            }
        }));

        // For each limb, constrain the lower and higher bits.
        let mut first_shift = AB::Expr::zero();
        let mut last_carry = AB::Expr::zero();
        for i in (0..WORD_SIZE).rev() {
            let limb = input_limbs_rotated[i].clone();
            // Break down the limb into lower and higher parts.
            //  - `limb = lower_limb + higher_limb * 2^bit_shift`
            //  - `lower_limb < 2^(bit_shift)`
            //  - `higher_limb < 2^(16 - bit_shift)`
            let lower_limb =
                limb - cols.higher_limb[i] * AB::Expr::from_canonical_u32(1 << nb_bits_to_shift);
            // Check that `lower_limb < 2^(bit_shift)`
            builder.send_byte(
                AB::F::from_canonical_u32(ByteOpcode::Range as u32),
                lower_limb.clone(),
                AB::F::from_canonical_u32(nb_bits_to_shift as u32),
                AB::Expr::zero(),
                is_real,
            );
            // Check that `higher_limb < 2^(16 - bit_shift)`
            builder.send_byte(
                AB::F::from_canonical_u32(ByteOpcode::Range as u32),
                cols.higher_limb[i],
                AB::Expr::from_canonical_u32(16 - nb_bits_to_shift as u32),
                AB::Expr::zero(),
                is_real,
            );
            if i == WORD_SIZE - 1 {
                first_shift = cols.higher_limb[i].into();
            } else {
                builder
                    .when(is_real)
                    .assert_eq(cols.value[i], cols.higher_limb[i] + last_carry * carry_multiplier);
            }
            last_carry = lower_limb;
        }

        // For the first limb, we don't move over the carry as this is a shift, not a rotate.
        builder.when(is_real).assert_eq(cols.value[WORD_SIZE - 1], first_shift);
    }
}
