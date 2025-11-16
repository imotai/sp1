use crate::air::WordAirBuilder;
use slop_air::AirBuilder;
use slop_algebra::{AbstractExtensionField, AbstractField, Field, PrimeField32};
use sp1_core_executor::{
    events::{ByteLookupEvent, ByteRecord},
    ByteOpcode,
};
use sp1_derive::AlignedBorrow;
use sp1_hypercube::{
    air::SP1AirBuilder,
    operations::poseidon2::{
        air::{eval_external_round, eval_internal_rounds},
        permutation::Poseidon2Cols,
        trace::populate_perm_deg3,
        Poseidon2Operation, NUM_EXTERNAL_ROUNDS,
    },
    septic_curve::{SepticCurve, CURVE_WITNESS_DUMMY_POINT_X, CURVE_WITNESS_DUMMY_POINT_Y},
    septic_extension::{SepticBlock, SepticExtension},
};

/// A set of columns needed to compute the global interaction elliptic curve digest.
#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct GlobalInteractionOperation<T: Copy> {
    pub x_coordinate: SepticBlock<T>,
    pub y_coordinate: SepticBlock<T>,
    pub permutation: Poseidon2Operation<T>,
    pub offset: T,
    pub y6_byte_decomp: [T; 4],
}

impl<F: PrimeField32> GlobalInteractionOperation<F> {
    pub fn get_digest(
        values: [u32; 8],
        is_receive: bool,
        kind: u8,
    ) -> (SepticCurve<F>, u8, [F; 16], [F; 16]) {
        let mut new_values = values.map(|x| F::from_canonical_u32(x));
        new_values[0] = new_values[0] + F::from_canonical_u32((kind as u32) << 24);
        let (point, offset, m_trial, m_hash) = SepticCurve::<F>::lift_x(new_values);
        if !is_receive {
            return (point.neg(), offset, m_trial, m_hash);
        }
        (point, offset, m_trial, m_hash)
    }

    pub fn populate(
        &mut self,
        blu: &mut impl ByteRecord,
        values: [u32; 8],
        is_receive: bool,
        is_real: bool,
        kind: u8,
    ) {
        if is_real {
            let (point, offset, m_trial, m_hash) = Self::get_digest(values, is_receive, kind);
            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::U8Range,
                a: 0,
                b: 0,
                c: offset,
            });
            self.offset = F::from_canonical_u8(offset);
            self.x_coordinate = SepticBlock::<F>::from(point.x.0);
            self.y_coordinate = SepticBlock::<F>::from(point.y.0);
            let range_check_value = if is_receive {
                point.y.0[6].as_canonical_u32() - 1
            } else {
                F::ORDER_U32 - point.y.0[6].as_canonical_u32() - 1
            };
            assert!(range_check_value < 63 * (1 << 24));
            for i in 0..3 {
                let byte = ((range_check_value >> (8 * i)) & 0xFF) as u8;
                self.y6_byte_decomp[i] = F::from_canonical_u8(byte);
                blu.add_byte_lookup_event(ByteLookupEvent {
                    opcode: ByteOpcode::U8Range,
                    a: 0,
                    b: 0,
                    c: byte,
                });
            }
            let last_byte = (range_check_value >> 24) as u8;
            self.y6_byte_decomp[3] = F::from_canonical_u8(last_byte);
            blu.add_byte_lookup_event(ByteLookupEvent {
                opcode: ByteOpcode::LTU,
                a: 1,
                b: last_byte,
                c: 63,
            });
            self.permutation = populate_perm_deg3(m_trial, Some(m_hash));

            assert_eq!(self.x_coordinate.0[0], self.permutation.permutation.perm_output()[0]);
        } else {
            self.populate_dummy();
            assert_eq!(self.x_coordinate.0[0], self.permutation.permutation.perm_output()[0]);
        }
    }

    pub fn populate_dummy(&mut self) {
        self.x_coordinate = SepticBlock::<F>::from_base_fn(|i| {
            F::from_canonical_u32(CURVE_WITNESS_DUMMY_POINT_X[i])
        });
        self.y_coordinate = SepticBlock::<F>::from_base_fn(|i| {
            F::from_canonical_u32(CURVE_WITNESS_DUMMY_POINT_Y[i])
        });
        self.offset = F::zero();
        for i in 0..4 {
            self.y6_byte_decomp[i] = F::zero();
        }
        self.permutation = populate_perm_deg3([F::zero(); 16], None);
    }
}

impl<F: Field> GlobalInteractionOperation<F> {
    /// Constrain that the elliptic curve point for the global interaction is correctly derived.
    #[allow(clippy::too_many_arguments)]
    pub fn eval_single_digest<AB: SP1AirBuilder + slop_air::PairBuilder>(
        builder: &mut AB,
        values: [AB::Expr; 8],
        cols: GlobalInteractionOperation<AB::Var>,
        is_receive: AB::Expr,
        is_send: AB::Expr,
        is_real: AB::Var,
        kind: AB::Var,
        message_0_limbs: [AB::Var; 2],
    ) {
        // Constrain that the `is_real` is boolean.
        builder.assert_bool(is_real);
        builder.when(is_real).assert_eq(is_receive.clone() + is_send.clone(), AB::Expr::one());
        builder.assert_bool(is_receive.clone());
        builder.assert_bool(is_send.clone());

        // Ensure that the offset is a byte.
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::U8Range as u32),
            AB::Expr::zero(),
            AB::Expr::zero(),
            cols.offset.into(),
            is_real.into(),
        );

        // Range check the first element in the message to be 24 bits so that we can encode the
        // interaction kind in the upper bits.
        builder.when(is_real).assert_eq(
            values[0].clone(),
            message_0_limbs[0] + message_0_limbs[1] * AB::F::from_canonical_u32(1 << 16),
        );
        builder.slice_range_check_u16(&[message_0_limbs[0].into(), values[7].clone()], is_real);
        builder.slice_range_check_u8(&[message_0_limbs[1]], is_real);
        // Range check that the `kind` is at most 6 bits.
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::Range as u32),
            kind.into(),
            AB::Expr::from_canonical_u32(6),
            AB::Expr::zero(),
            is_real.into(),
        );

        // Turn the message into a hash input. Only the first 8 elements are non-zero, as the rate
        // of the Poseidon2 hash is 8. Combining `values[0]` with `kind` is safe, as
        // `values[0]` is range checked to be 24 bits, and `kind` is known to be 6 bits.
        // Combining `values[7]` with `offset` is also safe, since `values[7]` is range checked
        // to be 16 bits, while `offset` is known to be 8 bits.
        let m_trial = [
            values[0].clone() + AB::Expr::from_canonical_u32(1 << 24) * kind,
            values[1].clone(),
            values[2].clone(),
            values[3].clone(),
            values[4].clone(),
            values[5].clone(),
            values[6].clone(),
            values[7].clone() + AB::Expr::from_canonical_u32(1 << 16) * cols.offset,
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
            AB::Expr::zero(),
        ];

        // Constrain the input of the permutation to be the message.
        for i in 0..16 {
            builder.when(is_real).assert_eq(
                cols.permutation.permutation.external_rounds_state()[0][i].into(),
                m_trial[i].clone(),
            );
        }

        // Constrain the permutation.
        for r in 0..NUM_EXTERNAL_ROUNDS {
            eval_external_round(builder, &cols.permutation.permutation, r);
        }
        eval_internal_rounds(builder, &cols.permutation.permutation);

        // Constrain that when `is_real` is true, the x-coordinate is the hash of the message.
        let m_hash = cols.permutation.permutation.perm_output();
        for i in 0..7 {
            builder.when(is_real).assert_eq(cols.x_coordinate[i].into(), m_hash[i]);
        }
        let x = SepticExtension::<AB::Expr>::from_base_fn(|i| cols.x_coordinate[i].into());
        let y = SepticExtension::<AB::Expr>::from_base_fn(|i| cols.y_coordinate[i].into());

        // Constrain that `(x, y)` is a valid point on the curve.
        let y2 = y.square();
        let x3_2x_26z5 = SepticCurve::<AB::Expr>::curve_formula(x);
        builder.assert_septic_ext_eq(y2, x3_2x_26z5);

        // Constrain that `0 <= y6_value < 63 * 2^24 < (p - 1) / 2`.
        let mut y6_value = AB::Expr::zero();
        for i in 0..3 {
            y6_value =
                y6_value + cols.y6_byte_decomp[i] * AB::Expr::from_canonical_u32(1 << (8 * i));
            builder.send_byte(
                AB::Expr::from_canonical_u32(ByteOpcode::U8Range as u32),
                AB::Expr::zero(),
                AB::Expr::zero(),
                cols.y6_byte_decomp[i].into(),
                is_real.into(),
            );
        }
        y6_value = y6_value + cols.y6_byte_decomp[3] * AB::Expr::from_canonical_u32(1 << 24);
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::LTU as u32),
            AB::Expr::one(),
            cols.y6_byte_decomp[3].into(),
            AB::Expr::from_canonical_u8(63),
            is_real.into(),
        );

        // Constrain that y has correct sign.
        // If it's a receive: `1 <= y_6 <= 63 * 2^24`, and `y_6 == y6_value + 1`.
        // If it's a send: `p - 63 * 2^24 <= y_6 <= p - 1`, and `y_6 = p - 1 - y6_value`.
        builder.when(is_receive).assert_eq(y.0[6].clone(), AB::Expr::one() + y6_value.clone());
        builder.when(is_send).assert_zero(y.0[6].clone() + AB::Expr::one() + y6_value.clone());
    }
}
