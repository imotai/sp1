use p3_field::{AbstractField, Field};
use sp1_core_executor::{events::ByteRecord, ByteOpcode};
use sp1_derive::AlignedBorrow;
use sp1_stark::air::SP1AirBuilder;

/// A set of columns to describe the state of the CPU.
/// The state is composed of the shard, clock, and the program counter.
/// The clock is split into two 14-limb bits to range check it to 28 bits.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct CPUState<T> {
    pub shard: T,
    pub clk_high_limb: T,
    pub clk_low_limb: T,
    pub pc: T,
}

impl<T: Copy> CPUState<T> {
    pub fn shard<AB>(&self) -> AB::Expr
    where
        AB: SP1AirBuilder<Var = T>,
        T: Into<AB::Expr>,
    {
        self.shard.into()
    }
    pub fn clk<AB: SP1AirBuilder<Var = T>>(&self) -> AB::Expr {
        AB::Expr::from_canonical_u32(1 << 14) * self.clk_high_limb + self.clk_low_limb
    }
}

impl<F: Field> CPUState<F> {
    #[allow(clippy::too_many_arguments)]
    pub fn populate(&mut self, blu_events: &mut impl ByteRecord, shard: u32, clk: u32, pc: u64) {
        let clk_high_limb = (clk >> 14) as u16;
        let clk_low_limb = (clk & ((1 << 14) - 1)) as u16;
        self.shard = F::from_canonical_u32(shard);
        self.clk_high_limb = F::from_canonical_u16(clk_high_limb);
        self.clk_low_limb = F::from_canonical_u16(clk_low_limb);
        self.pc = F::from_canonical_u64(pc);
        blu_events.add_bit_range_check(clk_high_limb, 14);
        blu_events.add_bit_range_check(clk_low_limb, 14);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn eval<AB: SP1AirBuilder>(
        builder: &mut AB,
        cols: CPUState<AB::Var>,
        next_pc: AB::Expr,
        clk_increment: AB::Expr,
        is_real: AB::Expr,
    ) {
        let shard = cols.shard::<AB>();
        let clk = cols.clk::<AB>();
        builder.assert_bool(is_real.clone());
        builder.receive_state(shard.clone(), clk.clone(), cols.pc.into(), is_real.clone());
        builder.send_state(shard, clk + clk_increment, next_pc, is_real.clone());
        // Range check the clock to be 28 bits.
        // Since the clock increment is bounded, the clock will not overflow.
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::Range as u32),
            cols.clk_high_limb.into(),
            AB::Expr::from_canonical_u32(14),
            AB::Expr::zero(),
            is_real.clone(),
        );
        builder.send_byte(
            AB::Expr::from_canonical_u32(ByteOpcode::Range as u32),
            cols.clk_low_limb.into(),
            AB::Expr::from_canonical_u32(14),
            AB::Expr::zero(),
            is_real.clone(),
        );
    }
}
