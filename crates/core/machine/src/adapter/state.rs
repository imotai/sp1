use p3_field::{AbstractField, Field};
use sp1_derive::AlignedBorrow;

use crate::air::WordAirBuilder;
use sp1_core_executor::events::ByteRecord;
use sp1_stark::air::SP1AirBuilder;

/// A set of columns to describe the state of the CPU.
/// The state is composed of the shard, clock, and the program counter.
/// The clock is split into a 16-bit limb and an 8-bit limb to range check it to 24 bits.
#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
#[repr(C)]
pub struct CPUState<T> {
    pub shard: T,
    pub clk_16bit_limb: T,
    pub clk_8bit_limb: T,
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
        AB::Expr::from_canonical_u32(1 << 16) * self.clk_8bit_limb + self.clk_16bit_limb
    }
}

impl<F: Field> CPUState<F> {
    #[allow(clippy::too_many_arguments)]
    pub fn populate(&mut self, blu_events: &mut impl ByteRecord, shard: u32, clk: u32, pc: u32) {
        let clk_16bit_limb = (clk & 0xffff) as u16;
        let clk_8bit_limb = (clk >> 16) as u8;
        self.shard = F::from_canonical_u32(shard);
        self.clk_16bit_limb = F::from_canonical_u16(clk_16bit_limb);
        self.clk_8bit_limb = F::from_canonical_u8(clk_8bit_limb);
        self.pc = F::from_canonical_u32(pc);
        blu_events.add_u16_range_check(clk_16bit_limb);
        blu_events.add_u8_range_checks(&[clk_8bit_limb]);
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
        // Range check the clock to be 24 bits.
        // Since the clock increment is bounded, the clock will not overflow.
        builder.slice_range_check_u16(&[cols.clk_16bit_limb], is_real.clone());
        builder.slice_range_check_u8(&[cols.clk_8bit_limb], is_real);
    }
}
