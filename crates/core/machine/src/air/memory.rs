use std::iter::once;

use itertools::Itertools;
use p3_air::AirBuilder;
use p3_field::AbstractField;
use sp1_core_executor::ByteOpcode;
use sp1_stark::{
    air::{AirInteraction, BaseAirBuilder, ByteAirBuilder, InteractionScope},
    InteractionKind, Word,
};

use crate::memory::{MemoryAccessCols, MemoryAccessTimestamp};

pub trait MemoryAirBuilder: BaseAirBuilder {
    /// Constrain a memory read, by using the read value as the write value.
    /// The constraints enforce that the new (shard, timestamp) tuple is greater than the previous one.
    fn eval_memory_access_read<E: Into<Self::Expr> + Clone>(
        &mut self,
        shard: impl Into<Self::Expr>,
        clk: impl Into<Self::Expr>,
        addr: impl Into<Self::Expr>,
        mem_access: MemoryAccessCols<E>,
        do_check: impl Into<Self::Expr>,
    ) {
        let do_check: Self::Expr = do_check.into();
        let shard: Self::Expr = shard.into();
        let clk: Self::Expr = clk.into();

        self.assert_bool(do_check.clone());
        // Verify that the current memory access time is greater than the previous's.
        self.eval_memory_access_timestamp(
            &mem_access.access_timestamp,
            do_check.clone(),
            shard.clone(),
            clk.clone(),
        );

        // Add to the memory argument.
        let addr = addr.into();
        let prev_shard = mem_access.access_timestamp.prev_shard.clone().into();
        let prev_clk = mem_access.access_timestamp.prev_clk.clone().into();
        let prev_values = once(prev_shard)
            .chain(once(prev_clk))
            .chain(once(addr.clone()))
            .chain(mem_access.prev_value.clone().map(Into::into))
            .collect();
        let current_values = once(shard)
            .chain(once(clk))
            .chain(once(addr.clone()))
            .chain(mem_access.prev_value.clone().map(Into::into))
            .collect();

        // The previous values get sent with multiplicity = 1, for "read".
        self.send(
            AirInteraction::new(prev_values, do_check.clone(), InteractionKind::Memory),
            InteractionScope::Local,
        );

        // The current values get "received", i.e. multiplicity = -1
        self.receive(
            AirInteraction::new(current_values, do_check.clone(), InteractionKind::Memory),
            InteractionScope::Local,
        );
    }

    /// Constrain a memory write, given the write value.
    /// The constraints enforce that the new (shard, timestamp) tuple is greater than the previous one.
    fn eval_memory_access_write<E: Into<Self::Expr> + Clone>(
        &mut self,
        shard: impl Into<Self::Expr>,
        clk: impl Into<Self::Expr>,
        addr: impl Into<Self::Expr>,
        mem_access: MemoryAccessCols<E>,
        write_value: Word<impl Into<Self::Expr>>,
        do_check: impl Into<Self::Expr>,
    ) {
        let do_check: Self::Expr = do_check.into();
        let shard: Self::Expr = shard.into();
        let clk: Self::Expr = clk.into();

        self.assert_bool(do_check.clone());
        // Verify that the current memory access time is greater than the previous's.
        self.eval_memory_access_timestamp(
            &mem_access.access_timestamp,
            do_check.clone(),
            shard.clone(),
            clk.clone(),
        );

        // Add to the memory argument.
        let addr = addr.into();
        let prev_shard = mem_access.access_timestamp.prev_shard.clone().into();
        let prev_clk = mem_access.access_timestamp.prev_clk.clone().into();
        let prev_values = once(prev_shard)
            .chain(once(prev_clk))
            .chain(once(addr.clone()))
            .chain(mem_access.prev_value.clone().map(Into::into))
            .collect();
        let current_values = once(shard)
            .chain(once(clk))
            .chain(once(addr.clone()))
            .chain(write_value.map(Into::into))
            .collect();

        // The previous values get sent with multiplicity = 1, for "read".
        self.send(
            AirInteraction::new(prev_values, do_check.clone(), InteractionKind::Memory),
            InteractionScope::Local,
        );

        // The current values get "received", i.e. multiplicity = -1
        self.receive(
            AirInteraction::new(current_values, do_check.clone(), InteractionKind::Memory),
            InteractionScope::Local,
        );
    }

    /// Constraints a memory read to a slice of `MemoryAccessCols`.
    fn eval_memory_access_slice_read<E: Into<Self::Expr> + Copy>(
        &mut self,
        shard: impl Into<Self::Expr> + Copy,
        clk: impl Into<Self::Expr> + Clone,
        initial_addr: impl Into<Self::Expr> + Clone,
        memory_access_slice: &[MemoryAccessCols<E>],
        verify_memory_access: impl Into<Self::Expr> + Copy,
    ) {
        for (i, access_slice) in memory_access_slice.iter().enumerate() {
            self.eval_memory_access_read(
                shard,
                clk.clone(),
                initial_addr.clone().into() + Self::Expr::from_canonical_usize(i * 4),
                *access_slice,
                verify_memory_access,
            );
        }
    }

    /// Constraints a memory write to a slice of `MemoryAccessCols`.
    fn eval_memory_access_slice_write<E: Into<Self::Expr> + Copy>(
        &mut self,
        shard: impl Into<Self::Expr> + Copy,
        clk: impl Into<Self::Expr> + Clone,
        initial_addr: impl Into<Self::Expr> + Clone,
        memory_access_slice: &[MemoryAccessCols<E>],
        write_values: Vec<Word<impl Into<Self::Expr>>>,
        verify_memory_access: impl Into<Self::Expr> + Copy,
    ) {
        for (i, (access_slice, write_value)) in
            memory_access_slice.iter().zip_eq(write_values).enumerate()
        {
            self.eval_memory_access_write(
                shard,
                clk.clone(),
                initial_addr.clone().into() + Self::Expr::from_canonical_usize(i * 4),
                *access_slice,
                write_value,
                verify_memory_access,
            );
        }
    }

    /// Verifies the memory access timestamp.
    ///
    /// This method verifies that the current memory access happened after the previous one's.
    /// Specifically it will ensure that if the current and previous access are in the same shard,
    /// then the current's clk val is greater than the previous's.  If they are not in the same
    /// shard, then it will ensure that the current's shard val is greater than the previous's.
    fn eval_memory_access_timestamp(
        &mut self,
        mem_access: &MemoryAccessTimestamp<impl Into<Self::Expr> + Clone>,
        do_check: impl Into<Self::Expr>,
        shard: impl Into<Self::Expr> + Clone,
        clk: impl Into<Self::Expr>,
    ) {
        let do_check: Self::Expr = do_check.into();
        let compare_clk: Self::Expr = mem_access.compare_clk.clone().into();
        let shard: Self::Expr = shard.clone().into();
        let prev_shard: Self::Expr = mem_access.prev_shard.clone().into();

        // First verify that compare_clk's value is correct.
        self.when(do_check.clone()).assert_bool(compare_clk.clone());
        self.when(do_check.clone()).when(compare_clk.clone()).assert_eq(shard.clone(), prev_shard);

        // Get the comparison timestamp values for the current and previous memory access.
        let prev_comp_value = self.if_else(
            mem_access.compare_clk.clone(),
            mem_access.prev_clk.clone(),
            mem_access.prev_shard.clone(),
        );

        let current_comp_val = self.if_else(compare_clk.clone(), clk.into(), shard.clone());

        // Assert `current_comp_val > prev_comp_val`. We check this by asserting that
        // `0 <= current_comp_val-prev_comp_val-1 < 2^28`.
        //
        // The equivalence of these statements comes from the fact that if
        // `current_comp_val <= prev_comp_val`, then `current_comp_val-prev_comp_val-1 < 0` and will
        // underflow in the prime field, resulting in a value that is `>= 2^28` as long as both
        // `current_comp_val, prev_comp_val` are range-checked to be `<2^28` and as long as we're
        // working in a field larger than `2 * 2^28` (which is true of the BabyBear and Mersenne31
        // prime).
        let diff_minus_one = current_comp_val - prev_comp_value - Self::Expr::one();

        // Verify that value = limb_low + limb_high * 2^14.
        self.when(do_check.clone()).assert_eq(
            diff_minus_one,
            mem_access.diff_low_limb.clone().into()
                + mem_access.diff_high_limb.clone().into()
                    * Self::Expr::from_canonical_u32(1 << 14),
        );

        // Send the range checks for the limbs.
        self.send_byte(
            Self::Expr::from_canonical_u8(ByteOpcode::Range as u8),
            mem_access.diff_low_limb.clone(),
            Self::Expr::from_canonical_u32(14),
            Self::Expr::zero(),
            do_check.clone(),
        );

        self.send_byte(
            Self::Expr::from_canonical_u8(ByteOpcode::Range as u8),
            mem_access.diff_high_limb.clone(),
            Self::Expr::from_canonical_u32(14),
            Self::Expr::zero(),
            do_check,
        )
    }
}
