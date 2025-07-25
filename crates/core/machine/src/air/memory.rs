use std::iter::once;

use itertools::Itertools;
use slop_air::AirBuilder;
use slop_algebra::{AbstractField, Field};
use sp1_core_executor::ByteOpcode;
use sp1_stark::{
    air::{AirInteraction, BaseAirBuilder, ByteAirBuilder, InteractionScope},
    InteractionKind, Word,
};

use crate::memory::{
    MemoryAccessCols, MemoryAccessInShardCols, MemoryAccessInShardTimestamp, MemoryAccessTimestamp,
};

pub trait MemoryAirBuilder: BaseAirBuilder {
    /// Constrain a memory read, by using the read value as the write value.
    /// The constraints enforce that the new timestamp is greater than the previous one.
    fn eval_memory_access_read<E: Into<Self::Expr> + Clone>(
        &mut self,
        clk_high: impl Into<Self::Expr>,
        clk_low: impl Into<Self::Expr>,
        addr: &[Self::Expr; 3],
        mem_access: MemoryAccessCols<E>,
        do_check: impl Into<Self::Expr>,
    ) {
        let do_check: Self::Expr = do_check.into();
        let clk_high: Self::Expr = clk_high.into();
        let clk_low: Self::Expr = clk_low.into();

        self.assert_bool(do_check.clone());
        // Verify that the current memory access time is greater than the previous's.
        self.eval_memory_access_timestamp(
            &mem_access.access_timestamp,
            do_check.clone(),
            clk_high.clone(),
            clk_low.clone(),
        );

        // Add to the memory argument.
        // let addr = addr.into();
        let prev_high = mem_access.access_timestamp.prev_high.clone().into();
        let prev_low = mem_access.access_timestamp.prev_low.clone().into();
        let prev_values = once(prev_high)
            .chain(once(prev_low))
            .chain(addr.clone())
            .chain(mem_access.prev_value.clone().map(Into::into))
            .collect();
        let current_values = once(clk_high)
            .chain(once(clk_low))
            .chain(addr.clone())
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
    /// The constraints enforce that the new (shard, timestamp) tuple is greater than the previous
    /// one.
    fn eval_memory_access_write<E: Into<Self::Expr> + Clone>(
        &mut self,
        clk_high: impl Into<Self::Expr>,
        clk_low: impl Into<Self::Expr>,
        addr: &[Self::Expr; 3],
        mem_access: MemoryAccessCols<E>,
        write_value: Word<impl Into<Self::Expr>>,
        do_check: impl Into<Self::Expr>,
    ) {
        let do_check: Self::Expr = do_check.into();
        let clk_high: Self::Expr = clk_high.into();
        let clk_low: Self::Expr = clk_low.into();

        self.assert_bool(do_check.clone());
        // Verify that the current memory access time is greater than the previous's.
        self.eval_memory_access_timestamp(
            &mem_access.access_timestamp,
            do_check.clone(),
            clk_high.clone(),
            clk_low.clone(),
        );

        // Add to the memory argument.
        let prev_high = mem_access.access_timestamp.prev_high.clone().into();
        let prev_low = mem_access.access_timestamp.prev_low.clone().into();
        let prev_values = once(prev_high)
            .chain(once(prev_low))
            .chain(addr.clone())
            .chain(mem_access.prev_value.clone().map(Into::into))
            .collect();
        let current_values = once(clk_high)
            .chain(once(clk_low))
            .chain(addr.clone())
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

    /// Constrain a memory read, by using the read value as the write value.
    /// The constraints enforce that the new (shard, timestamp) tuple is greater than the previous
    /// one. Used for cases where the previous shard is equal to the shard.
    fn eval_memory_access_in_shard_read<E: Into<Self::Expr> + Clone>(
        &mut self,
        clk_high: impl Into<Self::Expr>,
        clk_low: impl Into<Self::Expr>,
        addr: [Self::Expr; 3],
        mem_access: MemoryAccessInShardCols<E>,
        do_check: impl Into<Self::Expr>,
    ) {
        let do_check: Self::Expr = do_check.into();
        let clk_high: Self::Expr = clk_high.into();
        let clk_low: Self::Expr = clk_low.into();

        self.assert_bool(do_check.clone());
        // Verify that the current memory access time is greater than the previous's.
        self.eval_memory_access_in_shard_timestamp(
            &mem_access.access_timestamp,
            do_check.clone(),
            clk_low.clone(),
        );

        // Add to the memory argument.
        // let addr = addr.into();

        let prev_low = mem_access.access_timestamp.prev_low.clone().into();
        let prev_values = once(clk_high.clone())
            .chain(once(prev_low))
            .chain(addr.clone())
            .chain(mem_access.prev_value.clone().map(Into::into))
            .collect();
        let current_values = once(clk_high)
            .chain(once(clk_low))
            .chain(addr.clone())
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
    /// The constraints enforce that the new (shard, timestamp) tuple is greater than the previous
    /// one. Used for cases where the previous shard is equal to the shard.
    fn eval_memory_access_in_shard_write<E: Into<Self::Expr> + Clone>(
        &mut self,
        clk_high: impl Into<Self::Expr>,
        clk_low: impl Into<Self::Expr>,
        addr: [Self::Expr; 3],
        mem_access: MemoryAccessInShardCols<E>,
        write_value: Word<impl Into<Self::Expr>>,
        do_check: impl Into<Self::Expr>,
    ) {
        let do_check: Self::Expr = do_check.into();
        let clk_high: Self::Expr = clk_high.into();
        let clk_low: Self::Expr = clk_low.into();

        self.assert_bool(do_check.clone());
        // Verify that the current memory access time is greater than the previous's.
        self.eval_memory_access_in_shard_timestamp(
            &mem_access.access_timestamp,
            do_check.clone(),
            clk_low.clone(),
        );

        // Add to the memory argument.
        // let addr = addr.into();

        let prev_low = mem_access.access_timestamp.prev_low.clone().into();
        let prev_values = once(clk_high.clone())
            .chain(once(prev_low))
            .chain(addr.clone())
            .chain(mem_access.prev_value.clone().map(Into::into))
            .collect();
        let current_values = once(clk_high)
            .chain(once(clk_low))
            .chain(addr.clone())
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
        clk_high: impl Into<Self::Expr> + Clone,
        clk_low: impl Into<Self::Expr> + Clone,
        addr_slice: &[[Self::Expr; 3]],
        memory_access_slice: &[MemoryAccessCols<E>],
        verify_memory_access: impl Into<Self::Expr> + Copy,
    ) {
        for (access_slice, addr) in memory_access_slice.iter().zip(addr_slice) {
            self.eval_memory_access_read(
                clk_high.clone(),
                clk_low.clone(),
                addr,
                *access_slice,
                verify_memory_access,
            );
        }
    }

    /// Constraints a memory write to a slice of `MemoryAccessCols`.
    fn eval_memory_access_slice_write<E: Into<Self::Expr> + Copy>(
        &mut self,
        clk_high: impl Into<Self::Expr> + Clone,
        clk_low: impl Into<Self::Expr> + Clone,
        addr_slice: &[[Self::Expr; 3]],
        memory_access_slice: &[MemoryAccessCols<E>],
        write_values: Vec<Word<impl Into<Self::Expr>>>,
        verify_memory_access: impl Into<Self::Expr> + Copy,
    ) {
        for ((access_slice, addr), write_value) in
            memory_access_slice.iter().zip_eq(addr_slice).zip_eq(write_values)
        {
            self.eval_memory_access_write(
                clk_high.clone(),
                clk_low.clone(),
                addr,
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
        clk_high: impl Into<Self::Expr> + Clone,
        clk_low: impl Into<Self::Expr>,
    ) {
        let do_check: Self::Expr = do_check.into();
        let compare_low: Self::Expr = mem_access.compare_low.clone().into();
        let clk_high: Self::Expr = clk_high.clone().into();
        let prev_high: Self::Expr = mem_access.prev_high.clone().into();

        // First verify that compare_clk's value is correct.
        self.when(do_check.clone()).assert_bool(compare_low.clone());
        self.when(do_check.clone())
            .when(compare_low.clone())
            .assert_eq(clk_high.clone(), prev_high);

        // Get the comparison timestamp values for the current and previous memory access.
        let prev_comp_value = self.if_else(
            compare_low.clone(),
            mem_access.prev_low.clone(),
            mem_access.prev_high.clone(),
        );

        let current_comp_val = self.if_else(compare_low.clone(), clk_low.into(), clk_high.clone());

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
                    * Self::Expr::from_canonical_u32(1 << 16),
        );

        // Send the range checks for the limbs.
        self.send_byte(
            Self::Expr::from_canonical_u8(ByteOpcode::Range as u8),
            mem_access.diff_low_limb.clone(),
            Self::Expr::from_canonical_u32(16),
            Self::Expr::zero(),
            do_check.clone(),
        );

        self.send_byte(
            Self::Expr::from_canonical_u8(ByteOpcode::U8Range as u8),
            Self::Expr::zero(),
            mem_access.diff_high_limb.clone(),
            Self::Expr::zero(),
            do_check,
        )
    }

    /// Verifies the in-shard memory access timestamp.
    ///
    /// This method verifies that the current memory access happened after the previous one's.
    /// Specifically it will ensure that the current's clk val is greater than the previous's.
    fn eval_memory_access_in_shard_timestamp(
        &mut self,
        mem_access: &MemoryAccessInShardTimestamp<impl Into<Self::Expr> + Clone>,
        do_check: impl Into<Self::Expr>,
        clk: impl Into<Self::Expr>,
    ) {
        let do_check: Self::Expr = do_check.into();

        let diff_minus_one = clk.into() - mem_access.prev_low.clone().into() - Self::Expr::one();
        let diff_high_limb = (diff_minus_one.clone() - mem_access.diff_low_limb.clone().into())
            * Self::F::from_canonical_u32(1 << 16).inverse();

        // Send the range checks for the limbs.
        self.send_byte(
            Self::Expr::from_canonical_u8(ByteOpcode::Range as u8),
            mem_access.diff_low_limb.clone(),
            Self::Expr::from_canonical_u32(16),
            Self::Expr::zero(),
            do_check.clone(),
        );

        self.send_byte(
            Self::Expr::from_canonical_u8(ByteOpcode::U8Range as u8),
            Self::Expr::zero(),
            diff_high_limb,
            Self::Expr::zero(),
            do_check,
        )
    }
}
