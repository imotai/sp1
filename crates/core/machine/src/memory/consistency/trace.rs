use p3_field::PrimeField32;
use sp1_core_executor::events::{ByteRecord, MemoryRecord, MemoryRecordEnum};

use super::{
    MemoryAccessCols, MemoryAccessColsU8, MemoryAccessInShardCols, MemoryAccessInShardTimestamp,
    MemoryAccessTimestamp,
};

impl<F: PrimeField32> MemoryAccessCols<F> {
    pub fn populate(&mut self, record: MemoryRecordEnum, output: &mut impl ByteRecord) {
        let prev_record = record.previous_record();
        let current_record = record.current_record();
        self.prev_value = prev_record.value.into();
        self.access_timestamp.populate_timestamp(prev_record, current_record, output);
    }
}

impl<F: PrimeField32> MemoryAccessInShardCols<F> {
    pub fn populate(&mut self, record: MemoryRecordEnum, output: &mut impl ByteRecord) {
        let prev_record = record.previous_record();
        let current_record = record.current_record();
        self.prev_value = prev_record.value.into();
        self.access_timestamp.populate_timestamp(prev_record, current_record, output);
    }
}

impl<F: PrimeField32> MemoryAccessColsU8<F> {
    pub fn populate(&mut self, record: MemoryRecordEnum, output: &mut impl ByteRecord) {
        let prev_record = record.previous_record();
        let current_record = record.current_record();
        self.memory_access.prev_value = prev_record.value.into();
        // self.prev_value_u8.populate_u16_to_u8_safe(output, prev_record.value);
        self.memory_access.access_timestamp.populate_timestamp(prev_record, current_record, output);
    }
}

impl<F: PrimeField32> MemoryAccessTimestamp<F> {
    pub fn populate_timestamp(
        &mut self,
        prev_record: MemoryRecord,
        current_record: MemoryRecord,
        output: &mut impl ByteRecord,
    ) {
        self.prev_shard = F::from_canonical_u32(prev_record.shard);
        self.prev_clk = F::from_canonical_u32(prev_record.timestamp);

        // Fill columns used for verifying current memory access time value is greater than
        // previous's.
        let use_clk_comparison = prev_record.shard == current_record.shard;
        self.compare_clk = F::from_bool(use_clk_comparison);
        let prev_time_value =
            if use_clk_comparison { prev_record.timestamp } else { prev_record.shard };
        let current_time_value =
            if use_clk_comparison { current_record.timestamp } else { current_record.shard };

        let diff_minus_one = current_time_value - prev_time_value - 1;
        let diff_low_limb = (diff_minus_one & ((1 << 14) - 1)) as u16;
        self.diff_low_limb = F::from_canonical_u16(diff_low_limb);
        let diff_high_limb = (diff_minus_one >> 14) as u16;
        self.diff_high_limb = F::from_canonical_u16(diff_high_limb);

        // Add a byte table lookup with the u16 range check.
        output.add_bit_range_check(diff_low_limb, 14);
        output.add_bit_range_check(diff_high_limb, 14);
    }
}

impl<F: PrimeField32> MemoryAccessInShardTimestamp<F> {
    pub fn populate_timestamp(
        &mut self,
        prev_record: MemoryRecord,
        current_record: MemoryRecord,
        output: &mut impl ByteRecord,
    ) {
        let old_timestamp =
            if prev_record.shard == current_record.shard { prev_record.timestamp } else { 0 };
        self.prev_clk = F::from_canonical_u32(old_timestamp);
        let diff_minus_one = current_record.timestamp - old_timestamp - 1;
        let diff_low_limb = (diff_minus_one & ((1 << 14) - 1)) as u16;
        self.diff_low_limb = F::from_canonical_u16(diff_low_limb);
        let diff_high_limb = (diff_minus_one >> 14) as u16;

        // Add a byte table lookup with the u16 range check.
        output.add_bit_range_check(diff_low_limb, 14);
        output.add_bit_range_check(diff_high_limb, 14);
    }
}
