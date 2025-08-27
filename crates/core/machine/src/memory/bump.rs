use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use crate::{
    air::SP1CoreAirBuilder,
    utils::{next_multiple_of_32, zeroed_f_vec},
};

use super::MemoryAccessCols;
use hashbrown::HashMap;
use itertools::Itertools;
use slop_air::{Air, BaseAir};
use slop_algebra::{AbstractField, PrimeField32};
use slop_matrix::{dense::RowMajorMatrix, Matrix};
use sp1_core_executor::{
    events::{ByteLookupEvent, ByteRecord, MemoryReadRecord, MemoryRecordEnum},
    ExecutionRecord, Program,
};
use sp1_derive::AlignedBorrow;
use sp1_hypercube::air::MachineAir;

pub(crate) const NUM_MEMORY_BUMP_COLS: usize = size_of::<MemoryBumpCols<u8>>();

#[derive(AlignedBorrow, Clone, Copy)]
#[repr(C)]
pub struct MemoryBumpCols<T: Copy> {
    pub access: MemoryAccessCols<T>,
    pub clk_high: T,
    pub addr: [T; 3],
    pub is_real: T,
}

pub struct MemoryBumpChip {}

impl MemoryBumpChip {
    pub const fn new() -> Self {
        Self {}
    }
}

impl<F> BaseAir<F> for MemoryBumpChip {
    fn width(&self) -> usize {
        NUM_MEMORY_BUMP_COLS
    }
}

impl<F: PrimeField32> MachineAir<F> for MemoryBumpChip {
    type Record = ExecutionRecord;

    type Program = Program;

    fn name(&self) -> String {
        "MemoryBump".to_string()
    }

    fn generate_dependencies(&self, input: &Self::Record, output: &mut Self::Record) {
        let chunk_size = 1;
        let event_iter = input.bump_memory_events.chunks(chunk_size);

        let blu_batches = event_iter
            .map(|events| {
                let mut blu: HashMap<ByteLookupEvent, usize> = HashMap::new();
                events.iter().for_each(|(event, addr)| {
                    let mut row = [F::zero(); NUM_MEMORY_BUMP_COLS];
                    let cols: &mut MemoryBumpCols<F> = row.as_mut_slice().borrow_mut();
                    let bump_event = MemoryRecordEnum::Read(MemoryReadRecord {
                        value: event.prev_value(),
                        prev_timestamp: event.previous_record().timestamp,
                        timestamp: (event.current_record().timestamp >> 24) << 24,
                        prev_page_prot_record: None,
                    });
                    cols.access.populate(bump_event, &mut blu);
                    blu.add_u16_range_checks(&[
                        (addr & 0xFFFF) as u16,
                        (addr >> 16) as u16,
                        (addr >> 32) as u16,
                    ]);
                });
                blu
            })
            .collect::<Vec<_>>();

        output.add_byte_lookup_events_from_maps(blu_batches.iter().collect_vec());
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        let nb_rows = input.bump_memory_events.len();
        let size_log2 = input.fixed_log2_rows::<F, _>(self);
        Some(next_multiple_of_32(nb_rows, size_log2))
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> RowMajorMatrix<F> {
        let chunk_size = 1;
        let padded_nb_rows = <MemoryBumpChip as MachineAir<F>>::num_rows(self, input).unwrap();
        let mut values = zeroed_f_vec(padded_nb_rows * NUM_MEMORY_BUMP_COLS);

        values.chunks_mut(chunk_size * NUM_MEMORY_BUMP_COLS).enumerate().for_each(|(i, rows)| {
            rows.chunks_mut(NUM_MEMORY_BUMP_COLS).enumerate().for_each(|(j, row)| {
                let idx = i * chunk_size + j;
                let cols: &mut MemoryBumpCols<F> = row.borrow_mut();

                if idx < input.bump_memory_events.len() {
                    let mut byte_lookup_events = Vec::new();
                    let (event, addr) = input.bump_memory_events[idx];
                    let bump_event = MemoryRecordEnum::Read(MemoryReadRecord {
                        value: event.prev_value(),
                        prev_timestamp: event.previous_record().timestamp,
                        timestamp: (event.current_record().timestamp >> 24) << 24,
                        prev_page_prot_record: None,
                    });
                    cols.access.populate(bump_event, &mut byte_lookup_events);
                    cols.clk_high =
                        F::from_canonical_u32((event.current_record().timestamp >> 24) as u32);
                    cols.addr = [
                        F::from_canonical_u16((addr & 0xFFFF) as u16),
                        F::from_canonical_u16((addr >> 16) as u16),
                        F::from_canonical_u16((addr >> 32) as u16),
                    ];
                    cols.is_real = F::one();
                }
            });
        });

        // Convert the trace to a row major matrix.
        RowMajorMatrix::new(values, NUM_MEMORY_BUMP_COLS)
    }

    fn included(&self, shard: &Self::Record) -> bool {
        shard.cpu_event_count != 0
    }
}

impl<AB> Air<AB> for MemoryBumpChip
where
    AB: SP1CoreAirBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &MemoryBumpCols<AB::Var> = (*local).borrow();

        // Check that `is_real` is a boolean value.
        builder.assert_bool(local.is_real);

        // Check that the address is a valid u48 address.
        builder.slice_range_check_u16(&local.addr.map(Into::into), local.is_real);

        // Bump the memory timestamp to a multiple of 2^24, by doing an additional read.
        builder.eval_memory_access_read(
            local.clk_high,
            AB::Expr::zero(),
            &local.addr.map(Into::into),
            local.access,
            local.is_real,
        );
    }
}
