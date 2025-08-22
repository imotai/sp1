use crate::{events::MemoryInitializeFinalizeEvent, ExecutionRecord, MinimalExecutor};

impl MinimalExecutor {
    #[must_use]
    /// Postprocess the [`JitFunction`] to create an [`ExecutionRecord`],
    /// consisting of all the [`MemoryInitializeFinalizeEvent`]s.
    pub fn postprocess(&self) -> ExecutionRecord {
        // From the program, we can get the initial memory image and create those reocrds
        // inputs / hints are considred uninit memory so those should also be considreded init
        // memory
        //
        // from there we can use "touched_address" to get all the finalized memory records by
        // looking up into the memory.

        // todo get register init and finalize.
        // due to register refresh the init hsould all be 0,0 and the finalize should be the final
        // value , last clk

        // Registers always 0 initzialized.
        let _register_init_events =
            (0..32).map(|reg| MemoryInitializeFinalizeEvent::initialize(reg as u64, 0));

        let _addr_0_init_event = MemoryInitializeFinalizeEvent::initialize(0, 0);

        // todo!!!!
        let _register_finalize_events: Vec<MemoryInitializeFinalizeEvent> = vec![];

        let _memory_image_init_events = self
            .program
            .memory_image
            .iter()
            .map(|(addr, value)| MemoryInitializeFinalizeEvent::initialize(*addr, *value));

        let _hints_init_events =
            self.hints().iter().flat_map(|(addr, value)| chunked_memory_init_events(*addr, value));

        let memory = self.memory();
        let _memory_finalize_events = self.touched_addresses().lock().unwrap().iter().map(|addr| {
            let record = memory.get(*addr);
            MemoryInitializeFinalizeEvent::finalize(*addr, record.value, record.clk)
        });

        todo!()
    }
}

/// Given some contiguous memory, create a series of initialize and finalize events.
///
/// The events are created in chunks of 8 bytes.
///
/// The last chunk is not guaranteed to be 8 bytes, so we need to handle that case by padding with
/// 0s.
fn chunked_memory_init_events(start: u64, bytes: &[u8]) -> Vec<MemoryInitializeFinalizeEvent> {
    let chunks = bytes.chunks_exact(8);
    let num_chunks = chunks.len();
    let last = chunks.remainder();

    let mut output = Vec::with_capacity(num_chunks + 1);

    for (i, chunk) in chunks.enumerate() {
        let addr = start + i as u64 * 8;
        let value = u64::from_le_bytes(chunk.try_into().unwrap());
        output.push(MemoryInitializeFinalizeEvent::initialize(addr, value));
    }

    if !last.is_empty() {
        let addr = start + num_chunks as u64 * 8;
        let buf = {
            let mut buf = [0u8; 8];
            buf[..last.len()].copy_from_slice(last);
            buf
        };

        let value = u64::from_le_bytes(buf);
        output.push(MemoryInitializeFinalizeEvent::initialize(addr, value));
    }

    output
}
