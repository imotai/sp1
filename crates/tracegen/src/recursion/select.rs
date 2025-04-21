use csl_cuda::{args, TaskScope};
use csl_cuda::{TracegenPreprocessedRecursionSelectKernel, TracegenRecursionSelectKernel};
use slop_air::BaseAir;
use slop_alloc::mem::CopyError;
use slop_alloc::Buffer;
use slop_multilinear::Mle;
use slop_tensor::Tensor;
use sp1_recursion_executor::Instruction;
use sp1_recursion_machine::chips::select::SelectChip;
use sp1_stark::air::MachineAir;

use crate::{CudaTracegenAir, F};

impl CudaTracegenAir<F> for SelectChip {
    fn supports_device_preprocessed_tracegen(&self) -> bool {
        true
    }

    async fn generate_preprocessed_trace_device(
        &self,
        program: &Self::Program,
        scope: &TaskScope,
    ) -> Result<Option<Mle<F, TaskScope>>, CopyError> {
        let instrs = program
            .inner
            .iter() // Faster than using `rayon` for some reason. Maybe vectorization?
            .filter_map(|instruction| match instruction {
                Instruction::Select(instr) => Some(*instr),
                _ => None,
            })
            .collect::<Vec<_>>();

        let instrs_device = {
            let mut buf = Buffer::try_with_capacity_in(instrs.len(), scope.clone()).unwrap();
            buf.extend_from_host_slice(&instrs)?;
            buf
        };

        let width = MachineAir::<F>::preprocessed_width(self);

        let height = MachineAir::<F>::preprocessed_num_rows(self, program, instrs.len())
            .expect("preprocessed_num_rows(...) should be Some(_)");

        let mut trace = Tensor::<F, TaskScope>::zeros_in([width, height], scope.clone());

        unsafe {
            const BLOCK_DIM: usize = 64;
            let grid_dim = height.div_ceil(BLOCK_DIM);
            // args:
            // T *trace,
            // uintptr_t trace_height,
            // const csl_sys::SelectInstr<T> *instructions,
            // uintptr_t nb_instructions
            let args = args!(trace.as_mut_ptr(), height, instrs_device.as_ptr(), instrs.len());
            scope
                .launch_kernel(
                    TaskScope::tracegen_preprocessed_recursion_select_kernel(),
                    grid_dim,
                    BLOCK_DIM,
                    &args,
                    0,
                )
                .unwrap();
        }

        Ok(Some(Mle::new(trace)))
    }

    fn supports_device_main_tracegen(&self) -> bool {
        true
    }

    async fn generate_trace_device(
        &self,
        input: &Self::Record,
        _: &mut Self::Record,
        scope: &TaskScope,
    ) -> Result<Mle<F, TaskScope>, CopyError> {
        let events = &input.select_events;

        let events_device = {
            let mut buf = Buffer::try_with_capacity_in(events.len(), scope.clone()).unwrap();
            buf.extend_from_host_slice(events)?;
            buf
        };

        let width = <Self as BaseAir<F>>::width(self);

        let height = <Self as MachineAir<F>>::num_rows(self, input)
            .expect("num_rows(...) should be Some(_)");

        let mut trace = Tensor::<F, TaskScope>::zeros_in([width, height], scope.clone());

        unsafe {
            const BLOCK_DIM: usize = 64;
            let grid_dim = height.div_ceil(BLOCK_DIM);
            // args:
            // T *trace,
            // uintptr_t trace_height,
            // const csl_sys::SelectEvent<T> *events,
            // uintptr_t nb_events
            let args = args!(trace.as_mut_ptr(), height, events_device.as_ptr(), events.len());
            scope
                .launch_kernel(
                    TaskScope::tracegen_recursion_select_kernel(),
                    grid_dim,
                    BLOCK_DIM,
                    &args,
                    0,
                )
                .unwrap();
        }

        Ok(Mle::new(trace))
    }
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use slop_algebra::{AbstractField, Field};
    use sp1_recursion_executor::{
        Address, ExecutionRecord, Instruction, SelectEvent, SelectInstr, SelectIo,
    };
    use sp1_recursion_machine::chips::select::SelectChip;

    use crate::F;

    #[tokio::test]
    async fn test_select_generate_preprocessed_trace() {
        let task = csl_cuda::task().await.unwrap();
        task.run(|scope| {
            crate::recursion::tests::test_preprocessed_tracegen(
                SelectChip,
                |rng| {
                    Instruction::Select(SelectInstr {
                        addrs: SelectIo {
                            bit: Address(rng.gen()),
                            out1: Address(rng.gen()),
                            out2: Address(rng.gen()),
                            in1: Address(rng.gen()),
                            in2: Address(rng.gen()),
                        },
                        mult1: rng.gen(),
                        mult2: rng.gen(),
                    })
                },
                scope,
            )
        })
        .await;
    }

    #[tokio::test]
    async fn test_select_generate_main_trace() {
        let task = csl_cuda::task().await.unwrap();
        task.run(|scope| {
            crate::tests::test_main_tracegen(
                SelectChip,
                |rng| {
                    let bit = F::from_bool(rng.gen());
                    let in1 = rng.gen();
                    let in2 = rng.gen();
                    let (out1, out2) = if bit.is_one() { (in1, in2) } else { (in2, in1) };
                    SelectEvent { bit, out1, out2, in1, in2 }
                },
                |select_events| ExecutionRecord { select_events, ..Default::default() },
                scope,
            )
        })
        .await;
    }
}
