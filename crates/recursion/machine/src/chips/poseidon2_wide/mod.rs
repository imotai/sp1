#![allow(clippy::needless_range_loop)]

use std::{borrow::Borrow, ops::Deref};

use sp1_core_machine::operations::poseidon2::permutation::{
    Poseidon2Cols, Poseidon2Degree3Cols, Poseidon2Degree9Cols,
};

pub mod air;
pub mod columns;
pub mod trace;

/// A chip that implements addition for the opcode Poseidon2Wide.
#[derive(Default, Debug, Clone, Copy)]
pub struct Poseidon2WideChip<const DEGREE: usize>;

impl<'a, const DEGREE: usize> Poseidon2WideChip<DEGREE> {
    /// Transmute a row it to an immutable [`Poseidon2Cols`] instance.
    pub(crate) fn convert<T>(row: impl Deref<Target = [T]>) -> Box<dyn Poseidon2Cols<T> + 'a>
    where
        T: Copy + 'a,
    {
        if DEGREE == 3 {
            let convert: &Poseidon2Degree3Cols<T> = (*row).borrow();
            Box::new(*convert)
        } else if DEGREE == 9 || DEGREE == 17 {
            let convert: &Poseidon2Degree9Cols<T> = (*row).borrow();
            Box::new(*convert)
        } else {
            panic!("Unsupported degree");
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {

    use std::iter::once;

    use slop_algebra::{AbstractField, PrimeField32};
    use slop_baby_bear::BabyBear;
    use slop_symmetric::Permutation;
    use sp1_core_machine::{operations::poseidon2::WIDTH, utils::setup_logger};
    use sp1_primitives::poseidon2_init;
    use sp1_recursion_executor::{instruction as instr, MemAccessKind};
    use zkhash::ark_ff::UniformRand;

    use crate::machine::tests::test_recursion_linear_program;

    #[tokio::test]
    async fn test_poseidon2() {
        setup_logger();
        let input = [1; WIDTH];
        let output = poseidon2_init()
            .permute(input.map(BabyBear::from_canonical_u32))
            .map(|x| BabyBear::as_canonical_u32(&x));

        let rng = &mut rand::thread_rng();
        let input_1: [BabyBear; WIDTH] = std::array::from_fn(|_| BabyBear::rand(rng));
        let output_1 = poseidon2_init().permute(input_1).map(|x| BabyBear::as_canonical_u32(&x));
        let input_1 = input_1.map(|x| BabyBear::as_canonical_u32(&x));

        let instructions =
            (0..WIDTH)
                .map(|i| instr::mem(MemAccessKind::Write, 1, i as u32, input[i]))
                .chain(once(instr::poseidon2(
                    [1; WIDTH],
                    std::array::from_fn(|i| (i + WIDTH) as u32),
                    std::array::from_fn(|i| i as u32),
                )))
                .chain(
                    (0..WIDTH)
                        .map(|i| instr::mem(MemAccessKind::Read, 1, (i + WIDTH) as u32, output[i])),
                )
                .chain((0..WIDTH).map(|i| {
                    instr::mem(MemAccessKind::Write, 1, (2 * WIDTH + i) as u32, input_1[i])
                }))
                .chain(once(instr::poseidon2(
                    [1; WIDTH],
                    std::array::from_fn(|i| (i + 3 * WIDTH) as u32),
                    std::array::from_fn(|i| (i + 2 * WIDTH) as u32),
                )))
                .chain((0..WIDTH).map(|i| {
                    instr::mem(MemAccessKind::Read, 1, (i + 3 * WIDTH) as u32, output_1[i])
                }))
                .collect::<Vec<_>>();

        test_recursion_linear_program(instructions).await;
    }
}
