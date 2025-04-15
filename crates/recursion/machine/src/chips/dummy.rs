use p3_matrix::dense::RowMajorMatrix;
use slop_air::{Air, AirBuilder, BaseAir};
use slop_algebra::{Field, PrimeField32};
use slop_matrix::Matrix;
use sp1_recursion_executor::{ExecutionRecord, RecursionProgram};
use sp1_stark::air::MachineAir;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct PreprocessedDummyChip<const PREP_WIDTH: usize, const MAIN_WIDTH: usize>;

impl<F: Field, const PREP_WIDTH: usize, const MAIN_WIDTH: usize> BaseAir<F>
    for PreprocessedDummyChip<PREP_WIDTH, MAIN_WIDTH>
{
    fn width(&self) -> usize {
        MAIN_WIDTH
    }
}

impl<AB: AirBuilder, const PREP_WIDTH: usize, const MAIN_WIDTH: usize> Air<AB>
    for PreprocessedDummyChip<PREP_WIDTH, MAIN_WIDTH>
{
    fn eval(&self, builder: &mut AB) {
        // Assert that all entries are zero.
        let main = builder.main();
        let local = main.row_slice(0);
        for val in local.iter() {
            builder.assert_zero(*val);
        }
    }
}

impl<F: PrimeField32, const PREP_WIDTH: usize, const MAIN_WIDTH: usize> MachineAir<F>
    for PreprocessedDummyChip<PREP_WIDTH, MAIN_WIDTH>
{
    type Record = ExecutionRecord<F>;

    type Program = RecursionProgram<F>;

    fn name(&self) -> String {
        "PreprocessedDummy".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        PREP_WIDTH
    }

    fn preprocessed_num_rows(&self, program: &Self::Program, _instrs_len: usize) -> Option<usize> {
        Some(program.dummy_preprocessed_height)
    }

    fn generate_preprocessed_trace(&self, program: &Self::Program) -> Option<RowMajorMatrix<F>> {
        let nb_rows = self.preprocessed_num_rows(program, 0).unwrap();
        let values = vec![F::zero(); nb_rows * PREP_WIDTH];
        Some(RowMajorMatrix::new(values, PREP_WIDTH))
    }

    fn generate_dependencies(&self, _record: &Self::Record, _record_out: &mut Self::Record) {
        // No dependencies.
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(input.program.dummy_preprocessed_height)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> RowMajorMatrix<F> {
        let nb_rows = self.num_rows(input).unwrap();
        let values = vec![F::zero(); nb_rows * MAIN_WIDTH];
        RowMajorMatrix::new(values, MAIN_WIDTH)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MainDummyChip<const WIDTH: usize>;

impl<F: Field, const WIDTH: usize> BaseAir<F> for MainDummyChip<WIDTH> {
    fn width(&self) -> usize {
        WIDTH
    }
}

impl<AB: AirBuilder, const WIDTH: usize> Air<AB> for MainDummyChip<WIDTH> {
    fn eval(&self, builder: &mut AB) {
        // Assert that all entries are zero.
        let main = builder.main();
        let local = main.row_slice(0);
        for val in local.iter() {
            builder.assert_zero(*val);
        }
    }
}

impl<F: PrimeField32, const WIDTH: usize> MachineAir<F> for MainDummyChip<WIDTH> {
    type Record = ExecutionRecord<F>;

    type Program = RecursionProgram<F>;

    fn name(&self) -> String {
        "MainDummy".to_string()
    }

    fn preprocessed_width(&self) -> usize {
        0
    }

    fn preprocessed_num_rows(&self, _program: &Self::Program, _instrs_len: usize) -> Option<usize> {
        None
    }

    fn generate_preprocessed_trace(&self, _program: &Self::Program) -> Option<RowMajorMatrix<F>> {
        None
    }

    fn generate_dependencies(&self, _record: &Self::Record, _record_out: &mut Self::Record) {
        // No dependencies.
    }

    fn num_rows(&self, input: &Self::Record) -> Option<usize> {
        Some(input.program.dummy_main_height)
    }

    fn generate_trace(
        &self,
        input: &Self::Record,
        _output: &mut Self::Record,
    ) -> RowMajorMatrix<F> {
        let nb_rows = self.num_rows(input).unwrap();
        let values = vec![F::zero(); nb_rows * WIDTH];
        RowMajorMatrix::new(values, WIDTH)
    }

    fn included(&self, _record: &Self::Record) -> bool {
        true
    }
}
