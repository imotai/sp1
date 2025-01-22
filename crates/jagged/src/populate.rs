use std::marker::PhantomData;

use csl_device::tensor::{TensorView, TensorViewMut};
use csl_mle::{Mle, MleBaseBackend};
use slop_algebra::Field;

pub trait PopulateJaggedPolynomialBackend<F: Field>: MleBaseBackend<F> {
    #[allow(clippy::too_many_arguments)]
    fn populate_table_row_major(
        &self,
        out: TensorViewMut<F, Self>,
        eq_z_tab: TensorView<F, Self>,
        eq_z_col: TensorView<F, Self>,
        eq_z_row: TensorView<F, Self>,
        offset: usize,
        table_idx: usize,
        height: usize,
        width: usize,
    );

    #[allow(clippy::too_many_arguments)]
    fn populate_table_col_major(
        &self,
        out: TensorViewMut<F, Self>,
        eq_z_tab: TensorView<F, Self>,
        eq_z_col: TensorView<F, Self>,
        eq_z_row: TensorView<F, Self>,
        offset: usize,
        table_idx: usize,
        height: usize,
        width: usize,
    );

    fn populate_column_row_major(
        &self,
        out: TensorViewMut<F, Self>,
        eq_z_col: TensorView<F, Self>,
        eq_z_row: TensorView<F, Self>,
        offset: usize,
        height: usize,
        width: usize,
    );

    fn populate_column_col_major(
        &self,
        out: TensorViewMut<F, Self>,
        eq_z_col: TensorView<F, Self>,
        eq_z_row: TensorView<F, Self>,
        offset: usize,
        height: usize,
        width: usize,
    );
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PopulateJagged<T, B>(PhantomData<(T, B)>);

pub enum JaggedStoreConfig {
    RowMajor,
    ColMajor,
}

#[derive(Debug, Clone, Copy)]
pub struct TableDimensions {
    pub width: usize,
    pub height: usize,
}

pub struct JaggedTableConfiguration {
    pub table_dimensions: Vec<TableDimensions>,
    pub offsets: Vec<usize>,
    pub storage: JaggedStoreConfig,
}

impl JaggedTableConfiguration {
    pub fn new(table_dimensions: Vec<TableDimensions>, storage: JaggedStoreConfig) -> Self {
        let offsets = std::iter::once(0)
            .chain(table_dimensions.iter().scan(0, |acc, t| {
                let total_size = t.width * t.height;
                *acc += total_size;
                Some(*acc)
            }))
            .collect::<Vec<_>>();
        Self { table_dimensions, offsets, storage }
    }

    pub fn total_size(&self) -> usize {
        *self.offsets.last().unwrap()
    }
}

impl<F: Field, B: PopulateJaggedPolynomialBackend<F>> PopulateJagged<F, B> {
    pub fn generate_table_jagged_polynomial(
        config: &JaggedTableConfiguration,
        eq_z_tab: &Mle<F, B>,
        eq_z_col: &Mle<F, B>,
        eq_z_row: &Mle<F, B>,
    ) -> Mle<F, B> {
        let total_size = config
            .table_dimensions
            .iter()
            .map(|t| t.height * t.width)
            .sum::<usize>()
            .next_power_of_two();
        let scope = eq_z_tab.scope().clone();
        let mut f = Mle::uninit(1, total_size.ilog2() as usize, &scope);

        let generate_for_table = match config.storage {
            JaggedStoreConfig::RowMajor => B::populate_table_row_major,
            JaggedStoreConfig::ColMajor => B::populate_table_col_major,
        };

        for (table_idx, (offset, table_dimensions)) in
            config.offsets.iter().zip(config.table_dimensions.iter()).enumerate()
        {
            unsafe {
                f.assume_init();
                generate_for_table(
                    &scope,
                    f.guts_mut().as_view_mut(),
                    eq_z_tab.guts().as_view(),
                    eq_z_col.guts().as_view(),
                    eq_z_row.guts().as_view(),
                    *offset,
                    table_idx,
                    table_dimensions.height,
                    table_dimensions.width,
                );
            }
        }
        f
    }

    pub fn generate_column_jagged_polynomial(
        config: JaggedTableConfiguration,
        eq_z_col: &Mle<F, B>,
        eq_z_row: &Mle<F, B>,
    ) -> Mle<F, B> {
        let total_size = config.offsets.iter().sum::<usize>().next_power_of_two();
        let scope = eq_z_col.scope().clone();
        let mut f = Mle::uninit(1, total_size.ilog2() as usize, &scope);

        let generate_for_column = match config.storage {
            JaggedStoreConfig::RowMajor => B::populate_column_row_major,
            JaggedStoreConfig::ColMajor => B::populate_column_col_major,
        };

        for (offset, table_dimensions) in config.offsets.iter().zip(config.table_dimensions.iter())
        {
            unsafe {
                f.assume_init();
                generate_for_column(
                    &scope,
                    f.guts_mut().as_view_mut(),
                    eq_z_col.guts().as_view(),
                    eq_z_row.guts().as_view(),
                    *offset,
                    table_dimensions.height,
                    table_dimensions.width,
                );
            }
        }
        f
    }
}
