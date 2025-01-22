use csl_baby_bear::config::BabyBearExt;
use csl_device::{
    args,
    cuda::TaskScope,
    sys::{
        jagged::{
            jagged_column_baby_bear_extension_populate_col_major,
            jagged_column_baby_bear_extension_populate_row_major,
            jagged_table_baby_bear_extension_populate_col_major,
            jagged_table_baby_bear_extension_populate_row_major,
        },
        runtime::Dim3,
    },
    KernelPtr,
};
use slop_algebra::Field;

use crate::PopulateJaggedPolynomialBackend;

/// # Safety
unsafe trait PopulateJaggedPolynomialBackendKernel<F: Field> {
    const BLOCK_SIZE_X: usize;
    const BLOCK_SIZE_Y: usize;
    const X_STRIDE: usize;
    const Y_STRIDE: usize;

    fn populate_table_col_major() -> KernelPtr;
    fn populate_table_row_major() -> KernelPtr;
    fn populate_column_row_major() -> KernelPtr;
    fn populate_column_col_major() -> KernelPtr;
}

unsafe impl PopulateJaggedPolynomialBackendKernel<BabyBearExt> for TaskScope {
    const BLOCK_SIZE_X: usize = 128;
    const BLOCK_SIZE_Y: usize = 8;
    const X_STRIDE: usize = 32;
    const Y_STRIDE: usize = 1;

    fn populate_table_row_major() -> KernelPtr {
        unsafe { jagged_table_baby_bear_extension_populate_row_major() }
    }

    fn populate_table_col_major() -> KernelPtr {
        unsafe { jagged_table_baby_bear_extension_populate_col_major() }
    }

    fn populate_column_col_major() -> KernelPtr {
        unsafe { jagged_column_baby_bear_extension_populate_col_major() }
    }

    fn populate_column_row_major() -> KernelPtr {
        unsafe { jagged_column_baby_bear_extension_populate_row_major() }
    }
}

impl PopulateJaggedPolynomialBackend<BabyBearExt> for TaskScope {
    fn populate_table_row_major(
        &self,
        mut out: csl_device::tensor::TensorViewMut<BabyBearExt, Self>,
        eq_z_tab: csl_device::tensor::TensorView<BabyBearExt, Self>,
        eq_z_col: csl_device::tensor::TensorView<BabyBearExt, Self>,
        eq_z_row: csl_device::tensor::TensorView<BabyBearExt, Self>,
        offset: usize,
        table_idx: usize,
        height: usize,
        width: usize,
    ) {
        let block_dim: Dim3 = (Self::BLOCK_SIZE_X, Self::BLOCK_SIZE_Y, 1).into();
        let grid_dim: Dim3 = (
            height.div_ceil(Self::BLOCK_SIZE_X * Self::X_STRIDE),
            width.div_ceil(Self::BLOCK_SIZE_Y * Self::Y_STRIDE),
            1,
        )
            .into();
        let args = args!(
            out.as_mut_ptr(),
            eq_z_tab.as_ptr(),
            eq_z_col.as_ptr(),
            eq_z_row.as_ptr(),
            offset,
            table_idx,
            height,
            width
        );

        unsafe {
            self.launch_kernel(
                <Self as PopulateJaggedPolynomialBackendKernel::<BabyBearExt>>::populate_table_row_major(),
                grid_dim,
                block_dim,
                &args,
                0,
            )
            .unwrap();
        }
    }

    fn populate_table_col_major(
        &self,
        mut out: csl_device::tensor::TensorViewMut<BabyBearExt, Self>,
        eq_z_tab: csl_device::tensor::TensorView<BabyBearExt, Self>,
        eq_z_col: csl_device::tensor::TensorView<BabyBearExt, Self>,
        eq_z_row: csl_device::tensor::TensorView<BabyBearExt, Self>,
        offset: usize,
        table_idx: usize,
        height: usize,
        width: usize,
    ) {
        let block_dim: Dim3 = (Self::BLOCK_SIZE_X, Self::BLOCK_SIZE_Y, 1).into();
        let grid_dim: Dim3 = (
            height.div_ceil(Self::BLOCK_SIZE_X * Self::X_STRIDE),
            width.div_ceil(Self::BLOCK_SIZE_Y * Self::Y_STRIDE),
            1,
        )
            .into();
        let args = args!(
            out.as_mut_ptr(),
            eq_z_tab.as_ptr(),
            eq_z_col.as_ptr(),
            eq_z_row.as_ptr(),
            offset,
            table_idx,
            height,
            width
        );

        unsafe {
            self.launch_kernel(
                <Self as PopulateJaggedPolynomialBackendKernel::<BabyBearExt>>::populate_table_col_major(),
                grid_dim,
                block_dim,
                &args,
                0,
            )
            .unwrap();
        }
    }

    fn populate_column_col_major(
        &self,
        mut out: csl_device::tensor::TensorViewMut<BabyBearExt, Self>,
        eq_z_col: csl_device::tensor::TensorView<BabyBearExt, Self>,
        eq_z_row: csl_device::tensor::TensorView<BabyBearExt, Self>,
        offset: usize,
        height: usize,
        width: usize,
    ) {
        let block_dim: Dim3 = (Self::BLOCK_SIZE_X, Self::BLOCK_SIZE_Y, 1).into();
        let grid_dim: Dim3 =
            (height.div_ceil(Self::BLOCK_SIZE_X), width.div_ceil(Self::BLOCK_SIZE_Y), 1).into();
        let args =
            args!(out.as_mut_ptr(), eq_z_col.as_ptr(), eq_z_row.as_ptr(), offset, height, width);

        unsafe {
            self.launch_kernel(
                <Self as PopulateJaggedPolynomialBackendKernel::<BabyBearExt>>::populate_column_col_major(),
                grid_dim,
                block_dim,
                &args,
                0,
            )
            .unwrap();
        }
    }

    fn populate_column_row_major(
        &self,
        mut out: csl_device::tensor::TensorViewMut<BabyBearExt, Self>,
        eq_z_col: csl_device::tensor::TensorView<BabyBearExt, Self>,
        eq_z_row: csl_device::tensor::TensorView<BabyBearExt, Self>,
        offset: usize,
        height: usize,
        width: usize,
    ) {
        let block_dim: Dim3 = (Self::BLOCK_SIZE_X, Self::BLOCK_SIZE_Y, 1).into();
        let grid_dim: Dim3 =
            (height.div_ceil(Self::BLOCK_SIZE_X), width.div_ceil(Self::BLOCK_SIZE_Y), 1).into();
        let args =
            args!(out.as_mut_ptr(), eq_z_col.as_ptr(), eq_z_row.as_ptr(), offset, height, width);

        unsafe {
            self.launch_kernel(
                <Self as PopulateJaggedPolynomialBackendKernel::<BabyBearExt>>::populate_column_row_major(),
                grid_dim,
                block_dim,
                &args,
                0,
            )
            .unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use csl_device::{DeviceTensor, HostTensor};
    use csl_mle::{Mle, Point};
    use rand::{thread_rng, Rng};
    use slop_algebra::AbstractField;

    use crate::{JaggedTableConfiguration, PopulateJagged, TableDimensions};

    use super::*;

    #[tokio::test]
    async fn test_populate_jagged() {
        let mut rng = thread_rng();

        type EF = BabyBearExt;

        let total_rows_log2 = 21;
        let max_cols_log2 = 10;
        let num_tables_log2 = 4;

        let total_rows = 1 << total_rows_log2;
        let num_tables = 1 << num_tables_log2;

        let z_tab_values =
            HostTensor::from((0..num_tables).map(|_| rng.gen::<EF>()).collect::<Vec<_>>());

        let z_col_values =
            HostTensor::from((0..max_cols_log2).map(|_| rng.gen::<EF>()).collect::<Vec<_>>());

        let z_row_values =
            HostTensor::from((0..total_rows_log2).map(|_| rng.gen::<EF>()).collect::<Vec<_>>());

        let table_dimensions = vec![
            TableDimensions { height: total_rows >> 2, width: 100 },
            TableDimensions { height: total_rows, width: 50 },
            TableDimensions { height: total_rows, width: 20 },
            TableDimensions { height: total_rows >> 2, width: 100 },
            TableDimensions { height: total_rows >> 2, width: 130 },
            TableDimensions { height: total_rows >> 4, width: 500 },
            TableDimensions { height: total_rows >> 10, width: 1000 },
        ];

        let config = JaggedTableConfiguration::new(
            table_dimensions.clone(),
            crate::JaggedStoreConfig::ColMajor,
        );

        println!("Total size: {}", config.total_size());
        println!("log2 total size: {}", config.total_size().next_power_of_two().ilog2());

        let f_host = csl_device::cuda::task()
            .await
            .unwrap()
            .run(|t| async move {
                let z_tab = Point::new(
                    DeviceTensor::from_host(z_tab_values.clone(), t.clone()).await.unwrap(),
                );
                let z_col = Point::new(
                    DeviceTensor::from_host(z_col_values.clone(), t.clone()).await.unwrap(),
                );
                let z_row = Point::new(
                    DeviceTensor::from_host(z_row_values.clone(), t.clone()).await.unwrap(),
                );
                let time = tokio::time::Instant::now();
                let eq_z_tab = Mle::partial_lagrange(&z_tab);
                let eq_z_col = Mle::partial_lagrange(&z_col);
                let eq_z_row = Mle::partial_lagrange(&z_row);
                t.synchronize().await.unwrap();
                let f = PopulateJagged::generate_table_jagged_polynomial(
                    &config, &eq_z_tab, &eq_z_col, &eq_z_row,
                );
                t.synchronize().await.unwrap();
                let elapsed = time.elapsed();
                println!("Device time: {:?}", elapsed);
                f.into_host().await
            })
            .await
            .await
            .unwrap();

        for val in f_host.guts().as_buffer().iter() {
            assert_eq!(*val, *val * EF::one());
        }
    }
}
