use crate::runtime::KernelPtr;

extern "C" {
    pub fn jagged_table_baby_bear_extension_populate_row_major() -> KernelPtr;
    pub fn jagged_table_baby_bear_extension_populate_col_major() -> KernelPtr;
    pub fn jagged_column_baby_bear_extension_populate_row_major() -> KernelPtr;
    pub fn jagged_column_baby_bear_extension_populate_col_major() -> KernelPtr;
}
