//! This module contains the implementation of the special multilinear polynomial appearing in the
//! jagged sumcheck protocol.
//!
//! More precisely, given a collection of L tables with areas [a_1, a_2, ..., a_L] and column counts
//! [c_1, c_2, ..., c_L], lay out those tables in a 3D array, aligning their top-left corners. Then,
//! imagine padding all the tables with zeroes so that the have the same number of rows. On the other
//! hand, imagine laying out all the tables (considered in RowMajor form) in a single long vector.
//! The jagged multilinear polynomial is the multilinear extension of the function which determines,
//! given a table, row, and column index in the 3D array, and an index in the long vector, whether
//! the index in the long vector corresponds to the table, row, and column index in the 3D array.
//! More explicitly, it's the function checking whether
//!
//! index = (a_1 + ... + a_{tab}) + row * c_{tab} + col.
//!
//! Since there is an efficient algorithm to implement this "indicator" function as a branching
//! program, following [HR18](https://eccc.weizmann.ac.il/report/2018/161/) there is a concise
//! algorithm for the evaluation of the corresponding multilinear polynomial. The algorithm to
//! compute the indicator uses the prefix sums [t_0=0, t_1=a_1, t_2 = a_1+a_2, ..., t_L], reads
//! t_{tab}, t_{tab+1}, index, tab, row, and col bit-by-bit from LSB to MSB, checks the equality
//! above, and also checks that index < t_{tab+1}. Assuming that c_{tab} is a power of 2, the
//! multiplication `row * c_{tab}` can be done by bit-shift, and the addition is checked via the
//! grade-school algorithm.
use std::collections::BTreeMap;

use rayon::prelude::*;

use itertools::all;
use rayon::iter::{repeat, IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use slop_algebra::{AbstractField, Field};
use slop_utils::{log2_ceil_usize, log2_strict_usize};

use slop_multilinear::{Mle, Point};

/// The state space of the carry in the branching program.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Trit {
    Zero = 0,
    One = 1,
    Two = 2,
}

impl From<usize> for Trit {
    fn from(x: usize) -> Self {
        match x {
            0 => Trit::Zero,
            1 => Trit::One,
            2 => Trit::Two,
            _ => panic!("Invalid trit value"),
        }
    }
}

impl From<Trit> for usize {
    fn from(x: Trit) -> Self {
        x as usize
    }
}

/// A struct recording the state of the memory of the branching program. Because the program performs
/// a three-way addition and one u32 comparison, the memory needed is a carry (which lies in {0,1,2})
/// and a boolean to store the comparison of the u32s up to the current bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MemoryState {
    pub carry: Trit,

    pub comparison_so_far: bool,
}

impl MemoryState {
    /// The memory state which indicates success in the last layer of the branching program.
    fn success() -> Self {
        MemoryState { carry: Trit::Zero, comparison_so_far: true }
    }
}

/// An enum to represent a potentially failed computation at a layer of the branching program.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum StateOrFail {
    State(MemoryState),
    Fail,
}

/// A struct representing the five bits the branching program needs to read in order to go to the next
/// layer of the program. The program streams the bits of the row, column, index, and the
/// "table area prefix sum".
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BitState<T> {
    pub row_bit: T,
    pub col_bit: T,
    pub index_bit: T,
    pub curr_tab_bit: T,
    pub next_tab_bit: T,
}

/// Enumerate all the possible memory states.
pub fn all_memory_states() -> Vec<MemoryState> {
    (0..3)
        .flat_map(|carry| {
            (0..2).map(move |comparison_so_far| MemoryState {
                carry: carry.into(),
                comparison_so_far: comparison_so_far != 0,
            })
        })
        .collect()
}

/// Enumerate all the possible bit states.
pub fn all_bit_states() -> Vec<BitState<bool>> {
    (0..2)
        .flat_map(|row_bit| {
            (0..2).flat_map(move |col_bit| {
                (0..2).flat_map(move |index_bit| {
                    (0..2).flat_map(move |last_tab_bit| {
                        (0..2).map(move |curr_tab_bit| BitState {
                            row_bit: row_bit != 0,
                            col_bit: col_bit != 0,
                            index_bit: index_bit != 0,
                            curr_tab_bit: last_tab_bit != 0,
                            next_tab_bit: curr_tab_bit != 0,
                        })
                    })
                })
            })
        })
        .collect()
}

/// The transition function that determines the next memory state given the current memory state and
/// the current bits being read. The branching program reads bits from LSB to MSB.
pub fn transition_function(
    bit_state: BitState<bool>,
    memory_state: MemoryState,
    layer: usize,
    log_column_count: usize,
) -> StateOrFail {
    // If the current (most significant bit read so far) index_bit matches the current next_tab_bit,
    // then defer to the comparison so far. Otherwise, the comparison is correct only if
    // `next_tab_bit` is 1 and `index_bit` is 0.
    let new_comparison_so_far = if bit_state.index_bit == bit_state.next_tab_bit {
        memory_state.comparison_so_far
    } else {
        bit_state.next_tab_bit
    };

    // Check that the column bit is off if reading bits after log_column_count.
    if layer >= log_column_count && bit_state.col_bit {
        return StateOrFail::Fail;
    }

    // Compute the carry according to the logic of three-way addition, or fail if the current bits
    // are not consistent with the three-way addition.
    //
    // However, we are checking that index = curr_tab + row * (1<<log_column_count) + col, so we
    // need to read the row bit only if the layer is after log_column_count.
    let new_carry = if layer < log_column_count {
        if (bit_state.index_bit as usize)
            != ((bit_state.col_bit as usize)
                + Into::<usize>::into(memory_state.carry)
                + bit_state.curr_tab_bit as usize)
                % 2
        {
            return StateOrFail::Fail;
        }
        (bit_state.col_bit as usize
            + Into::<usize>::into(memory_state.carry)
            + bit_state.curr_tab_bit as usize)
            / 2
    } else {
        if (bit_state.index_bit as usize)
            != ((bit_state.col_bit as usize)
                + Into::<usize>::into(memory_state.carry)
                + (bit_state.row_bit as usize)
                + (bit_state.curr_tab_bit as usize))
                % 2
        {
            return StateOrFail::Fail;
        }
        (bit_state.col_bit as usize
            + Into::<usize>::into(memory_state.carry)
            + (bit_state.row_bit as usize)
            + bit_state.curr_tab_bit as usize)
            / 2
    };
    // Successful transition.
    StateOrFail::State(MemoryState {
        carry: new_carry.into(),
        comparison_so_far: new_comparison_so_far,
    })
}

/// A struct to hold all the parameters sufficient to determine the special multilinear polynopmial
/// appearing in the jagged sumcheck protocol.
#[derive(Clone, Debug)]
pub struct JaggedLittlePolynomialProverParams {
    pub max_log_column_count: usize,
    pub table_count: usize,
    pub prefix_sums_usize: Vec<usize>,
    pub column_counts_usize: Vec<usize>,
    pub max_log_row_count: usize,
}

/// A struct to hold all the parameters sufficient to determine the special multilinear polynopmial
/// appearing in the jagged sumcheck protocol. All usize parameters are intended to be inferred from
/// the proving context, while the `Vec<Point<K>>` fields are intended to be recieved directly from
/// the prover as field elements. The verifier program thus depends only on the usize parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JaggedLittlePolynomialVerifierParams<K: AbstractField> {
    pub max_log_column_count: usize,
    pub table_count: usize,
    pub prefix_sums: Vec<Point<K>>,
    pub next_prefix_sums: Vec<Point<K>>,
    pub column_counts: Vec<Point<K>>,
    pub max_log_row_count: usize,
    pub max_log_col_count: usize,
}

impl<K: AbstractField> JaggedLittlePolynomialVerifierParams<K> {
    /// Given `z_index`, evaluate the special multilinear polynomial appearing in the jagged sumcheck
    /// protocol.
    pub fn full_jagged_little_polynomial_evaluation(
        &self,
        z_tab: &Point<K>,
        z_row: &Point<K>,
        z_col: &Point<K>,
        z_index: &Point<K>,
    ) -> K {
        // assert!(self.prefix_sums.len() == self.column_counts.len() + 1);
        let memory_states = all_memory_states();
        let bit_states = all_bit_states();
        // Iterate over all tables. For each table, we need to know the area of all the tables up tp
        // the current one, the area of the tables after the current one - 1, and the number of columns
        // in the current table.
        self.prefix_sums
            .iter()
            .cloned()
            .zip(self.next_prefix_sums.iter())
            .zip(self.column_counts.iter().cloned())
            .enumerate()
            .map(|(table_num, ((prefix_sum, next_prefix_sum), column_count))| {
                let log_m = z_index.dimension();

                // Initialize the answer.
                let mut ans = K::zero();

                // For `z_tab` on the Boolean hypercube, this is the delta function to pick out
                // table number `table_num`.
                let z_tab_correction = if z_tab.dimension() != 0 {
                    Mle::partial_lagrange(z_tab).guts().as_slice()[table_num].clone()
                } else {
                    K::one()
                };

                for log_column_count in 0..=self.max_log_column_count {
                    // For `z_col` on the Boolean hypercube, this is the delta function to pick out
                    // the right column count for the current table.
                    let c_tab_correction = if column_count.dimension() != 0 {
                        Mle::partial_lagrange(&column_count).guts().as_slice()
                            [1 << log_column_count]
                            .clone()
                    } else {
                        K::one()
                    };

                    // For `z_row` on the Boolean hypercube, this is the delta function to pick out
                    // the correct row count for the current table.
                    let z_row_correction: K = z_row
                        .reversed()
                        .iter()
                        .skip(log_m + 1 - log_column_count)
                        .map(|z| K::one() - z.clone())
                        .product();

                    let mut state_by_state_results: BTreeMap<StateOrFail, K> = BTreeMap::new();

                    // Initialize the state-by-state results for the last layer of the branching
                    // program, namely the success state returns 1 and all other states return 0.
                    for memory_state in all_memory_states().iter() {
                        state_by_state_results.insert(
                            StateOrFail::State(*memory_state),
                            if memory_state == &MemoryState::success() {
                                K::one()
                            } else {
                                K::zero()
                            },
                        );
                    }

                    // The dynamic programming algorithm to output the result of the branching
                    // iterates over the layers of the branching program in reverse order.
                    for layer in (0..log_m + 1).rev() {
                        let mut new_state_by_state_results: BTreeMap<StateOrFail, K> =
                            BTreeMap::new();
                        new_state_by_state_results.insert(StateOrFail::Fail, K::zero());
                        // For each memory state in the new layer, compute the result of the branching
                        // program that starts at that memory state and in the current layer.
                        for memory_state in &memory_states {
                            let mut accum = K::zero();

                            // We assume that bits are aligned in big-endian order. The algorithm,
                            // in the ith layer, looks at the ith least significant bit, which is
                            // the m - 1 - i th bit if the bits are in a bit array in big-endian.
                            let point = Point::<K>::from(vec![
                                layer
                                    .checked_sub(log_column_count)
                                    .and_then(|i| z_row.reversed().get(i).cloned())
                                    .unwrap_or_default(),
                                z_col.reversed().get(layer).cloned().unwrap_or_default(),
                                z_index.reversed().get(layer).cloned().unwrap_or_default(),
                                prefix_sum.reversed().get(layer).cloned().unwrap_or_default(),
                                next_prefix_sum.reversed().get(layer).cloned().unwrap_or_default(),
                            ]);
                            let five_var_eq = Mle::partial_lagrange(&point);

                            // For each possible bit state, compute the result of the branching
                            // program transition function and modify the accumulator accordingly.
                            for (i, elem) in five_var_eq.guts().as_slice().iter().enumerate() {
                                let bit_state = &bit_states[i];

                                let state_or_fail = transition_function(
                                    *bit_state,
                                    *memory_state,
                                    layer,
                                    log_column_count,
                                );

                                // TODO: Group these by target.
                                accum += match state_or_fail {
                                    StateOrFail::State(state) => {
                                        elem.clone()
                                            * state_by_state_results
                                                .get(&StateOrFail::State(state))
                                                .unwrap()
                                                .clone()
                                    }
                                    StateOrFail::Fail => K::zero(),
                                };
                            }
                            new_state_by_state_results
                                .insert(StateOrFail::State(*memory_state), accum);
                        }
                        state_by_state_results = new_state_by_state_results;
                    }

                    // Perform the multiplication outside of the main loop to avoid redundant
                    // multiplications.
                    ans += z_row_correction
                        * c_tab_correction.clone()
                        * z_tab_correction.clone()
                        * state_by_state_results[&StateOrFail::State(MemoryState::success())]
                            .clone();
                }

                ans
            })
            .sum()
    }
}

impl JaggedLittlePolynomialProverParams {
    pub fn new(
        row_counts_usize: Vec<usize>,
        column_counts_usize: Vec<usize>,
        max_log_row_count: usize,
    ) -> Self {
        let mut prefix_sums_usize = row_counts_usize
            .iter()
            .zip(column_counts_usize.iter())
            .scan(0, |state, (row, col)| {
                let result = *state;
                *state += row * col;
                Some(result)
            })
            .collect::<Vec<_>>();

        prefix_sums_usize.push(
            *prefix_sums_usize.last().unwrap()
                + row_counts_usize.last().unwrap() * column_counts_usize.last().unwrap(),
        );
        println!("{:?}", prefix_sums_usize);

        assert!(all(column_counts_usize.iter(), |&x| x.is_power_of_two()));

        assert!(prefix_sums_usize.last().unwrap().is_power_of_two());

        let max_log_column_count = log2_strict_usize(*column_counts_usize.iter().max().unwrap());
        JaggedLittlePolynomialProverParams {
            max_log_column_count,
            table_count: prefix_sums_usize.len() - 1,
            prefix_sums_usize,
            column_counts_usize,
            max_log_row_count,
        }
    }

    /// Compute the "guts" of the multilinear polynomial represented by the fixed prover parameters.
    pub fn partial_jagged_little_polynomial_evaluation<K: Field>(
        &self,
        z_tab: &Point<K>,
        z_row: &Point<K>,
        z_col: &Point<K>,
    ) -> Mle<K> {
        let log_total_area = log2_ceil_usize(*self.prefix_sums_usize.last().unwrap());

        let log_table_count = log2_ceil_usize(self.table_count);
        let tab_eq = Mle::partial_lagrange(&z_tab.last_k(log_table_count));
        let col_eq = Mle::partial_lagrange(&z_col.last_k(self.max_log_column_count));
        let row_eq = Mle::partial_lagrange(&z_row.last_k(self.max_log_row_count));

        let mut result = Vec::new();
        tracing::info_span!("compute jagged polynomial entries").in_scope(|| {
            (0..*self.prefix_sums_usize.last().unwrap())
                .into_par_iter()
                .map(|index| {
                    let tab = self.prefix_sums_usize.iter().rposition(|&x| index >= x).unwrap();
                    let row = (index - self.prefix_sums_usize[tab]) / self.column_counts_usize[tab];
                    let col = (index - self.prefix_sums_usize[tab]) % self.column_counts_usize[tab];

                    let tab_eq_val = tab_eq.guts().as_slice()[tab];
                    let col_eq_val = col_eq.guts().as_slice()[col];
                    let row_eq_val = row_eq.guts().as_slice()[row];

                    tab_eq_val * col_eq_val * row_eq_val
                })
                .chain(
                    repeat(K::zero())
                        .take((1 << log_total_area) - *self.prefix_sums_usize.last().unwrap()),
                )
                .collect_into_vec(&mut result)
        });

        result.into()
    }

    /// Convert the prover parameters into verifier parameters so that the verifier can run its
    /// evaluation algorithm.
    pub fn into_verifier_params<K: Field>(self) -> JaggedLittlePolynomialVerifierParams<K> {
        let log_m = log2_ceil_usize(*self.prefix_sums_usize.last().unwrap());
        let prefix_sums =
            self.prefix_sums_usize.iter().map(|&x| Point::from_usize(x, log_m + 1)).collect();
        let next_prefix_sums = self
            .prefix_sums_usize
            .iter()
            .skip(1)
            .map(|&x| Point::from_usize(x - 1, log_m + 1))
            .collect();
        let column_counts = self
            .column_counts_usize
            .iter()
            .map(|&x| Point::from_usize(x, self.max_log_column_count + 1))
            .collect();
        JaggedLittlePolynomialVerifierParams {
            max_log_column_count: self.max_log_column_count,
            table_count: self.table_count,
            prefix_sums,
            next_prefix_sums,
            column_counts,
            max_log_row_count: self.max_log_row_count,
            max_log_col_count: log2_strict_usize(*self.column_counts_usize.last().unwrap()),
        }
    }
}

#[cfg(test)]
pub mod tests {
    use rand::Rng;
    use slop_baby_bear::BabyBear;
    use slop_utils::log2_ceil_usize;
    type F = BabyBear;
    use slop_algebra::AbstractField;

    use slop_multilinear::Point;

    #[test]
    fn test_single_table_jagged_eval() {
        for log_num_rows in 0..5 {
            for log_num_cols in 0..5 {
                for index in 0..(1 << (log_num_cols + log_num_rows)) {
                    let log_m = log_num_cols + log_num_rows;
                    let row = index >> log_num_cols;
                    let col = index & ((1 << log_num_cols) - 1);

                    let mut z_row = Point::<F>::from_usize(row, log_num_rows + 1);
                    let mut z_col = Point::<F>::from_usize(col, log_num_cols + 1);
                    let z_index = Point::<F>::from_usize(index, log_m + 1);

                    let z_tab = Point::<F>::from_usize(0, 1);

                    let prover_params = super::JaggedLittlePolynomialProverParams::new(
                        vec![1 << log_num_rows],
                        vec![1 << log_num_cols],
                        log_num_rows,
                    );

                    let verifier_params = prover_params.clone().into_verifier_params();

                    let result = verifier_params.full_jagged_little_polynomial_evaluation(
                        &z_tab,
                        &z_row,
                        &z_col,
                        &z_index.clone(),
                    );
                    assert_eq!(result, F::one());

                    for other_index in 0..(1 << (log_num_cols + log_num_rows)) {
                        if other_index != index {
                            assert!(
                                verifier_params.full_jagged_little_polynomial_evaluation(
                                    &z_tab,
                                    &z_row,
                                    &z_col,
                                    &Point::<F>::from_usize(other_index, log_m)
                                ) == F::zero()
                            );
                        }
                    }

                    z_row = Point::<F>::from_usize(row ^ 1, log_num_rows + 1);

                    let wrong_result = verifier_params.full_jagged_little_polynomial_evaluation(
                        &z_tab,
                        &z_row,
                        &z_col,
                        &z_index.clone(),
                    );
                    assert_eq!(wrong_result, F::zero());

                    z_row = Point::<F>::from_usize(row, log_num_rows + 1);
                    z_col = Point::<F>::from_usize(col ^ 1, log_num_cols + 1);

                    let wrong_result = verifier_params
                        .full_jagged_little_polynomial_evaluation(&z_tab, &z_row, &z_col, &z_index);
                    assert_eq!(wrong_result, F::zero());

                    z_col = Point::<F>::from_usize(col, log_num_cols + 1);
                    let wrong_result = verifier_params.full_jagged_little_polynomial_evaluation(
                        &z_tab,
                        &z_row,
                        &z_col,
                        &Point::<F>::from_usize(index ^ 1, log_num_cols + 1),
                    );
                    assert_eq!(wrong_result, F::zero());

                    let mut rng = rand::thread_rng();

                    for _ in 0..3 {
                        let z_index: Point<F> = (0..log_m).map(|_| rng.gen::<F>()).collect();
                        assert_eq!(
                            verifier_params.full_jagged_little_polynomial_evaluation(
                                &z_tab, &z_row, &z_col, &z_index
                            ),
                            prover_params
                                .partial_jagged_little_polynomial_evaluation(&z_tab, &z_row, &z_col)
                                .eval_at(&z_index)[0]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_multi_table_jagged_eval() {
        let log_m = 5;
        let log_max_column_count = 2;
        let log_max_row_count = 3;

        let column_counts = [2, 1 << 2];
        let row_counts = [1 << 3, 1 << 2];

        let mut prefix_sums = column_counts
            .iter()
            .zip(row_counts.iter())
            .scan(0, |state, (col, row)| {
                let result = *state;
                *state += col * row;
                Some(result)
            })
            .collect::<Vec<_>>();

        prefix_sums.push(
            *prefix_sums.last().unwrap()
                + column_counts.last().unwrap() * row_counts.last().unwrap(),
        );

        for index in 0..*prefix_sums.last().unwrap() {
            let tab = prefix_sums.iter().rposition(|&x| index >= x).unwrap();
            let row = (index - prefix_sums[tab]) / column_counts[tab];
            let col = (index - prefix_sums[tab]) % column_counts[tab];
            let z_tab = Point::<F>::from_usize(tab, log2_ceil_usize(column_counts.len()));
            let z_row = Point::<F>::from_usize(row, log_max_row_count);
            let z_col = Point::<F>::from_usize(col, log_max_column_count);
            let params = super::JaggedLittlePolynomialProverParams::new(
                row_counts.to_vec(),
                column_counts.to_vec(),
                log_max_row_count,
            );
            let verifier_params = params.clone().into_verifier_params();

            for new_tab in 0..column_counts.len() {
                for new_row in 0..(1 << log_max_row_count) {
                    for new_col in 0..(1 << log_max_column_count) {
                        if !(new_col == col && new_row == row && new_tab == tab) {
                            let z_index = Point::<F>::from_usize(index, log_m);

                            let new_z_tab = Point::<F>::from_usize(
                                new_tab,
                                log2_ceil_usize(column_counts.len()),
                            );
                            let new_z_row = Point::<F>::from_usize(new_row, log_max_row_count);
                            let new_z_col = Point::<F>::from_usize(new_col, log_max_column_count);

                            let result = verifier_params.full_jagged_little_polynomial_evaluation(
                                &new_z_tab, &new_z_row, &new_z_col, &z_index,
                            );
                            assert_eq!(result, F::zero());
                            assert_eq!(
                                params
                                    .partial_jagged_little_polynomial_evaluation(
                                        &new_z_tab, &new_z_row, &new_z_col
                                    )
                                    .eval_at(&z_index)[0],
                                F::zero()
                            );
                        }
                    }
                }
            }

            let verifier_params = params.clone().into_verifier_params();

            let z_index = Point::from_usize(index, log_m);
            let result = verifier_params
                .full_jagged_little_polynomial_evaluation(&z_tab, &z_row, &z_col, &z_index);
            assert_eq!(result, F::one());

            for other_index in 0..*prefix_sums.last().unwrap() {
                if other_index != index {
                    assert!(
                        verifier_params.full_jagged_little_polynomial_evaluation(
                            &z_tab,
                            &z_row,
                            &z_col,
                            &Point::from_usize(other_index, log_m)
                        ) == F::zero()
                    );

                    assert!(
                        params
                            .partial_jagged_little_polynomial_evaluation(&z_tab, &z_row, &z_col)
                            .eval_at(&Point::<F>::from_usize(other_index, log_m))[0]
                            == F::zero()
                    );
                }
            }

            let z_index: Point<F> = (0..log_m).map(|_| F::zero()).collect();
            assert_eq!(
                verifier_params
                    .full_jagged_little_polynomial_evaluation(&z_tab, &z_row, &z_col, &z_index),
                params
                    .partial_jagged_little_polynomial_evaluation(&z_tab, &z_row, &z_col)
                    .eval_at(&z_index)[0]
            );
        }

        let mut rng = rand::thread_rng();

        let params = super::JaggedLittlePolynomialProverParams::new(
            row_counts.to_vec(),
            column_counts.to_vec(),
            log_max_row_count,
        );

        let z_tab: Point<F> =
            (0..log2_ceil_usize(column_counts.len())).map(|_| rng.gen::<F>()).collect();
        let z_row: Point<F> = (0..log_max_row_count).map(|_| rng.gen::<F>()).collect();
        let z_col = (0..log_max_column_count).map(|_| rng.gen::<F>()).collect::<Point<_>>();

        let verifier_params = params.clone().into_verifier_params();

        for _ in 0..100 {
            let z_index: Point<F> = (0..log_m).map(|_| rng.gen::<F>()).collect();
            assert_eq!(
                verifier_params
                    .full_jagged_little_polynomial_evaluation(&z_tab, &z_row, &z_col, &z_index),
                params
                    .partial_jagged_little_polynomial_evaluation(&z_tab, &z_row, &z_col)
                    .eval_at(&z_index)[0]
            );
        }
    }

    #[test]
    fn test_single_table_jagged_eval_off_boolean_hypercube() {
        let mut rng = rand::thread_rng();
        for log_num_rows in 0..5 {
            for log_num_cols in 0..5 {
                for log_num_tables in 0..2 {
                    let log_m = log_num_cols + log_num_rows + log_num_tables;
                    let z_row: Point<F> = (0..log_num_rows).map(|_| rng.gen::<F>()).collect();
                    let z_col: Point<F> = (0..log_num_cols).map(|_| rng.gen::<F>()).collect();
                    let z_tab: Point<F> = (0..log_num_tables).map(|_| rng.gen::<F>()).collect();

                    for index in 0..(1 << (log_num_cols + log_num_rows + log_num_cols)) {
                        let params = super::JaggedLittlePolynomialProverParams::new(
                            (0..(1 << log_num_tables)).map(|_| (1 << log_num_rows)).collect(),
                            (0..(1 << log_num_tables)).map(|_| 1 << log_num_cols).collect(),
                            log_num_rows,
                        );

                        let verifier_params = params.clone().into_verifier_params();

                        let z_index = Point::from_usize(index, log_m);
                        assert_eq!(
                            verifier_params.full_jagged_little_polynomial_evaluation(
                                &z_tab, &z_row, &z_col, &z_index
                            ),
                            params
                                .partial_jagged_little_polynomial_evaluation(&z_tab, &z_row, &z_col)
                                .eval_at(&z_index)[0]
                        );

                        let z_index: Point<F> = (0..log_m).map(|_| rng.gen::<F>()).collect();
                        assert_eq!(
                            verifier_params.full_jagged_little_polynomial_evaluation(
                                &z_tab, &z_row, &z_col, &z_index
                            ),
                            params
                                .partial_jagged_little_polynomial_evaluation(&z_tab, &z_row, &z_col)
                                .eval_at(&z_index)[0]
                        );
                    }
                }
            }
        }
    }
}
