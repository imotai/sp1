use std::{borrow::Borrow, marker::PhantomData};

use itertools::Itertools;
use rayon::prelude::*;

use serde::{Deserialize, Serialize};
use slop_algebra::{ExtensionField, TwoAdicField};
use slop_alloc::{Backend, CpuBackend};
use slop_multilinear::Mle;

pub trait MleBatcher<F: TwoAdicField, EF: ExtensionField<F>, A: Backend = CpuBackend> {
    fn batch<M>(&self, batching_challenge: EF, mles: &[M]) -> MleBatch<F, EF, A>
    where
        M: Borrow<Mle<F, A>>;
}

pub trait Folder<F: TwoAdicField, EF: ExtensionField<F>, A: Backend = CpuBackend> {
    fn fold(&self, data: &MleBatch<F, EF, A>) -> MleBatch<F, EF, A>;
}

pub trait FriIoppProver<F: TwoAdicField, EF: ExtensionField<F>, A: Backend = CpuBackend>:
    MleBatcher<F, EF, A> + Folder<F, EF, A> + Folder<F, EF, A>
{
}

pub struct MleBatch<F: TwoAdicField, EF: ExtensionField<F>, A: Backend = CpuBackend> {
    pub batched_poly: Mle<F, A>,
    _marker: PhantomData<EF>,
}

#[derive(
    Debug, Clone, Default, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct FriCpuProver;

impl<F: TwoAdicField, EF: ExtensionField<F>> MleBatcher<F, EF, CpuBackend> for FriCpuProver {
    fn batch<M: Borrow<Mle<F>>>(
        &self,
        batching_challenge: EF,
        mles: &[M],
    ) -> MleBatch<F, EF, CpuBackend> {
        // Compute all the batch challenge powers.
        let total_num_polynomials =
            mles.iter().map(|mle| mle.borrow().num_polynomials()).sum::<usize>();
        let mut batch_challenge_powers =
            batching_challenge.powers().take(total_num_polynomials).collect::<Vec<_>>();

        // Compute the random linear combination of the MLEs of the columns of the matrices
        let num_variables = mles.first().unwrap().borrow().num_variables() as usize;
        let mut batch_mle = Mle::from(vec![EF::zero(); num_variables]);
        for mle in mles.iter() {
            let mle: &Mle<_, _> = mle.borrow();
            let powers = batch_challenge_powers.split_off(mle.num_polynomials());
            batch_mle
                .guts_mut()
                .as_mut_slice()
                .par_iter_mut()
                .zip_eq(mle.hypercube_par_iter())
                .for_each(|(batch, row)| {
                    let batch_row = powers.iter().zip_eq(row).map(|(a, b)| *a * *b).sum::<EF>();
                    *batch += batch_row;
                });
        }
        // let batch_challenge_powers
        // let mut curr_batch_power = 0;
        // // Compute the random linear combination of the MLEs of the columns of the matrices.
        // let mut current_mle_vec: Vec<EF> =
        //     vec![EF::zero(); encoded_height >> self.pcs.fri_config().log_blowup];
        // tracing::info_span!("batch MLEs").in_scope(|| {
        //     orig_matrices.iter().for_each(|mat| {
        //         current_mle_vec.par_iter_mut().zip(mat.par_rows()).for_each(|(a, row)| {
        //             *a += batching_challenge.exp_u64(curr_batch_power as u64)
        //                 * batch_reducer.reduce_base(row.collect::<Vec<_>>().as_slice())
        //         });
        //         curr_batch_power += mat.width();
        //     })
        // });

        // let mut current_mle: Mle<EK> = current_mle_vec.into();
        todo!()
    }
}

impl<F: TwoAdicField, EF: ExtensionField<F>> Folder<F, EF> for FriCpuProver {
    fn fold(&self, _data: &MleBatch<F, EF, CpuBackend>) -> MleBatch<F, EF, CpuBackend> {
        todo!()
    }
}

impl<F: TwoAdicField, EF: ExtensionField<F>> FriIoppProver<F, EF, CpuBackend> for FriCpuProver {}
