use itertools::izip;
use serde::{Deserialize, Serialize};
use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};
use thiserror::Error;

use p3_air::{Air, AirBuilder, BaseAir, ExtensionBuilder};
use p3_commit::Pcs;
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_uni_stark::{StarkGenericConfig, Val};

use spl_algebra::AbstractField;
use spl_multi_pcs::{partial_lagrange_eval, Point};

use crate::{
    air_types::ChipOpenedValues, folder::MultivariateEvaluationAirBuilder, verifier::AdapterAir,
};

type Com<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Commitment;

pub type Dom<SC> = <<SC as StarkGenericConfig>::Pcs as Pcs<
    <SC as StarkGenericConfig>::Challenge,
    <SC as StarkGenericConfig>::Challenger,
>>::Domain;

/// A univariate-to-multivariate adapter generic in a STARK configuration type.
#[derive(Default)]
pub struct MultivariateAdapterPCS<SC: StarkGenericConfig> {
    /// The STARK config.
    pub(crate) config: SC,
}

/// The proof struct for the multivariate opening.
pub struct MultivariateAdapterProof<SC: StarkGenericConfig> {
    pub adapter_commit: Com<SC>,
    pub quotient_commit: Com<SC>,
    pub opening_proof: <SC::Pcs as Pcs<SC::Challenge, SC::Challenger>>::Proof,
    pub opened_values: Vec<ChipOpenedValues<SC::Challenge>>,
}

#[derive(Debug, Error)]
pub enum MultivariateAdapterError {
    #[error("Verification error")]
    Verification,

    #[error("Pcs error")]
    PcsError,

    #[error("Shape mismatch")]
    ShapeMismatch,
}

pub struct MultivariateAdapterAir {}
pub const NUM_ADAPTER_COLS: usize = size_of::<MultivariateAdapterCols<u8>>();

#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct MultivariateAdapterCols<F> {
    /// The column of the evaluations of eq(i, eval_point) as i varies over the Boolean hypercube.
    pub lagrange_eval: F,

    /// A prefix-sum column to compute the inner product of the previous two columns.
    pub accum: F,
}

impl<F> BaseAir<F> for MultivariateAdapterAir {
    fn width(&self) -> usize {
        // Assuming a single opening for now.
        1
    }
}

impl<F> AdapterAir<F> for MultivariateAdapterAir {
    fn adapter_width(&self) -> usize {
        NUM_ADAPTER_COLS
    }
}

impl<F> Borrow<MultivariateAdapterCols<F>> for [F] {
    fn borrow(&self) -> &MultivariateAdapterCols<F> {
        debug_assert_eq!(self.len(), NUM_ADAPTER_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<MultivariateAdapterCols<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<F> BorrowMut<MultivariateAdapterCols<F>> for [F] {
    fn borrow_mut(&mut self) -> &mut MultivariateAdapterCols<F> {
        debug_assert_eq!(self.len(), NUM_ADAPTER_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to_mut::<MultivariateAdapterCols<F>>() };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}

impl<AB: AirBuilder + MultivariateEvaluationAirBuilder> Air<AB> for MultivariateAdapterAir {
    fn eval(&self, builder: &mut AB) {
        let adapter = builder.adapter();
        let main = builder.main();

        let (local_adapter, next_adapter) = (adapter.row_slice(0), adapter.row_slice(1));
        let local_adapter: &MultivariateAdapterCols<AB::VarEF> = (*local_adapter).borrow();
        let next_adapter: &MultivariateAdapterCols<AB::VarEF> = (*next_adapter).borrow();

        let (local, next) = (main.row_slice(0), main.row_slice(1));

        // Assert that the first row accumulator is equal to the product of the lagrange_eval and
        // the main trace element.
        builder.when_first_row().assert_eq_ext(
            local_adapter.accum,
            local_adapter.lagrange_eval.into() * local[0].into(),
        );

        // Assert that the accumulator is correctly computed.
        builder.when_transition().assert_eq_ext(
            local_adapter.accum.into() + next_adapter.lagrange_eval.into() * next[0].into(),
            next_adapter.accum,
        );

        let expected_eval = builder.expected_eval();

        // Assert that the last row of the accumulator and the claimed evaluation match.
        builder.when_last_row().assert_eq_ext(local_adapter.accum, expected_eval);

        // We also need to constrain the lagrange_evals to be correctly computed, but that requires
        // functionality which the current STARK/AIR API does not provide.
    }
}

pub fn generate_adapter_trace<SC: StarkGenericConfig>(
    data: &[Val<SC>],
    eval_point: &Point<SC::Challenge>,
) -> RowMajorMatrix<SC::Challenge> {
    let mut trace = Vec::with_capacity(data.len() * NUM_ADAPTER_COLS);

    // The eq polynomial, with one set of variables fixed to `eval_point`.
    let lagrange = partial_lagrange_eval(eval_point);

    // Compute the cumulative sum of the coordinate-wise product of eq polynomial and the Mle data.
    let accum = data.iter().zip(lagrange.iter()).scan(
        <SC::Challenge as AbstractField>::zero(),
        |acc, (x, y)| {
            *acc += *y * *x;
            Some(*acc)
        },
    );

    for (lagrange_eval, accum) in izip!(lagrange.iter(), accum) {
        let mut row = [SC::Challenge::zero(); NUM_ADAPTER_COLS];

        let cols: &mut MultivariateAdapterCols<_> = row.as_mut_slice().borrow_mut();

        // cols.val = *val;
        cols.lagrange_eval = *lagrange_eval;
        cols.accum = accum;

        trace.extend(row);
    }

    RowMajorMatrix::new(trace, NUM_ADAPTER_COLS)
}

#[cfg(test)]
pub mod tests {

    use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
    use p3_challenger::DuplexChallenger;
    use p3_commit::ExtensionMmcs;
    use p3_dft::Radix2DitParallel;
    use p3_fri::{FriConfig, TwoAdicFriPcs};
    use p3_merkle_tree::FieldMerkleTreeMmcs;
    use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
    use p3_symmetric::{PaddingFreeSponge, TruncatedPermutation};
    use p3_uni_stark::StarkConfig;
    use rand::{thread_rng, Rng};
    use spl_algebra::{
        extension::BinomialExtensionField, AbstractExtensionField, AbstractField, Field,
    };
    use spl_multi_pcs::{Mle, MultilinearPcs, Point};

    use crate::prover::AdapterProver;

    use super::MultivariateAdapterPCS;

    type Val = BabyBear;
    type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    type ValMmcs = FieldMerkleTreeMmcs<
        <Val as Field>::Packing,
        <Val as Field>::Packing,
        MyHash,
        MyCompress,
        8,
    >;
    type Challenge = BinomialExtensionField<Val, 4>;
    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
    type Dft = Radix2DitParallel;
    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;

    #[test]
    fn test_generate_adapter_trace() {
        let data = vec![Val::one(), Val::one()];
        let eval_point = Point::new(vec![Val::two()]);
        let trace = crate::types::generate_adapter_trace::<MyConfig>(
            &data,
            &Point(eval_point.0.iter().map(|x| Challenge::from_base(*x)).collect()),
        );
        println!("{:?}", trace);
    }

    #[test]
    fn test_adapter_stark() {
        let perm = Perm::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabyBear,
            &mut thread_rng(),
        );
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let dft = Dft {};
        const NUM_VARIABLES: usize = 8;
        const HEIGHT: usize = 1 << NUM_VARIABLES;

        let vals = std::array::from_fn::<_, HEIGHT, _>(|_| thread_rng().gen()).to_vec();
        let eval_point: Point<Challenge> =
            Point::new(std::array::from_fn::<_, NUM_VARIABLES, _>(|_| thread_rng().gen()).to_vec());
        let fri_config = FriConfig {
            log_blowup: 2,
            num_queries: 28,
            proof_of_work_bits: 8,
            mmcs: challenge_mmcs,
        };
        let pcs = Pcs::new(NUM_VARIABLES, dft, val_mmcs, fri_config);
        let config = MyConfig::new(pcs);
        let mut challenger = Challenger::new(perm.clone());

        let pcs = MultivariateAdapterPCS { config };

        let prover = AdapterProver::new(pcs);

        let (commitment, data) = prover.commit(Mle::new(vals.clone()));

        let mle = Mle::new(vals);
        let expected_eval = mle.eval_at_point(&eval_point);

        let proof = prover.prove_evaluation(
            mle,
            Point(eval_point.clone().0.iter().map(|x| Challenge::from_base(*x)).collect()),
            Challenge::from_base(expected_eval),
            data,
            commitment,
            &mut challenger,
        );

        prover
            .pcs
            .verify(
                Point(eval_point.clone().0.iter().map(|x| Challenge::from_base(*x)).collect()),
                Challenge::from_base(expected_eval),
                commitment,
                &proof,
                &mut Challenger::new(perm.clone()),
            )
            .unwrap();
    }

    #[test]
    #[should_panic]
    fn test_adapter_stark_fails_on_non_matching_commitment() {
        let perm = Perm::new_from_rng_128(
            Poseidon2ExternalMatrixGeneral,
            DiffusionMatrixBabyBear,
            &mut thread_rng(),
        );
        let hash = MyHash::new(perm.clone());
        let compress = MyCompress::new(perm.clone());
        let val_mmcs = ValMmcs::new(hash, compress);
        let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
        let dft = Dft {};

        let vals = std::array::from_fn::<BabyBear, 8, _>(|_| thread_rng().gen()).to_vec();
        let eval_point: Point<Challenge> =
            Point::new(std::array::from_fn::<_, 3, _>(|_| thread_rng().gen()).to_vec());
        let fri_config = FriConfig {
            log_blowup: 2,
            num_queries: 28,
            proof_of_work_bits: 8,
            mmcs: challenge_mmcs,
        };
        let pcs = Pcs::new(3, dft, val_mmcs, fri_config);
        let config = MyConfig::new(pcs);
        let mut challenger = Challenger::new(perm.clone());

        let pcs = MultivariateAdapterPCS { config };

        let prover = AdapterProver::new(pcs);

        let (commit, data) = prover.commit(Mle::new(vals.clone()));
        let mle = Mle::new(vals);
        let expected_eval = mle.eval_at_point(&eval_point);

        let proof = prover.prove_evaluation(
            mle,
            Point(eval_point.clone().0.iter().map(|x| Challenge::from_base(*x)).collect()),
            Challenge::from_base(expected_eval),
            data,
            commit,
            &mut challenger,
        );

        prover
            .pcs
            .verify(
                Point(eval_point.clone().0.iter().map(|x| Challenge::from_base(*x)).collect()),
                // Put a wrong value here to make sure the verification fails.
                Challenge::from_base(Val::from_canonical_u16(0xDEAD)),
                commit,
                &proof,
                &mut Challenger::new(perm.clone()),
            )
            .unwrap();
    }
}
