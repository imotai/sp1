use crate::operations::GlobalInteractionOperation;
use slop_algebra::{AbstractExtensionField, AbstractField, Field, PrimeField32};
use sp1_derive::AlignedBorrow;
use sp1_hypercube::{
    air::{AirInteraction, InteractionScope, SP1AirBuilder, SepticExtensionAirBuilder},
    septic_curve::{SepticCurve, SepticCurveComplete},
    septic_extension::{SepticBlock, SepticExtension},
    InteractionKind,
};

/// A set of columns needed to compute the global interaction elliptic curve digest.
/// It is critical that this struct is at the end of the main trace, as the permutation constraints
/// will be dependent on this fact. It is also critical the the cumulative sum is at the end of this
/// struct, for the same reason.
#[derive(AlignedBorrow, Debug, Clone, Copy)]
#[repr(C)]
pub struct GlobalAccumulationOperation<T> {
    pub initial_digest: [SepticBlock<T>; 2],
    pub cumulative_sum: [SepticBlock<T>; 2],
}

impl<T: Default> Default for GlobalAccumulationOperation<T> {
    fn default() -> Self {
        Self {
            initial_digest: core::array::from_fn(|_| SepticBlock::<T>::default()),
            cumulative_sum: core::array::from_fn(|_| SepticBlock::<T>::default()),
        }
    }
}

impl<F: PrimeField32> GlobalAccumulationOperation<F> {
    pub fn populate_dummy(
        &mut self,
        start_digest: SepticCurve<F>,
        start_digest_plus_dummy: SepticCurve<F>,
    ) {
        self.initial_digest[0] = SepticBlock::from(start_digest.x.0);
        self.initial_digest[1] = SepticBlock::from(start_digest.y.0);
        self.cumulative_sum[0] = SepticBlock::from(start_digest_plus_dummy.x.0);
        self.cumulative_sum[1] = SepticBlock::from(start_digest_plus_dummy.y.0);
    }

    pub fn populate_real(&mut self, sums: &[SepticCurveComplete<F>]) {
        let len = sums.len();
        assert_eq!(len, 2);
        let sums = sums.iter().map(|complete_point| complete_point.point()).collect::<Vec<_>>();
        self.initial_digest[0] = SepticBlock::from(sums[0].x.0);
        self.initial_digest[1] = SepticBlock::from(sums[0].y.0);
        self.cumulative_sum[0] = SepticBlock::from(sums[1].x.0);
        self.cumulative_sum[1] = SepticBlock::from(sums[1].y.0);
    }
}

impl<F: Field> GlobalAccumulationOperation<F> {
    pub fn eval_accumulation<AB: SP1AirBuilder>(
        builder: &mut AB,
        global_interaction_cols: GlobalInteractionOperation<AB::Var>,
        local_is_real: AB::Var,
        local_index: AB::Var,
        local_accumulation: GlobalAccumulationOperation<AB::Var>,
    ) {
        // First, constrain the control flow regarding `is_real`.
        // Constrain that all `is_real` values are boolean.
        builder.assert_bool(local_is_real);

        // Receive the initial digest.
        builder.receive(
            AirInteraction::new(
                vec![local_index]
                    .into_iter()
                    .chain(
                        local_accumulation.initial_digest.into_iter().flat_map(|septic| septic.0),
                    )
                    .map(Into::into)
                    .collect(),
                local_is_real.into(),
                InteractionKind::GlobalAccumulation,
            ),
            InteractionScope::Local,
        );

        // Next, constrain the accumulation.
        let initial_digest = SepticCurve::<AB::Expr> {
            x: SepticExtension::<AB::Expr>::from_base_fn(|i| {
                local_accumulation.initial_digest[0][i].into()
            }),
            y: SepticExtension::<AB::Expr>::from_base_fn(|i| {
                local_accumulation.initial_digest[1][i].into()
            }),
        };

        let cumulative_sum = SepticCurve::<AB::Expr> {
            x: SepticExtension::<AB::Expr>::from_base_fn(|i| {
                local_accumulation.cumulative_sum[0].0[i].into()
            }),
            y: SepticExtension::<AB::Expr>::from_base_fn(|i| {
                local_accumulation.cumulative_sum[1].0[i].into()
            }),
        };

        let point_to_add = SepticCurve::<AB::Expr> {
            x: SepticExtension::<AB::Expr>::from_base_fn(|i| {
                global_interaction_cols.x_coordinate.0[i].into()
            }),
            y: SepticExtension::<AB::Expr>::from_base_fn(|i| {
                global_interaction_cols.y_coordinate.0[i].into()
            }),
        };

        // If `is_real == 1`, initial_digest + point_to_add == cumulative_sum must hold.
        // Constrain that `sum_checker_x` and `sum_checker_y` are both zero when `is_real == 1`.
        let sum_checker_x = SepticCurve::<AB::Expr>::sum_checker_x(
            initial_digest.clone(),
            point_to_add.clone(),
            cumulative_sum.clone(),
        );
        let sum_checker_y = SepticCurve::<AB::Expr>::sum_checker_y(
            initial_digest.clone(),
            point_to_add,
            cumulative_sum.clone(),
        );

        // We enforce `sum_checker_x == 0` always, by putting appropriate dummy rows.
        // If `local_is_real == 0`, then the state machine doesn't do anything already.
        builder.assert_septic_ext_eq(sum_checker_x, SepticExtension::<AB::Expr>::zero());
        builder
            .when(local_is_real)
            .assert_septic_ext_eq(sum_checker_y, SepticExtension::<AB::Expr>::zero());

        // Send the next digest, with the incremented `index`.
        builder.send(
            AirInteraction::new(
                vec![local_index + AB::Expr::one()]
                    .into_iter()
                    .chain(
                        local_accumulation
                            .cumulative_sum
                            .into_iter()
                            .flat_map(|septic| septic.0)
                            .map(Into::into),
                    )
                    .collect(),
                local_is_real.into(),
                InteractionKind::GlobalAccumulation,
            ),
            InteractionScope::Local,
        );
    }
}
