mod types;
mod verifier;

pub use types::*;

use slop_fri::verifier::FriError;

#[cfg(test)]
mod tests {
    use rand::Rng;
    use slop_algebra::AbstractField;
    use slop_baby_bear::BabyBear;
    use slop_multilinear::Mle;

    use slop_multilinear::Point;

    type F = BabyBear;

    #[test]
    pub fn test_lagrange_eval() {
        for _ in 0..10 {
            let rng = &mut rand::thread_rng();
            let point_1: Point<F> = Point::from(vec![rng.gen(), rng.gen(), rng.gen()]);
            let point_2: Point<F> = Point::from(vec![rng.gen(), rng.gen(), rng.gen()]);
            let eq_mle = Mle::<F>::partial_lagrange(&point_1);
            assert_eq!(Mle::full_lagrange_eval(&point_1, &point_2), eq_mle.eval_at(&point_2)[0]);
        }
    }

    #[test]
    fn test_eval_at_point() {
        // Choosing the entry corresponding to the bitstring 10.
        let mle = Mle::from(vec![F::zero(), F::zero(), F::one(), F::zero()]);

        let point = Point::from(vec![F::two(), F::one()]);

        assert_eq!(mle.eval_at(&point)[0], F::zero());
    }
}
