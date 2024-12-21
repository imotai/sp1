mod types;
mod verifier;

pub use types::*;

use p3_fri::verifier::FriError;

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use rand::Rng;
    use slop_algebra::AbstractField;
    use slop_multilinear::{full_lagrange_eval, partial_lagrange_eval, Mle};

    use slop_multilinear::Point;

    type F = BabyBear;

    #[test]
    pub fn test_lagrange_eval() {
        for _ in 0..10 {
            let rng = &mut rand::thread_rng();
            let point_1: Point<F> = Point::new(vec![rng.gen(), rng.gen(), rng.gen()]);
            let point_2: Point<F> = Point::new(vec![rng.gen(), rng.gen(), rng.gen()]);
            assert_eq!(
                full_lagrange_eval(&point_1, &point_2),
                Mle::eval_at_point(&partial_lagrange_eval(&point_1).into(), &point_2)
            );
        }
    }

    #[test]
    fn test_eval_at_point() {
        // Choosing the entry corresponding to the bitstring 10.
        let mle = Mle::new(vec![F::zero(), F::zero(), F::one(), F::zero()]);

        let point = Point::new(vec![F::two(), F::one()]);

        assert_eq!(mle.eval_at_point(&point), F::zero());
    }
}
