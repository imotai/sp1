use slop_algebra::{ExtensionField, Field};

use crate::{full_geq, Mle, Point};

#[derive(Debug, Copy, Clone)]
pub struct VirtualGeq<F> {
    pub threshold: u32,
    pub geq_coefficient: F,
    pub eq_coefficient: F,
    pub num_vars: u32,
}
impl<F: Field> VirtualGeq<F> {
    pub fn new(threshold: u32, geq_coefficient: F, eq_coefficient: F, num_vars: u32) -> Self {
        assert!(threshold <= (1 << num_vars));
        Self { threshold, geq_coefficient, eq_coefficient, num_vars }
    }
    pub fn fix_last_variable<EF: ExtensionField<F>>(&self, alpha: EF) -> VirtualGeq<EF> {
        let new_threshold = self.threshold >> 1;
        let new_geq_coefficient = self.geq_coefficient.into();
        let new_eq_coefficient = if self.threshold & 1 == 0 { EF::one() - alpha } else { alpha }
            * self.eq_coefficient
            + if self.threshold & 1 == 0 {
                EF::zero()
            } else {
                (alpha - EF::one()) * self.geq_coefficient
            };

        VirtualGeq {
            threshold: new_threshold,
            geq_coefficient: new_geq_coefficient,
            eq_coefficient: new_eq_coefficient,
            num_vars: self.num_vars.saturating_sub(1),
        }
    }

    pub fn eval_at<EF: ExtensionField<F>>(&self, point: &Point<EF>) -> EF {
        if self.threshold == 1 << self.num_vars {
            return EF::zero();
        }
        let threshold_point = Point::from_usize(self.threshold as usize, self.num_vars as usize);
        let eq_eval = Mle::<F>::full_lagrange_eval(&threshold_point, point);
        let geq_eval = full_geq(&threshold_point, point);
        eq_eval * self.eq_coefficient + geq_eval * self.geq_coefficient
    }

    pub fn sum(&self) -> F {
        F::from_canonical_usize((1 << self.num_vars) - self.threshold as usize)
            * self.geq_coefficient
            + self.eq_coefficient
    }

    pub fn to_extension<EF: ExtensionField<F>>(&self) -> VirtualGeq<EF> {
        VirtualGeq {
            threshold: self.threshold,
            geq_coefficient: self.geq_coefficient.into(),
            eq_coefficient: self.eq_coefficient.into(),
            num_vars: self.num_vars,
        }
    }

    pub fn eval_at_usize(&self, index: usize) -> F {
        assert!(index <= (1 << self.num_vars));
        if index < self.threshold as usize {
            F::zero()
        } else if index == self.threshold as usize {
            self.eq_coefficient + self.geq_coefficient
        } else if index < (1 << self.num_vars) {
            F::one()
        } else {
            F::zero()
        }
    }
}

#[cfg(test)]
pub mod tests {

    use crate::partial_geq;

    use super::*;
    use rand::Rng;
    use slop_algebra::AbstractField;
    use slop_baby_bear::BabyBear;

    type F = BabyBear;

    #[tokio::test]
    async fn test_virtual_geq() {
        let num_vars = 4;
        let mut rng = rand::thread_rng();
        for threshold in 0..(1 << num_vars) {
            let geq_coefficient = rng.gen::<F>();
            let eq_coefficient = rng.gen::<F>();
            let geq = VirtualGeq { threshold, geq_coefficient, eq_coefficient, num_vars };
            let threshold_point = Point::<F>::from_usize(threshold as usize, num_vars as usize);
            let partial_lagrange = Mle::blocking_partial_lagrange(&threshold_point);
            let partial_geq = partial_geq::<F>(threshold as usize, num_vars as usize);
            let point = Point::<F>::rand(&mut rng, num_vars);
            assert_eq!(
                geq.eval_at(&point),
                geq_coefficient
                    * Mle::from(partial_geq.clone()).blocking_eval_at(&point).to_vec()[0]
                    + eq_coefficient * partial_lagrange.blocking_eval_at(&point).to_vec()[0]
            );

            assert_eq!(
                geq.sum(),
                geq_coefficient * partial_geq.iter().copied().sum::<F>()
                    + eq_coefficient
                        * partial_lagrange.guts().as_slice().iter().copied().sum::<F>()
            );
            let alpha = rng.gen::<F>();
            let new_geq = geq.fix_last_variable(alpha);
            let new_lagrange = partial_lagrange.fix_last_variable(alpha).await;
            let new_partial_geq = Mle::from(partial_geq).fix_last_variable(alpha).await;
            let new_point = Point::<F>::rand(&mut rng, num_vars - 1);
            assert_eq!(
                new_geq.eval_at(&new_point),
                geq_coefficient * new_partial_geq.blocking_eval_at(&new_point).to_vec()[0]
                    + eq_coefficient * new_lagrange.blocking_eval_at(&new_point).to_vec()[0]
            );

            let mut new_virtual_geq = VirtualGeq {
                threshold,
                geq_coefficient: F::one(),
                eq_coefficient: F::zero(),
                num_vars,
            };

            let mut randomness = vec![];

            for _ in 0..num_vars {
                let alpha = rng.gen::<F>();
                randomness.insert(0, alpha);
                new_virtual_geq = new_virtual_geq.fix_last_variable(alpha);
            }
            assert_eq!(
                full_geq(&threshold_point, &Point::from(randomness.clone())),
                new_virtual_geq.eval_at(&Point::from(vec![]))
            );

            assert_eq!(
                full_geq(&threshold_point, &Point::from(randomness)),
                new_virtual_geq.eq_coefficient + new_virtual_geq.geq_coefficient
            );
        }
    }
}
