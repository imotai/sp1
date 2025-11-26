use std::{
    ops::{Add, Mul, Neg, Sub},
    sync::{Arc, OnceLock},
};

/// Global, thread-safe holder for the PCL prime field modulus.
///
/// This is initialized exactly once via [`set_field_modulus`]. Arithmetic
/// that combines only constants will be reduced modulo this value when set.
static FIELD_MODULUS: OnceLock<Arc<u64>> = OnceLock::new();

/// Sets the field modulus for PCL
pub fn set_field_modulus(p: u64) -> Result<(), u64> {
    // set only once; returns Err(p) if already set
    FIELD_MODULUS.set(Arc::new(p)).map_err(|arc| Arc::try_unwrap(arc).unwrap_or_else(|a| *a))
}

/// Get PCL field modulus
pub fn current_modulus() -> Option<u64> {
    FIELD_MODULUS.get().map(|a| **a)
}

/// Given an integer reduce it into the field
fn reduce_mod(c: u64) -> u64 {
    if let Some(p) = current_modulus() {
        c % p
    } else {
        c
    }
}

/// Arithmetic expressions over the Picus constraint language (PCL).
#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum PicusExpr {
    /// Constant field element (big integer). NOTE: could use `BigUint` library but just familiar
    /// with rug
    Const(u64),
    /// Variable identified by `(name, column_index)`, printed as `name_index`.
    Var(String, usize),
    /// Add.
    Add(Box<PicusExpr>, Box<PicusExpr>),
    /// Sub.
    Sub(Box<PicusExpr>, Box<PicusExpr>),
    /// Mul
    Mul(Box<PicusExpr>, Box<PicusExpr>),
    /// Div (probably can delete)
    Div(Box<PicusExpr>, Box<PicusExpr>),
    /// Unary negation.
    Neg(Box<PicusExpr>),
    /// Exponentiation
    Pow(u64, Box<PicusExpr>),
}

impl PicusExpr {
    /// Approximate tree size (number of nodes).
    ///
    /// Useful as a heuristic for introducing temporary variables (e.g., to keep
    /// expressions small for solvers). `Pow` is counted as 1 by design.
    #[must_use]
    pub fn size(&self) -> usize {
        match self {
            Self::Const(_) | Self::Var(_, _) | Self::Pow(_, _) => 1,
            Self::Add(a, b) | Self::Sub(a, b) | Self::Mul(a, b) | Self::Div(a, b) => {
                1 + a.size() + b.size()
            }
            Self::Neg(a) => 1 + a.size(),
        }
    }
    /// Helper to construct a `Var` with a name, index.
    pub fn var(name: impl Into<String>, idx: usize) -> Self {
        PicusExpr::Var(name.into(), idx)
    }
    #[must_use]
    /// Convenience for exponentiating by a non-negative `u32` power.
    pub fn pow(self, k: u32) -> Self {
        PicusExpr::Pow(k.into(), Box::new(self))
    }
    /// Returns `true` iff this is exactly the constant zero.
    #[inline]
    #[must_use]
    pub fn is_const_zero(&self) -> bool {
        matches!(self, PicusExpr::Const(c) if *c == 0)
    }
}

macro_rules! impl_from_ints {
    ($($t:ty),* $(,)?) => {$(
        impl From<$t> for PicusExpr {
            fn from(v: $t) -> Self {
                PicusExpr::Const(v as u64)
            }
        }
    )*}
}

impl_from_ints!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);

/// Pointwise addition with light constant folding.
///
/// - If both sides are constant, the sum is reduced modulo the current field (if set).
/// - Adding zero returns the other side.
/// - Otherwise, constructs `Add(lhs, rhs)`.
impl Add<PicusExpr> for PicusExpr {
    type Output = PicusExpr;
    fn add(self, rhs: PicusExpr) -> Self::Output {
        let lhs = self.clone();
        match (lhs.clone(), rhs.clone()) {
            (PicusExpr::Const(c_1), PicusExpr::Const(c_2)) => (reduce_mod(c_1 + c_2)).into(),
            (PicusExpr::Const(c), _) => {
                if c == 0 {
                    rhs
                } else {
                    PicusExpr::Add(Box::new(lhs), Box::new(rhs))
                }
            }
            (_, PicusExpr::Const(c)) => {
                if c == 0 {
                    lhs
                } else {
                    PicusExpr::Add(Box::new(lhs), Box::new(rhs))
                }
            }
            _ => PicusExpr::Add(Box::new(lhs), Box::new(rhs)),
        }
    }
}

/// Pointwise subtraction with light constant folding.
///
/// - If both sides are constant, the difference is reduced modulo the current field (if set).
/// - Subtracting zero returns the left-hand side.
/// - Otherwise, constructs `Sub(lhs, rhs)`.
impl Sub<PicusExpr> for PicusExpr {
    type Output = PicusExpr;
    fn sub(self, rhs: PicusExpr) -> Self::Output {
        let lhs = self.clone();
        match (lhs.clone(), rhs.clone()) {
            (PicusExpr::Const(c_1), PicusExpr::Const(c_2)) => reduce_mod(c_1 - c_2).into(),
            (_, PicusExpr::Const(c)) => {
                if c == 0 {
                    lhs
                } else {
                    PicusExpr::Sub(Box::new(self), Box::new(rhs))
                }
            }
            _ => PicusExpr::Sub(Box::new(self), Box::new(rhs)),
        }
    }
}

/// Unary negation with constant folding.
///
/// - If the input is a constant, returns the additive inverse reduced modulo the current field (if
///   set). Otherwise constructs `Neg`.
impl Neg for PicusExpr {
    type Output = PicusExpr;
    fn neg(self) -> Self::Output {
        let lhs = self.clone();
        match lhs.clone() {
            PicusExpr::Const(c) => reduce_mod(current_modulus().unwrap() - c).into(),
            _ => PicusExpr::Neg(Box::new(lhs)),
        }
    }
}

/// Pointwise multiplication with light constant folding and scalar routing.
///
/// - If either side is a constant, routes to the `(PicusExpr * Integer)` impl to share logic.
/// - Otherwise constructs `Mul(lhs, rhs)`.
impl Mul<PicusExpr> for PicusExpr {
    type Output = PicusExpr;
    fn mul(self, rhs: PicusExpr) -> Self::Output {
        let lhs = self.clone();
        match (lhs.clone(), rhs.clone()) {
            (PicusExpr::Const(c), _) => rhs * c,
            (_, PicusExpr::Const(c)) => lhs * c,
            _ => PicusExpr::Mul(Box::new(lhs), Box::new(rhs)),
        }
    }
}

/// Scalar multiplication with constant folding.
///
/// - Multiplying by `0` yields `0`.
/// - Multiplying by `1` yields the original expression.
/// - If the left is also a constant, multiply and reduce modulo the current field (if set).
/// - Otherwise constructs `Mul(lhs, Const(rhs))`.
impl Mul<u64> for PicusExpr {
    type Output = PicusExpr;
    fn mul(self, rhs: u64) -> Self::Output {
        if rhs == 0 {
            return PicusExpr::Const(0);
        }
        if rhs == 1 {
            return self.clone();
        }
        let lhs = self.clone();
        match lhs {
            PicusExpr::Const(c_1) => reduce_mod(c_1 * rhs).into(),
            _ => PicusExpr::Mul(Box::new(lhs), Box::new(rhs.into())),
        }
    }
}

/// Boolean/relational constraints over `PicusExpr`.
#[derive(Debug, Clone)]
pub enum PicusConstraint {
    /// x < y
    Lt(Box<PicusExpr>, Box<PicusExpr>),
    /// x <= y
    Leq(Box<PicusExpr>, Box<PicusExpr>),
    /// x > y
    Gt(Box<PicusExpr>, Box<PicusExpr>),
    /// x >= y
    Geq(Box<PicusExpr>, Box<PicusExpr>),
    /// p => q
    Implies(Box<PicusConstraint>, Box<PicusConstraint>),
    /// -p
    Not(Box<PicusConstraint>),
    /// p <=> q
    Iff(Box<PicusConstraint>, Box<PicusConstraint>),
    /// p && q
    And(Box<PicusConstraint>, Box<PicusConstraint>),
    /// p || q
    Or(Box<PicusConstraint>, Box<PicusConstraint>),
    /// Canonical equality-to-zero form: `Eq(e)` represents `e = 0`.
    Eq(Box<PicusExpr>),
}

impl PicusConstraint {
    /// Build an equality constraint `left = right` by moving to zero:
    /// returns `Eq(left - right)`.
    #[must_use]
    pub fn new_equality(left: PicusExpr, right: PicusExpr) -> PicusConstraint {
        PicusConstraint::Eq(Box::new(left - right))
    }

    /// Build a comparison constraint `left < right`
    #[must_use]
    pub fn new_lt(left: PicusExpr, right: PicusExpr) -> PicusConstraint {
        PicusConstraint::Lt(Box::new(left), Box::new(right))
    }

    /// Build a comparison constraint `left <= right`
    #[must_use]
    pub fn new_leq(left: PicusExpr, right: PicusExpr) -> PicusConstraint {
        PicusConstraint::Leq(Box::new(left), Box::new(right))
    }

    /// Build a comparison constraint `left > right`
    #[must_use]
    pub fn new_gt(left: PicusExpr, right: PicusExpr) -> PicusConstraint {
        PicusConstraint::Gt(Box::new(left), Box::new(right))
    }

    /// Build a comparison constraint `left >= right`
    #[must_use]
    pub fn new_geq(left: PicusExpr, right: PicusExpr) -> PicusConstraint {
        PicusConstraint::Geq(Box::new(left), Box::new(right))
    }

    /// Assumes ``l`` and ``u`` fit into the prime
    /// Generates constraints l <= e <= u
    #[must_use]
    pub fn in_range(e: PicusExpr, l: usize, u: usize) -> Vec<PicusConstraint> {
        assert!(l < u);
        vec![PicusConstraint::new_geq(e.clone(), l.into()), PicusConstraint::new_leq(e, u.into())]
    }
}
