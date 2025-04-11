use std::{convert::Infallible, fmt::Debug};

use serde::{Deserialize, Serialize};
use slop_algebra::{ExtensionField, Field};
use slop_multilinear::Point;

use crate::{JaggedLittlePolynomialProverParams, JaggedLittlePolynomialVerifierParams};

use super::{JaggedEvalConfig, JaggedEvalProver};

#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct TrivialJaggedEvalConfig;

impl<F: Field, EF: ExtensionField<F>, C: Send + Sync> JaggedEvalConfig<F, EF, C>
    for TrivialJaggedEvalConfig
{
    type JaggedEvalProof = ();
    type JaggedEvalError = Infallible;

    fn jagged_evaluation(
        &self,
        params: &JaggedLittlePolynomialVerifierParams<F>,
        z_row: &Point<EF>,
        z_col: &Point<EF>,
        z_trace: &Point<EF>,
        _proof: &Self::JaggedEvalProof,
        _challenger: &mut C,
    ) -> Result<EF, Self::JaggedEvalError> {
        let (result, _) = params.full_jagged_little_polynomial_evaluation(z_row, z_col, z_trace);
        Ok(result)
    }
}

impl<F: Field, EF: ExtensionField<F>, C: Send + Sync> JaggedEvalProver<F, EF, C>
    for TrivialJaggedEvalConfig
{
    type EvalProof = ();
    type EvalConfig = TrivialJaggedEvalConfig;
    async fn prove_jagged_evaluation(
        &self,
        _params: &JaggedLittlePolynomialProverParams,
        _z_row: &Point<EF>,
        _z_col: &Point<EF>,
        _z_trace: &Point<EF>,
        _challenger: &mut C,
    ) -> Self::EvalProof {
    }
}
