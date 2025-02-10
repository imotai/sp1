use slop_algebra::{ExtensionField, TwoAdicField};
use slop_challenger::{CanObserve, FieldChallenger, GrindingChallenger};
use slop_commit::TensorCs;

/// The configuration required for a Reed-Solomon-based Basefold.
pub trait BasefoldConfig {
    /// The base field.
    ///
    /// This is the field on which the MLEs committed to are defined over.
    type F: TwoAdicField;
    /// The field of random elements.
    ///
    /// This is an extension field of the base field which is of cryptographically secure size. The
    /// random evaluation points of the protocol are drawn from `EF`.
    type EF: ExtensionField<Self::F>;
    /// The tensor commitment scheme.
    ///
    /// The tensor commitment scheme is used to send long messages in the protocol by converting
    /// them to a tensor committment providing oracle acccess.
    type Tcs: TensorCs<Data = Self::F>;
    /// The challenger type that creates the random challenges via Fiat-Shamir.
    ///
    /// The challenger is observing all the messages sent throughout the protocol and uses this
    /// to create the verifier messages of the IOP.
    type Challenger: FieldChallenger<Self::F>
        + GrindingChallenger
        + CanObserve<<Self::Tcs as TensorCs>::Commitment>;
}
