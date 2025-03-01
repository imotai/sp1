// use std::marker::PhantomData;

// use p3_uni_stark::SymbolicAirBuilder;
// use serde::{Deserialize, Serialize};
// use slop_air::Air;
// use slop_algebra::extension::BinomialExtensionField;
// use slop_alloc::CpuBackend;
// use slop_baby_bear::BabyBear;
// use slop_jagged::{
//     JaggedConfig, Poseidon2BabyBearJaggedConfig, Poseidon2BabyBearJaggedCpuProverComponents,
// };

// use crate::{
//     air::MachineAir,
//     prover::{DefaultTraceGenerator, MachineProverComponents, ZerocheckCpuProverData},
//     ConstraintSumcheckFolder,
// };

// #[derive(Clone, Debug, Default, Serialize, Deserialize)]
// pub struct BabyBearPoseidon2CpuProverComponents<A>(PhantomData<A>);

// impl<A> MachineProverComponents for BabyBearPoseidon2CpuProverComponents<A>
// where
//     A: MachineAir<BabyBear>
//         + Air<SymbolicAirBuilder<BabyBear>>
//         + for<'b> Air<
//             ConstraintSumcheckFolder<'b, BabyBear, BabyBear, BinomialExtensionField<BabyBear, 4>>,
//         > + for<'b> Air<
//             ConstraintSumcheckFolder<
//                 'b,
//                 BabyBear,
//                 BinomialExtensionField<BabyBear, 4>,
//                 BinomialExtensionField<BabyBear, 4>,
//             >,
//         >,
// {
//     type F = BabyBear;
//     type EF = BinomialExtensionField<BabyBear, 4>;
//     type Air = A;
//     type B = CpuBackend;

//     type Config = Poseidon2BabyBearJaggedConfig;
//     type Program = <A as MachineAir<BabyBear>>::Program;
//     type Record = <A as MachineAir<BabyBear>>::Record;

//     type Commitment = <Self::Config as JaggedConfig>::Commitment;
//     type Challenger = <Self::Config as JaggedConfig>::Challenger;

//     type PcsProverComponents = Poseidon2BabyBearJaggedCpuProverComponents;
//     type TraceGenerator = DefaultTraceGenerator<Self::F, Self::Air, Self::B>;
//     type ZerocheckProverData = ZerocheckCpuProverData<A>;
// }

// #![allow(missing_docs)]

// use serde::{Deserialize, Serialize};
// use slop_jagged::{
//     JaggedConfig, JaggedPcsVerifier, JaggedProver, Poseidon2BabyBearJaggedConfig,
//     Poseidon2BabyBearJaggedCpuProverComponents,
// };
// use sp1_primitives::poseidon2_init;

// pub const DIGEST_SIZE: usize = 8;

// pub const LOG_STACKING_HEIGHT: usize = 21;
// pub const MAX_LOG_ROW_COUNT: usize = 21;

// /// A configuration for inner recursion.
// pub type InnerVal = BabyBear;
// pub type InnerChallenge = BinomialExtensionField<InnerVal, 4>;
// pub type InnerPerm =
//     Poseidon2<InnerVal, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
// pub type InnerHash = PaddingFreeSponge<InnerPerm, 16, 8, DIGEST_SIZE>;
// pub type InnerDigestHash = Hash<InnerVal, InnerVal, DIGEST_SIZE>;
// pub type InnerDigest = [InnerVal; DIGEST_SIZE];
// pub type InnerCompress = TruncatedPermutation<InnerPerm, 2, 8, 16>;
// pub type InnerValMmcs = FieldMerkleTreeMmcs<
//     <InnerVal as Field>::Packing,
//     <InnerVal as Field>::Packing,
//     InnerHash,
//     InnerCompress,
//     8,
// >;
// pub type InnerChallengeMmcs = ExtensionMmcs<InnerVal, InnerChallenge, InnerValMmcs>;
// pub type InnerChallenger = DuplexChallenger<InnerVal, InnerPerm, 16, 8>;
// pub type InnerDft = Radix2DitParallel;
// pub type InnerPcsProver = JaggedProver<Poseidon2BabyBearJaggedCpuProverComponents>;
// pub type InnerPcsVerifier = JaggedPcsVerifier<Poseidon2BabyBearJaggedConfig>;
// pub type InnerQueryProof = QueryProof<InnerChallenge, InnerChallengeMmcs>;
// pub type InnerCommitPhaseStep = CommitPhaseProofStep<InnerChallenge, InnerChallengeMmcs>;
// pub type InnerFriProof = FriProof<InnerChallenge, InnerChallengeMmcs, InnerVal>;
// pub type InnerBatchOpening = BatchOpening<InnerVal, InnerValMmcs>;
// pub type InnerPcsProof =
//     TwoAdicFriPcsProof<InnerVal, InnerChallenge, InnerValMmcs, InnerChallengeMmcs>;

// /// The permutation for inner recursion.
// #[must_use]
// pub fn inner_perm() -> InnerPerm {
//     poseidon2_init()
// }

// /// The FRI config for sp1 proofs.
// #[must_use]
// pub fn sp1_fri_config() -> FriConfig<InnerChallengeMmcs> {
//     let perm = inner_perm();
//     let hash = InnerHash::new(perm.clone());
//     let compress = InnerCompress::new(perm.clone());
//     let challenge_mmcs = InnerChallengeMmcs::new(InnerValMmcs::new(hash, compress));
//     let num_queries = match std::env::var("FRI_QUERIES") {
//         Ok(value) => value.parse().unwrap(),
//         Err(_) => 100,
//     };
//     FriConfig { log_blowup: 1, num_queries, proof_of_work_bits: 16, mmcs: challenge_mmcs }
// }

// /// The FRI config for inner recursion.
// #[must_use]
// pub fn inner_fri_config() -> FriConfig<InnerChallengeMmcs> {
//     let perm = inner_perm();
//     let hash = InnerHash::new(perm.clone());
//     let compress = InnerCompress::new(perm.clone());
//     let challenge_mmcs = InnerChallengeMmcs::new(InnerValMmcs::new(hash, compress));
//     let num_queries = match std::env::var("FRI_QUERIES") {
//         Ok(value) => value.parse().unwrap(),
//         Err(_) => 100,
//     };
//     FriConfig { log_blowup: 1, num_queries, proof_of_work_bits: 16, mmcs: challenge_mmcs }
// }

// /// The recursion config used for recursive reduce circuit.
// #[derive(Deserialize)]
// #[serde(from = "std::marker::PhantomData<BabyBearPoseidon2Inner>")]
// pub struct BabyBearPoseidon2Inner {
//     pub perm: InnerPerm,
//     pub pcs_prover: JaggedProver<Poseidon2BabyBearJaggedCpuProverComponents>,
//     pub pcs_verifier: JaggedPcsVerifier<Poseidon2BabyBearJaggedConfig>,
// }

// impl Clone for BabyBearPoseidon2Inner {
//     fn clone(&self) -> Self {
//         Self::new()
//     }
// }

// impl Serialize for BabyBearPoseidon2Inner {
//     fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
//     where
//         S: serde::Serializer,
//     {
//         std::marker::PhantomData::<BabyBearPoseidon2Inner>.serialize(serializer)
//     }
// }

// impl From<std::marker::PhantomData<BabyBearPoseidon2Inner>> for BabyBearPoseidon2Inner {
//     fn from(_: std::marker::PhantomData<BabyBearPoseidon2Inner>) -> Self {
//         Self::new()
//     }
// }

// impl BabyBearPoseidon2Inner {
//     #[must_use]
//     pub fn new() -> Self {
//         let perm = inner_perm();

//         let verifier =
//             JaggedPcsVerifier::<<Self as StarkGenericConfig>::JaggedConfig>::new(1, 21, 21);

//         let prover = JaggedProver::from_verifier(&verifier);

//         Self { perm, pcs_prover: prover, pcs_verifier: verifier }
//     }
// }

// impl Default for BabyBearPoseidon2Inner {
//     fn default() -> Self {
//         Self::new()
//     }
// }

// impl StarkGenericConfig for BabyBearPoseidon2Inner {
//     type Val = InnerVal;
//     type JaggedConfig = Poseidon2BabyBearJaggedConfig;
//     type Challenge = InnerChallenge;
//     type Challenger = InnerChallenger;
//     type CpuProverComponents = Poseidon2BabyBearJaggedCpuProverComponents;
//     type Commitment = <Self::JaggedConfig as JaggedConfig>::Commitment;

//     fn prover_pcs(&self) -> &JaggedProver<Self::CpuProverComponents> {
//         &self.pcs_prover
//     }

//     fn pcs_verifier(&self) -> &JaggedPcsVerifier<Self::JaggedConfig> {
//         &self.pcs_verifier
//     }

//     fn challenger(&self) -> Self::Challenger {
//         InnerChallenger::new(self.perm.clone())
//     }
// }

// impl ZeroCommitment<BabyBearPoseidon2Inner> for InnerPcsProver {
//     fn zero_commitment(&self) -> Com<BabyBearPoseidon2Inner> {
//         [InnerVal::zero(); DIGEST_SIZE]
//     }
// }

// impl ZeroCommitment<BabyBearPoseidon2Inner> for InnerPcsVerifier {
//     fn zero_commitment(&self) -> Com<BabyBearPoseidon2Inner> {
//         [InnerVal::zero(); DIGEST_SIZE]
//     }
// }

// pub mod baby_bear_poseidon2 {

//     use p3_baby_bear::{BabyBear, DiffusionMatrixBabyBear};
//     use p3_challenger::DuplexChallenger;
//     use p3_commit::ExtensionMmcs;
//     use p3_dft::Radix2DitParallel;
//     use p3_field::{extension::BinomialExtensionField, AbstractField, Field};
//     use p3_fri::{FriConfig, TwoAdicFriPcs};
//     use p3_merkle_tree::FieldMerkleTreeMmcs;
//     use p3_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
//     use p3_symmetric::{Hash, PaddingFreeSponge, TruncatedPermutation};
//     use serde::{Deserialize, Serialize};
//     use slop_jagged::{
//         JaggedConfig, JaggedPcsVerifier, JaggedProver, Poseidon2BabyBearJaggedConfig,
//         Poseidon2BabyBearJaggedCpuProverComponents,
//     };
//     use sp1_primitives::RC_16_30;

//     use crate::{Com, StarkGenericConfig, ZeroCommitment, DIGEST_SIZE};

//     pub type Val = BabyBear;
//     pub type Challenge = BinomialExtensionField<Val, 4>;

//     pub type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
//     pub type MyHash = PaddingFreeSponge<Perm, 16, 8, DIGEST_SIZE>;
//     pub type DigestHash = Hash<Val, Val, DIGEST_SIZE>;
//     pub type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
//     pub type ValMmcs = FieldMerkleTreeMmcs<
//         <Val as Field>::Packing,
//         <Val as Field>::Packing,
//         MyHash,
//         MyCompress,
//         8,
//     >;
//     pub type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
//     pub type Dft = Radix2DitParallel;
//     pub type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
//     type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

//     #[must_use]
//     pub fn my_perm() -> Perm {
//         const ROUNDS_F: usize = 8;
//         const ROUNDS_P: usize = 13;
//         let mut round_constants = RC_16_30.to_vec();
//         let internal_start = ROUNDS_F / 2;
//         let internal_end = (ROUNDS_F / 2) + ROUNDS_P;
//         let internal_round_constants = round_constants
//             .drain(internal_start..internal_end)
//             .map(|vec| vec[0])
//             .collect::<Vec<_>>();
//         let external_round_constants = round_constants;
//         Perm::new(
//             ROUNDS_F,
//             external_round_constants,
//             Poseidon2ExternalMatrixGeneral,
//             ROUNDS_P,
//             internal_round_constants,
//             DiffusionMatrixBabyBear,
//         )
//     }

//     #[must_use]
//     pub fn default_fri_config() -> FriConfig<ChallengeMmcs> {
//         let perm = my_perm();
//         let hash = MyHash::new(perm.clone());
//         let compress = MyCompress::new(perm.clone());
//         let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));
//         let num_queries = match std::env::var("FRI_QUERIES") {
//             Ok(value) => value.parse().unwrap(),
//             Err(_) => 100,
//         };
//         FriConfig { log_blowup: 1, num_queries, proof_of_work_bits: 16, mmcs: challenge_mmcs }
//     }

//     #[must_use]
//     pub fn compressed_fri_config() -> FriConfig<ChallengeMmcs> {
//         let perm = my_perm();
//         let hash = MyHash::new(perm.clone());
//         let compress = MyCompress::new(perm.clone());
//         let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));
//         let num_queries = match std::env::var("FRI_QUERIES") {
//             Ok(value) => value.parse().unwrap(),
//             Err(_) => 50,
//         };
//         FriConfig { log_blowup: 2, num_queries, proof_of_work_bits: 16, mmcs: challenge_mmcs }
//     }

//     #[must_use]
//     pub fn ultra_compressed_fri_config() -> FriConfig<ChallengeMmcs> {
//         let perm = my_perm();
//         let hash = MyHash::new(perm.clone());
//         let compress = MyCompress::new(perm.clone());
//         let challenge_mmcs = ChallengeMmcs::new(ValMmcs::new(hash, compress));
//         let num_queries = match std::env::var("FRI_QUERIES") {
//             Ok(value) => value.parse().unwrap(),
//             Err(_) => 33,
//         };
//         FriConfig { log_blowup: 3, num_queries, proof_of_work_bits: 16, mmcs: challenge_mmcs }
//     }

//     enum BabyBearPoseidon2Type {
//         Default,
//         Compressed,
//     }

//     #[derive(Deserialize)]
//     #[serde(from = "std::marker::PhantomData<BabyBearPoseidon2>")]
//     pub struct BabyBearPoseidon2 {
//         pub perm: Perm,
//         pcs_prover: JaggedProver<Poseidon2BabyBearJaggedCpuProverComponents>,
//         pcs_verifier: JaggedPcsVerifier<Poseidon2BabyBearJaggedConfig>,
//         config_type: BabyBearPoseidon2Type,
//     }

//     impl BabyBearPoseidon2 {
//         #[must_use]
//         pub fn new() -> Self {
//             let perm = my_perm();

//             let verifier =
//                 JaggedPcsVerifier::<<Self as StarkGenericConfig>::JaggedConfig>::new(1, 21, 21);

//             let prover = JaggedProver::from_verifier(&verifier);

//             Self {
//                 perm,
//                 pcs_prover: prover,
//                 pcs_verifier: verifier,
//                 config_type: BabyBearPoseidon2Type::Default,
//             }
//         }

//         #[must_use]
//         pub fn compressed() -> Self {
//             let perm = my_perm();

//             let verifier =
//                 JaggedPcsVerifier::<<Self as StarkGenericConfig>::JaggedConfig>::new(2, 21, 21);

//             let prover = JaggedProver::from_verifier(&verifier);

//             Self {
//                 perm,
//                 pcs_prover: prover,
//                 pcs_verifier: verifier,
//                 config_type: BabyBearPoseidon2Type::Compressed,
//             }
//         }

//         #[must_use]
//         pub fn ultra_compressed() -> Self {
//             let perm = my_perm();

//             let verifier =
//                 JaggedPcsVerifier::<<Self as StarkGenericConfig>::JaggedConfig>::new(3, 21, 21);

//             let prover = JaggedProver::from_verifier(&verifier);

//             Self {
//                 perm,
//                 pcs_prover: prover,
//                 pcs_verifier: verifier,
//                 config_type: BabyBearPoseidon2Type::Compressed,
//             }
//         }
//     }

//     impl Clone for BabyBearPoseidon2 {
//         fn clone(&self) -> Self {
//             match self.config_type {
//                 BabyBearPoseidon2Type::Default => Self::new(),
//                 BabyBearPoseidon2Type::Compressed => Self::compressed(),
//             }
//         }
//     }

//     impl Default for BabyBearPoseidon2 {
//         fn default() -> Self {
//             Self::new()
//         }
//     }

//     /// Implement serialization manually instead of using serde to avoid cloing the config.
//     impl Serialize for BabyBearPoseidon2 {
//         fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
//         where
//             S: serde::Serializer,
//         {
//             std::marker::PhantomData::<BabyBearPoseidon2>.serialize(serializer)
//         }
//     }

//     impl From<std::marker::PhantomData<BabyBearPoseidon2>> for BabyBearPoseidon2 {
//         fn from(_: std::marker::PhantomData<BabyBearPoseidon2>) -> Self {
//             Self::new()
//         }
//     }

//     impl StarkGenericConfig for BabyBearPoseidon2 {
//         type Val = BabyBear;
//         type JaggedConfig = Poseidon2BabyBearJaggedConfig;
//         type Challenge = Challenge;
//         type Challenger = Challenger;
//         type CpuProverComponents = Poseidon2BabyBearJaggedCpuProverComponents;
//         type Commitment = <Self::JaggedConfig as JaggedConfig>::Commitment;

//         fn prover_pcs(&self) -> &JaggedProver<Self::CpuProverComponents> {
//             &self.pcs_prover
//         }

//         fn pcs_verifier(&self) -> &JaggedPcsVerifier<Self::JaggedConfig> {
//             &self.pcs_verifier
//         }

//         fn challenger(&self) -> Self::Challenger {
//             Challenger::new(self.perm.clone())
//         }
//     }

//     impl ZeroCommitment<BabyBearPoseidon2> for Pcs {
//         fn zero_commitment(&self) -> Com<BabyBearPoseidon2> {
//             [Val::zero(); DIGEST_SIZE]
//         }
//     }
// }
