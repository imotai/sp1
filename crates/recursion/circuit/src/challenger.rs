use std::{borrow::BorrowMut, mem::MaybeUninit};

use p3_challenger::DuplexChallenger;
use p3_symmetric::CryptographicPermutation;
use serde::{Deserialize, Serialize};
use slop_algebra::{AbstractField, PrimeField32};
use slop_baby_bear::BabyBear;
use slop_multilinear::Point;
use sp1_derive::AlignedBorrow;
use sp1_recursion_compiler::{
    circuit::CircuitV2Builder,
    prelude::{Builder, Config, Ext, Felt},
};
use sp1_recursion_executor::{HASH_RATE, NUM_BITS, PERMUTATION_WIDTH};

use crate::CircuitConfig;

// Constants for the Multifield challenger.
pub const POSEIDON_2_BB_RATE: usize = 16;

// use crate::{DigestVariable, VerifyingKeyVariable};

pub trait CanCopyChallenger<C: Config> {
    fn copy(&self, builder: &mut Builder<C>) -> Self;
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpongeChallengerShape {
    pub input_buffer_len: usize,
    pub output_buffer_len: usize,
}

/// Reference: [p3_challenger::CanObserve].
pub trait CanObserveVariable<C: Config, V> {
    fn observe(&mut self, builder: &mut Builder<C>, value: V);

    fn observe_slice(&mut self, builder: &mut Builder<C>, values: impl IntoIterator<Item = V>) {
        for value in values {
            self.observe(builder, value);
        }
    }
}

pub trait CanSampleVariable<C: Config, V> {
    fn sample(&mut self, builder: &mut Builder<C>) -> V;
}

/// Reference: [p3_challenger::FieldChallenger].
pub trait FieldChallengerVariable<C: Config, Bit>:
    CanObserveVariable<C, Felt<C::F>> + CanSampleVariable<C, Felt<C::F>> + CanSampleBitsVariable<C, Bit>
{
    fn sample_ext(&mut self, builder: &mut Builder<C>) -> Ext<C::F, C::EF>;

    fn check_witness(&mut self, builder: &mut Builder<C>, nb_bits: usize, witness: Felt<C::F>);

    fn duplexing(&mut self, builder: &mut Builder<C>);

    fn sample_point(
        &mut self,
        builder: &mut Builder<C>,
        dimension: u32,
    ) -> Point<Ext<C::F, C::EF>> {
        (0..dimension).map(|_| self.sample_ext(builder)).collect()
    }

    fn observe_ext_element(&mut self, builder: &mut Builder<C>, element: Ext<C::F, C::EF>)
    where
        C: CircuitConfig,
    {
        let felts = C::ext2felt(builder, element);
        self.observe_slice(builder, felts);
    }
}

pub trait CanSampleBitsVariable<C: Config, V> {
    fn sample_bits(&mut self, builder: &mut Builder<C>, nb_bits: usize) -> Vec<V>;
}

/// Reference: [p3_challenger::DuplexChallenger]
#[derive(Clone, Debug)]
pub struct DuplexChallengerVariable<C: Config> {
    pub sponge_state: [Felt<C::F>; PERMUTATION_WIDTH],
    pub input_buffer: Vec<Felt<C::F>>,
    pub output_buffer: Vec<Felt<C::F>>,
}

impl<C: Config<F = BabyBear>> DuplexChallengerVariable<C> {
    /// Creates a new duplex challenger with the default state.
    pub fn new(builder: &mut Builder<C>) -> Self {
        DuplexChallengerVariable::<C> {
            sponge_state: core::array::from_fn(|_| builder.eval(C::F::zero())),
            input_buffer: vec![],
            output_buffer: vec![],
        }
    }

    /// Creates a new challenger variable with the same state as an existing challenger.
    pub fn from_challenger<P: CryptographicPermutation<[BabyBear; PERMUTATION_WIDTH]>>(
        builder: &mut Builder<C>,
        challenger: &DuplexChallenger<BabyBear, P, PERMUTATION_WIDTH, HASH_RATE>,
    ) -> Self {
        let sponge_state = challenger.sponge_state.map(|x| builder.eval(x));
        let input_buffer = challenger.input_buffer.iter().map(|x| builder.eval(*x)).collect();
        let output_buffer = challenger.output_buffer.iter().map(|x| builder.eval(*x)).collect();
        DuplexChallengerVariable::<C> { sponge_state, input_buffer, output_buffer }
    }

    /// Creates a new challenger with the same state as an existing challenger.
    pub fn copy(&self, builder: &mut Builder<C>) -> Self {
        let DuplexChallengerVariable { sponge_state, input_buffer, output_buffer } = self;
        let sponge_state = sponge_state.map(|x| builder.eval(x));
        let mut copy_vec = |v: &Vec<Felt<C::F>>| v.iter().map(|x| builder.eval(*x)).collect();
        DuplexChallengerVariable::<C> {
            sponge_state,
            input_buffer: copy_vec(input_buffer),
            output_buffer: copy_vec(output_buffer),
        }
    }

    fn observe(&mut self, builder: &mut Builder<C>, value: Felt<C::F>) {
        self.output_buffer.clear();

        self.input_buffer.push(value);

        if self.input_buffer.len() == HASH_RATE {
            self.duplexing(builder);
        }
    }

    fn sample(&mut self, builder: &mut Builder<C>) -> Felt<C::F> {
        if !self.input_buffer.is_empty() || self.output_buffer.is_empty() {
            self.duplexing(builder);
        }

        self.output_buffer.pop().expect("output buffer should be non-empty")
    }

    fn sample_bits(&mut self, builder: &mut Builder<C>, nb_bits: usize) -> Vec<Felt<C::F>> {
        assert!(nb_bits <= NUM_BITS);
        let rand_f = self.sample(builder);
        let mut rand_f_bits = builder.num2bits_v2_f(rand_f, NUM_BITS);
        rand_f_bits.truncate(nb_bits);
        rand_f_bits
    }

    pub fn public_values(&self, builder: &mut Builder<C>) -> ChallengerPublicValues<Felt<C::F>> {
        assert!(self.input_buffer.len() <= PERMUTATION_WIDTH);
        assert!(self.output_buffer.len() <= PERMUTATION_WIDTH);

        let sponge_state = self.sponge_state;
        let num_inputs = builder.eval(C::F::from_canonical_usize(self.input_buffer.len()));
        let num_outputs = builder.eval(C::F::from_canonical_usize(self.output_buffer.len()));

        let input_buffer: [_; PERMUTATION_WIDTH] = self
            .input_buffer
            .iter()
            .copied()
            .chain((self.input_buffer.len()..PERMUTATION_WIDTH).map(|_| builder.eval(C::F::zero())))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let output_buffer: [_; PERMUTATION_WIDTH] = self
            .output_buffer
            .iter()
            .copied()
            .chain(
                (self.output_buffer.len()..PERMUTATION_WIDTH).map(|_| builder.eval(C::F::zero())),
            )
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        ChallengerPublicValues {
            sponge_state,
            num_inputs,
            input_buffer,
            num_outputs,
            output_buffer,
        }
    }
}

impl<C: Config<F = BabyBear>> CanCopyChallenger<C> for DuplexChallengerVariable<C> {
    fn copy(&self, builder: &mut Builder<C>) -> Self {
        DuplexChallengerVariable::copy(self, builder)
    }
}

impl<C: Config<F = BabyBear>> CanObserveVariable<C, Felt<C::F>> for DuplexChallengerVariable<C> {
    fn observe(&mut self, builder: &mut Builder<C>, value: Felt<C::F>) {
        DuplexChallengerVariable::observe(self, builder, value);
    }

    fn observe_slice(
        &mut self,
        builder: &mut Builder<C>,
        values: impl IntoIterator<Item = Felt<C::F>>,
    ) {
        for value in values {
            self.observe(builder, value);
        }
    }
}

impl<C: Config<F = BabyBear>, const N: usize> CanObserveVariable<C, [Felt<C::F>; N]>
    for DuplexChallengerVariable<C>
{
    fn observe(&mut self, builder: &mut Builder<C>, values: [Felt<C::F>; N]) {
        for value in values {
            self.observe(builder, value);
        }
    }
}

impl<C: Config<F = BabyBear>> CanSampleVariable<C, Felt<C::F>> for DuplexChallengerVariable<C> {
    fn sample(&mut self, builder: &mut Builder<C>) -> Felt<C::F> {
        DuplexChallengerVariable::sample(self, builder)
    }
}

impl<C: Config<F = BabyBear>> CanSampleBitsVariable<C, Felt<C::F>> for DuplexChallengerVariable<C> {
    fn sample_bits(&mut self, builder: &mut Builder<C>, nb_bits: usize) -> Vec<Felt<C::F>> {
        DuplexChallengerVariable::sample_bits(self, builder, nb_bits)
    }
}

impl<C: Config<F = BabyBear>> FieldChallengerVariable<C, Felt<C::F>>
    for DuplexChallengerVariable<C>
{
    fn sample_ext(&mut self, builder: &mut Builder<C>) -> Ext<C::F, C::EF> {
        let a = self.sample(builder);
        let b = self.sample(builder);
        let c = self.sample(builder);
        let d = self.sample(builder);
        builder.ext_from_base_slice(&[a, b, c, d])
    }

    fn check_witness(
        &mut self,
        builder: &mut Builder<C>,
        nb_bits: usize,
        witness: Felt<<C as Config>::F>,
    ) {
        self.observe(builder, witness);
        let element_bits = self.sample_bits(builder, nb_bits);
        for bit in element_bits {
            builder.assert_felt_eq(bit, C::F::zero());
        }
    }

    fn duplexing(&mut self, builder: &mut Builder<C>) {
        assert!(self.input_buffer.len() <= HASH_RATE);

        self.sponge_state[0..self.input_buffer.len()].copy_from_slice(self.input_buffer.as_slice());
        self.input_buffer.clear();

        self.sponge_state = builder.poseidon2_permute_v2(self.sponge_state);

        self.output_buffer.clear();
        self.output_buffer.extend_from_slice(&self.sponge_state);
    }
}

pub const CHALLENGER_STATE_NUM_ELTS: usize = size_of::<ChallengerPublicValues<u8>>();

#[derive(AlignedBorrow, Serialize, Deserialize, Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct ChallengerPublicValues<T> {
    pub sponge_state: [T; PERMUTATION_WIDTH],
    pub num_inputs: T,
    pub input_buffer: [T; PERMUTATION_WIDTH],
    pub num_outputs: T,
    pub output_buffer: [T; PERMUTATION_WIDTH],
}

impl<T: Clone> ChallengerPublicValues<T> {
    pub fn set_challenger<P: CryptographicPermutation<[T; PERMUTATION_WIDTH]>>(
        &self,
        challenger: &mut DuplexChallenger<T, P, PERMUTATION_WIDTH, HASH_RATE>,
    ) where
        T: PrimeField32,
    {
        challenger.sponge_state = self.sponge_state;
        let num_inputs = self.num_inputs.as_canonical_u32() as usize;
        challenger.input_buffer = self.input_buffer[..num_inputs].to_vec();
        let num_outputs = self.num_outputs.as_canonical_u32() as usize;
        challenger.output_buffer = self.output_buffer[..num_outputs].to_vec();
    }

    pub fn as_array(&self) -> [T; CHALLENGER_STATE_NUM_ELTS]
    where
        T: Copy,
    {
        unsafe {
            let mut ret = [MaybeUninit::<T>::zeroed().assume_init(); CHALLENGER_STATE_NUM_ELTS];
            let pv: &mut ChallengerPublicValues<T> = ret.as_mut_slice().borrow_mut();
            *pv = *self;
            ret
        }
    }
}

impl<T: Copy> IntoIterator for ChallengerPublicValues<T> {
    type Item = T;
    type IntoIter = std::array::IntoIter<T, CHALLENGER_STATE_NUM_ELTS>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_array().into_iter()
    }
}

// #[cfg(test)]
// pub(crate) mod tests {
//     #![allow(clippy::print_stdout)]

//     use std::iter::zip;

//     use crate::{
//         challenger::{CanCopyChallenger, MultiField32ChallengerVariable},
//         hash::{FieldHasherVariable, BN254_DIGEST_SIZE},
//         utils::tests::run_test_recursion,
//     };
//     use p3_baby_bear::BabyBear;
//     use p3_bn254_fr::Bn254Fr;
//     use p3_challenger::{CanObserve, CanSample, CanSampleBits, FieldChallenger};
//     use p3_field::AbstractField;
//     use p3_symmetric::{CryptographicHasher, Hash, PseudoCompressionFunction};
//     use sp1_recursion_compiler::{
//         circuit::{AsmBuilder, AsmConfig},
//         config::OuterConfig,
//         constraints::ConstraintCompiler,
//         ir::{Builder, Config, Ext, ExtConst, Felt, Var},
//     };
//     use sp1_recursion_core::stark::{outer_perm, BabyBearPoseidon2Outer, OuterCompress,
// OuterHash};     use sp1_recursion_gnark_ffi::PlonkBn254Prover;
//     use sp1_stark::{baby_bear_poseidon2::BabyBearPoseidon2, StarkGenericConfig};

//     use crate::{
//         challenger::{DuplexChallengerVariable, FieldChallengerVariable},
//         witness::OuterWitness,
//     };

//     type SC = BabyBearPoseidon2;
//     type C = OuterConfig;
//     type F = <SC as StarkGenericConfig>::Val;
//     type EF = <SC as StarkGenericConfig>::Challenge;

//     #[test]
//     fn test_compiler_challenger() {
//         let config = SC::default();
//         let mut challenger = config.challenger();
//         challenger.observe(F::one());
//         challenger.observe(F::two());
//         challenger.observe(F::two());
//         challenger.observe(F::two());
//         let result: F = challenger.sample();
//         println!("expected result: {}", result);
//         let result_ef: EF = challenger.sample_ext_element();
//         println!("expected result_ef: {}", result_ef);

//         let mut builder = AsmBuilder::<F, EF>::default();

//         let mut challenger = DuplexChallengerVariable::<AsmConfig<F, EF>> {
//             sponge_state: core::array::from_fn(|_| builder.eval(F::zero())),
//             input_buffer: vec![],
//             output_buffer: vec![],
//         };
//         let one: Felt<_> = builder.eval(F::one());
//         let two: Felt<_> = builder.eval(F::two());

//         challenger.observe(&mut builder, one);
//         challenger.observe(&mut builder, two);
//         challenger.observe(&mut builder, two);
//         challenger.observe(&mut builder, two);
//         let element = challenger.sample(&mut builder);
//         let element_ef = challenger.sample_ext(&mut builder);

//         let expected_result: Felt<_> = builder.eval(result);
//         let expected_result_ef: Ext<_, _> = builder.eval(result_ef.cons());
//         builder.print_f(element);
//         builder.assert_felt_eq(expected_result, element);
//         builder.print_e(element_ef);
//         builder.assert_ext_eq(expected_result_ef, element_ef);

//         run_test_recursion(builder.into_root_block(), None);
//     }

//     #[test]
//     fn test_challenger_outer() {
//         type SC = BabyBearPoseidon2Outer;
//         type F = <SC as StarkGenericConfig>::Val;
//         type EF = <SC as StarkGenericConfig>::Challenge;
//         type N = <C as Config>::N;

//         let config = SC::default();
//         let mut challenger = config.challenger();
//         challenger.observe(F::one());
//         challenger.observe(F::two());
//         challenger.observe(F::two());
//         challenger.observe(F::two());
//         let commit = Hash::from([N::two()]);
//         challenger.observe(commit);
//         let result: F = challenger.sample();
//         println!("expected result: {}", result);
//         let result_ef: EF = challenger.sample_ext_element();
//         println!("expected result_ef: {}", result_ef);
//         let mut bits = challenger.sample_bits(30);
//         let mut bits_vec = vec![];
//         for _ in 0..30 {
//             bits_vec.push(bits % 2);
//             bits >>= 1;
//         }
//         println!("expected bits: {:?}", bits_vec);

//         let mut builder = Builder::<C>::default();

//         // let width: Var<_> = builder.eval(F::from_canonical_usize(PERMUTATION_WIDTH));
//         let mut challenger = MultiField32ChallengerVariable::<C>::new(&mut builder);
//         let one: Felt<_> = builder.eval(F::one());
//         let two: Felt<_> = builder.eval(F::two());
//         let two_var: Var<_> = builder.eval(N::two());
//         // builder.halt();
//         challenger.observe(&mut builder, one);
//         challenger.observe(&mut builder, two);
//         challenger.observe(&mut builder, two);
//         challenger.observe(&mut builder, two);
//         challenger.observe_commitment(&mut builder, [two_var]);

//         // Check to make sure the copying works.
//         challenger = challenger.copy(&mut builder);
//         let element = challenger.sample(&mut builder);
//         let element_ef = challenger.sample_ext(&mut builder);
//         let bits = challenger.sample_bits(&mut builder, 31);

//         let expected_result: Felt<_> = builder.eval(result);
//         let expected_result_ef: Ext<_, _> = builder.eval(result_ef.cons());
//         builder.print_f(element);
//         builder.assert_felt_eq(expected_result, element);
//         builder.print_e(element_ef);
//         builder.assert_ext_eq(expected_result_ef, element_ef);
//         for (expected_bit, bit) in zip(bits_vec.iter(), bits.iter()) {
//             let expected_bit: Var<_> = builder.eval(N::from_canonical_usize(*expected_bit));
//             builder.print_v(*bit);
//             builder.assert_var_eq(expected_bit, *bit);
//         }

//         let mut backend = ConstraintCompiler::<C>::default();
//         let constraints = backend.emit(builder.into_operations());
//         let witness = OuterWitness::default();
//         PlonkBn254Prover::test::<C>(constraints, witness);
//     }

//     #[test]
//     fn test_select_chain_digest() {
//         type N = <C as Config>::N;

//         let mut builder = Builder::<C>::default();

//         let one: Var<_> = builder.eval(N::one());
//         let two: Var<_> = builder.eval(N::two());

//         let to_swap = [[one], [two]];
//         let result = BabyBearPoseidon2Outer::select_chain_digest(&mut builder, one, to_swap);

//         builder.assert_var_eq(result[0][0], two);
//         builder.assert_var_eq(result[1][0], one);

//         let mut backend = ConstraintCompiler::<C>::default();
//         let constraints = backend.emit(builder.into_operations());
//         let witness = OuterWitness::default();
//         PlonkBn254Prover::test::<C>(constraints, witness);
//     }

//     #[test]
//     fn test_p2_hash() {
//         let perm = outer_perm();
//         let hasher = OuterHash::new(perm.clone()).unwrap();

//         let input: [BabyBear; 7] = [
//             BabyBear::from_canonical_u32(0),
//             BabyBear::from_canonical_u32(1),
//             BabyBear::from_canonical_u32(2),
//             BabyBear::from_canonical_u32(2),
//             BabyBear::from_canonical_u32(2),
//             BabyBear::from_canonical_u32(2),
//             BabyBear::from_canonical_u32(2),
//         ];
//         let output = hasher.hash_iter(input);

//         let mut builder = Builder::<C>::default();
//         let a: Felt<_> = builder.eval(input[0]);
//         let b: Felt<_> = builder.eval(input[1]);
//         let c: Felt<_> = builder.eval(input[2]);
//         let d: Felt<_> = builder.eval(input[3]);
//         let e: Felt<_> = builder.eval(input[4]);
//         let f: Felt<_> = builder.eval(input[5]);
//         let g: Felt<_> = builder.eval(input[6]);
//         let result = BabyBearPoseidon2Outer::hash(&mut builder, &[a, b, c, d, e, f, g]);

//         builder.assert_var_eq(result[0], output[0]);

//         let mut backend = ConstraintCompiler::<C>::default();
//         let constraints = backend.emit(builder.into_operations());
//         PlonkBn254Prover::test::<C>(constraints.clone(), OuterWitness::default());
//     }

//     #[test]
//     fn test_p2_compress() {
//         type OuterDigestVariable = [Var<<C as Config>::N>; BN254_DIGEST_SIZE];
//         let perm = outer_perm();
//         let compressor = OuterCompress::new(perm.clone());

//         let a: [Bn254Fr; 1] = [Bn254Fr::two()];
//         let b: [Bn254Fr; 1] = [Bn254Fr::two()];
//         let gt = compressor.compress([a, b]);

//         let mut builder = Builder::<C>::default();
//         let a: OuterDigestVariable = [builder.eval(a[0])];
//         let b: OuterDigestVariable = [builder.eval(b[0])];
//         let result = BabyBearPoseidon2Outer::compress(&mut builder, [a, b]);
//         builder.assert_var_eq(result[0], gt[0]);

//         let mut backend = ConstraintCompiler::<C>::default();
//         let constraints = backend.emit(builder.into_operations());
//         PlonkBn254Prover::test::<C>(constraints.clone(), OuterWitness::default());
//     }
// }
