use std::{borrow::BorrowMut, mem::MaybeUninit};

use serde::{Deserialize, Serialize};
use slop_algebra::{AbstractField, Field, PrimeField32};
use slop_baby_bear::BabyBear;
use slop_challenger::DuplexChallenger;
use slop_merkle_tree::{OUTER_CHALLENGER_RATE, OUTER_DIGEST_SIZE};
use slop_multilinear::Point;
use slop_symmetric::CryptographicPermutation;
use sp1_derive::AlignedBorrow;
use sp1_recursion_compiler::{
    circuit::CircuitV2Builder,
    ir::{DslIr, Var},
    prelude::{Builder, Config, Ext, Felt},
};
use sp1_recursion_executor::{HASH_RATE, NUM_BITS, PERMUTATION_WIDTH};

use crate::CircuitConfig;

// Constants for the Multifield challenger.
pub const POSEIDON_2_BB_RATE: usize = 16;

// use crate::{DigestVariable, VerifyingKeyVariable};

pub trait CanCopyChallenger<C: CircuitConfig> {
    fn copy(&self, builder: &mut Builder<C>) -> Self;
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpongeChallengerShape {
    pub input_buffer_len: usize,
    pub output_buffer_len: usize,
}

/// Reference: [p3_challenger::CanObserve].
pub trait CanObserveVariable<C: CircuitConfig, V> {
    fn observe(&mut self, builder: &mut Builder<C>, value: V);

    fn observe_slice(&mut self, builder: &mut Builder<C>, values: impl IntoIterator<Item = V>) {
        for value in values {
            self.observe(builder, value);
        }
    }
}

pub trait CanSampleVariable<C: CircuitConfig, V> {
    fn sample(&mut self, builder: &mut Builder<C>) -> V;
}

/// Reference: [p3_challenger::FieldChallenger].
pub trait FieldChallengerVariable<C: CircuitConfig, Bit>:
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

pub trait CanSampleBitsVariable<C: CircuitConfig, V> {
    fn sample_bits(&mut self, builder: &mut Builder<C>, nb_bits: usize) -> Vec<V>;
}

/// Reference: [p3_challenger::DuplexChallenger]
#[derive(Clone, Debug)]
pub struct DuplexChallengerVariable<C: CircuitConfig> {
    pub sponge_state: [Felt<C::F>; PERMUTATION_WIDTH],
    pub input_buffer: Vec<Felt<C::F>>,
    pub output_buffer: Vec<Felt<C::F>>,
}

impl<C: CircuitConfig<F = BabyBear>> DuplexChallengerVariable<C> {
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

impl<C: CircuitConfig<F = BabyBear>> CanCopyChallenger<C> for DuplexChallengerVariable<C> {
    fn copy(&self, builder: &mut Builder<C>) -> Self {
        DuplexChallengerVariable::copy(self, builder)
    }
}

impl<C: CircuitConfig<F = BabyBear>> CanObserveVariable<C, Felt<C::F>>
    for DuplexChallengerVariable<C>
{
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

impl<C: CircuitConfig<F = BabyBear>, const N: usize> CanObserveVariable<C, [Felt<C::F>; N]>
    for DuplexChallengerVariable<C>
{
    fn observe(&mut self, builder: &mut Builder<C>, values: [Felt<C::F>; N]) {
        for value in values {
            self.observe(builder, value);
        }
    }
}

impl<C: CircuitConfig<F = BabyBear>> CanSampleVariable<C, Felt<C::F>>
    for DuplexChallengerVariable<C>
{
    fn sample(&mut self, builder: &mut Builder<C>) -> Felt<C::F> {
        DuplexChallengerVariable::sample(self, builder)
    }
}

impl<C: CircuitConfig<F = BabyBear>> CanSampleBitsVariable<C, Felt<C::F>>
    for DuplexChallengerVariable<C>
{
    fn sample_bits(&mut self, builder: &mut Builder<C>, nb_bits: usize) -> Vec<Felt<C::F>> {
        DuplexChallengerVariable::sample_bits(self, builder, nb_bits)
    }
}

impl<C: CircuitConfig<F = BabyBear>> FieldChallengerVariable<C, Felt<C::F>>
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

        self.sponge_state = C::poseidon2_permute_v2(builder, self.sponge_state);

        self.output_buffer.clear();
        self.output_buffer.extend_from_slice(&self.sponge_state);
    }
}

#[derive(Clone)]
pub struct MultiField32ChallengerVariable<C: CircuitConfig> {
    sponge_state: [Var<C::N>; 3],
    input_buffer: Vec<Felt<C::F>>,
    output_buffer: Vec<Felt<C::F>>,
    output_var_buffer: Vec<Var<C::N>>,
    num_duplex_elms: usize,
    num_f_elms: usize,
}

impl<C: CircuitConfig> MultiField32ChallengerVariable<C> {
    pub fn new(builder: &mut Builder<C>) -> Self {
        MultiField32ChallengerVariable::<C> {
            sponge_state: [
                builder.eval(C::N::zero()),
                builder.eval(C::N::zero()),
                builder.eval(C::N::zero()),
            ],
            input_buffer: vec![],
            output_buffer: vec![],
            output_var_buffer: vec![],
            num_duplex_elms: C::N::bits() / C::F::bits(),
            num_f_elms: C::N::bits() / C::F::bits() / 2,
        }
    }

    pub fn duplexing(&mut self, builder: &mut Builder<C>) {
        assert!(self.input_buffer.len() <= self.num_duplex_elms * OUTER_CHALLENGER_RATE);

        for (i, f_chunk) in self.input_buffer.chunks(self.num_duplex_elms).enumerate() {
            self.sponge_state[i] = reduce_31(builder, f_chunk);
        }
        self.input_buffer.clear();

        // TODO make this a method for the builder.
        builder.push_op(DslIr::CircuitPoseidon2Permute(self.sponge_state));

        self.output_buffer.clear();
        self.output_var_buffer.clear();
        for &pf_val in self.sponge_state.iter() {
            self.output_var_buffer.push(pf_val);
        }
    }

    pub fn split_var(&mut self, builder: &mut Builder<C>) {
        assert!(self.output_buffer.is_empty());
        assert!(!self.output_var_buffer.is_empty());
        let pf_val = self.output_var_buffer.pop().expect("output var buffer shouldn't be empty");
        let f_vals = split_32(builder, pf_val, self.num_f_elms);
        for f_val in f_vals {
            self.output_buffer.push(f_val);
        }
    }

    pub fn observe(&mut self, builder: &mut Builder<C>, value: Felt<C::F>) {
        self.output_buffer.clear();
        self.output_var_buffer.clear();

        self.input_buffer.push(value);
        if self.input_buffer.len() == self.num_duplex_elms * OUTER_CHALLENGER_RATE {
            self.duplexing(builder);
        }
    }

    pub fn observe_commitment(
        &mut self,
        builder: &mut Builder<C>,
        value: [Var<C::N>; OUTER_DIGEST_SIZE],
    ) {
        for val in value {
            let f_vals: Vec<Felt<C::F>> = split_32(builder, val, self.num_f_elms);
            for f_val in f_vals {
                self.observe(builder, f_val);
            }
        }
    }

    pub fn sample(&mut self, builder: &mut Builder<C>) -> Felt<C::F> {
        if !self.input_buffer.is_empty()
            || (self.output_buffer.is_empty() && self.output_var_buffer.is_empty())
        {
            self.duplexing(builder);
        }

        if self.output_buffer.is_empty() {
            self.split_var(builder);
        }
        self.output_buffer.pop().expect("output buffer should be non-empty")
    }

    pub fn sample_ext(&mut self, builder: &mut Builder<C>) -> Ext<C::F, C::EF> {
        let a = self.sample(builder);
        let b = self.sample(builder);
        let c = self.sample(builder);
        let d = self.sample(builder);
        builder.felts2ext(&[a, b, c, d])
    }

    pub fn sample_bits(&mut self, builder: &mut Builder<C>, bits: usize) -> Vec<Var<C::N>> {
        let rand_f = self.sample(builder);
        builder.num2bits_f_circuit(rand_f)[0..bits].to_vec()
    }

    pub fn check_witness(&mut self, builder: &mut Builder<C>, bits: usize, witness: Felt<C::F>) {
        self.observe(builder, witness);
        let element = self.sample_bits(builder, bits);
        for bit in element {
            builder.assert_var_eq(bit, C::N::from_canonical_usize(0));
        }
    }
}

impl<C: CircuitConfig> CanCopyChallenger<C> for MultiField32ChallengerVariable<C> {
    /// Creates a new challenger with the same state as an existing challenger.
    fn copy(&self, builder: &mut Builder<C>) -> Self {
        let MultiField32ChallengerVariable {
            sponge_state,
            input_buffer,
            output_buffer,
            output_var_buffer,
            num_duplex_elms,
            num_f_elms,
        } = self;
        let sponge_state = sponge_state.map(|x| builder.eval(x));
        let mut copy_vec = |v: &Vec<Felt<C::F>>| v.iter().map(|x| builder.eval(*x)).collect();
        MultiField32ChallengerVariable::<C> {
            sponge_state,
            num_duplex_elms: *num_duplex_elms,
            num_f_elms: *num_f_elms,
            input_buffer: copy_vec(input_buffer),
            output_buffer: copy_vec(output_buffer),
            output_var_buffer: output_var_buffer.iter().map(|x| builder.eval(*x)).collect(),
        }
    }
}

impl<C: CircuitConfig> CanObserveVariable<C, Felt<C::F>> for MultiField32ChallengerVariable<C> {
    fn observe(&mut self, builder: &mut Builder<C>, value: Felt<C::F>) {
        MultiField32ChallengerVariable::observe(self, builder, value);
    }
}

impl<C: CircuitConfig> CanObserveVariable<C, [Var<C::N>; OUTER_DIGEST_SIZE]>
    for MultiField32ChallengerVariable<C>
{
    fn observe(&mut self, builder: &mut Builder<C>, value: [Var<C::N>; OUTER_DIGEST_SIZE]) {
        self.observe_commitment(builder, value)
    }
}

impl<C: CircuitConfig> CanObserveVariable<C, Var<C::N>> for MultiField32ChallengerVariable<C> {
    fn observe(&mut self, builder: &mut Builder<C>, value: Var<C::N>) {
        self.observe_commitment(builder, [value])
    }
}

impl<C: CircuitConfig> CanSampleVariable<C, Felt<C::F>> for MultiField32ChallengerVariable<C> {
    fn sample(&mut self, builder: &mut Builder<C>) -> Felt<C::F> {
        MultiField32ChallengerVariable::sample(self, builder)
    }
}

impl<C: CircuitConfig> CanSampleBitsVariable<C, Var<C::N>> for MultiField32ChallengerVariable<C> {
    fn sample_bits(&mut self, builder: &mut Builder<C>, bits: usize) -> Vec<Var<C::N>> {
        MultiField32ChallengerVariable::sample_bits(self, builder, bits)
    }
}

impl<C: CircuitConfig> FieldChallengerVariable<C, Var<C::N>> for MultiField32ChallengerVariable<C> {
    fn sample_ext(&mut self, builder: &mut Builder<C>) -> Ext<C::F, C::EF> {
        MultiField32ChallengerVariable::sample_ext(self, builder)
    }

    fn check_witness(&mut self, builder: &mut Builder<C>, bits: usize, witness: Felt<C::F>) {
        MultiField32ChallengerVariable::check_witness(self, builder, bits, witness);
    }

    fn duplexing(&mut self, builder: &mut Builder<C>) {
        MultiField32ChallengerVariable::duplexing(self, builder);
    }
}

pub fn reduce_31<C: CircuitConfig>(builder: &mut Builder<C>, vals: &[Felt<C::F>]) -> Var<C::N> {
    let mut power = C::N::one();
    let result: Var<C::N> = builder.eval(C::N::zero());
    for val in vals.iter() {
        let val = builder.felt2var_circuit(*val);
        builder.assign(result, result + val * power);
        power *= C::N::from_canonical_u64(1u64 << 31);
    }
    result
}

pub fn split_32<C: CircuitConfig>(
    builder: &mut Builder<C>,
    val: Var<C::N>,
    n: usize,
) -> Vec<Felt<C::F>> {
    let bits = builder.num2bits_v_circuit(val, 256);
    let mut results = Vec::new();
    for i in 0..n {
        let result: Felt<C::F> = builder.eval(C::F::zero());
        for j in 0..64 {
            let bit = bits[i * 64 + j];
            let t = builder.eval(result + C::F::from_wrapped_u64(1 << j));
            let z = builder.select_f(bit, t, result);
            builder.assign(result, z);
        }
        results.push(result);
    }
    results
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

#[cfg(test)]
pub(crate) mod tests {
    #![allow(clippy::print_stdout)]

    use std::iter::zip;

    use crate::{
        challenger::{CanCopyChallenger, MultiField32ChallengerVariable},
        hash::{FieldHasherVariable, BN254_DIGEST_SIZE},
        witness::OuterWitness,
    };
    use slop_algebra::AbstractField;
    use slop_baby_bear::BabyBear;
    use slop_bn254::Bn254Fr;
    use slop_challenger::{
        CanObserve, CanSample, CanSampleBits, DuplexChallenger, FieldChallenger,
        MultiField32Challenger,
    };
    use slop_jagged::JaggedConfig;
    use slop_merkle_tree::{
        my_bb_16_perm, outer_perm, DefaultMerkleTreeConfig, OuterPerm, Perm, Poseidon2Bn254Config,
    };
    use slop_symmetric::{CryptographicHasher, Hash, PseudoCompressionFunction};
    use sp1_recursion_compiler::{
        circuit::{AsmBuilder, AsmCompiler, AsmConfig},
        config::OuterConfig,
        constraints::ConstraintCompiler,
        ir::{Builder, Config, Ext, ExtConst, Felt, Var},
    };
    use sp1_recursion_gnark_ffi::PlonkBn254Prover;
    use sp1_recursion_machine::test::run_recursion_test_machines;
    use sp1_stark::{BabyBearPoseidon2, Bn254JaggedConfig};

    use crate::challenger::{DuplexChallengerVariable, FieldChallengerVariable};

    type SC = BabyBearPoseidon2;
    type C = OuterConfig;
    type F = <SC as JaggedConfig>::F;
    type EF = <SC as JaggedConfig>::EF;

    #[tokio::test]
    #[allow(clippy::uninlined_format_args)]
    async fn test_compiler_challenger() {
        let default_perm = my_bb_16_perm();
        let mut challenger = DuplexChallenger::<BabyBear, Perm, 16, 8>::new(default_perm.clone());
        challenger.observe(F::one());
        challenger.observe(F::two());
        challenger.observe(F::two());
        challenger.observe(F::two());
        let result: F = challenger.sample();
        println!("expected result: {result}");
        let result_ef: EF = challenger.sample_ext_element();
        println!("expected result_ef: {result_ef}");

        let mut builder = AsmBuilder::<F, EF>::default();

        let mut challenger = DuplexChallengerVariable::<AsmConfig<F, EF>> {
            sponge_state: core::array::from_fn(|_| builder.eval(F::zero())),
            input_buffer: vec![],
            output_buffer: vec![],
        };
        let one: Felt<_> = builder.eval(F::one());
        let two: Felt<_> = builder.eval(F::two());

        challenger.observe(&mut builder, one);
        challenger.observe(&mut builder, two);
        challenger.observe(&mut builder, two);
        challenger.observe(&mut builder, two);
        let element = challenger.sample(&mut builder);
        let element_ef = challenger.sample_ext(&mut builder);

        let expected_result: Felt<_> = builder.eval(result);
        let expected_result_ef: Ext<_, _> = builder.eval(result_ef.cons());
        builder.print_f(element);
        builder.assert_felt_eq(expected_result, element);
        builder.print_e(element_ef);
        builder.assert_ext_eq(expected_result_ef, element_ef);

        let block = builder.into_root_block();
        let mut compiler = AsmCompiler::<AsmConfig<F, EF>>::default();
        let program = compiler.compile_inner(block).validate().unwrap();

        let witness_stream = Vec::new();
        run_recursion_test_machines(program.clone(), witness_stream).await;
    }

    #[tokio::test]
    #[allow(clippy::uninlined_format_args)]
    async fn test_challenger_outer() {
        type SC = Bn254JaggedConfig;
        type F = <SC as JaggedConfig>::F;
        type EF = <SC as JaggedConfig>::EF;
        type N = <C as Config>::N;

        let default_perm = outer_perm();
        let mut challenger =
            MultiField32Challenger::<BabyBear, Bn254Fr, OuterPerm, 3, 2>::new(default_perm.clone())
                .unwrap();
        challenger.observe(F::one());
        challenger.observe(F::two());
        challenger.observe(F::two());
        challenger.observe(F::two());
        let commit = Hash::from([N::two()]);
        challenger.observe(commit);
        let result: F = challenger.sample();
        println!("expected result: {result}");
        let result_ef: EF = challenger.sample_ext_element();
        println!("expected result_ef: {result_ef}");
        let mut bits = challenger.sample_bits(30);
        let mut bits_vec = vec![];
        for _ in 0..30 {
            bits_vec.push(bits % 2);
            bits >>= 1;
        }
        println!("expected bits: {bits_vec:?}");

        let mut builder = Builder::<C>::default();

        // let width: Var<_> = builder.eval(F::from_canonical_usize(PERMUTATION_WIDTH));
        let mut challenger = MultiField32ChallengerVariable::<C>::new(&mut builder);
        let one: Felt<_> = builder.eval(F::one());
        let two: Felt<_> = builder.eval(F::two());
        let two_var: Var<_> = builder.eval(N::two());
        // builder.halt();
        challenger.observe(&mut builder, one);
        challenger.observe(&mut builder, two);
        challenger.observe(&mut builder, two);
        challenger.observe(&mut builder, two);
        challenger.observe_commitment(&mut builder, [two_var]);

        // Check to make sure the copying works.
        challenger = challenger.copy(&mut builder);
        let element = challenger.sample(&mut builder);
        let element_ef = challenger.sample_ext(&mut builder);
        let bits = challenger.sample_bits(&mut builder, 31);

        let expected_result: Felt<_> = builder.eval(result);
        let expected_result_ef: Ext<_, _> = builder.eval(result_ef.cons());
        builder.print_f(element);
        builder.assert_felt_eq(expected_result, element);
        builder.print_e(element_ef);
        builder.assert_ext_eq(expected_result_ef, element_ef);
        for (expected_bit, bit) in zip(bits_vec.iter(), bits.iter()) {
            let expected_bit: Var<_> = builder.eval(N::from_canonical_usize(*expected_bit));
            builder.print_v(*bit);
            builder.assert_var_eq(expected_bit, *bit);
        }

        let mut backend = ConstraintCompiler::<C>::default();
        let constraints = backend.emit(builder.into_operations());
        let witness = OuterWitness::default();
        PlonkBn254Prover::test::<C>(constraints, witness);
    }

    #[test]
    fn test_select_chain_digest() {
        type N = <C as Config>::N;

        let mut builder = Builder::<C>::default();

        let one: Var<_> = builder.eval(N::one());
        let two: Var<_> = builder.eval(N::two());

        let to_swap = [[one], [two]];
        let result = Bn254JaggedConfig::select_chain_digest(&mut builder, one, to_swap);

        builder.assert_var_eq(result[0][0], two);
        builder.assert_var_eq(result[1][0], one);

        let mut backend = ConstraintCompiler::<C>::default();
        let constraints = backend.emit(builder.into_operations());
        let witness = OuterWitness::default();
        PlonkBn254Prover::test::<C>(constraints, witness);
    }

    #[test]
    fn test_p2_hash() {
        let (hasher, _) = Poseidon2Bn254Config::default_hasher_and_compressor();

        let input: [BabyBear; 7] = [
            BabyBear::from_canonical_u32(0),
            BabyBear::from_canonical_u32(1),
            BabyBear::from_canonical_u32(2),
            BabyBear::from_canonical_u32(2),
            BabyBear::from_canonical_u32(2),
            BabyBear::from_canonical_u32(2),
            BabyBear::from_canonical_u32(2),
        ];
        let output: [Bn254Fr; 1] = hasher.hash_iter(input);

        let mut builder = Builder::<C>::default();
        let a: Felt<_> = builder.eval(input[0]);
        let b: Felt<_> = builder.eval(input[1]);
        let c: Felt<_> = builder.eval(input[2]);
        let d: Felt<_> = builder.eval(input[3]);
        let e: Felt<_> = builder.eval(input[4]);
        let f: Felt<_> = builder.eval(input[5]);
        let g: Felt<_> = builder.eval(input[6]);
        let result = Bn254JaggedConfig::hash(&mut builder, &[a, b, c, d, e, f, g]);

        builder.assert_var_eq(result[0], output[0]);

        let mut backend = ConstraintCompiler::<C>::default();
        let constraints = backend.emit(builder.into_operations());
        PlonkBn254Prover::test::<C>(constraints.clone(), OuterWitness::default());
    }

    #[test]
    fn test_p2_compress() {
        type OuterDigestVariable = [Var<<C as Config>::N>; BN254_DIGEST_SIZE];
        let (_, compressor) = Poseidon2Bn254Config::default_hasher_and_compressor();

        let a: [Bn254Fr; 1] = [Bn254Fr::two()];
        let b: [Bn254Fr; 1] = [Bn254Fr::two()];
        let gt = compressor.compress([a, b]);

        let mut builder = Builder::<C>::default();
        let a: OuterDigestVariable = [builder.eval(a[0])];
        let b: OuterDigestVariable = [builder.eval(b[0])];
        let result = Bn254JaggedConfig::compress(&mut builder, [a, b]);
        builder.assert_var_eq(result[0], gt[0]);

        let mut backend = ConstraintCompiler::<C>::default();
        let constraints = backend.emit(builder.into_operations());
        PlonkBn254Prover::test::<C>(constraints.clone(), OuterWitness::default());
    }
}
