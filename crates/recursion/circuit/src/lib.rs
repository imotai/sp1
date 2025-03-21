//! Copied from [`sp1_recursion_program`].

use challenger::{
    CanCopyChallenger, CanObserveVariable, DuplexChallengerVariable, FieldChallengerVariable,
};
use hash::{FieldHasherVariable, Posedion2BabyBearHasherVariable};
use itertools::izip;
use slop_algebra::AbstractField;
use sp1_recursion_compiler::{
    circuit::CircuitV2Builder,
    config::{InnerConfig, OuterConfig},
    ir::{Builder, Config, DslIr, Ext, Felt, SymbolicFelt, Var, Variable},
};
use std::iter::{repeat, zip};

use slop_basefold::{
    BasefoldConfig, BasefoldProof, BasefoldVerifier, Poseidon2BabyBear16BasefoldConfig,
};
use slop_commit::TensorCs;
use slop_merkle_tree::{MerkleTreeConfig, MerkleTreeTcs, Poseidon2BabyBearConfig};
use sp1_stark::BabyBearPoseidon2;
pub mod basefold;
pub mod challenger;
pub mod hash;
pub mod jagged;
pub mod primitives;
pub mod stark;
pub mod witness;
pub mod zerocheck;
pub const D: usize = 4;
use p3_challenger::{CanObserve, CanSample, FieldChallenger, GrindingChallenger};
use slop_baby_bear::BabyBear;
use slop_jagged::JaggedConfig;

type EF = <BabyBearPoseidon2 as JaggedConfig>::EF;

pub type Digest<C, SC> = <SC as FieldHasherVariable<C>>::DigestVariable;

pub trait AsRecursive<C: CircuitConfig> {
    type Recursive;
}

pub trait BabyBearFriConfig:
    JaggedConfig<
    F = BabyBear,
    EF = EF,
    Commitment = <MerkleTreeTcs<Self::MerkleTreeConfig> as TensorCs>::Commitment,
    BatchPcsProof = BasefoldProof<Self::BasefoldConfig>,
    Challenger = Self::FriChallenger,
    BatchPcsVerifier = BasefoldVerifier<Self::BasefoldConfig>,
>
{
    type BasefoldConfig: BasefoldConfig<
        F = BabyBear,
        EF = EF,
        Tcs = MerkleTreeTcs<Self::MerkleTreeConfig>,
        Commitment = <MerkleTreeTcs<Self::MerkleTreeConfig> as TensorCs>::Commitment,
        Challenger = Self::FriChallenger,
    >;
    type MerkleTreeConfig: MerkleTreeConfig<Data = BabyBear>;
    type FriChallenger: CanObserve<<Self::BasefoldConfig as BasefoldConfig>::Commitment>
        + CanSample<EF>
        + GrindingChallenger<Witness = BabyBear>
        + FieldChallenger<BabyBear>;
}

pub trait BabyBearFriConfigVariable<C: CircuitConfig<F = BabyBear>>:
    BabyBearFriConfig + FieldHasherVariable<C> + Posedion2BabyBearHasherVariable<C>
{
    type FriChallengerVariable: FieldChallengerVariable<C, <C as CircuitConfig>::Bit>
        + CanObserveVariable<C, <Self as FieldHasherVariable<C>>::DigestVariable>
        + CanCopyChallenger<C>;

    /// Get a new challenger corresponding to the given config.
    fn challenger_variable(&self, builder: &mut Builder<C>) -> Self::FriChallengerVariable;
}

pub trait CircuitConfig: Config {
    type Bit: Copy + Variable<Self>;

    fn read_bit(builder: &mut Builder<Self>) -> Self::Bit;

    fn read_felt(builder: &mut Builder<Self>) -> Felt<Self::F>;

    fn read_ext(builder: &mut Builder<Self>) -> Ext<Self::F, Self::EF>;

    fn assert_bit_zero(builder: &mut Builder<Self>, bit: Self::Bit);

    fn assert_bit_one(builder: &mut Builder<Self>, bit: Self::Bit);

    fn ext2felt(
        builder: &mut Builder<Self>,
        ext: Ext<<Self as Config>::F, <Self as Config>::EF>,
    ) -> [Felt<<Self as Config>::F>; D];

    fn exp_reverse_bits(
        builder: &mut Builder<Self>,
        input: Felt<<Self as Config>::F>,
        power_bits: Vec<Self::Bit>,
    ) -> Felt<<Self as Config>::F>;

    /// Exponentiates a felt x to a list of bits in little endian. Uses precomputed powers
    /// of x.
    fn exp_f_bits_precomputed(
        builder: &mut Builder<Self>,
        power_bits: &[Self::Bit],
        two_adic_powers_of_x: &[Felt<Self::F>],
    ) -> Felt<Self::F>;

    fn batch_fri(
        builder: &mut Builder<Self>,
        alpha_pows: Vec<Ext<Self::F, Self::EF>>,
        p_at_zs: Vec<Ext<Self::F, Self::EF>>,
        p_at_xs: Vec<Felt<Self::F>>,
    ) -> Ext<Self::F, Self::EF>;

    fn num2bits(
        builder: &mut Builder<Self>,
        num: Felt<<Self as Config>::F>,
        num_bits: usize,
    ) -> Vec<Self::Bit>;

    fn bits2num(
        builder: &mut Builder<Self>,
        bits: impl IntoIterator<Item = Self::Bit>,
    ) -> Felt<<Self as Config>::F>;

    #[allow(clippy::type_complexity)]
    fn select_chain_f(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
        second: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
    ) -> Vec<Felt<<Self as Config>::F>>;

    #[allow(clippy::type_complexity)]
    fn select_chain_ef(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
        second: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
    ) -> Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>;

    fn range_check_felt(builder: &mut Builder<Self>, value: Felt<Self::F>, num_bits: usize) {
        let bits = Self::num2bits(builder, value, 31);
        for bit in bits.into_iter().skip(num_bits) {
            Self::assert_bit_zero(builder, bit);
        }
    }
}

impl CircuitConfig for InnerConfig {
    type Bit = Felt<<Self as Config>::F>;

    fn assert_bit_zero(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_felt_eq(bit, Self::F::zero());
    }

    fn assert_bit_one(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_felt_eq(bit, Self::F::one());
    }

    fn read_bit(builder: &mut Builder<Self>) -> Self::Bit {
        builder.hint_felt_v2()
    }

    fn read_felt(builder: &mut Builder<Self>) -> Felt<Self::F> {
        builder.hint_felt_v2()
    }

    fn read_ext(builder: &mut Builder<Self>) -> Ext<Self::F, Self::EF> {
        builder.hint_ext_v2()
    }

    fn ext2felt(
        builder: &mut Builder<Self>,
        ext: Ext<<Self as Config>::F, <Self as Config>::EF>,
    ) -> [Felt<<Self as Config>::F>; D] {
        builder.ext2felt_v2(ext)
    }

    fn exp_reverse_bits(
        builder: &mut Builder<Self>,
        input: Felt<<Self as Config>::F>,
        power_bits: Vec<Felt<<Self as Config>::F>>,
    ) -> Felt<<Self as Config>::F> {
        builder.exp_reverse_bits_v2(input, power_bits)
    }

    fn batch_fri(
        builder: &mut Builder<Self>,
        alpha_pows: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
        p_at_zs: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
        p_at_xs: Vec<Felt<<Self as Config>::F>>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        builder.batch_fri_v2(alpha_pows, p_at_zs, p_at_xs)
    }

    fn num2bits(
        builder: &mut Builder<Self>,
        num: Felt<<Self as Config>::F>,
        num_bits: usize,
    ) -> Vec<Felt<<Self as Config>::F>> {
        builder.num2bits_v2_f(num, num_bits)
    }

    fn bits2num(
        builder: &mut Builder<Self>,
        bits: impl IntoIterator<Item = Felt<<Self as Config>::F>>,
    ) -> Felt<<Self as Config>::F> {
        builder.bits2num_v2_f(bits)
    }

    fn select_chain_f(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
        second: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
    ) -> Vec<Felt<<Self as Config>::F>> {
        let one: Felt<_> = builder.constant(Self::F::one());
        let shouldnt_swap: Felt<_> = builder.eval(one - should_swap);

        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(zip(id_branch, swap_branch), zip(repeat(shouldnt_swap), repeat(should_swap)))
            .map(|((id_v, sw_v), (id_c, sw_c))| builder.eval(id_v * id_c + sw_v * sw_c))
            .collect()
    }

    fn select_chain_ef(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
        second: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
    ) -> Vec<Ext<<Self as Config>::F, <Self as Config>::EF>> {
        let one: Felt<_> = builder.constant(Self::F::one());
        let shouldnt_swap: Felt<_> = builder.eval(one - should_swap);

        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(zip(id_branch, swap_branch), zip(repeat(shouldnt_swap), repeat(should_swap)))
            .map(|((id_v, sw_v), (id_c, sw_c))| builder.eval(id_v * id_c + sw_v * sw_c))
            .collect()
    }

    fn exp_f_bits_precomputed(
        builder: &mut Builder<Self>,
        power_bits: &[Self::Bit],
        two_adic_powers_of_x: &[Felt<Self::F>],
    ) -> Felt<Self::F> {
        Self::exp_reverse_bits(
            builder,
            two_adic_powers_of_x[0],
            power_bits.iter().rev().copied().collect(),
        )
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WrapConfig;

impl Config for WrapConfig {
    type F = <InnerConfig as Config>::F;
    type EF = <InnerConfig as Config>::EF;
    type N = <InnerConfig as Config>::N;
}

impl CircuitConfig for WrapConfig {
    type Bit = <InnerConfig as CircuitConfig>::Bit;

    fn assert_bit_zero(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_felt_eq(bit, Self::F::zero());
    }

    fn assert_bit_one(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_felt_eq(bit, Self::F::one());
    }

    fn read_bit(builder: &mut Builder<Self>) -> Self::Bit {
        builder.hint_felt_v2()
    }

    fn read_felt(builder: &mut Builder<Self>) -> Felt<Self::F> {
        builder.hint_felt_v2()
    }

    fn read_ext(builder: &mut Builder<Self>) -> Ext<Self::F, Self::EF> {
        builder.hint_ext_v2()
    }

    fn ext2felt(
        builder: &mut Builder<Self>,
        ext: Ext<<Self as Config>::F, <Self as Config>::EF>,
    ) -> [Felt<<Self as Config>::F>; D] {
        builder.ext2felt_v2(ext)
    }

    fn exp_reverse_bits(
        builder: &mut Builder<Self>,
        input: Felt<<Self as Config>::F>,
        power_bits: Vec<Felt<<Self as Config>::F>>,
    ) -> Felt<<Self as Config>::F> {
        // builder.exp_reverse_bits_v2(input, power_bits)
        let mut result = builder.constant(Self::F::one());
        let mut power_f = input;
        let bit_len = power_bits.len();

        for i in 1..=bit_len {
            let index = bit_len - i;
            let bit = power_bits[index];
            let prod: Felt<_> = builder.eval(result * power_f);
            result = builder.eval(bit * prod + (SymbolicFelt::one() - bit) * result);
            power_f = builder.eval(power_f * power_f);
        }
        result
    }

    fn batch_fri(
        builder: &mut Builder<Self>,
        alpha_pows: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
        p_at_zs: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
        p_at_xs: Vec<Felt<<Self as Config>::F>>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        // builder.batch_fri_v2(alpha_pows, p_at_zs, p_at_xs)
        // Initialize the `acc` to zero.
        let mut acc: Ext<_, _> = builder.uninit();
        builder.push_op(DslIr::ImmE(acc, <Self as Config>::EF::zero()));
        for (alpha_pow, p_at_z, p_at_x) in izip!(alpha_pows, p_at_zs, p_at_xs) {
            // Set `temp_1 = p_at_z - p_at_x`
            let temp_1: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::SubEF(temp_1, p_at_z, p_at_x));
            // Set `temp_2 = alpha_pow * temp_1 = alpha_pow * (p_at_z - p_at_x)`
            let temp_2: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::MulE(temp_2, alpha_pow, temp_1));
            // Set `acc += temp_2`, so that `acc` becomes the sum of `alpha_pow * (p_at_z - p_at_x)`
            let temp_3: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::AddE(temp_3, acc, temp_2));
            acc = temp_3;
        }
        acc
    }

    fn num2bits(
        builder: &mut Builder<Self>,
        num: Felt<<Self as Config>::F>,
        num_bits: usize,
    ) -> Vec<Felt<<Self as Config>::F>> {
        builder.num2bits_v2_f(num, num_bits)
    }

    fn bits2num(
        builder: &mut Builder<Self>,
        bits: impl IntoIterator<Item = Felt<<Self as Config>::F>>,
    ) -> Felt<<Self as Config>::F> {
        builder.bits2num_v2_f(bits)
    }

    fn select_chain_f(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
        second: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
    ) -> Vec<Felt<<Self as Config>::F>> {
        let one: Felt<_> = builder.constant(Self::F::one());
        let shouldnt_swap: Felt<_> = builder.eval(one - should_swap);

        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(zip(id_branch, swap_branch), zip(repeat(shouldnt_swap), repeat(should_swap)))
            .map(|((id_v, sw_v), (id_c, sw_c))| builder.eval(id_v * id_c + sw_v * sw_c))
            .collect()
    }

    fn select_chain_ef(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
        second: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
    ) -> Vec<Ext<<Self as Config>::F, <Self as Config>::EF>> {
        let one: Felt<_> = builder.constant(Self::F::one());
        let shouldnt_swap: Felt<_> = builder.eval(one - should_swap);

        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(zip(id_branch, swap_branch), zip(repeat(shouldnt_swap), repeat(should_swap)))
            .map(|((id_v, sw_v), (id_c, sw_c))| builder.eval(id_v * id_c + sw_v * sw_c))
            .collect()
    }

    fn exp_f_bits_precomputed(
        builder: &mut Builder<Self>,
        power_bits: &[Self::Bit],
        two_adic_powers_of_x: &[Felt<Self::F>],
    ) -> Felt<Self::F> {
        Self::exp_reverse_bits(
            builder,
            two_adic_powers_of_x[0],
            power_bits.iter().rev().copied().collect(),
        )
    }
}

impl CircuitConfig for OuterConfig {
    type Bit = Var<<Self as Config>::N>;

    fn assert_bit_zero(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_var_eq(bit, Self::N::zero());
    }

    fn assert_bit_one(builder: &mut Builder<Self>, bit: Self::Bit) {
        builder.assert_var_eq(bit, Self::N::one());
    }

    fn read_bit(builder: &mut Builder<Self>) -> Self::Bit {
        builder.witness_var()
    }

    fn read_felt(builder: &mut Builder<Self>) -> Felt<Self::F> {
        builder.witness_felt()
    }

    fn read_ext(builder: &mut Builder<Self>) -> Ext<Self::F, Self::EF> {
        builder.witness_ext()
    }

    fn ext2felt(
        builder: &mut Builder<Self>,
        ext: Ext<<Self as Config>::F, <Self as Config>::EF>,
    ) -> [Felt<<Self as Config>::F>; D] {
        let felts = core::array::from_fn(|_| builder.uninit());
        builder.push_op(DslIr::CircuitExt2Felt(felts, ext));
        felts
    }

    fn exp_reverse_bits(
        builder: &mut Builder<Self>,
        input: Felt<<Self as Config>::F>,
        power_bits: Vec<Var<<Self as Config>::N>>,
    ) -> Felt<<Self as Config>::F> {
        let mut result = builder.constant(Self::F::one());
        let power_f = input;
        let bit_len = power_bits.len();

        for i in 1..=bit_len {
            let index = bit_len - i;
            let bit = power_bits[index];
            let prod = builder.eval(result * power_f);
            result = builder.select_f(bit, prod, result);
            builder.assign(power_f, power_f * power_f);
        }
        result
    }

    fn batch_fri(
        builder: &mut Builder<Self>,
        alpha_pows: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
        p_at_zs: Vec<Ext<<Self as Config>::F, <Self as Config>::EF>>,
        p_at_xs: Vec<Felt<<Self as Config>::F>>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        // Initialize the `acc` to zero.
        let mut acc: Ext<_, _> = builder.uninit();
        builder.push_op(DslIr::ImmE(acc, <Self as Config>::EF::zero()));
        for (alpha_pow, p_at_z, p_at_x) in izip!(alpha_pows, p_at_zs, p_at_xs) {
            // Set `temp_1 = p_at_z - p_at_x`
            let temp_1: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::SubEF(temp_1, p_at_z, p_at_x));
            // Set `temp_2 = alpha_pow * temp_1 = alpha_pow * (p_at_z - p_at_x)`
            let temp_2: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::MulE(temp_2, alpha_pow, temp_1));
            // Set `acc += temp_2`, so that `acc` becomes the sum of `alpha_pow * (p_at_z - p_at_x)`
            let temp_3: Ext<_, _> = builder.uninit();
            builder.push_op(DslIr::AddE(temp_3, acc, temp_2));
            acc = temp_3;
        }
        acc
    }

    fn num2bits(
        builder: &mut Builder<Self>,
        num: Felt<<Self as Config>::F>,
        num_bits: usize,
    ) -> Vec<Var<<Self as Config>::N>> {
        builder.num2bits_f_circuit(num)[..num_bits].to_vec()
    }

    fn bits2num(
        builder: &mut Builder<Self>,
        bits: impl IntoIterator<Item = Var<<Self as Config>::N>>,
    ) -> Felt<<Self as Config>::F> {
        let result = builder.eval(Self::F::zero());
        for (i, bit) in bits.into_iter().enumerate() {
            let to_add: Felt<_> = builder.uninit();
            let pow2 = builder.constant(Self::F::from_canonical_u32(1 << i));
            let zero = builder.constant(Self::F::zero());
            builder.push_op(DslIr::CircuitSelectF(bit, pow2, zero, to_add));
            builder.assign(result, result + to_add);
        }
        result
    }

    fn select_chain_f(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
        second: impl IntoIterator<Item = Felt<<Self as Config>::F>> + Clone,
    ) -> Vec<Felt<<Self as Config>::F>> {
        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(id_branch, swap_branch)
            .map(|(id_v, sw_v): (Felt<_>, Felt<_>)| -> Felt<_> {
                let result: Felt<_> = builder.uninit();
                builder.push_op(DslIr::CircuitSelectF(should_swap, sw_v, id_v, result));
                result
            })
            .collect()
    }

    fn select_chain_ef(
        builder: &mut Builder<Self>,
        should_swap: Self::Bit,
        first: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
        second: impl IntoIterator<Item = Ext<<Self as Config>::F, <Self as Config>::EF>> + Clone,
    ) -> Vec<Ext<<Self as Config>::F, <Self as Config>::EF>> {
        let id_branch = first.clone().into_iter().chain(second.clone());
        let swap_branch = second.into_iter().chain(first);
        zip(id_branch, swap_branch)
            .map(|(id_v, sw_v): (Ext<_, _>, Ext<_, _>)| -> Ext<_, _> {
                let result: Ext<_, _> = builder.uninit();
                builder.push_op(DslIr::CircuitSelectE(should_swap, sw_v, id_v, result));
                result
            })
            .collect()
    }

    fn exp_f_bits_precomputed(
        builder: &mut Builder<Self>,
        power_bits: &[Self::Bit],
        two_adic_powers_of_x: &[Felt<Self::F>],
    ) -> Felt<Self::F> {
        let mut result: Felt<_> = builder.eval(Self::F::one());
        let one = builder.constant(Self::F::one());
        for (&bit, &power) in power_bits.iter().zip(two_adic_powers_of_x) {
            let multiplier = builder.select_f(bit, power, one);
            result = builder.eval(multiplier * result);
        }
        result
    }
}

impl BabyBearFriConfig for BabyBearPoseidon2 {
    type BasefoldConfig = Poseidon2BabyBear16BasefoldConfig;
    type MerkleTreeConfig = Poseidon2BabyBearConfig;
    type FriChallenger = <Self as JaggedConfig>::Challenger;
}

impl<C: CircuitConfig<F = BabyBear, Bit = Felt<BabyBear>>> BabyBearFriConfigVariable<C>
    for BabyBearPoseidon2
{
    type FriChallengerVariable = DuplexChallengerVariable<C>;

    fn challenger_variable(&self, builder: &mut Builder<C>) -> Self::FriChallengerVariable {
        DuplexChallengerVariable::new(builder)
    }
}
