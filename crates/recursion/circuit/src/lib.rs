use challenger::{
    CanCopyChallenger, CanObserveVariable, DuplexChallengerVariable, FieldChallengerVariable,
    MultiField32ChallengerVariable,
};
use hash::{FieldHasherVariable, Poseidon2MyFieldHasherVariable};
use itertools::izip;
use slop_algebra::{AbstractExtensionField, AbstractField};
use slop_bn254::Bn254Fr;
use sp1_core_machine::operations::poseidon2::{NUM_EXTERNAL_ROUNDS, NUM_INTERNAL_ROUNDS};
use sp1_recursion_compiler::{
    circuit::CircuitV2Builder,
    config::{InnerConfig, OuterConfig},
    ir::{Builder, Config, DslIr, Ext, Felt, SymbolicExt, SymbolicFelt, Var, Variable},
};
use sp1_recursion_executor::{RecursionPublicValues, DIGEST_SIZE, NUM_BITS, PERMUTATION_WIDTH};
use std::iter::{repeat, zip};
use utils::{felt_bytes_to_bn254_var, felts_to_bn254_var, words_to_bytes};

use slop_basefold::{
    BasefoldConfig, BasefoldProof, BasefoldVerifier, Poseidon2Bn254FrBasefoldConfig,
};
use slop_commit::TensorCs;
use slop_merkle_tree::{MerkleTreeConfig, MerkleTreeTcs, Poseidon2Bn254Config};
use sp1_hypercube::{SP1BasefoldConfig, SP1CoreJaggedConfig, SP1MerkleTreeConfig, SP1OuterConfig};
pub mod basefold;
pub mod challenger;
pub mod dummy;
pub mod hash;
pub mod jagged;
pub mod logup_gkr;
pub mod machine;
pub mod shard;
pub mod sumcheck;
mod symbolic;
pub mod utils;
pub mod witness;
pub mod zerocheck;
pub const D: usize = 4;
use slop_challenger::{CanObserve, CanSample, FieldChallenger, GrindingChallenger};
use slop_jagged::JaggedConfig;
use sp1_primitives::{SP1Field, RC_16_30_U32};
type EF = <SP1CoreJaggedConfig as JaggedConfig>::EF;

pub type Digest<C, SC> = <SC as FieldHasherVariable<C>>::DigestVariable;

pub type InnerSC = SP1CoreJaggedConfig;

pub trait AsRecursive<C: CircuitConfig> {
    type Recursive;
}
pub trait SP1FieldFriConfig:
    JaggedConfig<
    F = SP1Field,
    EF = EF,
    Commitment = <MerkleTreeTcs<Self::MerkleTreeConfig> as TensorCs>::Commitment,
    BatchPcsProof = BasefoldProof<Self::BasefoldConfig>,
    Challenger = Self::FriChallenger,
    BatchPcsVerifier = BasefoldVerifier<Self::BasefoldConfig>,
>
{
    type BasefoldConfig: BasefoldConfig<
        F = SP1Field,
        EF = EF,
        Tcs = MerkleTreeTcs<Self::MerkleTreeConfig>,
        Commitment = <MerkleTreeTcs<Self::MerkleTreeConfig> as TensorCs>::Commitment,
        Challenger = Self::FriChallenger,
    >;
    type MerkleTreeConfig: MerkleTreeConfig<Data = SP1Field>;
    type FriChallenger: CanObserve<<Self::BasefoldConfig as BasefoldConfig>::Commitment>
        + CanSample<EF>
        + GrindingChallenger<Witness = SP1Field>
        + FieldChallenger<SP1Field>;
}

pub trait SP1FieldConfigVariable<C: CircuitConfig<F = SP1Field>>:
    SP1FieldFriConfig + FieldHasherVariable<C> + Poseidon2MyFieldHasherVariable<C> + Send + Sync
{
    type FriChallengerVariable: FieldChallengerVariable<C, <C as CircuitConfig>::Bit>
        + CanObserveVariable<C, <Self as FieldHasherVariable<C>>::DigestVariable>
        + CanCopyChallenger<C>;

    /// Get a new challenger corresponding to the given config.
    fn challenger_variable(builder: &mut Builder<C>) -> Self::FriChallengerVariable;

    fn commit_recursion_public_values(
        builder: &mut Builder<C>,
        public_values: RecursionPublicValues<Felt<C::F>>,
    );
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

    fn felt2ext(
        builder: &mut Builder<Self>,
        felt: [Felt<<Self as Config>::F>; D],
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF>;

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

    #[allow(clippy::type_complexity)]
    fn prefix_sum_checks(
        builder: &mut Builder<Self>,
        x1: Vec<Felt<Self::F>>,
        x2: Vec<Ext<Self::F, Self::EF>>,
    ) -> (Ext<Self::F, Self::EF>, Felt<Self::F>);

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
        let bits = Self::num2bits(builder, value, NUM_BITS);
        for bit in bits.into_iter().skip(num_bits) {
            Self::assert_bit_zero(builder, bit);
        }
    }

    fn poseidon2_permute_v2(
        builder: &mut Builder<Self>,
        input: [Felt<Self::F>; PERMUTATION_WIDTH],
    ) -> [Felt<Self::F>; PERMUTATION_WIDTH];

    fn poseidon2_compress_v2(
        builder: &mut Builder<Self>,
        input: impl IntoIterator<Item = Felt<Self::F>>,
    ) -> [Felt<Self::F>; DIGEST_SIZE] {
        let mut pre_iter = input.into_iter().chain(repeat(builder.eval(Self::F::zero())));
        let pre = core::array::from_fn(move |_| pre_iter.next().unwrap());
        let post = Self::poseidon2_permute_v2(builder, pre);
        let post: [Felt<Self::F>; DIGEST_SIZE] = post[..DIGEST_SIZE].try_into().unwrap();
        post
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

    fn felt2ext(
        builder: &mut Builder<Self>,
        felt: [Felt<<Self as Config>::F>; D],
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        let mut reconstructed_ext: Ext<Self::F, Self::EF> = builder.constant(Self::EF::zero());
        for i in 0..D {
            let mut monomial_slice = [Self::F::zero(); D];
            monomial_slice[i] = Self::F::one();
            let monomial: Ext<Self::F, Self::EF> =
                builder.constant(Self::EF::from_base_slice(&monomial_slice));
            reconstructed_ext = builder.eval(reconstructed_ext + monomial * felt[i]);
        }
        reconstructed_ext
    }

    fn exp_reverse_bits(
        builder: &mut Builder<Self>,
        input: Felt<<Self as Config>::F>,
        power_bits: Vec<Felt<<Self as Config>::F>>,
    ) -> Felt<<Self as Config>::F> {
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

    fn prefix_sum_checks(
        builder: &mut Builder<Self>,
        x1: Vec<Felt<Self::F>>,
        x2: Vec<Ext<Self::F, Self::EF>>,
    ) -> (Ext<Self::F, Self::EF>, Felt<Self::F>) {
        builder.prefix_sum_checks_v2(x1, x2)
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

    fn poseidon2_permute_v2(
        builder: &mut Builder<Self>,
        input: [Felt<Self::F>; PERMUTATION_WIDTH],
    ) -> [Felt<Self::F>; PERMUTATION_WIDTH] {
        builder.poseidon2_permute_v2(input)
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WrapConfig;

impl Config for WrapConfig {
    type F = <InnerConfig as Config>::F;
    type EF = <InnerConfig as Config>::EF;
    type N = <InnerConfig as Config>::N;
    fn initialize(builder: &mut Builder<Self>) {
        for round in 0..RC_16_30_U32.len() {
            for i in 0..PERMUTATION_WIDTH / D {
                let add_rc = if (NUM_EXTERNAL_ROUNDS / 2
                    ..NUM_EXTERNAL_ROUNDS / 2 + NUM_INTERNAL_ROUNDS)
                    .contains(&round)
                {
                    builder.constant(<Self as Config>::EF::from_base(
                        <Self as Config>::F::from_wrapped_u32(RC_16_30_U32[round][i * D]),
                    ))
                } else {
                    builder.constant(<Self as Config>::EF::from_base_fn(|idx| {
                        <Self as Config>::F::from_wrapped_u32(RC_16_30_U32[round][i * D + idx])
                    }))
                };

                builder.poseidon2_constants.push(add_rc);
            }
        }
    }
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
        let felts = core::array::from_fn(|_| builder.uninit());
        builder.push_op(DslIr::CircuitChipExt2Felt(felts, ext));
        felts
    }

    fn felt2ext(
        builder: &mut Builder<Self>,
        felt: [Felt<<Self as Config>::F>; D],
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        let ext = builder.uninit();
        builder.push_op(DslIr::CircuitChipFelt2Ext(ext, felt));
        ext
    }

    fn exp_reverse_bits(
        builder: &mut Builder<Self>,
        input: Felt<<Self as Config>::F>,
        power_bits: Vec<Felt<<Self as Config>::F>>,
    ) -> Felt<<Self as Config>::F> {
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

    fn prefix_sum_checks(
        builder: &mut Builder<Self>,
        point_1: Vec<Felt<Self::F>>,
        point_2: Vec<Ext<Self::F, Self::EF>>,
    ) -> (Ext<Self::F, Self::EF>, Felt<Self::F>) {
        let mut acc: Ext<_, _> = builder.uninit();
        builder.push_op(DslIr::ImmE(acc, <Self as Config>::EF::one()));
        let mut acc_felt: Felt<_> = builder.uninit();
        builder.push_op(DslIr::ImmF(acc_felt, Self::F::zero()));
        let one: Felt<_> = builder.constant(Self::F::one());
        for (i, (x1, x2)) in izip!(point_1.clone(), point_2).enumerate() {
            let prod = builder.uninit();
            builder.push_op(DslIr::MulEF(prod, x2, x1));
            let lagrange_term: Ext<_, _> = builder.eval(SymbolicExt::one() - x1 - x2 + prod + prod);
            // Check that x1 is boolean.
            builder.assert_felt_eq(x1 * (x1 - one), SymbolicFelt::zero());
            acc = builder.eval(acc * lagrange_term);
            // Only need felt of first half of point_1 (current prefix sum).
            if i < point_1.len() / 2 {
                acc_felt = builder.eval(x1 + acc_felt * SymbolicFelt::from_canonical_u32(2));
            }
        }
        (acc, acc_felt)
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

    fn poseidon2_permute_v2(
        builder: &mut Builder<Self>,
        input: [Felt<Self::F>; PERMUTATION_WIDTH],
    ) -> [Felt<Self::F>; PERMUTATION_WIDTH] {
        // builder.poseidon2_permute_v2(input)
        let mut state = Self::blockify(builder, input);
        for i in 0..NUM_EXTERNAL_ROUNDS / 2 {
            state = Self::external_round(builder, state, i);
        }
        for i in 0..NUM_INTERNAL_ROUNDS {
            state[0] = Self::internal_constant_addition(builder, state[0], i);
            state[0] = Self::pow7_internal(builder, state[0]);
            state = Self::internal_linear_layer(builder, state);
        }
        for i in NUM_EXTERNAL_ROUNDS / 2..NUM_EXTERNAL_ROUNDS {
            state = Self::external_round(builder, state, i);
        }
        Self::unblockify(builder, state)
    }
}

impl WrapConfig {
    fn blockify(
        builder: &mut Builder<Self>,
        input: [Felt<<Self as Config>::F>; PERMUTATION_WIDTH],
    ) -> [Ext<<Self as Config>::F, <Self as Config>::EF>; PERMUTATION_WIDTH / D] {
        let mut ret: [Ext<<Self as Config>::F, <Self as Config>::EF>; PERMUTATION_WIDTH / D] =
            core::array::from_fn(|_| builder.uninit());
        for i in 0..PERMUTATION_WIDTH / D {
            ret[i] = Self::felt2ext(builder, input[i * D..i * D + D].try_into().unwrap());
        }
        ret
    }

    fn unblockify(
        builder: &mut Builder<Self>,
        input: [Ext<<Self as Config>::F, <Self as Config>::EF>; PERMUTATION_WIDTH / D],
    ) -> [Felt<<Self as Config>::F>; PERMUTATION_WIDTH] {
        let mut ret = core::array::from_fn(|_| builder.uninit());
        for i in 0..PERMUTATION_WIDTH / D {
            let felts = Self::ext2felt(builder, input[i]);
            for j in 0..D {
                ret[i * D + j] = felts[j];
            }
        }
        ret
    }

    fn external_round(
        builder: &mut Builder<Self>,
        input: [Ext<<Self as Config>::F, <Self as Config>::EF>; PERMUTATION_WIDTH / D],
        round_index: usize,
    ) -> [Ext<<Self as Config>::F, <Self as Config>::EF>; PERMUTATION_WIDTH / D] {
        let mut state = input;
        if round_index == 0 {
            state = Self::external_linear_layer(builder, state);
        }
        state = Self::external_constant_addition(builder, state, round_index);
        #[allow(clippy::needless_range_loop)]
        for i in 0..PERMUTATION_WIDTH / D {
            state[i] = Self::pow7(builder, state[i]);
        }
        state = Self::external_linear_layer(builder, state);
        state
    }

    fn external_linear_layer(
        builder: &mut Builder<Self>,
        input: [Ext<<Self as Config>::F, <Self as Config>::EF>; PERMUTATION_WIDTH / D],
    ) -> [Ext<<Self as Config>::F, <Self as Config>::EF>; PERMUTATION_WIDTH / D] {
        let output = core::array::from_fn(|_| builder.uninit());
        builder.push_op(DslIr::Poseidon2ExternalLinearLayer(Box::new((output, input))));
        output
    }

    fn internal_linear_layer(
        builder: &mut Builder<Self>,
        input: [Ext<<Self as Config>::F, <Self as Config>::EF>; PERMUTATION_WIDTH / D],
    ) -> [Ext<<Self as Config>::F, <Self as Config>::EF>; PERMUTATION_WIDTH / D] {
        let output = core::array::from_fn(|_| builder.uninit());
        builder.push_op(DslIr::Poseidon2InternalLinearLayer(Box::new((output, input))));
        output
    }

    fn pow7(
        builder: &mut Builder<Self>,
        input: Ext<<Self as Config>::F, <Self as Config>::EF>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        let output = builder.uninit();
        builder.push_op(DslIr::Poseidon2ExternalSBOX(output, input));
        output
    }

    fn pow7_internal(
        builder: &mut Builder<Self>,
        input: Ext<<Self as Config>::F, <Self as Config>::EF>,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        let output = builder.uninit();
        builder.push_op(DslIr::Poseidon2InternalSBOX(output, input));
        output
    }

    fn external_constant_addition(
        builder: &mut Builder<Self>,
        input: [Ext<<Self as Config>::F, <Self as Config>::EF>; PERMUTATION_WIDTH / D],
        round_index: usize,
    ) -> [Ext<<Self as Config>::F, <Self as Config>::EF>; PERMUTATION_WIDTH / D] {
        let output = core::array::from_fn(|_| builder.uninit());
        let round = if round_index < NUM_EXTERNAL_ROUNDS / 2 {
            round_index
        } else {
            round_index + NUM_INTERNAL_ROUNDS
        };
        for i in 0..PERMUTATION_WIDTH / D {
            let add_rc = builder.poseidon2_constants[(PERMUTATION_WIDTH / D) * round + i];
            builder.push_op(DslIr::AddE(output[i], input[i], add_rc));
        }

        output
    }

    fn internal_constant_addition(
        builder: &mut Builder<Self>,
        input: Ext<<Self as Config>::F, <Self as Config>::EF>,
        round_index: usize,
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        let round = round_index + NUM_EXTERNAL_ROUNDS / 2;
        let add_rc = builder.poseidon2_constants[(PERMUTATION_WIDTH / D) * round];
        let output = builder.uninit();
        builder.push_op(DslIr::AddE(output, input, add_rc));
        output
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

    fn felt2ext(
        builder: &mut Builder<Self>,
        felt: [Felt<<Self as Config>::F>; D],
    ) -> Ext<<Self as Config>::F, <Self as Config>::EF> {
        let ext = builder.uninit();
        builder.push_op(DslIr::CircuitFelts2Ext(felt, ext));
        ext
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

    fn prefix_sum_checks(
        builder: &mut Builder<Self>,
        point_1: Vec<Felt<Self::F>>,
        point_2: Vec<Ext<Self::F, Self::EF>>,
    ) -> (Ext<Self::F, Self::EF>, Felt<Self::F>) {
        let acc: Ext<_, _> = builder.uninit();
        builder.push_op(DslIr::ImmE(acc, <Self as Config>::EF::one()));
        let mut acc_felt: Felt<_> = builder.uninit();
        builder.push_op(DslIr::ImmF(acc_felt, Self::F::zero()));
        for (i, (x1, x2)) in izip!(point_1.clone(), point_2).enumerate() {
            builder.push_op(DslIr::EqEval(x1, x2, acc));
            // Only need felt of first half of point_1 (current prefix sum).
            if i < point_1.len() / 2 {
                acc_felt = builder.eval(x1 + acc_felt * SymbolicFelt::from_canonical_u32(2));
            }
        }
        (acc, acc_felt)
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

    fn poseidon2_permute_v2(
        _: &mut Builder<Self>,
        _: [Felt<Self::F>; PERMUTATION_WIDTH],
    ) -> [Felt<Self::F>; PERMUTATION_WIDTH] {
        unimplemented!();
    }
}

impl SP1FieldFriConfig for SP1CoreJaggedConfig {
    type BasefoldConfig = SP1BasefoldConfig;
    type MerkleTreeConfig = SP1MerkleTreeConfig;
    type FriChallenger = <Self as JaggedConfig>::Challenger;
}

impl<C: CircuitConfig<F = SP1Field, Bit = Felt<SP1Field>>> SP1FieldConfigVariable<C>
    for SP1CoreJaggedConfig
{
    type FriChallengerVariable = DuplexChallengerVariable<C>;

    fn challenger_variable(builder: &mut Builder<C>) -> Self::FriChallengerVariable {
        DuplexChallengerVariable::new(builder)
    }

    fn commit_recursion_public_values(
        builder: &mut Builder<C>,
        public_values: RecursionPublicValues<Felt<<C>::F>>,
    ) {
        builder.commit_public_values_v2(public_values);
    }
}

impl SP1FieldFriConfig for SP1OuterConfig {
    type BasefoldConfig = Poseidon2Bn254FrBasefoldConfig;
    type MerkleTreeConfig = Poseidon2Bn254Config;
    type FriChallenger = <Self as JaggedConfig>::Challenger;
}

impl<C: CircuitConfig<F = SP1Field, N = Bn254Fr, Bit = Var<Bn254Fr>>> SP1FieldConfigVariable<C>
    for SP1OuterConfig
{
    type FriChallengerVariable = MultiField32ChallengerVariable<C>;

    fn challenger_variable(builder: &mut Builder<C>) -> Self::FriChallengerVariable {
        MultiField32ChallengerVariable::new(builder)
    }

    fn commit_recursion_public_values(
        builder: &mut Builder<C>,
        public_values: RecursionPublicValues<Felt<<C>::F>>,
    ) {
        let committed_values_digest_bytes_felts: [Felt<_>; 32] =
            words_to_bytes(&public_values.committed_value_digest).try_into().unwrap();
        let committed_values_digest_bytes: Var<_> =
            felt_bytes_to_bn254_var(builder, &committed_values_digest_bytes_felts);
        builder.commit_committed_values_digest_circuit(committed_values_digest_bytes);

        let vkey_hash = felts_to_bn254_var(builder, &public_values.sp1_vk_digest);
        builder.commit_vkey_hash_circuit(vkey_hash);

        let exit_code = public_values.exit_code;
        let var_exit_code: Var<C::N> = builder.felt2var_circuit(exit_code);
        builder.commit_exit_code_circuit(var_exit_code);

        let vk_root: Var<_> = felts_to_bn254_var(builder, &public_values.vk_root);
        builder.commit_vk_root_circuit(vk_root);
    }
}
