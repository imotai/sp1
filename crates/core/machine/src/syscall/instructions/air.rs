use std::borrow::Borrow;

use itertools::Itertools;
use p3_air::{Air, AirBuilder};
use p3_field::AbstractField;
use p3_matrix::Matrix;
use sp1_core_executor::{syscalls::SyscallCode, Opcode, DEFAULT_CLK_INC};
use sp1_stark::{
    air::{
        BaseAirBuilder, InteractionScope, PublicValues, SP1AirBuilder, POSEIDON_NUM_WORDS,
        PV_DIGEST_NUM_WORDS, SP1_PROOF_NUM_PV_ELTS,
    },
    Word,
};

use crate::{
    adapter::{register::r_type::RTypeReader, state::CPUState},
    air::{SP1CoreAirBuilder, SP1Operation, WordAirBuilder},
    operations::{
        BabyBearWordRangeChecker, IsZeroOperation, U16toU8OperationSafe, U16toU8OperationSafeInput,
    },
};

use super::{columns::SyscallInstrColumns, SyscallInstrsChip};

impl<AB> Air<AB> for SyscallInstrsChip
where
    AB: SP1CoreAirBuilder,
    AB::Var: Sized,
{
    #[inline(never)]
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &SyscallInstrColumns<AB::Var> = (*local).borrow();

        let public_values_slice: [AB::PublicVar; SP1_PROOF_NUM_PV_ELTS] =
            core::array::from_fn(|i| builder.public_values()[i]);
        let public_values: &PublicValues<
            [AB::PublicVar; 4],
            Word<AB::PublicVar>,
            [AB::PublicVar; 4],
            AB::PublicVar,
        > = public_values_slice.as_slice().borrow();

        // Convert the syscall code to four bytes using the safe API.
        let a_input = U16toU8OperationSafeInput::new(
            local.adapter.prev_a().0.map(Into::into),
            local.a_low_bytes,
            local.is_real.into(),
        );
        let a = U16toU8OperationSafe::eval(builder, a_input);

        // SAFETY: Only `ECALL` opcode can be received in this chip.
        // `is_real` is checked to be boolean, and the `opcode` matches the corresponding opcode.
        builder.assert_bool(local.is_real);

        // Verify that local.is_halt is correct.
        self.eval_is_halt_syscall(builder, &a, local);

        // Constrain the state of the CPU.
        // The extra timestamp increment is `num_extra_cycles`.
        // The `next_pc` is constrained in the AIR.
        CPUState::<AB::F>::eval(
            builder,
            local.state,
            local.next_pc.into(),
            AB::Expr::from_canonical_u32(DEFAULT_CLK_INC + 256),
            local.is_real.into(),
        );

        // Constrain the program and register reads.
        RTypeReader::<AB::F>::eval(
            builder,
            local.state.clk_high::<AB>(),
            local.state.clk_low::<AB>(),
            local.state.pc,
            AB::Expr::from_canonical_u32(Opcode::ECALL as u32),
            local.op_a_value,
            local.adapter,
            local.is_real.into(),
        );
        builder.when(local.is_real).assert_zero(local.adapter.op_a_0);

        // If the syscall is not halt, then next_pc should be pc + 4.
        // `next_pc` is constrained for the case where `is_halt` is false to be `pc + 4`.
        builder
            .when(local.is_real)
            .when(AB::Expr::one() - local.is_halt)
            .assert_eq(local.next_pc, local.state.pc + AB::Expr::from_canonical_u32(4));

        // `num_extra_cycles` is checked to be equal to the return value of
        // `get_num_extra_ecall_cycles`
        builder.assert_eq::<AB::Var, AB::Expr>(
            local.num_extra_cycles,
            self.get_num_extra_ecall_cycles::<AB>(&a, local),
        );

        // ECALL instruction.
        self.eval_ecall(builder, &a, local);

        // COMMIT/COMMIT_DEFERRED_PROOFS ecall instruction.
        self.eval_commit(
            builder,
            &a,
            local,
            public_values.committed_value_digest,
            public_values.deferred_proofs_digest,
        );

        // HALT ecall and UNIMPL instruction.
        self.eval_halt_unimpl(builder, local, public_values);
    }
}

impl SyscallInstrsChip {
    /// Constraints related to the ECALL opcode.
    ///
    /// This method will do the following:
    /// 1. Send the syscall to the precompile table, if needed.
    /// 2. Check for valid op_a values.
    pub(crate) fn eval_ecall<AB: SP1CoreAirBuilder>(
        &self,
        builder: &mut AB,
        prev_a_byte: &[AB::Expr; 4],
        local: &SyscallInstrColumns<AB::Var>,
    ) {
        // We interpret the syscall_code as little-endian bytes and interpret each byte as a u8
        // with different information.
        let syscall_id = prev_a_byte[0].clone();
        let send_to_table = prev_a_byte[1].clone();

        // SAFETY: Assert that for padding rows, the interactions from `send_syscall` and
        // BabyBearWordRangeChecker do not have non-zero multiplicities.
        builder.when_not(local.is_real).assert_zero(send_to_table.clone());
        builder.when_not(local.is_real).assert_zero(local.is_halt);
        builder.when_not(local.is_real).assert_zero(local.is_commit_deferred_proofs.result);

        builder.send_syscall(
            local.state.clk_high::<AB>(),
            local.state.clk_low::<AB>(),
            syscall_id.clone(),
            local.adapter.b().reduce::<AB>(),
            local.adapter.c().reduce::<AB>(),
            send_to_table.clone(),
            InteractionScope::Local,
        );

        // Check if `op_b` and `op_c` are a valid BabyBear words.
        // SAFETY: The multiplicities are zero when `is_real = 0`.
        // Note that `send_to_table = 1` implies `is_halt, is_commit_deferred_proofs` are zero,
        // since the syscall cannot be `HALT` or `COMMIT_DEFERRED_PROOFS`.
        BabyBearWordRangeChecker::<AB::F>::range_check::<AB>(
            builder,
            *local.adapter.b(),
            local.op_b_range_check,
            send_to_table.clone() + local.is_halt,
        );

        // Check if `op_c` is a valid BabyBear word.
        BabyBearWordRangeChecker::<AB::F>::range_check::<AB>(
            builder,
            *local.adapter.c(),
            local.op_c_range_check,
            send_to_table.clone() + local.is_commit_deferred_proofs.result,
        );

        // Compute whether this ecall is ENTER_UNCONSTRAINED.
        let is_enter_unconstrained = {
            IsZeroOperation::<AB::F>::eval(
                builder,
                (
                    syscall_id.clone()
                        - AB::Expr::from_canonical_u32(
                            SyscallCode::ENTER_UNCONSTRAINED.syscall_id(),
                        ),
                    local.is_enter_unconstrained,
                    local.is_real.into(),
                ),
            );
            local.is_enter_unconstrained.result
        };

        // Compute whether this ecall is HINT_LEN.
        let is_hint_len = {
            IsZeroOperation::<AB::F>::eval(
                builder,
                (
                    syscall_id.clone()
                        - AB::Expr::from_canonical_u32(SyscallCode::HINT_LEN.syscall_id()),
                    local.is_hint_len,
                    local.is_real.into(),
                ),
            );
            local.is_hint_len.result
        };

        // `op_a_val` is constrained.
        // When syscall_id is ENTER_UNCONSTRAINED, the new value of op_a should be 0.
        let zero_word = Word::<AB::F>::from(0);
        builder
            .when(local.is_real)
            .when(is_enter_unconstrained)
            .assert_word_eq(local.op_a_value, zero_word);

        // When the syscall is not one of ENTER_UNCONSTRAINED or HINT_LEN, op_a shouldn't change.
        builder
            .when(local.is_real)
            .when_not(is_enter_unconstrained + is_hint_len)
            .assert_word_eq(local.op_a_value, *local.adapter.prev_a());

        // SAFETY: This leaves the case where syscall is `HINT_LEN`.
        // In this case, `op_a`'s value can be arbitrary, but it still must be a valid word.
        // As this is a syscall for HINT, the value itself being arbitrary is fine, as long as it is
        // a valid word.
        builder.slice_range_check_u16(&local.op_a_value.0, local.is_real);
    }

    /// Constraints related to the COMMIT and COMMIT_DEFERRED_PROOFS instructions.
    pub(crate) fn eval_commit<AB: SP1CoreAirBuilder>(
        &self,
        builder: &mut AB,
        prev_a_byte: &[AB::Expr; 4],
        local: &SyscallInstrColumns<AB::Var>,
        commit_digest: [[AB::PublicVar; 4]; PV_DIGEST_NUM_WORDS],
        deferred_proofs_digest: [AB::PublicVar; POSEIDON_NUM_WORDS],
    ) {
        let (is_commit, is_commit_deferred_proofs) =
            self.get_is_commit_related_syscall(builder, prev_a_byte, local);

        // Verify the index bitmap.
        let mut bitmap_sum = AB::Expr::zero();
        // They should all be bools.
        for bit in local.index_bitmap.iter() {
            builder.when(local.is_real).assert_bool(*bit);
            bitmap_sum = bitmap_sum.clone() + (*bit).into();
        }
        // When the syscall is COMMIT or COMMIT_DEFERRED_PROOFS, there should be one set bit.
        builder
            .when(local.is_real)
            .when(is_commit.clone() + is_commit_deferred_proofs.clone())
            .assert_one(bitmap_sum.clone());
        // When it's some other syscall, there should be no set bits.
        builder
            .when(local.is_real)
            .when(AB::Expr::one() - (is_commit.clone() + is_commit_deferred_proofs.clone()))
            .assert_zero(bitmap_sum);

        // Verify that word_idx corresponds to the set bit in index bitmap.
        for (i, bit) in local.index_bitmap.iter().enumerate() {
            builder
                .when(local.is_real)
                .when(*bit)
                .assert_eq(local.adapter.b()[0], AB::Expr::from_canonical_u32(i as u32));
        }
        // Verify that the upper limb of the word_idx is 0.
        builder
            .when(local.is_real)
            .when(is_commit.clone() + is_commit_deferred_proofs.clone())
            .assert_zero(local.adapter.b()[1]);

        // Retrieve the expected public values digest word to check against the one passed into the
        // commit ecall. Note that for the interaction builder, it will not have any digest words,
        // since it's used during AIR compilation time to parse for all send/receives. Since
        // that interaction builder will ignore the other constraints of the air, it is safe
        // to not include the verification check of the expected public values digest word.

        // First, get the expected public value digest from the bitmap and public values.
        let mut expected_pv_digest =
            [AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero(), AB::Expr::zero()];
        for i in 0..4 {
            expected_pv_digest[i] = builder.index_array(
                commit_digest.iter().map(|word| word[i]).collect_vec().as_slice(),
                &local.index_bitmap,
            );
        }
        // Then, combine the bytes into u16 Word form, which will be compared to `op_c` value.
        let expected_pv_digest_word = Word([
            expected_pv_digest[0].clone()
                + expected_pv_digest[1].clone() * AB::F::from_canonical_u32(1 << 8),
            expected_pv_digest[2].clone()
                + expected_pv_digest[3].clone() * AB::F::from_canonical_u32(1 << 8),
        ]);

        // Assert that the expected public value digest are valid bytes.
        builder.assert_bool(is_commit.clone());
        for i in 0..4 {
            builder
                .when(is_commit.clone())
                .assert_eq(expected_pv_digest[i].clone(), local.expected_public_values_digest[i]);
        }
        builder.slice_range_check_u8(&local.expected_public_values_digest, is_commit.clone());

        let digest_word = local.adapter.c();

        // Verify the public_values_digest_word.
        builder
            .when(local.is_real)
            .when(is_commit.clone())
            .assert_word_eq(expected_pv_digest_word, *digest_word);

        let expected_deferred_proofs_digest_element =
            builder.index_array(&deferred_proofs_digest, &local.index_bitmap);

        builder
            .when(local.is_real)
            .when(is_commit_deferred_proofs.clone())
            .assert_eq(expected_deferred_proofs_digest_element, digest_word.reduce::<AB>());
    }

    /// Constraint related to the halt and unimpl instruction.
    pub(crate) fn eval_halt_unimpl<AB: SP1AirBuilder>(
        &self,
        builder: &mut AB,
        local: &SyscallInstrColumns<AB::Var>,
        public_values: &PublicValues<
            [AB::PublicVar; 4],
            Word<AB::PublicVar>,
            [AB::PublicVar; 4],
            AB::PublicVar,
        >,
    ) {
        // `next_pc` is constrained for the case where `is_halt` is true to be `0`
        builder.when(local.is_halt).assert_zero(local.next_pc);

        // Check that the `op_b_value` reduced is the `public_values.exit_code`.
        builder
            .when(local.is_halt)
            .assert_eq(local.adapter.b().reduce::<AB>(), public_values.exit_code);
    }

    /// Returns a boolean expression indicating whether the instruction is a HALT instruction.
    pub(crate) fn eval_is_halt_syscall<AB: SP1CoreAirBuilder>(
        &self,
        builder: &mut AB,
        prev_a_byte: &[AB::Expr; 4],
        local: &SyscallInstrColumns<AB::Var>,
    ) {
        // `is_halt` is checked to be correct in `eval_is_halt_syscall`.
        let syscall_id = prev_a_byte[0].clone();

        // Compute whether this ecall is HALT.
        let is_halt = {
            IsZeroOperation::<AB::F>::eval(
                builder,
                (
                    syscall_id.clone()
                        - AB::Expr::from_canonical_u32(SyscallCode::HALT.syscall_id()),
                    local.is_halt_check,
                    local.is_real.into(),
                ),
            );
            local.is_halt_check.result
        };

        // Verify that the is_halt flag is correct.
        // If `is_real = 0`, then `local.is_halt = 0`.
        // If `is_real = 1`, then `is_halt_check.result` will be correct, so `local.is_halt` is
        // correct.
        builder.assert_eq(local.is_halt, is_halt * local.is_real);
    }

    /// Returns two boolean expression indicating whether the instruction is a COMMIT or
    /// COMMIT_DEFERRED_PROOFS instruction.
    pub(crate) fn get_is_commit_related_syscall<AB: SP1CoreAirBuilder>(
        &self,
        builder: &mut AB,
        prev_a_byte: &[AB::Expr; 4],
        local: &SyscallInstrColumns<AB::Var>,
    ) -> (AB::Expr, AB::Expr) {
        let syscall_id = prev_a_byte[0].clone();

        // Compute whether this ecall is COMMIT.
        let is_commit = {
            IsZeroOperation::<AB::F>::eval(
                builder,
                (
                    syscall_id.clone()
                        - AB::Expr::from_canonical_u32(SyscallCode::COMMIT.syscall_id()),
                    local.is_commit,
                    local.is_real.into(),
                ),
            );
            local.is_commit.result
        };

        // Compute whether this ecall is COMMIT_DEFERRED_PROOFS.
        let is_commit_deferred_proofs = {
            IsZeroOperation::<AB::F>::eval(
                builder,
                (
                    syscall_id.clone()
                        - AB::Expr::from_canonical_u32(
                            SyscallCode::COMMIT_DEFERRED_PROOFS.syscall_id(),
                        ),
                    local.is_commit_deferred_proofs,
                    local.is_real.into(),
                ),
            );
            local.is_commit_deferred_proofs.result
        };

        (is_commit.into(), is_commit_deferred_proofs.into())
    }

    /// Returns the number of extra cycles from an ECALL instruction.
    pub(crate) fn get_num_extra_ecall_cycles<AB: SP1AirBuilder>(
        &self,
        prev_a_byte: &[AB::Expr; 4],
        local: &SyscallInstrColumns<AB::Var>,
    ) -> AB::Expr {
        let num_extra_cycles = prev_a_byte[2].clone();

        // If `is_real = 0`, then the return value is `0` regardless of `num_extra_cycles`.
        // If `is_real = 1`, `prev_a_byte` is constrained, and `num_extra_cycles` is correct.
        num_extra_cycles * local.is_real
    }
}
