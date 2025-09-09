use crate::syscalls::SyscallCode;

use super::{
    hint::{hint_len, hint_read},
    precompiles::{
        edwards::{edwards_add, edwards_decompress_syscall},
        fptower::{fp2_addsub_syscall, fp2_mul_syscall, fp_op_syscall},
        keccak::keccak_permute,
        poseidon2::poseidon2,
        sha256::{sha256_compress, sha256_extend},
        uint256::uint256_mul,
        uint256_ops::uint256_ops,
        uint256x2048::u256x2048_mul,
        weierstrass::{
            weierstrass_add_assign_syscall, weierstrass_decompress_syscall,
            weierstrass_double_assign_syscall,
        },
    },
    unconstrained::{enter_unconstrained, exit_unconstrained},
    write::write,
};

use sp1_curves::{
    edwards::ed25519::Ed25519,
    weierstrass::{
        bls12_381::{Bls12381, Bls12381BaseField},
        bn254::{Bn254, Bn254BaseField},
        secp256k1::Secp256k1,
        secp256r1::Secp256r1,
    },
};
use sp1_jit::JitContext;

pub(super) extern "C" fn sp1_ecall_handler(ctx: *mut JitContext) -> u64 {
    let ctx = unsafe { &mut *ctx };
    let registers = ctx.registers();
    let arg1 = registers[10];
    let arg2 = registers[11];

    let code = SyscallCode::from_u32(registers[5] as u32);
    let clk = ctx.clk;

    let res = match code {
        SyscallCode::SHA_EXTEND => unsafe { sha256_extend(ctx, arg1, arg2) },
        SyscallCode::SHA_COMPRESS => unsafe { sha256_compress(ctx, arg1, arg2) },
        SyscallCode::KECCAK_PERMUTE => unsafe { keccak_permute(ctx, arg1, arg2) },
        SyscallCode::SECP256K1_ADD => unsafe {
            weierstrass_add_assign_syscall::<Secp256k1>(ctx, arg1, arg2)
        },
        SyscallCode::SECP256K1_DOUBLE => unsafe {
            weierstrass_double_assign_syscall::<Secp256k1>(ctx, arg1, arg2)
        },
        SyscallCode::SECP256K1_DECOMPRESS => {
            weierstrass_decompress_syscall::<Secp256k1>(ctx, arg1, arg2)
        }
        SyscallCode::SECP256R1_ADD => unsafe {
            weierstrass_add_assign_syscall::<Secp256r1>(ctx, arg1, arg2)
        },
        SyscallCode::SECP256R1_DOUBLE => unsafe {
            weierstrass_double_assign_syscall::<Secp256r1>(ctx, arg1, arg2)
        },
        SyscallCode::SECP256R1_DECOMPRESS => {
            weierstrass_decompress_syscall::<Secp256r1>(ctx, arg1, arg2)
        }
        SyscallCode::BLS12381_ADD => unsafe {
            weierstrass_add_assign_syscall::<Bls12381>(ctx, arg1, arg2)
        },
        SyscallCode::BLS12381_DOUBLE => unsafe {
            weierstrass_double_assign_syscall::<Bls12381>(ctx, arg1, arg2)
        },
        SyscallCode::BLS12381_DECOMPRESS => {
            weierstrass_decompress_syscall::<Bls12381>(ctx, arg1, arg2)
        }
        SyscallCode::BN254_ADD => unsafe {
            weierstrass_add_assign_syscall::<Bn254>(ctx, arg1, arg2)
        },
        SyscallCode::BN254_DOUBLE => unsafe {
            weierstrass_double_assign_syscall::<Bn254>(ctx, arg1, arg2)
        },
        SyscallCode::ED_ADD => unsafe { edwards_add::<Ed25519>(ctx, arg1, arg2) },
        SyscallCode::ED_DECOMPRESS => unsafe { edwards_decompress_syscall(ctx, arg1, arg2) },
        SyscallCode::BLS12381_FP_ADD
        | SyscallCode::BLS12381_FP_SUB
        | SyscallCode::BLS12381_FP_MUL => unsafe {
            fp_op_syscall::<Bls12381BaseField>(ctx, arg1, arg2, code.fp_op_map())
        },
        SyscallCode::BLS12381_FP2_ADD | SyscallCode::BLS12381_FP2_SUB => unsafe {
            fp2_addsub_syscall::<Bls12381BaseField>(ctx, arg1, arg2, code.fp_op_map())
        },
        SyscallCode::BLS12381_FP2_MUL => unsafe {
            fp2_mul_syscall::<Bls12381BaseField>(ctx, arg1, arg2)
        },
        SyscallCode::BN254_FP_ADD | SyscallCode::BN254_FP_SUB | SyscallCode::BN254_FP_MUL => unsafe {
            fp_op_syscall::<Bn254BaseField>(ctx, arg1, arg2, code.fp_op_map())
        },
        SyscallCode::BN254_FP2_ADD | SyscallCode::BN254_FP2_SUB => unsafe {
            fp2_addsub_syscall::<Bn254BaseField>(ctx, arg1, arg2, code.fp_op_map())
        },
        SyscallCode::BN254_FP2_MUL => unsafe { fp2_mul_syscall::<Bn254BaseField>(ctx, arg1, arg2) },
        SyscallCode::UINT256_MUL => unsafe { uint256_mul(ctx, arg1, arg2) },
        SyscallCode::U256XU2048_MUL => unsafe { u256x2048_mul(ctx, arg1, arg2) },
        SyscallCode::ENTER_UNCONSTRAINED => {
            // Note we return directly from this syscall, as it does not fall the usual syscall
            // path, which is adding 256 to the clock.
            return unsafe {
                enter_unconstrained(ctx, arg1, arg2).expect("Enter unconstrained failed")
            };
        }
        SyscallCode::EXIT_UNCONSTRAINED => {
            // Note: The `exit_unconstrained` syscall does not fall through to the normal syscall
            // path because this syscall directly modifies the PC and CLK.
            let code = unsafe { exit_unconstrained(ctx, arg1, arg2) };
            ctx.pc += 4;
            ctx.clk += 256;

            return code.expect("Exit unconstrained failed");
        }
        SyscallCode::HINT_LEN => unsafe { hint_len(ctx, arg1, arg2) },
        SyscallCode::HINT_READ => unsafe { hint_read(ctx, arg1, arg2) },
        SyscallCode::WRITE => unsafe { write(ctx, arg1, arg2) },
        SyscallCode::HALT => {
            ctx.pc = 1;
            ctx.clk += 256;
            return code as u64;
        }
        SyscallCode::UINT256_MUL_CARRY | SyscallCode::UINT256_ADD_CARRY => unsafe {
            uint256_ops(ctx, arg1, arg2)
        },
        SyscallCode::POSEIDON2 => unsafe { poseidon2(ctx, arg1, arg2) },
        SyscallCode::MPROTECT
        | SyscallCode::VERIFY_SP1_PROOF
        | SyscallCode::COMMIT
        | SyscallCode::COMMIT_DEFERRED_PROOFS => None,
    };

    // Default syscall behavior
    ctx.pc += 4;
    ctx.clk = clk + 256;

    res.unwrap_or(code as u64)
}
