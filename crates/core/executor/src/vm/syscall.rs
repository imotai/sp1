use crate::{syscalls::SyscallCode, tracing::LocalMemoryAccess, TracingVM};
use sp1_curves::{
    edwards::ed25519::Ed25519,
    weierstrass::{
        bls12_381::{Bls12381, Bls12381BaseField},
        bn254::{Bn254, Bn254BaseField},
        secp256k1::Secp256k1,
        secp256r1::Secp256r1,
    },
};
use sp1_jit::{MemValue, RiscRegister};

use super::CoreVM;

mod commit;
mod deferred;
mod halt;
mod hint;
mod poseidon2;
mod precompiles;
mod u256x2048_mul;
mod uint256;
mod uint256_ops;
mod unconstrained;
mod write;

pub trait SyscallRuntime<'a> {
    fn core(&self) -> &CoreVM<'a>;
    fn core_mut(&mut self) -> &mut CoreVM<'a>;
    fn push_public_values(&mut self, value: &[u8]);
}

impl<'a> SyscallRuntime<'a> for CoreVM<'a> {
    fn core(&self) -> &CoreVM<'a> {
        self
    }

    fn core_mut(&mut self) -> &mut CoreVM<'a> {
        self
    }

    fn push_public_values(&mut self, _: &[u8]) {
        // no op by default.
    }
}

/// The default syscall handler for the core VM.
///
/// Note that mostly syscalls actually do nothing in the core VM.
pub(crate) fn core_syscall_handler<'a, RT: SyscallRuntime<'a>>(
    rt: &mut RT,
    code: SyscallCode,
    args1: u64,
    args2: u64,
) -> Option<u64> {
    match code {
        SyscallCode::HINT_LEN => hint::hint_len_syscall(rt, code, args1, args2),
        SyscallCode::HALT => halt::halt_syscall(rt, code, args1, args2),
        SyscallCode::WRITE => write::write_syscall(rt, code, args1, args2),
        SyscallCode::SECP256K1_ADD => {
            precompiles::weirstrass::core_weirstrass_add::<RT, Secp256k1>(rt, code, args1, args2)
        }
        SyscallCode::SECP256K1_DOUBLE => {
            precompiles::weirstrass::core_weirstrass_double::<RT, Secp256k1>(rt, code, args1, args2)
        }
        SyscallCode::BLS12381_ADD => {
            precompiles::weirstrass::core_weirstrass_add::<RT, Bls12381>(rt, code, args1, args2)
        }
        SyscallCode::BLS12381_DOUBLE => {
            precompiles::weirstrass::core_weirstrass_double::<RT, Bls12381>(rt, code, args1, args2)
        }
        SyscallCode::BN254_ADD => {
            precompiles::weirstrass::core_weirstrass_add::<RT, Bn254>(rt, code, args1, args2)
        }
        SyscallCode::BN254_DOUBLE => {
            precompiles::weirstrass::core_weirstrass_double::<RT, Bn254>(rt, code, args1, args2)
        }
        SyscallCode::SECP256R1_ADD => {
            precompiles::weirstrass::core_weirstrass_add::<RT, Secp256r1>(rt, code, args1, args2)
        }
        SyscallCode::SECP256R1_DOUBLE => {
            precompiles::weirstrass::core_weirstrass_double::<RT, Secp256r1>(rt, code, args1, args2)
        }
        // Edwards curve operations
        SyscallCode::ED_ADD => {
            precompiles::edwards::core_edwards_add::<RT, Ed25519>(rt, code, args1, args2)
        }
        SyscallCode::ED_DECOMPRESS => {
            precompiles::edwards::core_edwards_decompress::<RT, Ed25519>(rt, code, args1, args2)
        }
        SyscallCode::UINT256_MUL => uint256::core_uint256_mul(rt, code, args1, args2),
        SyscallCode::UINT256_MUL_CARRY | SyscallCode::UINT256_ADD_CARRY => {
            uint256_ops::core_uint256_ops(rt, code, args1, args2)
        }
        SyscallCode::U256XU2048_MUL => u256x2048_mul::core_u256xu2048_mul(rt, code, args1, args2),
        SyscallCode::SHA_COMPRESS => {
            precompiles::sha256::core_sha256_compress(rt, code, args1, args2)
        }
        SyscallCode::SHA_EXTEND => precompiles::sha256::core_sha256_extend(rt, code, args1, args2),
        SyscallCode::KECCAK_PERMUTE => {
            precompiles::keccak256::core_keccak256_permute(rt, code, args1, args2)
        }
        SyscallCode::BLS12381_FP2_ADD | SyscallCode::BLS12381_FP2_SUB => {
            precompiles::fptower::core_fp2_add::<RT, Bls12381BaseField>(rt, code, args1, args2)
        }
        SyscallCode::BN254_FP2_ADD | SyscallCode::BN254_FP2_SUB => {
            precompiles::fptower::core_fp2_add::<RT, Bn254BaseField>(rt, code, args1, args2)
        }
        SyscallCode::BLS12381_FP2_MUL => {
            precompiles::fptower::core_fp2_mul::<RT, Bls12381BaseField>(rt, code, args1, args2)
        }
        SyscallCode::BN254_FP2_MUL => {
            precompiles::fptower::core_fp2_mul::<RT, Bn254BaseField>(rt, code, args1, args2)
        }
        SyscallCode::BLS12381_FP_ADD
        | SyscallCode::BLS12381_FP_SUB
        | SyscallCode::BLS12381_FP_MUL => {
            precompiles::fptower::core_fp_op::<RT, Bls12381BaseField>(rt, code, args1, args2)
        }
        SyscallCode::BN254_FP_ADD | SyscallCode::BN254_FP_SUB | SyscallCode::BN254_FP_MUL => {
            precompiles::fptower::core_fp_op::<RT, Bn254BaseField>(rt, code, args1, args2)
        }
        SyscallCode::ENTER_UNCONSTRAINED => {
            unconstrained::enter_unconstrained(rt, code, args1, args2)
        }
        SyscallCode::EXIT_UNCONSTRAINED => {
            unconstrained::exit_unconstrained(rt, code, args1, args2).1
        }
        SyscallCode::POSEIDON2 => poseidon2::core_poseidon2(rt, code, args1, args2),
        SyscallCode::VERIFY_SP1_PROOF | SyscallCode::MPROTECT => None,
        _ => None,
    }
}

pub(crate) fn tracing_syscall_handler(
    rt: &mut TracingVM<'_>,
    code: SyscallCode,
    args1: u64,
    args2: u64,
) -> Option<u64> {
    // If the syscall is not retained, we need to track the local memory access separately.
    if rt.core().is_retained_syscall(code) {
        rt.precompile_local_memory_access = None;
    } else {
        rt.precompile_local_memory_access = Some(LocalMemoryAccess::default());
    }

    let mut clk = rt.core.clk();

    #[allow(clippy::match_same_arms)]
    let ret = match code {
        // Noop: This method just writes to uninitialized memory.
        // Since the tracing VM relies on oracled memory, this method is a no-op.
        SyscallCode::HINT_READ => None,
        SyscallCode::HINT_LEN => hint::hint_len_syscall(rt, code, args1, args2),
        SyscallCode::HALT => halt::halt_syscall(rt, code, args1, args2),
        SyscallCode::WRITE => write::write_syscall(rt, code, args1, args2),
        SyscallCode::COMMIT => commit::commit_syscall(rt, code, args1, args2),
        SyscallCode::COMMIT_DEFERRED_PROOFS => {
            deferred::commit_deferred_proofs_syscall(rt, code, args1, args2)
        }
        // Weierstrass curve operations
        SyscallCode::SECP256K1_ADD => {
            precompiles::weirstrass::tracing_weirstrass_add::<Secp256k1>(rt, code, args1, args2)
        }
        SyscallCode::SECP256K1_DOUBLE => {
            precompiles::weirstrass::tracing_weirstrass_double::<Secp256k1>(rt, code, args1, args2)
        }
        SyscallCode::BLS12381_ADD => {
            precompiles::weirstrass::tracing_weirstrass_add::<Bls12381>(rt, code, args1, args2)
        }
        SyscallCode::BLS12381_DOUBLE => {
            precompiles::weirstrass::tracing_weirstrass_double::<Bls12381>(rt, code, args1, args2)
        }
        SyscallCode::BN254_ADD => {
            precompiles::weirstrass::tracing_weirstrass_add::<Bn254>(rt, code, args1, args2)
        }
        SyscallCode::BN254_DOUBLE => {
            precompiles::weirstrass::tracing_weirstrass_double::<Bn254>(rt, code, args1, args2)
        }
        SyscallCode::SECP256R1_ADD => {
            precompiles::weirstrass::tracing_weirstrass_add::<Secp256r1>(rt, code, args1, args2)
        }
        SyscallCode::SECP256R1_DOUBLE => {
            precompiles::weirstrass::tracing_weirstrass_double::<Secp256r1>(rt, code, args1, args2)
        }
        // Edwards curve operations
        SyscallCode::ED_ADD => {
            precompiles::edwards::tracing_edwards_add::<Ed25519>(rt, code, args1, args2)
        }
        SyscallCode::ED_DECOMPRESS => {
            precompiles::edwards::tracing_edwards_decompress::<Ed25519>(rt, code, args1, args2)
        }
        SyscallCode::UINT256_MUL => uint256::tracing_uint256_mul(rt, code, args1, args2),
        SyscallCode::UINT256_MUL_CARRY | SyscallCode::UINT256_ADD_CARRY => {
            uint256_ops::tracing_uint256_ops(rt, code, args1, args2)
        }
        SyscallCode::U256XU2048_MUL => {
            u256x2048_mul::tracing_u256xu2048_mul(rt, code, args1, args2)
        }
        SyscallCode::SHA_COMPRESS => {
            precompiles::sha256::tracing_sha256_compress(rt, code, args1, args2)
        }
        SyscallCode::SHA_EXTEND => {
            precompiles::sha256::tracing_sha256_extend(rt, code, args1, args2)
        }
        SyscallCode::KECCAK_PERMUTE => {
            precompiles::keccak256::tracing_keccak256_permute(rt, code, args1, args2)
        }
        SyscallCode::BLS12381_FP2_ADD | SyscallCode::BLS12381_FP2_SUB => {
            precompiles::fptower::tracing_fp2_add::<Bls12381BaseField>(rt, code, args1, args2)
        }
        SyscallCode::BN254_FP2_ADD | SyscallCode::BN254_FP2_SUB => {
            precompiles::fptower::tracing_fp2_add::<Bn254BaseField>(rt, code, args1, args2)
        }
        SyscallCode::BLS12381_FP2_MUL => {
            precompiles::fptower::tracing_fp2_mul::<Bls12381BaseField>(rt, code, args1, args2)
        }
        SyscallCode::BN254_FP2_MUL => {
            precompiles::fptower::tracing_fp2_mul::<Bn254BaseField>(rt, code, args1, args2)
        }
        SyscallCode::BLS12381_FP_ADD
        | SyscallCode::BLS12381_FP_SUB
        | SyscallCode::BLS12381_FP_MUL => {
            precompiles::fptower::tracing_fp_op::<Bls12381BaseField>(rt, code, args1, args2)
        }
        SyscallCode::BN254_FP_ADD | SyscallCode::BN254_FP_SUB | SyscallCode::BN254_FP_MUL => {
            precompiles::fptower::tracing_fp_op::<Bn254BaseField>(rt, code, args1, args2)
        }
        SyscallCode::ENTER_UNCONSTRAINED => {
            unconstrained::enter_unconstrained(rt, code, args1, args2)
        }
        SyscallCode::EXIT_UNCONSTRAINED => {
            let (new_clk, ret_val) = unconstrained::exit_unconstrained(rt, code, args1, args2);
            clk = new_clk;
            ret_val
        }
        SyscallCode::POSEIDON2 => poseidon2::tracing_poseidon2(rt, code, args1, args2),
        SyscallCode::VERIFY_SP1_PROOF | SyscallCode::MPROTECT => None,
        _ => None,
    };

    rt.core.set_clk(clk);

    ret
}
