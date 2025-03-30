pub use riscv_chips::*;

use core::fmt;
use std::collections::BTreeSet;

use hashbrown::{HashMap, HashSet};
use itertools::Itertools;
use p3_field::PrimeField32;
use sp1_core_executor::{
    events::PrecompileLocalMemory, syscalls::SyscallCode, ExecutionRecord, Program, RiscvAirId,
};
use sp1_curves::weierstrass::{bls12_381::Bls12381BaseField, bn254::Bn254BaseField};
use sp1_stark::{
    air::{InteractionScope, MachineAir, SP1_PROOF_NUM_PV_ELTS},
    Chip, InteractionKind, Machine, MachineShape,
};
use strum_macros::{EnumDiscriminants, EnumIter};

use crate::{
    bytes::trace::NUM_ROWS as BYTE_CHIP_NUM_ROWS,
    control_flow::{AuipcChip, BranchChip, JalChip, JalrChip},
    global::GlobalChip,
    memory::{
        load::{
            load_byte::LoadByteChip, load_half::LoadHalfChip, load_word::LoadWordChip,
            load_x0::LoadX0Chip,
        },
        store::{store_byte::StoreByteChip, store_half::StoreHalfChip, store_word::StoreWordChip},
        MemoryChipType, MemoryLocalChip, NUM_LOCAL_MEMORY_ENTRIES_PER_ROW,
    },
    range::{trace::NUM_ROWS as RANGE_CHIP_NUM_ROWS, RangeChip},
    syscall::{
        instructions::SyscallInstrsChip,
        precompiles::fptower::{Fp2AddSubAssignChip, Fp2MulAssignChip, FpOpChip},
    },
};

/// A module for importing all the different RISC-V chips.
pub(crate) mod riscv_chips {
    pub use crate::{
        alu::{
            add::AddChip, addi::AddiChip, sub::SubChip, BitwiseChip, DivRemChip, LtChip, MulChip,
            ShiftLeft, ShiftRightChip,
        },
        bytes::ByteChip,
        cpu::CpuChip,
        memory::MemoryGlobalChip,
        program::ProgramChip,
        syscall::{
            chip::SyscallChip,
            precompiles::{
                edwards::{EdAddAssignChip, EdDecompressChip},
                keccak256::{KeccakPermuteChip, KeccakPermuteControlChip},
                sha256::{
                    ShaCompressChip, ShaCompressControlChip, ShaExtendChip, ShaExtendControlChip,
                },
                u256x2048_mul::U256x2048MulChip,
                uint256::Uint256MulChip,
                weierstrass::{
                    WeierstrassAddAssignChip, WeierstrassDecompressChip,
                    WeierstrassDoubleAssignChip,
                },
            },
        },
    };
    pub use sp1_curves::{
        edwards::{ed25519::Ed25519Parameters, EdwardsCurve},
        weierstrass::{
            bls12_381::Bls12381Parameters, bn254::Bn254Parameters, secp256k1::Secp256k1Parameters,
            secp256r1::Secp256r1Parameters, SwCurve,
        },
    };
}

/// The maximum log number of shards in core.
pub const MAX_LOG_NUMBER_OF_SHARDS: usize = 24;

/// The maximum number of shards in core.
pub const MAX_NUMBER_OF_SHARDS: usize = 1 << MAX_LOG_NUMBER_OF_SHARDS;

/// An AIR for encoding RISC-V execution.
///
/// This enum contains all the different AIRs that are used in the Sp1 RISC-V IOP. Each variant is
/// a different AIR that is used to encode a different part of the RISC-V execution, and the
/// different AIR variants have a joint lookup argument.
#[derive(sp1_derive::MachineAir, EnumDiscriminants)]
#[strum_discriminants(derive(Hash, EnumIter))]
pub enum RiscvAir<F: PrimeField32> {
    /// An AIR that contains a preprocessed program table and a lookup for the instructions.
    Program(ProgramChip),
    /// An AIR for the RISC-V CPU. Each row represents a cpu cycle.
    // Cpu(CpuChip),
    /// An AIR for the RISC-V Add instruction.
    Add(AddChip),
    /// An AIR for the RISC-V Addi instruction.
    Addi(AddiChip),
    // An AIR for the RISC-V Sub instruction.
    Sub(SubChip),
    /// An AIR for RISC-V Bitwise instructions.
    Bitwise(BitwiseChip),
    /// An AIR for RISC-V Mul instruction.
    Mul(MulChip),
    /// An AIR for RISC-V Div and Rem instructions.
    DivRem(DivRemChip),
    /// An AIR for RISC-V Lt instruction.
    Lt(LtChip),
    /// An AIR for RISC-V SLL instruction.
    ShiftLeft(ShiftLeft),
    /// An AIR for RISC-V SRL and SRA instruction.
    ShiftRight(ShiftRightChip),
    /// An AIR for RISC-V memory load byte instructions.
    LoadByte(LoadByteChip),
    /// An AIR for RISC-V memory load half instructions.
    LoadHalf(LoadHalfChip),
    /// An AIR for RISC-V memory load word instructions.
    LoadWord(LoadWordChip),
    /// An AIR for RISC-V memory load x0 instructions.
    LoadX0(LoadX0Chip),
    /// An AIR for RISC-V memory store byte instructions.
    StoreByte(StoreByteChip),
    /// An AIR for RISC-V memory store half instructions.
    StoreHalf(StoreHalfChip),
    /// An AIR for RISC-V memory store word instructions.
    StoreWord(StoreWordChip),
    /// An AIR for RISC-V AUIPC instruction.
    AUIPC(AuipcChip),
    /// An AIR for RISC-V branch instructions.
    Branch(BranchChip),
    /// An AIR for RISC-V jal instructions.
    Jal(JalChip),
    /// An AIR for RISC-V jalr instructions.
    Jalr(JalrChip),
    /// An AIR for RISC-V ecall instructions.
    SyscallInstrs(SyscallInstrsChip),
    /// A lookup table for byte operations.
    ByteLookup(ByteChip<F>),
    /// A lookup table for range operations.
    RangeLookup(RangeChip<F>),
    /// A table for initializing the global memory state.
    MemoryGlobalInit(MemoryGlobalChip),
    /// A table for finalizing the global memory state.
    MemoryGlobalFinal(MemoryGlobalChip),
    /// A table for the local memory state.
    MemoryLocal(MemoryLocalChip),
    /// A table for all the syscall invocations.
    SyscallCore(SyscallChip),
    /// A table for all the precompile invocations.
    SyscallPrecompile(SyscallChip),
    /// A table for all the global interactions.
    Global(GlobalChip),
    /// A precompile for sha256 extend.
    Sha256Extend(ShaExtendChip),
    /// A controller for sha256 extend.
    Sha256ExtendControl(ShaExtendControlChip),
    /// A precompile for sha256 compress.
    Sha256Compress(ShaCompressChip),
    /// A controller for sha256 compress.
    Sha256CompressControl(ShaCompressControlChip),
    /// A precompile for addition on the Elliptic curve ed25519.
    Ed25519Add(EdAddAssignChip<EdwardsCurve<Ed25519Parameters>>),
    /// A precompile for decompressing a point on the Edwards curve ed25519.
    Ed25519Decompress(EdDecompressChip<Ed25519Parameters>),
    /// A precompile for decompressing a point on the K256 curve.
    K256Decompress(WeierstrassDecompressChip<SwCurve<Secp256k1Parameters>>),
    /// A precompile for decompressing a point on the P256 curve.
    P256Decompress(WeierstrassDecompressChip<SwCurve<Secp256r1Parameters>>),
    /// A precompile for addition on the Elliptic curve secp256k1.
    Secp256k1Add(WeierstrassAddAssignChip<SwCurve<Secp256k1Parameters>>),
    /// A precompile for doubling a point on the Elliptic curve secp256k1.
    Secp256k1Double(WeierstrassDoubleAssignChip<SwCurve<Secp256k1Parameters>>),
    /// A precompile for addition on the Elliptic curve secp256r1.
    Secp256r1Add(WeierstrassAddAssignChip<SwCurve<Secp256r1Parameters>>),
    /// A precompile for doubling a point on the Elliptic curve secp256r1.
    Secp256r1Double(WeierstrassDoubleAssignChip<SwCurve<Secp256r1Parameters>>),
    /// A precompile for the Keccak permutation.
    KeccakP(KeccakPermuteChip),
    /// A controller for the Keccak permutation.
    KeccakPControl(KeccakPermuteControlChip),
    /// A precompile for addition on the Elliptic curve bn254.
    Bn254Add(WeierstrassAddAssignChip<SwCurve<Bn254Parameters>>),
    /// A precompile for doubling a point on the Elliptic curve bn254.
    Bn254Double(WeierstrassDoubleAssignChip<SwCurve<Bn254Parameters>>),
    /// A precompile for addition on the Elliptic curve bls12_381.
    Bls12381Add(WeierstrassAddAssignChip<SwCurve<Bls12381Parameters>>),
    /// A precompile for doubling a point on the Elliptic curve bls12_381.
    Bls12381Double(WeierstrassDoubleAssignChip<SwCurve<Bls12381Parameters>>),
    /// A precompile for uint256 mul.
    Uint256Mul(Uint256MulChip),
    /// A precompile for u256x2048 mul.
    U256x2048Mul(U256x2048MulChip),
    /// A precompile for decompressing a point on the BLS12-381 curve.
    Bls12381Decompress(WeierstrassDecompressChip<SwCurve<Bls12381Parameters>>),
    /// A precompile for BLS12-381 fp operation.
    Bls12381Fp(FpOpChip<Bls12381BaseField>),
    /// A precompile for BLS12-381 fp2 multiplication.
    Bls12381Fp2Mul(Fp2MulAssignChip<Bls12381BaseField>),
    /// A precompile for BLS12-381 fp2 addition/subtraction.
    Bls12381Fp2AddSub(Fp2AddSubAssignChip<Bls12381BaseField>),
    /// A precompile for BN-254 fp operation.
    Bn254Fp(FpOpChip<Bn254BaseField>),
    /// A precompile for BN-254 fp2 multiplication.
    Bn254Fp2Mul(Fp2MulAssignChip<Bn254BaseField>),
    /// A precompile for BN-254 fp2 addition/subtraction.
    Bn254Fp2AddSub(Fp2AddSubAssignChip<Bn254BaseField>),
}

impl<F: PrimeField32> RiscvAir<F> {
    pub fn machine() -> Machine<F, Self> {
        let mut preprocessed_chips = BTreeSet::new();
        let mut core_chips = BTreeSet::new();
        let mut memory_chips = BTreeSet::new();
        let mut precompile_clusters = Vec::new();
        let mut base_precompile_cluster = BTreeSet::new();

        // The order of the chips is used to determine the order of trace generation.
        let mut chips = vec![];

        let program = Chip::new(RiscvAir::Program(ProgramChip::default()));
        chips.push(program.clone());
        preprocessed_chips.insert(program);

        let sha_extend = Chip::new(RiscvAir::Sha256Extend(ShaExtendChip::default()));
        chips.push(sha_extend.clone());
        let sha_extend_control =
            Chip::new(RiscvAir::Sha256ExtendControl(ShaExtendControlChip::default()));
        chips.push(sha_extend_control.clone());
        precompile_clusters.push(BTreeSet::from([sha_extend, sha_extend_control]));

        let sha_compress = Chip::new(RiscvAir::Sha256Compress(ShaCompressChip::default()));
        chips.push(sha_compress.clone());
        let sha_compress_control =
            Chip::new(RiscvAir::Sha256CompressControl(ShaCompressControlChip::default()));
        chips.push(sha_compress_control.clone());
        precompile_clusters.push(BTreeSet::from([sha_compress, sha_compress_control]));

        let ed_add_assign = Chip::new(RiscvAir::Ed25519Add(EdAddAssignChip::<
            EdwardsCurve<Ed25519Parameters>,
        >::new()));
        chips.push(ed_add_assign.clone());
        precompile_clusters.push(BTreeSet::from([ed_add_assign]));

        let ed_decompress = Chip::new(RiscvAir::Ed25519Decompress(EdDecompressChip::<
            Ed25519Parameters,
        >::default()));
        chips.push(ed_decompress.clone());
        precompile_clusters.push(BTreeSet::from([ed_decompress]));

        let k256_decompress = Chip::new(RiscvAir::K256Decompress(WeierstrassDecompressChip::<
            SwCurve<Secp256k1Parameters>,
        >::with_lsb_rule()));
        chips.push(k256_decompress.clone());
        precompile_clusters.push(BTreeSet::from([k256_decompress]));

        let secp256k1_add_assign = Chip::new(RiscvAir::Secp256k1Add(WeierstrassAddAssignChip::<
            SwCurve<Secp256k1Parameters>,
        >::new()));
        chips.push(secp256k1_add_assign.clone());
        precompile_clusters.push(BTreeSet::from([secp256k1_add_assign]));

        let secp256k1_double_assign =
            Chip::new(RiscvAir::Secp256k1Double(WeierstrassDoubleAssignChip::<
                SwCurve<Secp256k1Parameters>,
            >::new()));
        chips.push(secp256k1_double_assign.clone());
        precompile_clusters.push(BTreeSet::from([secp256k1_double_assign]));

        let p256_decompress = Chip::new(RiscvAir::P256Decompress(WeierstrassDecompressChip::<
            SwCurve<Secp256r1Parameters>,
        >::with_lsb_rule()));
        chips.push(p256_decompress.clone());
        precompile_clusters.push(BTreeSet::from([p256_decompress]));

        let secp256r1_add_assign = Chip::new(RiscvAir::Secp256r1Add(WeierstrassAddAssignChip::<
            SwCurve<Secp256r1Parameters>,
        >::new()));
        chips.push(secp256r1_add_assign.clone());
        precompile_clusters.push(BTreeSet::from([secp256r1_add_assign]));

        let secp256r1_double_assign =
            Chip::new(RiscvAir::Secp256r1Double(WeierstrassDoubleAssignChip::<
                SwCurve<Secp256r1Parameters>,
            >::new()));
        chips.push(secp256r1_double_assign.clone());
        precompile_clusters.push(BTreeSet::from([secp256r1_double_assign]));

        let keccak_permute = Chip::new(RiscvAir::KeccakP(KeccakPermuteChip::new()));
        chips.push(keccak_permute.clone());
        let keccak_permute_control =
            Chip::new(RiscvAir::KeccakPControl(KeccakPermuteControlChip::new()));
        chips.push(keccak_permute_control.clone());
        precompile_clusters.push(BTreeSet::from([keccak_permute_control, keccak_permute]));

        let bn254_add_assign = Chip::new(RiscvAir::Bn254Add(WeierstrassAddAssignChip::<
            SwCurve<Bn254Parameters>,
        >::new()));
        chips.push(bn254_add_assign.clone());
        precompile_clusters.push(BTreeSet::from([bn254_add_assign]));

        let bn254_double_assign = Chip::new(RiscvAir::Bn254Double(WeierstrassDoubleAssignChip::<
            SwCurve<Bn254Parameters>,
        >::new()));
        chips.push(bn254_double_assign.clone());
        precompile_clusters.push(BTreeSet::from([bn254_double_assign]));

        let bls12381_add = Chip::new(RiscvAir::Bls12381Add(WeierstrassAddAssignChip::<
            SwCurve<Bls12381Parameters>,
        >::new()));
        chips.push(bls12381_add.clone());
        precompile_clusters.push(BTreeSet::from([bls12381_add]));

        let bls12381_double = Chip::new(RiscvAir::Bls12381Double(WeierstrassDoubleAssignChip::<
            SwCurve<Bls12381Parameters>,
        >::new()));
        chips.push(bls12381_double.clone());
        precompile_clusters.push(BTreeSet::from([bls12381_double]));

        let uint256_mul = Chip::new(RiscvAir::Uint256Mul(Uint256MulChip::default()));
        chips.push(uint256_mul.clone());
        precompile_clusters.push(BTreeSet::from([uint256_mul]));

        let u256x2048_mul = Chip::new(RiscvAir::U256x2048Mul(U256x2048MulChip::default()));
        chips.push(u256x2048_mul.clone());
        precompile_clusters.push(BTreeSet::from([u256x2048_mul]));

        let bls12381_fp = Chip::new(RiscvAir::Bls12381Fp(FpOpChip::<Bls12381BaseField>::new()));
        chips.push(bls12381_fp.clone());
        precompile_clusters.push(BTreeSet::from([bls12381_fp]));

        let bls12381_fp2_addsub =
            Chip::new(RiscvAir::Bls12381Fp2AddSub(Fp2AddSubAssignChip::<Bls12381BaseField>::new()));
        chips.push(bls12381_fp2_addsub.clone());
        precompile_clusters.push(BTreeSet::from([bls12381_fp2_addsub]));

        let bls12381_fp2_mul =
            Chip::new(RiscvAir::Bls12381Fp2Mul(Fp2MulAssignChip::<Bls12381BaseField>::new()));
        chips.push(bls12381_fp2_mul.clone());
        precompile_clusters.push(BTreeSet::from([bls12381_fp2_mul]));

        let bn254_fp = Chip::new(RiscvAir::Bn254Fp(FpOpChip::<Bn254BaseField>::new()));
        chips.push(bn254_fp.clone());
        precompile_clusters.push(BTreeSet::from([bn254_fp]));

        let bn254_fp2_addsub =
            Chip::new(RiscvAir::Bn254Fp2AddSub(Fp2AddSubAssignChip::<Bn254BaseField>::new()));
        chips.push(bn254_fp2_addsub.clone());
        precompile_clusters.push(BTreeSet::from([bn254_fp2_addsub]));

        let bn254_fp2_mul =
            Chip::new(RiscvAir::Bn254Fp2Mul(Fp2MulAssignChip::<Bn254BaseField>::new()));
        chips.push(bn254_fp2_mul.clone());
        precompile_clusters.push(BTreeSet::from([bn254_fp2_mul]));

        let bls12381_decompress =
            Chip::new(RiscvAir::Bls12381Decompress(WeierstrassDecompressChip::<
                SwCurve<Bls12381Parameters>,
            >::with_lexicographic_rule()));
        chips.push(bls12381_decompress.clone());
        precompile_clusters.push(BTreeSet::from([bls12381_decompress]));

        let syscall_core = Chip::new(RiscvAir::SyscallCore(SyscallChip::core()));
        chips.push(syscall_core.clone());
        core_chips.insert(syscall_core);

        let syscall_precompile = Chip::new(RiscvAir::SyscallPrecompile(SyscallChip::precompile()));
        chips.push(syscall_precompile.clone());
        base_precompile_cluster.insert(syscall_precompile);

        let div_rem = Chip::new(RiscvAir::DivRem(DivRemChip::default()));
        chips.push(div_rem.clone());
        core_chips.insert(div_rem);

        let add = Chip::new(RiscvAir::Add(AddChip::default()));
        chips.push(add.clone());
        core_chips.insert(add);

        let addi = Chip::new(RiscvAir::Addi(AddiChip::default()));
        chips.push(addi.clone());
        core_chips.insert(addi);

        let sub = Chip::new(RiscvAir::Sub(SubChip::default()));
        chips.push(sub.clone());
        core_chips.insert(sub);

        let bitwise = Chip::new(RiscvAir::Bitwise(BitwiseChip::default()));
        chips.push(bitwise.clone());
        core_chips.insert(bitwise);

        let mul = Chip::new(RiscvAir::Mul(MulChip::default()));
        chips.push(mul.clone());
        core_chips.insert(mul);

        let shift_right = Chip::new(RiscvAir::ShiftRight(ShiftRightChip::default()));
        chips.push(shift_right.clone());
        core_chips.insert(shift_right);

        let shift_left = Chip::new(RiscvAir::ShiftLeft(ShiftLeft::default()));
        chips.push(shift_left.clone());
        core_chips.insert(shift_left);

        let lt = Chip::new(RiscvAir::Lt(LtChip::default()));
        chips.push(lt.clone());
        core_chips.insert(lt);

        let load_byte = Chip::new(RiscvAir::LoadByte(LoadByteChip::default()));
        chips.push(load_byte.clone());
        core_chips.insert(load_byte);

        let load_half = Chip::new(RiscvAir::LoadHalf(LoadHalfChip::default()));
        chips.push(load_half.clone());
        core_chips.insert(load_half);

        let load_word = Chip::new(RiscvAir::LoadWord(LoadWordChip::default()));
        chips.push(load_word.clone());
        core_chips.insert(load_word);

        let load_x0 = Chip::new(RiscvAir::LoadX0(LoadX0Chip::default()));
        chips.push(load_x0.clone());
        core_chips.insert(load_x0);

        let store_byte = Chip::new(RiscvAir::StoreByte(StoreByteChip::default()));
        chips.push(store_byte.clone());
        core_chips.insert(store_byte);

        let store_half = Chip::new(RiscvAir::StoreHalf(StoreHalfChip::default()));
        chips.push(store_half.clone());
        core_chips.insert(store_half);

        let store_word = Chip::new(RiscvAir::StoreWord(StoreWordChip::default()));
        chips.push(store_word.clone());
        core_chips.insert(store_word);

        let auipc = Chip::new(RiscvAir::AUIPC(AuipcChip::default()));
        chips.push(auipc.clone());
        core_chips.insert(auipc);

        let branch = Chip::new(RiscvAir::Branch(BranchChip::default()));
        chips.push(branch.clone());
        core_chips.insert(branch);

        let jal = Chip::new(RiscvAir::Jal(JalChip::default()));
        chips.push(jal.clone());
        core_chips.insert(jal);

        let jalr = Chip::new(RiscvAir::Jalr(JalrChip::default()));
        chips.push(jalr.clone());
        core_chips.insert(jalr);

        let syscall_instrs = Chip::new(RiscvAir::SyscallInstrs(SyscallInstrsChip::default()));
        chips.push(syscall_instrs.clone());
        core_chips.insert(syscall_instrs);

        let memory_global_init = Chip::new(RiscvAir::MemoryGlobalInit(MemoryGlobalChip::new(
            MemoryChipType::Initialize,
        )));
        chips.push(memory_global_init.clone());
        memory_chips.insert(memory_global_init);

        let memory_global_finalize =
            Chip::new(RiscvAir::MemoryGlobalFinal(MemoryGlobalChip::new(MemoryChipType::Finalize)));
        chips.push(memory_global_finalize.clone());
        memory_chips.insert(memory_global_finalize);

        let memory_local = Chip::new(RiscvAir::MemoryLocal(MemoryLocalChip::new()));
        chips.push(memory_local.clone());
        core_chips.insert(memory_local.clone());
        base_precompile_cluster.insert(memory_local);

        let global = Chip::new(RiscvAir::Global(GlobalChip));
        chips.push(global.clone());
        core_chips.insert(global.clone());
        base_precompile_cluster.insert(global.clone());
        memory_chips.insert(global.clone());

        let byte = Chip::new(RiscvAir::ByteLookup(ByteChip::default()));
        chips.push(byte.clone());
        preprocessed_chips.insert(byte);

        let range = Chip::new(RiscvAir::RangeLookup(RangeChip::default()));
        chips.push(range.clone());
        preprocessed_chips.insert(range);

        // Insert all preprocessed chips to the core cluster.
        core_chips.extend(preprocessed_chips.clone());
        memory_chips.extend(preprocessed_chips.clone());
        base_precompile_cluster.extend(preprocessed_chips.clone());

        for cluster in precompile_clusters.iter_mut() {
            cluster.extend(base_precompile_cluster.clone());
        }

        let mut chip_clusters = vec![core_chips, memory_chips];
        chip_clusters.extend(precompile_clusters);
        let shape = MachineShape::new(chip_clusters);

        Machine::new(chips, SP1_PROOF_NUM_PV_ELTS, shape)
    }

    /// Get all the different RISC-V AIRs.
    pub fn chips() -> Vec<Chip<F, Self>> {
        let (chips, _) = Self::get_chips_and_costs();
        chips
    }

    /// Get all the costs of the different RISC-V AIRs.
    pub fn costs() -> HashMap<String, u64> {
        let (_, costs) = Self::get_chips_and_costs();
        costs
    }

    /// Get all the different RISC-V AIRs and their costs.
    pub fn get_airs_and_costs() -> (Vec<Self>, HashMap<String, u64>) {
        let (chips, costs) = Self::get_chips_and_costs();
        (chips.into_iter().map(|chip| chip.into_inner().unwrap()).collect(), costs)
    }

    /// Get all the different RISC-V chips and their costs.
    pub fn get_chips_and_costs() -> (Vec<Chip<F, Self>>, HashMap<String, u64>) {
        let mut costs: HashMap<String, u64> = HashMap::new();

        // The order of the chips is used to determine the order of trace generation.
        let mut chips = vec![];

        let program = Chip::new(RiscvAir::Program(ProgramChip::default()));
        costs.insert(program.name(), program.cost());
        chips.push(program);

        let sha_extend = Chip::new(RiscvAir::Sha256Extend(ShaExtendChip::default()));
        costs.insert(sha_extend.name(), 48 * sha_extend.cost());
        chips.push(sha_extend);

        let sha_extend_control =
            Chip::new(RiscvAir::Sha256ExtendControl(ShaExtendControlChip::default()));
        costs.insert(sha_extend_control.name(), sha_extend_control.cost());
        chips.push(sha_extend_control);

        let sha_compress = Chip::new(RiscvAir::Sha256Compress(ShaCompressChip::default()));
        costs.insert(sha_compress.name(), 80 * sha_compress.cost());
        chips.push(sha_compress);

        let sha_compress_control =
            Chip::new(RiscvAir::Sha256CompressControl(ShaCompressControlChip::default()));
        costs.insert(sha_compress_control.name(), sha_compress_control.cost());
        chips.push(sha_compress_control);

        let ed_add_assign = Chip::new(RiscvAir::Ed25519Add(EdAddAssignChip::<
            EdwardsCurve<Ed25519Parameters>,
        >::new()));
        costs.insert(ed_add_assign.name(), ed_add_assign.cost());
        chips.push(ed_add_assign);

        let ed_decompress = Chip::new(RiscvAir::Ed25519Decompress(EdDecompressChip::<
            Ed25519Parameters,
        >::default()));
        costs.insert(ed_decompress.name(), ed_decompress.cost());
        chips.push(ed_decompress);

        let k256_decompress = Chip::new(RiscvAir::K256Decompress(WeierstrassDecompressChip::<
            SwCurve<Secp256k1Parameters>,
        >::with_lsb_rule()));
        costs.insert(k256_decompress.name(), k256_decompress.cost());
        chips.push(k256_decompress);

        let secp256k1_add_assign = Chip::new(RiscvAir::Secp256k1Add(WeierstrassAddAssignChip::<
            SwCurve<Secp256k1Parameters>,
        >::new()));
        costs.insert(secp256k1_add_assign.name(), secp256k1_add_assign.cost());
        chips.push(secp256k1_add_assign);

        let secp256k1_double_assign =
            Chip::new(RiscvAir::Secp256k1Double(WeierstrassDoubleAssignChip::<
                SwCurve<Secp256k1Parameters>,
            >::new()));
        costs.insert(secp256k1_double_assign.name(), secp256k1_double_assign.cost());
        chips.push(secp256k1_double_assign);

        let p256_decompress = Chip::new(RiscvAir::P256Decompress(WeierstrassDecompressChip::<
            SwCurve<Secp256r1Parameters>,
        >::with_lsb_rule()));
        costs.insert(p256_decompress.name(), p256_decompress.cost());
        chips.push(p256_decompress);

        let secp256r1_add_assign = Chip::new(RiscvAir::Secp256r1Add(WeierstrassAddAssignChip::<
            SwCurve<Secp256r1Parameters>,
        >::new()));
        costs.insert(secp256r1_add_assign.name(), secp256r1_add_assign.cost());
        chips.push(secp256r1_add_assign);

        let secp256r1_double_assign =
            Chip::new(RiscvAir::Secp256r1Double(WeierstrassDoubleAssignChip::<
                SwCurve<Secp256r1Parameters>,
            >::new()));
        costs.insert(secp256r1_double_assign.name(), secp256r1_double_assign.cost());
        chips.push(secp256r1_double_assign);

        let keccak_permute = Chip::new(RiscvAir::KeccakP(KeccakPermuteChip::new()));
        costs.insert(keccak_permute.name(), 24 * keccak_permute.cost());
        chips.push(keccak_permute);

        let keccak_control = Chip::new(RiscvAir::KeccakPControl(KeccakPermuteControlChip::new()));
        costs.insert(keccak_control.name(), keccak_control.cost());
        chips.push(keccak_control);

        let bn254_add_assign = Chip::new(RiscvAir::Bn254Add(WeierstrassAddAssignChip::<
            SwCurve<Bn254Parameters>,
        >::new()));
        costs.insert(bn254_add_assign.name(), bn254_add_assign.cost());
        chips.push(bn254_add_assign);

        let bn254_double_assign = Chip::new(RiscvAir::Bn254Double(WeierstrassDoubleAssignChip::<
            SwCurve<Bn254Parameters>,
        >::new()));
        costs.insert(bn254_double_assign.name(), bn254_double_assign.cost());
        chips.push(bn254_double_assign);

        let bls12381_add = Chip::new(RiscvAir::Bls12381Add(WeierstrassAddAssignChip::<
            SwCurve<Bls12381Parameters>,
        >::new()));
        costs.insert(bls12381_add.name(), bls12381_add.cost());
        chips.push(bls12381_add);

        let bls12381_double = Chip::new(RiscvAir::Bls12381Double(WeierstrassDoubleAssignChip::<
            SwCurve<Bls12381Parameters>,
        >::new()));
        costs.insert(bls12381_double.name(), bls12381_double.cost());
        chips.push(bls12381_double);

        let uint256_mul = Chip::new(RiscvAir::Uint256Mul(Uint256MulChip::default()));
        costs.insert(uint256_mul.name(), uint256_mul.cost());
        chips.push(uint256_mul);

        let u256x2048_mul = Chip::new(RiscvAir::U256x2048Mul(U256x2048MulChip::default()));
        costs.insert(u256x2048_mul.name(), u256x2048_mul.cost());
        chips.push(u256x2048_mul);

        let bls12381_fp = Chip::new(RiscvAir::Bls12381Fp(FpOpChip::<Bls12381BaseField>::new()));
        costs.insert(bls12381_fp.name(), bls12381_fp.cost());
        chips.push(bls12381_fp);

        let bls12381_fp2_addsub =
            Chip::new(RiscvAir::Bls12381Fp2AddSub(Fp2AddSubAssignChip::<Bls12381BaseField>::new()));
        costs.insert(bls12381_fp2_addsub.name(), bls12381_fp2_addsub.cost());
        chips.push(bls12381_fp2_addsub);

        let bls12381_fp2_mul =
            Chip::new(RiscvAir::Bls12381Fp2Mul(Fp2MulAssignChip::<Bls12381BaseField>::new()));
        costs.insert(bls12381_fp2_mul.name(), bls12381_fp2_mul.cost());
        chips.push(bls12381_fp2_mul);

        let bn254_fp = Chip::new(RiscvAir::Bn254Fp(FpOpChip::<Bn254BaseField>::new()));
        costs.insert(bn254_fp.name(), bn254_fp.cost());
        chips.push(bn254_fp);

        let bn254_fp2_addsub =
            Chip::new(RiscvAir::Bn254Fp2AddSub(Fp2AddSubAssignChip::<Bn254BaseField>::new()));
        costs.insert(bn254_fp2_addsub.name(), bn254_fp2_addsub.cost());
        chips.push(bn254_fp2_addsub);

        let bn254_fp2_mul =
            Chip::new(RiscvAir::Bn254Fp2Mul(Fp2MulAssignChip::<Bn254BaseField>::new()));
        costs.insert(bn254_fp2_mul.name(), bn254_fp2_mul.cost());
        chips.push(bn254_fp2_mul);

        let bls12381_decompress =
            Chip::new(RiscvAir::Bls12381Decompress(WeierstrassDecompressChip::<
                SwCurve<Bls12381Parameters>,
            >::with_lexicographic_rule()));
        costs.insert(bls12381_decompress.name(), bls12381_decompress.cost());
        chips.push(bls12381_decompress);

        let syscall_core = Chip::new(RiscvAir::SyscallCore(SyscallChip::core()));
        costs.insert(syscall_core.name(), syscall_core.cost());
        chips.push(syscall_core);

        let syscall_precompile = Chip::new(RiscvAir::SyscallPrecompile(SyscallChip::precompile()));
        costs.insert(syscall_precompile.name(), syscall_precompile.cost());
        chips.push(syscall_precompile);

        let div_rem = Chip::new(RiscvAir::DivRem(DivRemChip::default()));
        costs.insert(div_rem.name(), div_rem.cost());
        chips.push(div_rem);

        let add = Chip::new(RiscvAir::Add(AddChip::default()));
        costs.insert(add.name(), add.cost());
        chips.push(add);

        let addi = Chip::new(RiscvAir::Addi(AddiChip::default()));
        costs.insert(addi.name(), addi.cost());
        chips.push(addi);

        let sub = Chip::new(RiscvAir::Sub(SubChip::default()));
        costs.insert(sub.name(), sub.cost());
        chips.push(sub);

        let bitwise = Chip::new(RiscvAir::Bitwise(BitwiseChip::default()));
        costs.insert(bitwise.name(), bitwise.cost());
        chips.push(bitwise);

        let mul = Chip::new(RiscvAir::Mul(MulChip::default()));
        costs.insert(mul.name(), mul.cost());
        chips.push(mul);

        let shift_right = Chip::new(RiscvAir::ShiftRight(ShiftRightChip::default()));
        costs.insert(shift_right.name(), shift_right.cost());
        chips.push(shift_right);

        let shift_left = Chip::new(RiscvAir::ShiftLeft(ShiftLeft::default()));
        costs.insert(shift_left.name(), shift_left.cost());
        chips.push(shift_left);

        let lt = Chip::new(RiscvAir::Lt(LtChip::default()));
        costs.insert(lt.name(), lt.cost());
        chips.push(lt);

        let load_byte = Chip::new(RiscvAir::LoadByte(LoadByteChip::default()));
        costs.insert(load_byte.name(), load_byte.cost());
        chips.push(load_byte);

        let load_half = Chip::new(RiscvAir::LoadHalf(LoadHalfChip::default()));
        costs.insert(load_half.name(), load_half.cost());
        chips.push(load_half);

        let load_word = Chip::new(RiscvAir::LoadWord(LoadWordChip::default()));
        costs.insert(load_word.name(), load_word.cost());
        chips.push(load_word);

        let load_x0 = Chip::new(RiscvAir::LoadX0(LoadX0Chip::default()));
        costs.insert(load_x0.name(), load_x0.cost());
        chips.push(load_x0);

        let store_byte = Chip::new(RiscvAir::StoreByte(StoreByteChip::default()));
        costs.insert(store_byte.name(), store_byte.cost());
        chips.push(store_byte);

        let store_half = Chip::new(RiscvAir::StoreHalf(StoreHalfChip::default()));
        costs.insert(store_half.name(), store_half.cost());
        chips.push(store_half);

        let store_word = Chip::new(RiscvAir::StoreWord(StoreWordChip::default()));
        costs.insert(store_word.name(), store_word.cost());
        chips.push(store_word);

        let auipc = Chip::new(RiscvAir::AUIPC(AuipcChip::default()));
        costs.insert(auipc.name(), auipc.cost());
        chips.push(auipc);

        let branch = Chip::new(RiscvAir::Branch(BranchChip::default()));
        costs.insert(branch.name(), branch.cost());
        chips.push(branch);

        let jal = Chip::new(RiscvAir::Jal(JalChip::default()));
        costs.insert(jal.name(), jal.cost());
        chips.push(jal);

        let jalr = Chip::new(RiscvAir::Jalr(JalrChip::default()));
        costs.insert(jalr.name(), jalr.cost());
        chips.push(jalr);

        let syscall_instrs = Chip::new(RiscvAir::SyscallInstrs(SyscallInstrsChip::default()));
        costs.insert(syscall_instrs.name(), syscall_instrs.cost());
        chips.push(syscall_instrs);

        let memory_global_init = Chip::new(RiscvAir::MemoryGlobalInit(MemoryGlobalChip::new(
            MemoryChipType::Initialize,
        )));
        costs.insert(memory_global_init.name(), memory_global_init.cost());
        chips.push(memory_global_init);

        let memory_global_finalize =
            Chip::new(RiscvAir::MemoryGlobalFinal(MemoryGlobalChip::new(MemoryChipType::Finalize)));
        costs.insert(memory_global_finalize.name(), memory_global_finalize.cost());
        chips.push(memory_global_finalize);

        let memory_local = Chip::new(RiscvAir::MemoryLocal(MemoryLocalChip::new()));
        costs.insert(memory_local.name(), memory_local.cost());
        chips.push(memory_local);

        let global = Chip::new(RiscvAir::Global(GlobalChip));
        costs.insert(global.name(), global.cost());
        chips.push(global);

        let byte = Chip::new(RiscvAir::ByteLookup(ByteChip::default()));
        costs.insert(byte.name(), byte.cost());
        chips.push(byte);

        let range = Chip::new(RiscvAir::RangeLookup(RangeChip::default()));
        costs.insert(range.name(), range.cost());
        chips.push(range);

        assert_eq!(chips.len(), costs.len(), "chips and costs must have the same length",);

        (chips, costs)
    }

    /// Get the heights of the preprocessed chips for a given program.
    pub(crate) fn preprocessed_heights(program: &Program) -> Vec<(RiscvAirId, usize)> {
        vec![
            (RiscvAirId::Program, program.instructions.len()),
            (RiscvAirId::Byte, BYTE_CHIP_NUM_ROWS),
            (RiscvAirId::Range, RANGE_CHIP_NUM_ROWS),
        ]
    }

    /// Get the heights of the chips for a given execution record.
    pub fn core_heights(record: &ExecutionRecord) -> Vec<(RiscvAirId, usize)> {
        vec![
            (RiscvAirId::DivRem, record.divrem_events.len()),
            (RiscvAirId::Add, record.add_events.len()),
            (RiscvAirId::Addi, record.addi_events.len()),
            (RiscvAirId::Sub, record.sub_events.len()),
            (RiscvAirId::Bitwise, record.bitwise_events.len()),
            (RiscvAirId::Mul, record.mul_events.len()),
            (RiscvAirId::ShiftRight, record.shift_right_events.len()),
            (RiscvAirId::ShiftLeft, record.shift_left_events.len()),
            (RiscvAirId::Lt, record.lt_events.len()),
            (
                RiscvAirId::MemoryLocal,
                record
                    .get_local_mem_events()
                    .chunks(NUM_LOCAL_MEMORY_ENTRIES_PER_ROW)
                    .into_iter()
                    .count(),
            ),
            (RiscvAirId::LoadByte, record.memory_load_byte_events.len()),
            (RiscvAirId::LoadHalf, record.memory_load_half_events.len()),
            (RiscvAirId::LoadWord, record.memory_load_word_events.len()),
            (RiscvAirId::LoadX0, record.memory_load_x0_events.len()),
            (RiscvAirId::StoreByte, record.memory_store_byte_events.len()),
            (RiscvAirId::StoreHalf, record.memory_store_half_events.len()),
            (RiscvAirId::StoreWord, record.memory_store_word_events.len()),
            (RiscvAirId::Auipc, record.auipc_events.len()),
            (RiscvAirId::Branch, record.branch_events.len()),
            (RiscvAirId::Jal, record.jal_events.len()),
            (RiscvAirId::Jalr, record.jalr_events.len()),
            (RiscvAirId::Global, record.global_interaction_events.len()),
            (RiscvAirId::SyscallCore, record.syscall_events.len()),
            (RiscvAirId::SyscallInstrs, record.syscall_events.len()),
        ]
    }

    pub(crate) fn memory_heights(record: &ExecutionRecord) -> Vec<(RiscvAirId, usize)> {
        vec![
            (RiscvAirId::MemoryGlobalInit, record.global_memory_initialize_events.len()),
            (RiscvAirId::MemoryGlobalFinalize, record.global_memory_finalize_events.len()),
            (
                RiscvAirId::Global,
                record.global_memory_finalize_events.len()
                    + record.global_memory_initialize_events.len(),
            ),
        ]
    }

    pub(crate) fn precompile_heights(
        &self,
        record: &ExecutionRecord,
    ) -> Option<(usize, usize, usize)> {
        record
            .precompile_events
            .get_events(self.syscall_code())
            .filter(|events| !events.is_empty())
            .map(|events| {
                (
                    events.len() * self.rows_per_event(),
                    events.get_local_mem_events().into_iter().count(),
                    record.global_interaction_events.len(),
                )
            })
    }

    pub(crate) fn get_all_core_airs() -> Vec<Self> {
        vec![
            RiscvAir::Add(AddChip::default()),
            RiscvAir::Addi(AddiChip::default()),
            RiscvAir::Sub(SubChip::default()),
            RiscvAir::Bitwise(BitwiseChip::default()),
            RiscvAir::Mul(MulChip::default()),
            RiscvAir::DivRem(DivRemChip::default()),
            RiscvAir::Lt(LtChip::default()),
            RiscvAir::ShiftLeft(ShiftLeft::default()),
            RiscvAir::ShiftRight(ShiftRightChip::default()),
            RiscvAir::LoadByte(LoadByteChip::default()),
            RiscvAir::LoadHalf(LoadHalfChip::default()),
            RiscvAir::LoadWord(LoadWordChip::default()),
            RiscvAir::LoadX0(LoadX0Chip::default()),
            RiscvAir::StoreByte(StoreByteChip::default()),
            RiscvAir::StoreHalf(StoreHalfChip::default()),
            RiscvAir::StoreWord(StoreWordChip::default()),
            RiscvAir::AUIPC(AuipcChip::default()),
            RiscvAir::Branch(BranchChip::default()),
            RiscvAir::Jal(JalChip::default()),
            RiscvAir::Jalr(JalrChip::default()),
            RiscvAir::SyscallInstrs(SyscallInstrsChip::default()),
            RiscvAir::MemoryLocal(MemoryLocalChip::new()),
            RiscvAir::Global(GlobalChip),
            RiscvAir::SyscallCore(SyscallChip::core()),
        ]
    }

    pub(crate) fn memory_init_final_airs() -> Vec<Self> {
        vec![
            RiscvAir::MemoryGlobalInit(MemoryGlobalChip::new(MemoryChipType::Initialize)),
            RiscvAir::MemoryGlobalFinal(MemoryGlobalChip::new(MemoryChipType::Finalize)),
            RiscvAir::Global(GlobalChip),
        ]
    }

    pub(crate) fn precompile_airs_with_memory_events_per_row() -> Vec<(Self, usize)> {
        let mut airs: HashSet<_> = Self::get_airs_and_costs().0.into_iter().collect();

        // Remove the core airs.
        for core_air in Self::get_all_core_airs() {
            airs.remove(&core_air);
        }

        // Remove the memory init/finalize airs.
        for memory_air in Self::memory_init_final_airs() {
            airs.remove(&memory_air);
        }

        // Remove the syscall, program, and byte lookup airs.
        airs.remove(&Self::SyscallPrecompile(SyscallChip::precompile()));
        airs.remove(&Self::Program(ProgramChip::default()));
        airs.remove(&Self::ByteLookup(ByteChip::default()));
        airs.remove(&Self::RangeLookup(RangeChip::default()));

        airs.into_iter()
            .map(|air| {
                let chip = Chip::new(air);
                let local_mem_events_per_row: usize = chip
                    .sends()
                    .iter()
                    .chain(chip.receives())
                    .filter(|interaction| {
                        interaction.kind == InteractionKind::Memory
                            && interaction.scope == InteractionScope::Local
                    })
                    .count();

                (chip.into_inner().unwrap(), local_mem_events_per_row)
            })
            .collect()
    }

    pub(crate) fn rows_per_event(&self) -> usize {
        match self {
            Self::Sha256Compress(_) => 80,
            Self::Sha256Extend(_) => 48,
            Self::KeccakP(_) => 24,
            _ => 1,
        }
    }

    pub(crate) fn syscall_code(&self) -> SyscallCode {
        match self {
            Self::Bls12381Add(_) => SyscallCode::BLS12381_ADD,
            Self::Bn254Add(_) => SyscallCode::BN254_ADD,
            Self::Bn254Double(_) => SyscallCode::BN254_DOUBLE,
            Self::Bn254Fp(_) => SyscallCode::BN254_FP_ADD,
            Self::Bn254Fp2AddSub(_) => SyscallCode::BN254_FP2_ADD,
            Self::Bn254Fp2Mul(_) => SyscallCode::BN254_FP2_MUL,
            Self::Ed25519Add(_) => SyscallCode::ED_ADD,
            Self::Ed25519Decompress(_) => SyscallCode::ED_DECOMPRESS,
            Self::KeccakP(_) => SyscallCode::KECCAK_PERMUTE,
            Self::KeccakPControl(_) => SyscallCode::KECCAK_PERMUTE,
            Self::Secp256k1Add(_) => SyscallCode::SECP256K1_ADD,
            Self::Secp256k1Double(_) => SyscallCode::SECP256K1_DOUBLE,
            Self::Secp256r1Add(_) => SyscallCode::SECP256R1_ADD,
            Self::Secp256r1Double(_) => SyscallCode::SECP256R1_DOUBLE,
            Self::Sha256Compress(_) => SyscallCode::SHA_COMPRESS,
            Self::Sha256CompressControl(_) => SyscallCode::SHA_COMPRESS,
            Self::Sha256Extend(_) => SyscallCode::SHA_EXTEND,
            Self::Sha256ExtendControl(_) => SyscallCode::SHA_EXTEND,
            Self::Uint256Mul(_) => SyscallCode::UINT256_MUL,
            Self::U256x2048Mul(_) => SyscallCode::U256XU2048_MUL,
            Self::Bls12381Decompress(_) => SyscallCode::BLS12381_DECOMPRESS,
            Self::K256Decompress(_) => SyscallCode::SECP256K1_DECOMPRESS,
            Self::P256Decompress(_) => SyscallCode::SECP256R1_DECOMPRESS,
            Self::Bls12381Double(_) => SyscallCode::BLS12381_DOUBLE,
            Self::Bls12381Fp(_) => SyscallCode::BLS12381_FP_ADD,
            Self::Bls12381Fp2Mul(_) => SyscallCode::BLS12381_FP2_MUL,
            Self::Bls12381Fp2AddSub(_) => SyscallCode::BLS12381_FP2_ADD,
            Self::Add(_) => unreachable!("Invalid for core chip"),
            Self::Addi(_) => unreachable!("Invalid for core chip"),
            Self::Sub(_) => unreachable!("Invalid for core chip"),
            Self::Bitwise(_) => unreachable!("Invalid for core chip"),
            Self::DivRem(_) => unreachable!("Invalid for core chip"),
            // Self::Cpu(_) => unreachable!("Invalid for core chip"),
            Self::MemoryGlobalInit(_) => unreachable!("Invalid for memory init/final"),
            Self::MemoryGlobalFinal(_) => unreachable!("Invalid for memory init/final"),
            Self::MemoryLocal(_) => unreachable!("Invalid for memory local"),
            Self::Global(_) => unreachable!("Invalid for global chip"),
            Self::Program(_) => unreachable!("Invalid for core chip"),
            Self::Mul(_) => unreachable!("Invalid for core chip"),
            Self::Lt(_) => unreachable!("Invalid for core chip"),
            Self::ShiftRight(_) => unreachable!("Invalid for core chip"),
            Self::ShiftLeft(_) => unreachable!("Invalid for core chip"),
            Self::LoadByte(_) => unreachable!("Invalid for memory chip"),
            Self::LoadHalf(_) => unreachable!("Invalid for memory chip"),
            Self::LoadWord(_) => unreachable!("Invalid for memory chip"),
            Self::LoadX0(_) => unreachable!("Invalid for memory chip"),
            Self::StoreByte(_) => unreachable!("Invalid for memory chip"),
            Self::StoreHalf(_) => unreachable!("Invalid for memory chip"),
            Self::StoreWord(_) => unreachable!("Invalid for memory chip"),
            Self::AUIPC(_) => unreachable!("Invalid for auipc chip"),
            Self::Branch(_) => unreachable!("Invalid for branch chip"),
            Self::Jal(_) => unreachable!("Invalid for jal chip"),
            Self::Jalr(_) => unreachable!("Invalid for jalr chip"),
            Self::SyscallInstrs(_) => unreachable!("Invalid for syscall instr chip"),
            Self::ByteLookup(_) => unreachable!("Invalid for core chip"),
            Self::RangeLookup(_) => unreachable!("Invalid for core chip"),
            Self::SyscallCore(_) => unreachable!("Invalid for core chip"),
            Self::SyscallPrecompile(_) => unreachable!("Invalid for syscall precompile chip"),
        }
    }
}

impl<F: PrimeField32> PartialEq for RiscvAir<F> {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl<F: PrimeField32> Eq for RiscvAir<F> {}

impl<F: PrimeField32> core::hash::Hash for RiscvAir<F> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.name().hash(state);
    }
}

impl<F: PrimeField32> fmt::Debug for RiscvAir<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
pub mod tests {

    use std::sync::Arc;

    use p3_air::BaseAir;
    use p3_baby_bear::BabyBear;
    use sp1_core_executor::{Instruction, Opcode, Program, SP1Context};
    use sp1_stark::prover::CpuProver;
    use sp1_stark::{MachineVerifier, SP1CoreOpts, ShardVerifier};

    use crate::riscv::RiscvAir;
    use crate::utils::prove_core;
    use crate::{programs::tests::*, utils::setup_logger};
    use sp1_stark::InteractionKind;
    //     use p3_baby_bear::BabyBear;
    //     use sp1_core_executor::{Instruction, Opcode, Program, SP1Context};
    //     use sp1_stark::{
    //         baby_bear_poseidon2::BabyBearPoseidon2, CpuProver, MachineProver, SP1CoreOpts,
    //         StarkProvingKey, StarkVerifyingKey,
    //     };

    // TODO:  Re-enable when we get all precompiles compatible w/ v6 (specifically the
    // first_row, last_row.
    // #[test]
    // fn test_primitives_and_machine_air_names_match() {
    //     let chips = RiscvAir::<BabyBear>::chips();
    //     for (a, b) in chips.iter().zip_eq(RiscvAirId::iter()) {
    //         assert_eq!(a.name(), b.to_string());
    //     }
    // }

    //     // TODO:  Re-enable when we get all precompiles compatible w/ v6 (specifically the
    // first_row, last_row,     // and next_row constraints).
    //     // #[test]
    //     // fn core_air_cost_consistency() {
    //     //     // Load air costs from file
    //     //     let file =
    // std::fs::File::open("../executor/src/artifacts/rv32im_costs.json").unwrap();     //
    // let costs: HashMap<String, u64> = serde_json::from_reader(file).unwrap();     //     //
    // Compare with costs computed by machine     //     let machine_costs =
    // RiscvAir::<BabyBear>::costs();     //     assert_eq!(costs, machine_costs);
    //     // }
    //     #[test]
    //     #[ignore]
    //     fn write_core_air_costs() {
    //         let costs = RiscvAir::<BabyBear>::costs();
    //         println!("{:?}", costs);
    //         // write to file
    //         // Create directory if it doesn't exist
    //         let dir = std::path::Path::new("../executor/src/artifacts");
    //         if !dir.exists() {
    //             std::fs::create_dir_all(dir).unwrap();
    //         }
    //         let file = std::fs::File::create(dir.join("rv32im_costs.json")).unwrap();
    //         serde_json::to_writer_pretty(file, &costs).unwrap();
    //     }

    //     #[test]
    //     fn test_simple_prove() {
    //         utils::setup_logger();
    //         let program = simple_program();
    //         let stdin = SP1Stdin::new();
    //         run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    //     }

    //     #[test]
    //     fn test_shift_prove() {
    //         utils::setup_logger();
    //         let shift_ops = [Opcode::SRL, Opcode::SRA, Opcode::SLL];
    //         let operands =
    //             [(1, 1), (1234, 5678), (0xffff, 0xffff - 1), (u32::MAX - 1, u32::MAX), (u32::MAX,
    // 0)];         for shift_op in shift_ops.iter() {
    //             for op in operands.iter() {
    //                 let instructions = vec![
    //                     Instruction::new(Opcode::ADDI, 29, 0, op.0, false, true),
    //                     Instruction::new(Opcode::ADDI, 30, 0, op.1, false, true),
    //                     Instruction::new(*shift_op, 31, 29, 3, false, false),
    //                 ];
    //                 let program = Program::new(instructions, 0, 0);
    //                 let stdin = SP1Stdin::new();
    //                 run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    //             }
    //         }
    //     }

    //     #[test]
    //     fn test_sub_prove() {
    //         utils::setup_logger();
    //         let instructions = vec![
    //             Instruction::new(Opcode::ADDI, 29, 0, 5, false, true),
    //             Instruction::new(Opcode::ADDI, 30, 0, 8, false, true),
    //             Instruction::new(Opcode::SUB, 31, 30, 29, false, false),
    //         ];
    //         let program = Program::new(instructions, 0, 0);
    //         let stdin = SP1Stdin::new();
    //         run_test::<CpuProver<_, _>>(program, stdin).unwrap();
    //     }

    use crate::{io::SP1Stdin, utils::run_test};

    // #[tokio::test]
    // async fn test_add_prove() {
    //     // setup_logger();
    //     let instructions = vec![
    //         Instruction::new(Opcode::ADDI, 29, 0, 5, false, true),
    //         Instruction::new(Opcode::ADDI, 30, 0, 8, false, true),
    //         Instruction::new(Opcode::ADD, 31, 30, 29, false, false),
    //     ];
    //     let program = Program::new(instructions, 0, 0);
    //     let stdin = SP1Stdin::new();
    //     run_test(program, stdin).await.unwrap();
    // }

    #[test]
    fn test_chips_main_width_interaction_ratio() {
        let chips = RiscvAir::<BabyBear>::chips();
        for chip in chips.iter() {
            let main_width = chip.air.width();
            for kind in InteractionKind::all_kinds() {
                let interaction_count =
                    chip.num_sends_by_kind(kind) + chip.num_receives_by_kind(kind);
                assert!(interaction_count <= main_width);
            }
        }
    }

    #[tokio::test]
    async fn test_mul_prove() {
        let mul_ops = [Opcode::MUL, Opcode::MULH, Opcode::MULHU, Opcode::MULHSU];
        setup_logger();
        let operands =
            [(1, 1), (1234, 5678), (8765, 4321), (0xffff, 0xffff - 1), (u32::MAX - 1, u32::MAX)];
        for mul_op in mul_ops.iter() {
            for operand in operands.iter() {
                let instructions = vec![
                    Instruction::new(Opcode::ADDI, 29, 0, operand.0, false, true),
                    Instruction::new(Opcode::ADDI, 30, 0, operand.1, false, true),
                    Instruction::new(*mul_op, 31, 30, 29, false, false),
                ];
                let program = Program::new(instructions, 0, 0);
                let stdin = SP1Stdin::new();
                run_test(program, stdin).await.unwrap();
            }
        }
    }

    #[tokio::test]
    async fn test_lt_prove() {
        setup_logger();
        let less_than = [Opcode::SLT, Opcode::SLTU];
        for lt_op in less_than.iter() {
            let instructions = vec![
                Instruction::new(Opcode::ADDI, 29, 0, 5, false, true),
                Instruction::new(Opcode::ADDI, 30, 0, 8, false, true),
                Instruction::new(*lt_op, 31, 30, 29, false, false),
            ];
            let program = Program::new(instructions, 0, 0);
            let stdin = SP1Stdin::new();
            run_test(program, stdin).await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_bitwise_prove() {
        setup_logger();
        let bitwise_opcodes = [Opcode::XOR, Opcode::OR, Opcode::AND];

        for bitwise_op in bitwise_opcodes.iter() {
            let instructions = vec![
                Instruction::new(Opcode::ADDI, 29, 0, 5, false, true),
                Instruction::new(Opcode::ADDI, 30, 0, 8, false, true),
                Instruction::new(*bitwise_op, 31, 30, 29, false, false),
            ];
            let program = Program::new(instructions, 0, 0);
            let stdin = SP1Stdin::new();
            run_test(program, stdin).await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_divrem_prove() {
        setup_logger();
        let div_rem_ops = [Opcode::DIV, Opcode::DIVU, Opcode::REM, Opcode::REMU];
        let operands = [
            (1, 1),
            (123, 456 * 789),
            (123 * 456, 789),
            (0xffff * (0xffff - 1), 0xffff),
            (u32::MAX - 5, u32::MAX - 7),
            (1 << 31, u32::MAX),
            (1, 0),
            (0, 0),
        ];
        let mut instructions = vec![];
        for div_rem_op in div_rem_ops.iter() {
            for op in operands.iter() {
                instructions.push(Instruction::new(Opcode::ADDI, 29, 0, op.0, false, true));
                instructions.push(Instruction::new(Opcode::ADDI, 30, 0, op.1, false, true));
                instructions.push(Instruction::new(*div_rem_op, 31, 29, 30, false, false));
            }
        }
        let program = Program::new(instructions.to_vec(), 0, 0);
        let stdin = SP1Stdin::new();
        run_test(program, stdin).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_fibonacci_prove_simple() {
        setup_logger();
        let program = fibonacci_program();
        let stdin = SP1Stdin::new();
        run_test(program, stdin).await.unwrap();
    }

    #[tokio::test]
    async fn test_keccak_permute_prove() {
        setup_logger();
        let program = keccak_permute_program();
        let stdin = SP1Stdin::new();
        run_test(program, stdin).await.unwrap();
    }

    #[tokio::test]
    async fn test_fibonacci_prove_checkpoints() {
        setup_logger();

        let program = fibonacci_program();
        let stdin = SP1Stdin::new();
        let mut opts = SP1CoreOpts::default();
        opts.shard_size = 1024;
        opts.shard_batch_size = 2;

        let log_blowup = 1;
        let log_stacking_height = 21;
        let max_log_row_count = 21;
        let machine = RiscvAir::machine();
        let verifier = ShardVerifier::from_basefold_parameters(
            log_blowup,
            log_stacking_height,
            max_log_row_count,
            machine,
        );
        let prover = CpuProver::new(verifier.clone());
        let challenger = verifier.pcs_verifier.challenger();

        let program = Arc::new(program);
        let (pk, vk) = prover.setup(program.clone()).await;
        let (proof, _) = prove_core(
            Arc::new(prover),
            Arc::new(pk),
            program,
            &stdin,
            opts,
            SP1Context::default(),
            challenger,
        )
        .await
        .unwrap();

        let mut challenger = verifier.pcs_verifier.challenger();
        let machine_verifier = MachineVerifier::new(verifier);
        tracing::debug_span!("verify the proof")
            .in_scope(|| machine_verifier.verify(&vk, &proof, &mut challenger))
            .unwrap();
    }

    #[tokio::test]
    async fn test_fibonacci_prove_batch() {
        setup_logger();
        let program = Arc::new(fibonacci_program());
        let stdin = SP1Stdin::new();

        let opts = SP1CoreOpts::default();
        let log_blowup = 1;
        let log_stacking_height = 21;
        let max_log_row_count = 21;
        let machine = RiscvAir::machine();
        let verifier = ShardVerifier::from_basefold_parameters(
            log_blowup,
            log_stacking_height,
            max_log_row_count,
            machine,
        );
        let prover = CpuProver::new(verifier.clone());
        let challenger = verifier.pcs_verifier.challenger();
        let (pk, vk) = prover.setup(program.clone()).await;
        let (proof, _) = prove_core(
            Arc::new(prover),
            Arc::new(pk),
            program,
            &stdin,
            opts,
            SP1Context::default(),
            challenger,
        )
        .await
        .unwrap();

        let mut challenger = verifier.pcs_verifier.challenger();
        let machine_verifier = MachineVerifier::new(verifier);
        tracing::debug_span!("verify the proof")
            .in_scope(|| machine_verifier.verify(&vk, &proof, &mut challenger))
            .unwrap();
    }

    #[tokio::test]
    async fn test_simple_memory_program_prove() {
        setup_logger();
        let program = simple_memory_program();
        let stdin = SP1Stdin::new();
        run_test(program, stdin).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_ssz_withdrawal() {
        setup_logger();
        let program = ssz_withdrawals_program();
        let stdin = SP1Stdin::new();
        run_test(program, stdin).await.unwrap();
    }

    // #[test]
    // fn test_key_serde() {
    //     let program = ssz_withdrawals_program();
    //     let config = BabyBearPoseidon2::new();
    //     let machine = RiscvAir::machine(config);
    //     let (pk, vk) = machine.setup(&program);

    //     let serialized_pk = bincode::serialize(&pk).unwrap();
    //     let deserialized_pk: StarkProvingKey<BabyBearPoseidon2> =
    //         bincode::deserialize(&serialized_pk).unwrap();
    //     assert_eq!(pk.preprocessed_commit, deserialized_pk.preprocessed_commit);
    //     assert_eq!(pk.pc_start, deserialized_pk.pc_start);
    //     assert_eq!(pk.traces, deserialized_pk.traces);
    //     // assert_eq!(pk.data, deserialized_pk.data);
    //     assert_eq!(pk.chip_ordering, deserialized_pk.chip_ordering);
    //     assert_eq!(pk.local_only, deserialized_pk.local_only);

    //     let serialized_vk = bincode::serialize(&vk).unwrap();
    //     let deserialized_vk: StarkVerifyingKey<BabyBearPoseidon2> =
    //         bincode::deserialize(&serialized_vk).unwrap();
    //     assert_eq!(vk.pc_start, deserialized_vk.pc_start);
    //     assert_eq!(vk.chip_information.len(), deserialized_vk.chip_information.len());
    //     for (a, b) in vk.chip_information.iter().zip(deserialized_vk.chip_information.iter()) {
    //         assert_eq!(a.0, b.0);
    //         assert_eq!(a.1.height, b.1.height);
    //         assert_eq!(a.1.width, b.1.width);
    //     }
    //     assert_eq!(vk.chip_ordering, deserialized_vk.chip_ordering);
    // }
}
