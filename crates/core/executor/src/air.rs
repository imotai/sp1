use std::{
    fmt::{Display, Formatter, Result as FmtResult},
    str::FromStr,
};

use enum_map::{Enum, EnumMap};
use serde::{Deserialize, Serialize};
use sp1_stark::shape::Shape;
use strum::{EnumIter, IntoEnumIterator};
use subenum::subenum;

/// RV32IM AIR Identifiers.
///
/// These identifiers are for the various chips in the rv32im prover. We need them in the
/// executor to compute the memory cost of the current shard of execution.
///
/// The [`CoreAirId`]s are the AIRs that are not part of precompile shards and not the program or
/// byte AIR.
#[subenum(CoreAirId)]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, EnumIter, PartialOrd, Ord, Enum,
)]
pub enum RiscvAirId {
    /// The cpu chip, which is a dummy for now, needed for shape loading.
    #[subenum(CoreAirId)]
    Cpu = 0,
    /// The program chip.
    Program = 1,
    /// The SHA-256 extend chip.
    ShaExtend = 2,
    /// The sha extend control chip.
    ShaExtendControl = 3,
    /// The SHA-256 compress chip.
    ShaCompress = 4,
    /// The sha compress control chip.
    ShaCompressControl = 5,
    /// The Edwards add assign chip.
    EdAddAssign = 6,
    /// The Edwards decompress chip.
    EdDecompress = 7,
    /// The secp256k1 decompress chip.
    Secp256k1Decompress = 8,
    /// The secp256k1 add assign chip.
    Secp256k1AddAssign = 9,
    /// The secp256k1 double assign chip.
    Secp256k1DoubleAssign = 10,
    /// The secp256r1 decompress chip.
    Secp256r1Decompress = 11,
    /// The secp256r1 add assign chip.
    Secp256r1AddAssign = 12,
    /// The secp256r1 double assign chip.
    Secp256r1DoubleAssign = 13,
    /// The Keccak permute chip.
    KeccakPermute = 14,
    /// The keccak permute control chip.
    KeccakPermuteControl = 15,
    /// The bn254 add assign chip.
    Bn254AddAssign = 16,
    /// The bn254 double assign chip.
    Bn254DoubleAssign = 17,
    /// The bls12-381 add assign chip.
    Bls12381AddAssign = 18,
    /// The bls12-381 double assign chip.
    Bls12381DoubleAssign = 19,
    /// The uint256 mul mod chip.
    Uint256MulMod = 20,
    /// The u256 xu2048 mul chip.
    U256XU2048Mul = 21,
    /// The bls12-381 fp op assign chip.
    Bls12381FpOpAssign = 22,
    /// The bls12-831 fp2 add sub assign chip.
    Bls12831Fp2AddSubAssign = 23,
    /// The bls12-831 fp2 mul assign chip.
    Bls12831Fp2MulAssign = 24,
    /// The bn254 fp2 add sub assign chip.
    Bn254FpOpAssign = 25,
    /// The bn254 fp op assign chip.
    Bn254Fp2AddSubAssign = 26,
    /// The bn254 fp2 mul assign chip.
    Bn254Fp2MulAssign = 27,
    /// The bls12-381 decompress chip.
    Bls12381Decompress = 28,
    /// The syscall core chip.
    #[subenum(CoreAirId)]
    SyscallCore = 29,
    /// The syscall precompile chip.
    SyscallPrecompile = 30,
    /// The div rem chip.
    #[subenum(CoreAirId)]
    DivRem = 31,
    /// The add chip.
    #[subenum(CoreAirId)]
    Add = 32,
    /// The addi chip.
    #[subenum(CoreAirId)]
    Addi = 33,
    /// The sub chip.
    #[subenum(CoreAirId)]
    Sub = 34,
    /// The bitwise chip.
    #[subenum(CoreAirId)]
    Bitwise = 35,
    /// The mul chip.
    #[subenum(CoreAirId)]
    Mul = 36,
    /// The shift right chip.
    #[subenum(CoreAirId)]
    ShiftRight = 37,
    /// The shift left chip.
    #[subenum(CoreAirId)]
    ShiftLeft = 38,
    /// The lt chip.
    #[subenum(CoreAirId)]
    Lt = 39,
    /// The load byte chip.
    #[subenum(CoreAirId)]
    LoadByte = 40,
    /// The load half chip.
    #[subenum(CoreAirId)]
    LoadHalf = 41,
    /// The load word chip.
    #[subenum(CoreAirId)]
    LoadWord = 42,
    /// The load x0 chip.
    #[subenum(CoreAirId)]
    LoadX0 = 43,
    /// The store byte chip.
    #[subenum(CoreAirId)]
    StoreByte = 44,
    /// The store half chip.
    #[subenum(CoreAirId)]
    StoreHalf = 45,
    /// The store word chip.
    #[subenum(CoreAirId)]
    StoreWord = 46,
    /// The auipc chip.
    #[subenum(CoreAirId)]
    Auipc = 47,
    /// The branch chip.
    #[subenum(CoreAirId)]
    Branch = 48,
    /// The jal chip.
    #[subenum(CoreAirId)]
    Jal = 49,
    /// The jalr chip.
    #[subenum(CoreAirId)]
    Jalr = 50,
    /// The syscall instructions chip.
    #[subenum(CoreAirId)]
    SyscallInstrs = 51,
    /// The memory global init chip.
    MemoryGlobalInit = 52,
    /// The memory global finalize chip.
    MemoryGlobalFinalize = 53,
    /// The memory local chip.
    #[subenum(CoreAirId)]
    MemoryLocal = 54,
    /// The global chip.
    #[subenum(CoreAirId)]
    Global = 55,
    /// The byte chip.
    Byte = 56,
    /// The range chip.
    Range = 57,
}

impl RiscvAirId {
    /// Returns the AIRs that are not part of precompile shards and not the program or byte AIR.
    #[must_use]
    pub fn core() -> Vec<RiscvAirId> {
        vec![
            RiscvAirId::Add,
            RiscvAirId::Addi,
            RiscvAirId::Sub,
            RiscvAirId::Mul,
            RiscvAirId::Bitwise,
            RiscvAirId::ShiftLeft,
            RiscvAirId::ShiftRight,
            RiscvAirId::DivRem,
            RiscvAirId::Lt,
            RiscvAirId::Auipc,
            RiscvAirId::MemoryLocal,
            RiscvAirId::LoadByte,
            RiscvAirId::LoadHalf,
            RiscvAirId::LoadWord,
            RiscvAirId::LoadX0,
            RiscvAirId::StoreByte,
            RiscvAirId::StoreHalf,
            RiscvAirId::StoreWord,
            RiscvAirId::Branch,
            RiscvAirId::Jal,
            RiscvAirId::Jalr,
            RiscvAirId::SyscallCore,
            RiscvAirId::SyscallInstrs,
            RiscvAirId::Global,
        ]
    }
    /// Returns the string representation of the AIR.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Cpu => "Cpu",
            Self::Program => "Program",
            Self::ShaExtend => "ShaExtend",
            Self::ShaExtendControl => "ShaExtendControl",
            Self::ShaCompress => "ShaCompress",
            Self::ShaCompressControl => "ShaCompressControl",
            Self::EdAddAssign => "EdAddAssign",
            Self::EdDecompress => "EdDecompress",
            Self::Secp256k1Decompress => "Secp256k1Decompress",
            Self::Secp256k1AddAssign => "Secp256k1AddAssign",
            Self::Secp256k1DoubleAssign => "Secp256k1DoubleAssign",
            Self::Secp256r1Decompress => "Secp256r1Decompress",
            Self::Secp256r1AddAssign => "Secp256r1AddAssign",
            Self::Secp256r1DoubleAssign => "Secp256r1DoubleAssign",
            Self::KeccakPermute => "KeccakPermute",
            Self::KeccakPermuteControl => "KeccakPermuteControl",
            Self::Bn254AddAssign => "Bn254AddAssign",
            Self::Bn254DoubleAssign => "Bn254DoubleAssign",
            Self::Bls12381AddAssign => "Bls12381AddAssign",
            Self::Bls12381DoubleAssign => "Bls12381DoubleAssign",
            Self::Uint256MulMod => "Uint256MulMod",
            Self::U256XU2048Mul => "U256XU2048Mul",
            Self::Bls12381FpOpAssign => "Bls12381FpOpAssign",
            Self::Bls12831Fp2AddSubAssign => "Bls12831Fp2AddSubAssign",
            Self::Bls12831Fp2MulAssign => "Bls12831Fp2MulAssign",
            Self::Bn254FpOpAssign => "Bn254FpOpAssign",
            Self::Bn254Fp2AddSubAssign => "Bn254Fp2AddSubAssign",
            Self::Bn254Fp2MulAssign => "Bn254Fp2MulAssign",
            Self::Bls12381Decompress => "Bls12381Decompress",
            Self::SyscallCore => "SyscallCore",
            Self::SyscallPrecompile => "SyscallPrecompile",
            Self::DivRem => "DivRem",
            Self::Add => "Add",
            Self::Addi => "Addi",
            Self::Sub => "Sub",
            Self::Bitwise => "Bitwise",
            Self::Mul => "Mul",
            Self::ShiftRight => "ShiftRight",
            Self::ShiftLeft => "ShiftLeft",
            Self::Lt => "Lt",
            Self::LoadByte => "LoadByte",
            Self::LoadHalf => "LoadHalf",
            Self::LoadWord => "LoadWord",
            Self::LoadX0 => "LoadX0",
            Self::StoreByte => "StoreByte",
            Self::StoreHalf => "StoreHalf",
            Self::StoreWord => "StoreWord",
            Self::Auipc => "Auipc",
            Self::Branch => "Branch",
            Self::Jal => "Jal",
            Self::Jalr => "Jalr",
            Self::SyscallInstrs => "SyscallInstrs",
            Self::MemoryGlobalInit => "MemoryGlobalInit",
            Self::MemoryGlobalFinalize => "MemoryGlobalFinalize",
            Self::MemoryLocal => "MemoryLocal",
            Self::Global => "Global",
            Self::Byte => "Byte",
            Self::Range => "Range",
        }
    }
}

impl FromStr for RiscvAirId {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let air = Self::iter().find(|chip| chip.as_str() == s);
        match air {
            Some(air) => Ok(air),
            None => Err(format!("Invalid RV32IMAir: {s}")),
        }
    }
}

impl Display for RiscvAirId {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.as_str())
    }
}

/// Defines a set of maximal shapes for generating core proofs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaximalShapes {
    inner: Vec<EnumMap<CoreAirId, u32>>,
}

impl FromIterator<Shape<RiscvAirId>> for MaximalShapes {
    fn from_iter<T: IntoIterator<Item = Shape<RiscvAirId>>>(iter: T) -> Self {
        let mut maximal_shapes = Vec::new();
        for shape in iter {
            let mut maximal_shape = EnumMap::<CoreAirId, u32>::default();
            for (air, height) in shape {
                if let Ok(core_air) = CoreAirId::try_from(air) {
                    maximal_shape[core_air] = height as u32;
                } else if air != RiscvAirId::Program
                    && air != RiscvAirId::Byte
                    && air != RiscvAirId::Range
                {
                    tracing::warn!("Invalid core air: {air}");
                }
            }
            maximal_shapes.push(maximal_shape);
        }
        Self { inner: maximal_shapes }
    }
}

impl MaximalShapes {
    /// Returns an iterator over the maximal shapes.
    pub fn iter(&self) -> impl Iterator<Item = &EnumMap<CoreAirId, u32>> {
        self.inner.iter()
    }
}
