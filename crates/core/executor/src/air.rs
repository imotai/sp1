use std::{
    fmt::{Display, Formatter, Result as FmtResult},
    str::FromStr,
};

use enum_map::{Enum, EnumMap};
use serde::{Deserialize, Serialize};
use sp1_stark::shape::Shape;
use strum::{EnumIter, IntoEnumIterator, IntoStaticStr};
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
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    Serialize,
    Deserialize,
    EnumIter,
    IntoStaticStr,
    PartialOrd,
    Ord,
    Enum,
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
    /// The uint256 ops chip.
    Uint256Ops = 21,
    /// The u256 xu2048 mul chip.
    U256XU2048Mul = 22,
    /// The bls12-381 fp op assign chip.
    Bls12381FpOpAssign = 23,
    /// The bls12-831 fp2 add sub assign chip.
    Bls12381Fp2AddSubAssign = 24,
    /// The bls12-831 fp2 mul assign chip.
    Bls12381Fp2MulAssign = 25,
    /// The bn254 fp2 add sub assign chip.
    Bn254FpOpAssign = 26,
    /// The bn254 fp op assign chip.
    Bn254Fp2AddSubAssign = 27,
    /// The bn254 fp2 mul assign chip.
    Bn254Fp2MulAssign = 28,
    /// The bls12-381 decompress chip.
    Bls12381Decompress = 29,
    /// The syscall core chip.
    #[subenum(CoreAirId)]
    SyscallCore = 30,
    /// The syscall precompile chip.
    SyscallPrecompile = 31,
    /// The div rem chip.
    #[subenum(CoreAirId)]
    DivRem = 32,
    /// The add chip.
    #[subenum(CoreAirId)]
    Add = 33,
    /// The addi chip.
    #[subenum(CoreAirId)]
    Addi = 34,
    /// The sub chip.
    #[subenum(CoreAirId)]
    Sub = 35,
    /// The bitwise chip.
    #[subenum(CoreAirId)]
    Bitwise = 36,
    /// The mul chip.
    #[subenum(CoreAirId)]
    Mul = 37,
    /// The shift right chip.
    #[subenum(CoreAirId)]
    ShiftRight = 38,
    /// The shift left chip.
    #[subenum(CoreAirId)]
    ShiftLeft = 39,
    /// The lt chip.
    #[subenum(CoreAirId)]
    Lt = 40,
    /// The load byte chip.
    #[subenum(CoreAirId)]
    LoadByte = 41,
    /// The load half chip.
    #[subenum(CoreAirId)]
    LoadHalf = 42,
    /// The load word chip.
    #[subenum(CoreAirId)]
    LoadWord = 43,
    /// The load x0 chip.
    #[subenum(CoreAirId)]
    LoadX0 = 44,
    /// The store byte chip.
    #[subenum(CoreAirId)]
    StoreByte = 45,
    /// The store half chip.
    #[subenum(CoreAirId)]
    StoreHalf = 46,
    /// The store word chip.
    #[subenum(CoreAirId)]
    StoreWord = 47,
    /// The auipc chip.
    #[subenum(CoreAirId)]
    Auipc = 48,
    /// The branch chip.
    #[subenum(CoreAirId)]
    Branch = 49,
    /// The jal chip.
    #[subenum(CoreAirId)]
    Jal = 50,
    /// The jalr chip.
    #[subenum(CoreAirId)]
    Jalr = 51,
    /// The syscall instructions chip.
    #[subenum(CoreAirId)]
    SyscallInstrs = 52,
    /// The memory bump chip.
    #[subenum(CoreAirId)]
    MemoryBump = 53,
    /// The state bump chip.
    #[subenum(CoreAirId)]
    StateBump = 54,
    /// The memory global init chip.
    MemoryGlobalInit = 55,
    /// The memory global finalize chip.
    MemoryGlobalFinalize = 56,
    /// The memory local chip.
    #[subenum(CoreAirId)]
    MemoryLocal = 57,
    /// The global chip.
    #[subenum(CoreAirId)]
    Global = 58,
    /// The byte chip.
    Byte = 59,
    /// The range chip.
    Range = 60,
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
            RiscvAirId::MemoryBump,
            RiscvAirId::StateBump,
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

    /// TODO replace these three with subenums or something
    /// Whether the ID represents a core AIR.
    #[must_use]
    pub fn is_core(self) -> bool {
        CoreAirId::try_from(self).is_ok()
    }

    /// Whether the ID represents a memory AIR.
    #[must_use]
    pub fn is_memory(self) -> bool {
        matches!(
            self,
            RiscvAirId::MemoryGlobalInit | RiscvAirId::MemoryGlobalFinalize | RiscvAirId::Global
        )
    }

    /// Whether the ID represents a precompile AIR.
    #[must_use]
    pub fn is_precompile(self) -> bool {
        matches!(
            self,
            RiscvAirId::ShaExtend
                | RiscvAirId::ShaCompress
                | RiscvAirId::EdAddAssign
                | RiscvAirId::EdDecompress
                | RiscvAirId::Secp256k1Decompress
                | RiscvAirId::Secp256k1AddAssign
                | RiscvAirId::Secp256k1DoubleAssign
                | RiscvAirId::Secp256r1Decompress
                | RiscvAirId::Secp256r1AddAssign
                | RiscvAirId::Secp256r1DoubleAssign
                | RiscvAirId::KeccakPermute
                | RiscvAirId::Bn254AddAssign
                | RiscvAirId::Bn254DoubleAssign
                | RiscvAirId::Bls12381AddAssign
                | RiscvAirId::Bls12381DoubleAssign
                | RiscvAirId::Uint256MulMod
                | RiscvAirId::Uint256Ops
                | RiscvAirId::U256XU2048Mul
                | RiscvAirId::Bls12381FpOpAssign
                | RiscvAirId::Bls12381Fp2AddSubAssign
                | RiscvAirId::Bls12381Fp2MulAssign
                | RiscvAirId::Bn254FpOpAssign
                | RiscvAirId::Bn254Fp2AddSubAssign
                | RiscvAirId::Bn254Fp2MulAssign
                | RiscvAirId::Bls12381Decompress
        )
    }

    /// The number of rows in the AIR produced by each event.
    #[must_use]
    pub fn rows_per_event(&self) -> usize {
        match self {
            Self::ShaCompress => 80,
            Self::ShaExtend => 48,
            Self::KeccakPermute => 24,
            _ => 1,
        }
    }

    /// Get the ID of the AIR used in the syscall control implementation.
    #[must_use]
    pub fn control_air_id(self) -> Option<RiscvAirId> {
        match self {
            RiscvAirId::ShaCompress => Some(RiscvAirId::ShaCompressControl),
            RiscvAirId::ShaExtend => Some(RiscvAirId::ShaExtendControl),
            RiscvAirId::KeccakPermute => Some(RiscvAirId::KeccakPermuteControl),
            _ => None,
        }
    }

    /// Returns the string representation of the AIR.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        self.into()
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
