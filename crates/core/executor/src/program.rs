//! Programs that can be executed by the SP1 zkVM.

use std::{fs::File, io::Read, str::FromStr};

use crate::{
    disassembler::{transpile, Elf},
    instruction::Instruction,
    RiscvAirId,
};
use hashbrown::HashMap;
use p3_field::{AbstractExtensionField, Field, PrimeField32};
use p3_maybe_rayon::prelude::{IntoParallelIterator, ParallelBridge, ParallelIterator};
use serde::{Deserialize, Serialize};
use sp1_stark::{
    air::{MachineAir, MachineProgram},
    septic_curve::{SepticCurve, SepticCurveComplete},
    septic_digest::SepticDigest,
    septic_extension::SepticExtension,
    shape::Shape,
    InteractionKind,
};

/// A program that can be executed by the SP1 zkVM.
///
/// Contains a series of instructions along with the initial memory image. It also contains the
/// start address and base address of the program.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Program {
    /// The instructions of the program.
    pub instructions: Vec<Instruction>,
    /// The start address of the program. It is absolute, meaning not relative to `pc_base`.
    pub pc_start_abs: u64,
    /// The base address of the program.
    pub pc_base: u64,
    /// The initial memory image, useful for global constants.
    pub memory_image: HashMap<u64, u64>,
    /// The shape for the preprocessed tables.
    pub preprocessed_shape: Option<Shape<RiscvAirId>>,
}

impl Program {
    /// Create a new [Program].
    #[must_use]
    pub fn new(instructions: Vec<Instruction>, pc_start_abs: u64, pc_base: u64) -> Self {
        Self {
            instructions,
            pc_start_abs,
            pc_base,
            memory_image: HashMap::new(),
            preprocessed_shape: None,
        }
    }

    /// Disassemble a RV32IM ELF to a program that be executed by the VM.
    ///
    /// # Errors
    ///
    /// This function may return an error if the ELF is not valid.
    pub fn from(input: &[u8]) -> eyre::Result<Self> {
        // Decode the bytes as an ELF.
        let elf = Elf::decode(input)?;

        assert!(elf.pc_base != 0, "elf with pc_base == 0 is not supported");

        // Transpile the RV32IM instructions.
        let instructions = transpile(&elf.instructions);

        // Return the program.
        Ok(Program {
            instructions,
            pc_start_abs: elf.pc_start,
            pc_base: elf.pc_base,
            memory_image: elf.memory_image,
            preprocessed_shape: None,
        })
    }

    /// Disassemble a RV32IM ELF to a program that be executed by the VM from a file path.
    ///
    /// # Errors
    ///
    /// This function will return an error if the file cannot be opened or read.
    pub fn from_elf(path: &str) -> eyre::Result<Self> {
        let mut elf_code = Vec::new();
        File::open(path)?.read_to_end(&mut elf_code)?;
        Program::from(&elf_code)
    }

    /// Custom logic for padding the trace to a power of two according to the proof shape.
    pub fn fixed_log2_rows<F: Field, A: MachineAir<F>>(&self, air: &A) -> Option<usize> {
        let id = RiscvAirId::from_str(&air.name()).unwrap();
        self.preprocessed_shape.as_ref().map(|shape| {
            shape
                .log2_height(&id)
                .unwrap_or_else(|| panic!("Chip {} not found in specified shape", air.name()))
        })
    }

    #[must_use]
    /// Fetch the instruction at the given program counter.
    pub fn fetch(&self, pc_rel: u64) -> &Instruction {
        let idx = (pc_rel / 4) as usize;
        &self.instructions[idx]
    }

    /// Returns `self.pc_start - self.pc_base`, that is, the relative `pc_start`.
    #[must_use]
    pub fn pc_start_rel_u64(&self) -> u64 {
        self.pc_start_abs.checked_sub(self.pc_base).expect("expected pc_base <= pc_start")
    }
}

impl<F: PrimeField32> MachineProgram<F> for Program {
    fn pc_start_rel(&self) -> F {
        let pc_start_rel = self.pc_start_rel_u64();
        assert!(pc_start_rel < F::ORDER_U64);
        // Casting to u32 and truncating is okay, since F: PrimeField32 means it fits.
        F::from_wrapped_u32(pc_start_rel as u32)
    }

    fn initial_global_cumulative_sum(&self) -> SepticDigest<F> {
        let mut digests: Vec<SepticCurveComplete<F>> = self
            .memory_image
            .iter()
            .par_bridge()
            .map(|(&addr, &word)| {
                let limb_1 = (word & 0xFFFF) as u32 + (1 << 16) * ((word >> 32) & 0xFF) as u32;
                let limb_2 =
                    ((word >> 16) & 0xFFFF) as u32 + (1 << 16) * ((word >> 40) & 0xFF) as u32;
                let values = [
                    (InteractionKind::Memory as u32) << 24,
                    0,
                    (addr & 0xFFFF) as u32,
                    ((addr >> 16) & 0xFFFF) as u32,
                    ((addr >> 32) & 0xFFFF) as u32,
                    limb_1,
                    limb_2,
                    ((word >> 48) & 0xFFFF) as u32,
                ];
                // let x_start =
                //     SepticExtension::<F>::from_base_fn(|i| F::from_canonical_u32(values[i]));
                let (point, _, _, _) =
                    SepticCurve::<F>::lift_x(values.map(|x| F::from_canonical_u32(x)));
                SepticCurveComplete::Affine(point.neg())
            })
            .collect();
        digests.push(SepticCurveComplete::Affine(SepticDigest::<F>::zero().0));
        SepticDigest(
            digests.into_par_iter().reduce(|| SepticCurveComplete::Infinity, |a, b| a + b).point(),
        )
    }
}
