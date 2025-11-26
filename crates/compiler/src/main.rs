use std::{path::PathBuf, str::FromStr, sync::Arc};

use clap::{ArgAction, Parser, ValueEnum, ValueHint};
use slop_air::Air;
use sp1_core_machine::riscv::RiscvAir;
use sp1_hypercube::{
    air::{MachineAir, PicusInfo},
    ir::{
        ConstraintCompiler, FieldBlastSpec, PicusExtractConfig, Shape,
        SuccinctChipToPicusTranslator,
    },
    Chip,
};
use sp1_primitives::SP1Field;
use sp1_prover::{CompressAir, ShrinkAir, WrapAir};
type F = SP1Field;

#[derive(ValueEnum, Clone, Debug)]
enum OutputFormat {
    Text,
    Json,
    Lean,
    Picus,
}

pub enum SP1Air<'a> {
    Riscv(&'a RiscvAir<F>),
    Compress(&'a CompressAir<F>),
    Shrink(&'a ShrinkAir<F>),
    Wrap(&'a WrapAir<F>),
}

#[derive(Clone, Debug)]
pub enum SP1AirArc {
    Riscv(Arc<RiscvAir<F>>),
    Compress(Arc<CompressAir<F>>),
    Shrink(Arc<ShrinkAir<F>>),
    Wrap(Arc<WrapAir<F>>),
}

#[derive(Clone, Debug)]
pub enum SP1Chip {
    Riscv(Chip<F, RiscvAir<F>>),
    Compress(Chip<F, CompressAir<F>>),
    Shrink(Chip<F, ShrinkAir<F>>),
    Wrap(Chip<F, WrapAir<F>>),
}

impl SP1Chip {
    pub fn name(&self) -> String {
        match self {
            SP1Chip::Riscv(chip) => chip.name(),
            SP1Chip::Compress(chip) => chip.name(),
            SP1Chip::Shrink(chip) => chip.name(),
            SP1Chip::Wrap(chip) => chip.name(),
        }
    }

    pub fn air(&self) -> SP1AirArc {
        match self {
            SP1Chip::Riscv(chip) => SP1AirArc::Riscv(chip.air.clone()),
            SP1Chip::Compress(chip) => SP1AirArc::Compress(chip.air.clone()),
            SP1Chip::Shrink(chip) => SP1AirArc::Shrink(chip.air.clone()),
            SP1Chip::Wrap(chip) => SP1AirArc::Wrap(chip.air.clone()),
        }
    }
}

impl SP1AirArc {
    pub fn as_ref(&self) -> SP1Air<'_> {
        match self {
            SP1AirArc::Riscv(air) => SP1Air::Riscv(air.as_ref()),
            SP1AirArc::Compress(air) => SP1Air::Compress(air.as_ref()),
            SP1AirArc::Shrink(air) => SP1Air::Shrink(air.as_ref()),
            SP1AirArc::Wrap(air) => SP1Air::Wrap(air.as_ref()),
        }
    }
}

impl SP1Air<'_> {
    pub fn picus_info(&self) -> PicusInfo {
        match self {
            SP1Air::Riscv(air) => air.picus_info(),
            SP1Air::Compress(air) => air.picus_info(),
            SP1Air::Shrink(air) => air.picus_info(),
            SP1Air::Wrap(air) => air.picus_info(),
        }
    }

    pub fn preprocessed_width(&self) -> usize {
        match self {
            SP1Air::Riscv(air) => air.preprocessed_width(),
            SP1Air::Compress(air) => air.preprocessed_width(),
            SP1Air::Shrink(air) => air.preprocessed_width(),
            SP1Air::Wrap(air) => air.preprocessed_width(),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, help = "Chip name to compile")]
    pub chip: Option<String>,

    #[arg(long, help = "Operation name to compile")]
    pub operation: Option<String>,

    #[arg(long, default_value = "target/constraints/")]
    pub out_dir: String,

    #[arg(long, value_enum, default_value = "text", help = "Output format")]
    pub format: OutputFormat,

    /// Picus-specific options (flattened into the CLI)
    #[command(flatten)]
    pub picus_args: PicusArgs,
}

/// Defines a set of arguments specific to the tranlsation from SP1 IR
/// to Picus.
#[derive(Parser, Debug, Default)]
#[command(next_help_heading = "Picus options")]
pub struct PicusArgs {
    /// Enable modular extraction (default: false)
    #[arg(long, default_value_t = false)]
    pub modular_extraction: bool,

    /// Comma-separated list of operation names that must always be inlined
    #[arg(long, value_delimiter = ',',
        default_values_t = [
            "RTypeReader".to_string(),
            "CPUState".to_string(),
            "ALUTypeReader".to_string(),
            "JTypeReader".to_string(),
            "ITypeReaderImmutable".to_string()]
    )]
    pub inline_ops: Vec<String>,

    /// Field blasting spec (repeatable). Forms:
    ///   name=lo..=hi
    ///   42=lo..=hi or #42=lo..=hi
    ///   name[a:b]=lo..=hi
    #[arg(
        long = "field-blast",
        action = ArgAction::Append,
        value_parser = FieldBlastSpec::from_str
    )]
    pub field_blast: Vec<FieldBlastSpec>,

    /// Directory to write the extracted Picus program(s).
    ///
    /// Can be overridden with PICUS_OUT_DIR.
    #[arg(
        long = "picus-out-dir",
        value_name = "DIR",
        value_hint = ValueHint::DirPath,
        env = "PICUS_OUT_DIR",
        // Any string literal works as a default for PathBuf
        default_value = "picus_out"
    )]

    /// Directory to write the extracted Picus program(s).
    ///
    /// Can be overridden with PICUS_OUT_DIR.
    pub picus_out_dir: PathBuf,

    /// Do not use builtin op summaries (small formulas that are equivalent to the constraints)
    /// instead of translating the ops. Use this flag when verifying chips whose main functionality
    /// is the op or if the op gets changed
    #[arg(long = "no-apply-summaries", default_value_t = false)]
    pub no_apply_summaries: bool,
}

impl<'a> From<&'a PicusArgs> for PicusExtractConfig<'a> {
    fn from(args: &'a PicusArgs) -> Self {
        Self {
            modular_extraction: args.modular_extraction,
            ops_to_inline: args.inline_ops.iter().cloned().collect(),
            output_dir: args.picus_out_dir.clone(),
            field_blast: &args.field_blast,
            no_apply_summaries: args.no_apply_summaries,
        }
    }
}

#[allow(clippy::print_stdout)]
fn main() {
    let args = Args::parse();
    let _out_dir = args.out_dir;

    // Validate arguments and dispatch
    match (&args.chip, &args.operation) {
        (Some(chip_name), Some(operation_name)) => {
            // Both specified: compile specific operation from chip
            compile_operation(chip_name, operation_name, &args.format);
        }
        (Some(chip_name), None) => {
            // Only chip specified: compile entire chip
            compile_chip(chip_name, &args.format, &args.picus_args);
        }
        (None, Some(_)) => {
            eprintln!("Error: When using --operation, you must also specify --chip");
            eprintln!("Example: --chip Add --operation AddOperation");
            std::process::exit(1);
        }
        (None, None) => {
            eprintln!("Error: Must specify --chip (and optionally --operation)");
            std::process::exit(1);
        }
    }
}

#[allow(clippy::print_stdout)]
#[allow(clippy::uninlined_format_args)]
fn compile_chip(chip_name: &str, output_format: &OutputFormat, picus_args: &PicusArgs) {
    let riscv_machine = RiscvAir::<F>::machine();
    let compress_machine = CompressAir::<F>::compress_machine();
    let shrink_machine = ShrinkAir::<F>::shrink_machine();
    let wrap_machine = WrapAir::<F>::wrap_machine();
    let all_chips = riscv_machine
        .chips()
        .to_vec()
        .into_iter()
        .map(SP1Chip::Riscv)
        .chain(compress_machine.chips().to_vec().into_iter().map(SP1Chip::Compress))
        .chain(shrink_machine.chips().to_vec().into_iter().map(SP1Chip::Shrink))
        .chain(wrap_machine.chips().to_vec().into_iter().map(SP1Chip::Wrap))
        .collect::<Vec<_>>();
    let chip = all_chips.iter().find(|c| c.name() == chip_name).cloned().unwrap_or_else(|| {
        eprintln!("Error: Chip '{}' not found", chip_name);
        eprintln!("Available chips:");
        for chip in all_chips {
            eprintln!("  {}", chip.name());
        }
        std::process::exit(1);
    });
    let air = match &chip {
        SP1Chip::Riscv(chip) => SP1AirArc::Riscv(chip.air.clone()),
        SP1Chip::Compress(chip) => SP1AirArc::Compress(chip.air.clone()),
        SP1Chip::Shrink(chip) => SP1AirArc::Shrink(chip.air.clone()),
        SP1Chip::Wrap(chip) => SP1AirArc::Wrap(chip.air.clone()),
    };

    let num_public_values = match chip {
        SP1Chip::Riscv(_) => riscv_machine.num_pv_elts(),
        SP1Chip::Compress(_) => compress_machine.num_pv_elts(),
        SP1Chip::Shrink(_) => shrink_machine.num_pv_elts(),
        SP1Chip::Wrap(_) => wrap_machine.num_pv_elts(),
    };

    let mut builder = match air.as_ref() {
        SP1Air::Riscv(air) => ConstraintCompiler::new(air, num_public_values),
        SP1Air::Compress(air) => ConstraintCompiler::new(air, num_public_values),
        SP1Air::Shrink(air) => ConstraintCompiler::new(air, num_public_values),
        SP1Air::Wrap(air) => ConstraintCompiler::new(air, num_public_values),
    };

    match &air {
        SP1AirArc::Riscv(air) => {
            air.eval(&mut builder);
        }
        SP1AirArc::Compress(air) => {
            air.eval(&mut builder);
        }
        SP1AirArc::Shrink(air) => {
            air.eval(&mut builder);
        }
        SP1AirArc::Wrap(air) => {
            air.eval(&mut builder);
        }
    }

    match output_format {
        OutputFormat::Text => {
            let ast = builder.ast();
            let ast_str = ast.to_string_pretty("   ");

            // Display Picus annotation info if available
            let picus_info = air.as_ref().picus_info();
            // Display field map (all fields with their ranges)
            if !picus_info.field_map.is_empty() {
                println!("All field ranges:");
                for (name, start, end) in &picus_info.field_map {
                    println!("  Main({}-{}) => {}", start, end - 1, name);
                }
                println!();
            }

            if !picus_info.input_ranges.is_empty() {
                println!("Input column ranges:");
                for (start, end, name) in &picus_info.input_ranges {
                    println!("  Main({}-{}) => {}", start, end - 1, name);
                }
                println!();
            }
            if !picus_info.selector_indices.is_empty() {
                println!("Selector columns:");
                for (index, name) in &picus_info.selector_indices {
                    println!("  Main({}) => {}", index, name);
                }
                println!();
            }

            println!("Constraints for chip {chip_name} (main):");
            println!("{ast_str}");

            for func in builder.modules().values() {
                println!("{func}");
            }
        }
        OutputFormat::Lean => {
            let input_mapping = Default::default();
            let (steps, constraints, num_calls) = builder.ast().to_lean_components(&input_mapping);
            // println!("{:#?}", lean_components);

            println!();
            println!("-- Generated Lean code for chip {}Chip", chip_name);
            // println!("import SP1Foundations");
            // println!();
            //
            // println!("namespace {}Chip", chip_name);
            // println!();

            println!(
                "def constraints (Main : Vector (Fin BB) {}) : SP1ConstraintList :=",
                builder.num_cols()
            );

            for step in steps {
                println!("  {}", step)
            }

            let calls_constraints: String = (0..num_calls).map(|i| format!("CS{i} ++ ")).collect();
            println!("  {calls_constraints}[");
            for constraint in constraints {
                println!("    {},", constraint);
            }
            println!("  ]");

            println!();
            // println!("end {}Chip", chip_name);
        }
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(&builder.ast()).unwrap());
        }
        OutputFormat::Picus => {
            let ast = builder.ast();
            let cfg: PicusExtractConfig = picus_args.into();
            let mut picus_extractor = SuccinctChipToPicusTranslator::new(
                &chip.name(),
                &ast,
                builder.modules(),
                &air.as_ref().picus_info(),
                cfg,
            );
            picus_extractor.generate_picus_program();
        }
    }
}

#[allow(clippy::print_stdout)]
#[allow(clippy::uninlined_format_args)]
fn compile_operation(chip_name: &str, operation_name: &str, output_format: &OutputFormat) {
    // Step 1: Compile the chip normally to register all operations
    let machine = RiscvAir::<F>::machine();
    let air = machine
        .chips()
        .iter()
        .find(|c| c.name() == chip_name)
        .cloned()
        .unwrap_or_else(|| {
            eprintln!("Error: Chip '{}' not found", chip_name);
            eprintln!("Available chips:");
            for chip in machine.chips() {
                eprintln!("  {}", chip.name());
            }
            std::process::exit(1);
        })
        .air
        .clone();

    let num_public_values = machine.num_pv_elts();
    let mut builder = ConstraintCompiler::new(air.as_ref(), num_public_values);

    // Step 2: Evaluate the chip (this registers all operations in modules)
    air.eval(&mut builder);

    // Step 3: Extract only the requested operation
    let operation = builder.modules().get(operation_name).unwrap_or_else(|| {
        eprintln!("Error: Operation '{}' not found in chip '{}'", operation_name, chip_name);
        eprintln!("Available operations in this chip:");
        for name in builder.modules().keys() {
            eprintln!("  {}", name);
        }
        std::process::exit(1);
    });

    // Step 4: Generate output for just this operation
    match output_format {
        OutputFormat::Text => {
            println!("{}", operation);
        }
        OutputFormat::Lean => {
            let input_mapping = operation.decl.input_mapping();
            let (steps, constraints, num_calls) = operation.body.to_lean_components(&input_mapping);
            // println!("{:#?}", lean_components);

            // println!(
            //     "-- Generated Lean code for operation {} (from chip {})",
            //     operation_name, chip_name
            // );
            // println!("import SP1Foundations");
            // println!();

            // println!("namespace {}", operation_name);
            println!();

            println!("def constraints");
            for (param_name, _, param) in &operation.decl.input {
                println!(
                    "  ({} : {})",
                    // In Mathlib, c[i] is pre-defined...
                    if param_name == "c" { "cc" } else { param_name },
                    param.to_lean_type()
                );
            }

            println!("  : {} :=", operation.decl.to_output_lean_type());

            for step in steps {
                println!("  {}", step)
            }

            let calls_constraints: String = (0..num_calls).map(|i| format!("CS{i} ++ ")).collect();
            match operation.decl.output {
                Shape::Unit => {
                    println!("  {calls_constraints}[");
                    for constraint in constraints {
                        println!("    {},", constraint);
                    }
                    println!("  ]");
                }
                _ => {
                    println!(
                        "  ⟨{}, {calls_constraints}[",
                        operation.decl.output.to_lean_constructor(&input_mapping)
                    );
                    for constraint in constraints {
                        println!("    {},", constraint);
                    }
                    println!("  ]⟩");
                }
            }

            println!();
            // println!("end {}", operation_name);
        }
        OutputFormat::Json => {
            println!("{}", serde_json::to_string_pretty(operation).unwrap());
        }
        OutputFormat::Picus => {}
    }
}
