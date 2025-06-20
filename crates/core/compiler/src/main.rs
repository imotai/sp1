use clap::Parser;
use slop_air::Air;
use slop_baby_bear::BabyBear;
use sp1_constraint_compiler::ir::ConstraintCompiler;
use sp1_core_machine::riscv::RiscvAir;
use sp1_stark::air::MachineAir;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "Add")]
    pub chip: String,
    #[arg(long, default_value = "target/constraints/")]
    pub out_dir: String,
}

type F = BabyBear;

#[allow(clippy::print_stdout)]
fn main() {
    let args = Args::parse();
    let chip = args.chip;
    let _out_dir = args.out_dir;

    let machine = RiscvAir::<F>::machine();
    let air = machine
        .chips()
        .iter()
        .find(|c| c.name() == chip)
        .cloned()
        .expect("Chip not found")
        .air
        .clone();

    let num_public_values = machine.num_pv_elts();
    let mut builder = ConstraintCompiler::new(air.as_ref(), num_public_values);

    air.eval(&mut builder);

    let ast = builder.ast();
    let ast_str = ast.to_string_pretty("   ");
    println!("Constraints for chip {chip} (main):");
    println!("{ast_str}");

    for func in builder.modules().values() {
        println!("{func}");
    }
}
