use std::sync::Arc;

use clap::Parser;
use rand::Rng;
use slop_dft::Radix2DitParallel;
use slop_jagged::MachineJaggedPcs;
use slop_matrix::dense::RowMajorMatrix;

use slop_baby_bear::{my_perm, BabyBear, DiffusionMatrixBabyBear};
use slop_challenger::{CanObserve, DuplexChallenger};
use slop_commit::ExtensionMmcs;
use slop_merkle_tree::FieldMerkleTreeMmcs;
use slop_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use slop_symmetric::{PaddingFreeSponge, TruncatedPermutation};

use slop_algebra::{extension::BinomialExtensionField, AbstractField, Field};
use slop_multilinear::{Mle, Point};
use slop_utils::{log2_ceil_usize, setup_logger};

use slop_basefold_prover::default_jagged_basefold_config;

// type F = BabyBear;
// type EF = BinomialExtensionField<F, 4>;

pub type Val = BabyBear;
pub type Challenge = BinomialExtensionField<Val, 4>;

pub type Perm = Poseidon2<Val, Poseidon2ExternalMatrixGeneral, DiffusionMatrixBabyBear, 16, 7>;
pub type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
// pub type DigestHash = Hash<Val, Val, DIGEST_SIZE>;
pub type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
pub type ValMmcs =
    FieldMerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 8>;
pub type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
pub type Dft = Radix2DitParallel;
pub type Challenger = DuplexChallenger<Val, Perm, 16, 8>;
// type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;

#[derive(Parser, Debug)]
struct Args {
    #[clap(short, long, value_delimiter = ' ', default_value = "2000000 4000000 17")]
    row_counts: Vec<usize>,
    #[clap(short, long, value_delimiter = ' ', default_value = "128 32 512")]
    col_counts: Vec<usize>,
    #[clap(short, long, default_value = "23")]
    max_log_row_count: usize,
    #[clap(short, long, value_delimiter = ' ', default_value = "21")]
    log_stacking_height: usize,
}

fn main() {
    setup_logger();
    let mut rng = rand::thread_rng();
    let args = Args::parse();

    let row_counts = args.row_counts;
    let column_counts = args.col_counts;

    assert!(row_counts.len() == column_counts.len());

    let vals = tracing::info_span!("construct big vecs").in_scope(|| {
        row_counts
            .iter()
            .zip(column_counts.iter())
            .map(|(num_rows, num_columns)| {
                let mut vec = vec![Val::zero(); num_columns * num_rows];
                vec.iter_mut().for_each(|v| *v = rng.gen::<Val>());
                vec
            })
            .collect::<Vec<_>>()
    });

    let batch_split_point = 1;

    let (jagged_prover, jagged_verifier) =
        default_jagged_basefold_config(args.log_stacking_height, args.max_log_row_count);
    let jagged_verifier = MachineJaggedPcs::new(
        &jagged_verifier,
        vec![
            column_counts[0..batch_split_point].to_vec(),
            column_counts[batch_split_point..].to_vec(),
        ],
    );

    let new_eval_point =
        (0..args.max_log_row_count).map(|_| rng.gen::<Challenge>()).collect::<Point<_>>();

    let mats = tracing::info_span!("construct matrices").in_scope(|| {
        vals.into_iter()
            .zip(column_counts.iter())
            .flat_map(|(vals, &num_columns)| vec![RowMajorMatrix::new(vals, num_columns)])
            .collect::<Vec<_>>()
    });

    let eval_claims =
        mats.iter().map(|mat| Mle::eval_matrix_at_point(mat, &new_eval_point)).collect::<Vec<_>>();

    let mut challenger = Challenger::new(my_perm());

    let now = std::time::Instant::now();
    let (commit_1, data_1) = tracing::info_span!("commit")
        .in_scope(|| jagged_prover.commit_multilinears(mats[0..batch_split_point].to_vec()));

    challenger.observe(commit_1);

    let (commit_2, data_2) = tracing::info_span!("commit")
        .in_scope(|| jagged_prover.commit_multilinears(mats[batch_split_point..].to_vec()));
    challenger.observe(commit_2);

    let commit_time = now.elapsed();

    let mut data = vec![Arc::new(data_1), Arc::new(data_2)];

    let mut commits = vec![commit_1, commit_2];

    // Don't time finalize for now because the commits will be pre-computed.
    tracing::info_span!("finalize")
        .in_scope(|| jagged_prover.finalize(&mut data, &mut commits, &mut challenger));

    let now = std::time::Instant::now();
    let proof = tracing::info_span!("prove evaluations").in_scope(|| {
        jagged_prover.prove_trusted_evaluations(
            new_eval_point.clone(),
            &[&eval_claims.iter().map(Vec::as_slice).collect::<Vec<_>>()],
            &data,
            &mut challenger.clone(),
        )
    });

    let prove_time = now.elapsed();

    let now = std::time::Instant::now();
    let result = jagged_verifier.verify_trusted_evaluations(
        new_eval_point,
        &[&eval_claims.iter().map(Vec::as_slice).collect::<Vec<_>>()],
        &commits,
        &proof,
        &mut challenger,
    );

    let verify_time = now.elapsed();

    println!("Result: {:?}", result);

    assert!(result.is_ok());

    println!("+-------------------------+------------------+");
    println!("| Variables               | {:>16} |", args.max_log_row_count);
    println!("| Log Stacking Height     | {:>16} |", args.log_stacking_height);
    println!(
        "| Log Total Area          | {:>16} |",
        log2_ceil_usize(
            column_counts.iter().zip(row_counts.iter()).map(|(c, r)| c * r).sum::<usize>()
        )
    );
    println!("| Commit Time             | {:>16?} |", commit_time);
    println!("| Prove Time              | {:>16?} |", prove_time);
    println!("| Verify Time             | {:>16?} |", verify_time);
    println!("+-------------------------+------------------+");
}
