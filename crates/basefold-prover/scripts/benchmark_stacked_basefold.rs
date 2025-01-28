use clap::Parser;
use rand::Rng;
use slop_dft::Radix2DitParallel;
use slop_matrix::{dense::RowMajorMatrix, Matrix};

use slop_baby_bear::{my_perm, BabyBear, DiffusionMatrixBabyBear};
use slop_challenger::DuplexChallenger;
use slop_commit::{ExtensionMmcs, Pcs};
use slop_fri::{FriConfig, TwoAdicFriPcs};
use slop_merkle_tree::FieldMerkleTreeMmcs;
use slop_poseidon2::{Poseidon2, Poseidon2ExternalMatrixGeneral};
use slop_symmetric::{PaddingFreeSponge, TruncatedPermutation};

use slop_algebra::{extension::BinomialExtensionField, AbstractField, Field};
use slop_basefold::BaseFoldPcs;
use slop_multilinear::{MultilinearPcsProver, Point, StackedPcsProver};
use slop_utils::setup_logger;

use slop_basefold_prover::BaseFoldProver;

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

#[derive(Parser, Debug)]
struct Args {
    #[clap(short, long, value_delimiter = ' ', default_value = "28 29")]
    nums_variables: Vec<usize>,
    #[clap(short, long, value_delimiter = ' ', default_value = "21 22")]
    log_stacking_heights: Vec<usize>,
}

fn main() {
    setup_logger();
    let mut rng = rand::thread_rng();
    let args = Args::parse();
    let nums_variables = args.nums_variables;
    let num_matrices = 1;

    let log_stacking_heights = args.log_stacking_heights;

    nums_variables.iter().for_each(|num_variables| {
        log_stacking_heights.iter().for_each(|log_stacking_height| {
            let num_columns = 1 << (num_variables - log_stacking_height);

            let vals = tracing::info_span!("construct big vecs").in_scope(|| {
                (0..num_matrices)
                    .map(|_| {
                        let mut vec = vec![Val::zero(); num_columns * (1 << log_stacking_height)];
                        vec.iter_mut().for_each(|v| *v = rng.gen::<Val>());
                        vec
                    })
                    .collect::<Vec<_>>()
            });
            let perm = my_perm();
            let hash = MyHash::new(perm.clone());
            let compress = MyCompress::new(perm.clone());
            let inner_mmcs = ValMmcs::new(hash, compress);
            let mmcs = ChallengeMmcs::new(inner_mmcs.clone());
            let config = FriConfig {
                log_blowup: 1,
                num_queries: 100,
                proof_of_work_bits: 16,
                mmcs: mmcs.clone(),
            };

            let config_clone = FriConfig {
                log_blowup: 1,
                num_queries: 100,
                proof_of_work_bits: 16,
                mmcs: mmcs.clone(),
            };

            let pcs =
                BaseFoldPcs::<Val, Challenge, ValMmcs, Challenger>::new(config, inner_mmcs.clone());

            let new_eval_point =
                Point::new((0..*num_variables).map(|_| rng.gen::<Challenge>()).collect());

            let prover = BaseFoldProver::new(pcs);

            let stacked_prover =
                StackedPcsProver { pcs: prover, log_stacking_height: *log_stacking_height };

            let vals_clone = vals.clone();

            let now = std::time::Instant::now();
            let (_, data) = tracing::info_span!("commit")
                .in_scope(|| stacked_prover.commit_multilinear(vals_clone));
            let commit_time = now.elapsed();

            let now = std::time::Instant::now();
            let _ = tracing::info_span!("prove evaluations").in_scope(|| {
                stacked_prover.prove_trusted_evaluation(
                    new_eval_point.clone(),
                    rng.gen::<Challenge>(),
                    data,
                    &mut Challenger::new(perm.clone()),
                )
            });

            let prove_time = now.elapsed();

            let mats = tracing::info_span!("construct matrices").in_scope(|| {
                vals.into_iter()
                    .flat_map(|vals| vec![RowMajorMatrix::new(vals, num_columns)])
                    .collect::<Vec<_>>()
            });

            let fri_pcs = TwoAdicFriPcs::<
                Val,
                Radix2DitParallel,
                ValMmcs,
                ExtensionMmcs<Val, Challenge, ValMmcs>,
            >::new(27, Radix2DitParallel, inner_mmcs, config_clone);

            let now = std::time::Instant::now();
            let (_, fri_data) = tracing::info_span!("Plonky3 commit").in_scope(|| {
                // let tall_domain =
                <TwoAdicFriPcs<
                    Val,
                    Radix2DitParallel,
                    ValMmcs,
                    ExtensionMmcs<Val, Challenge, ValMmcs>,
                > as Pcs<Challenge, Challenger>>::commit(
                    &fri_pcs,
                    mats.into_iter()
                        .map(|mat| {
                            let domain = <TwoAdicFriPcs<
                                Val,
                                Radix2DitParallel,
                                ValMmcs,
                                ExtensionMmcs<Val, Challenge, ValMmcs>,
                            > as Pcs<Challenge, Challenger>>::natural_domain_for_degree(
                                &fri_pcs,
                                mat.height(),
                            );
                            (domain, mat)
                        })
                        .collect(),
                )
            });
            let plonky3_commit_time = now.elapsed();

            let now = std::time::Instant::now();
            tracing::info_span!("prove Plonky3 evaluations").in_scope(|| {
                fri_pcs.open(
                    vec![(
                        &fri_data,
                        (0..num_matrices).map(|_| vec![rng.gen::<Challenge>()]).collect(),
                    )],
                    &mut Challenger::new(perm.clone()),
                );
            });
            let plonky3_prove_time = now.elapsed();

            println!("+-------------------------+------------------+");
            println!("| Variables              | {:>16} |", num_variables);
            println!("| Log Stacking Height    | {:>16} |", log_stacking_height);
            println!("| Commit Time            | {:>16?} |", commit_time);
            println!("| Prove Time             | {:>16?} |", prove_time);
            println!("| Plonky3 Commit Time    | {:>16?} |", plonky3_commit_time);
            println!("| Plonky3 Prove Time     | {:>16?} |", plonky3_prove_time);
            println!("+-------------------------+------------------+");
        });
    });
}
