mod artifacts;
pub use artifacts::*;

pub mod cluster {
    tonic::include_proto!("cluster");
}
pub use cluster::*;

pub mod worker {
    tonic::include_proto!("worker");
}
pub use worker::*;

mod utils;
