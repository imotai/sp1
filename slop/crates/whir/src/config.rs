/// A fully expanded WHIR configuration.
#[derive(Debug, Clone)]
pub struct WhirProofShape<F> {
    /// The number of variables in the polynomial committed.
    pub num_variables: usize,

    pub domain_generator: F,

    /// The OOD samples used in the commitment.
    pub starting_ood_samples: usize,

    /// The rate of the initial RS code used during the protocol.
    pub starting_log_inv_rate: usize,

    /// The initial folding factor.
    pub starting_folding_factor: usize,

    /// The initial domain size
    pub starting_domain_log_size: usize,

    /// The initial pow bits used in the first fold.
    pub starting_folding_pow_bits: Vec<f64>,

    /// The round-specific parameters.
    pub round_parameters: Vec<RoundConfig>,

    /// Degree of the final polynomial sent over.
    pub final_poly_log_degree: usize,

    /// Number of queries in the last round
    pub final_queries: usize,

    /// Number of final bits of proof of work (for the queries).
    pub final_pow_bits: f64,

    /// Number of final bits of proof of work (for the sumcheck).
    pub final_folding_pow_bits: Vec<f64>,
}

/// Round specific configuration
#[derive(Debug, Clone)]
pub struct RoundConfig {
    /// Folding factor for this round.
    pub folding_factor: usize,
    /// Size of evaluation domain (of oracle sent in this round)
    pub evaluation_domain_log_size: usize,
    /// Number of bits of proof of work (for the queries).
    pub queries_pow_bits: f64,
    /// Number of bits of proof of work (for the folding).
    pub pow_bits: Vec<f64>,
    /// Number of queries in this round
    pub num_queries: usize,
    /// Number of OOD samples in this round
    pub ood_samples: usize,
    /// Rate of current RS codeword
    pub log_inv_rate: usize,
}
