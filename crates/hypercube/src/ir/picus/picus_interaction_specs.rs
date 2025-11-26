use crate::{
    ir::{
        expr_impl::{Expr, ExprExt},
        picus::{PicusConstraint, PicusExpr, PicusModuleTranslator},
    },
    InteractionKind,
};

/// Does the interaction specify inputs or outputs
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub enum IoPort {
    /// Values are appended to Picus module inputs.
    Inputs,
    /// Values are appended to Picus module outputs.
    Outputs,
    /// Some indices are inputs, some are outputs (by slices).
    Mixed {
        /// The indices of the interactions values corresponding to inputs.
        inputs: &'static [IndexSlice],
        /// The indices of the interaction values corresponding to outputs.
        outputs: &'static [IndexSlice],
    },
    /// Values are neither inputs nor outputs
    Neither,
}

/// Direction of the interaction (send or receive).
#[derive(Clone, Copy)]
pub enum Direction {
    /// A send interaction.
    Send,
    /// A receive interaction.
    Receive,
}

/// How to wire values on send/receive.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct DirectionSpec {
    /// Where to place the values for this direction (Inputs/Outputs).
    pub port: IoPort,
    /// Index subsets on which to apply range assumptions.
    ///
    /// Example: `[((2..8), (< 65536)), ((9..25) (< 128))]` means: for these slices of the `values`
    /// array, apply the predicates in the tuple.
    pub range_slice_predicates: &'static [(IndexSlice, Predicate)],
}

impl Default for DirectionSpec {
    fn default() -> Self {
        Self { port: IoPort::Neither, range_slice_predicates: Default::default() }
    }
}

/// A selection of indices inside `values`.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub enum IndexSlice {
    /// A continuous half-open range [start, end). If end is `usize::MAX` then
    /// it represents [start, ``values.len()``)
    Range {
        /// Range start
        start: usize,
        /// Range end
        end: usize,
    },
    /// A single position
    Single(usize),
}

/// A small catalogue of built-in predicates you already use in bespoke code.
#[derive(Clone, Copy, Debug)]
#[allow(dead_code, missing_docs)]
pub enum Predicate {
    LtConst(u64),
    LeqConst(u64),
    Bit, // encodes v*(v-1)=0
    InRange { lo: u64, hi: u64 },
}

impl Predicate {
    pub(crate) fn build_picus_pred(&self, val: &PicusExpr) -> PicusConstraint {
        match self {
            Predicate::LtConst(c) => PicusConstraint::new_lt(val.clone(), (*c).into()),
            Predicate::LeqConst(c) => PicusConstraint::new_leq(val.clone(), (*c).into()),
            Predicate::Bit => {
                PicusConstraint::new_equality(val.clone() * (val.clone() - 1.into()), 0.into())
            }
            Predicate::InRange { lo, hi } => PicusConstraint::And(
                Box::new(PicusConstraint::new_leq((*lo).into(), val.clone())),
                Box::new(PicusConstraint::new_geq((*hi).into(), val.clone())),
            ),
        }
    }
}

/// Expand an ``IndexSlice`` list into concrete indices, clamped to len.
pub fn expand_indices(slices: &[IndexSlice], len: usize) -> impl Iterator<Item = usize> + '_ {
    slices.iter().flat_map(move |sl| {
        match *sl {
            IndexSlice::Single(i) => {
                // in-bounds: i..i+1; out-of-bounds: empty i..i
                let s = i.min(len);
                let e = if i < len { i + 1 } else { i };
                s..e
            }
            IndexSlice::Range { start, end } => {
                // clamp to [0, len]
                let s = start.min(len);
                let e = end.min(len);
                s..e
            }
        }
    })
}

/// Build a per-index port assignment for the given ``IoPort``.
pub(crate) fn per_index_ports(port: IoPort, len: usize) -> Vec<IoPort> {
    match port {
        IoPort::Inputs => vec![IoPort::Inputs; len],
        IoPort::Outputs => vec![IoPort::Outputs; len],
        IoPort::Neither => vec![IoPort::Neither; len],
        IoPort::Mixed { inputs, outputs } => {
            let mut v = vec![IoPort::Neither; len];
            for i in expand_indices(inputs, len) {
                v[i] = IoPort::Inputs;
            }
            for i in expand_indices(outputs, len) {
                debug_assert!(
                    !matches!(v[i], IoPort::Inputs),
                    "Mixed IoPort: index {i} marked as both input and output"
                );
                v[i] = IoPort::Outputs;
            }
            v
        }
    }
}

/// Optional hook for interactions that still need custom logic (e.g., Byte opcodes).
/// If present, it runs *after* the generic wiring/assumptions are applied.
#[allow(dead_code)]
type PostHook =
    fn(ctx: &mut PicusModuleTranslator<'_, Expr, ExprExt>, values: &[PicusExpr], is_send: bool);

/// Picus specification for the interaction.
#[derive(Clone, Debug, Default)]
#[allow(dead_code)]
pub struct InteractionSpec {
    /// Whether this is treated as an I/O interaction at all.
    pub is_io: bool,
    /// Assumptions on arguments on the send side of the interaction.
    pub on_send: DirectionSpec,
    /// Assumptions on arguments on the receive side of the interaction.
    pub on_recv: DirectionSpec,
    /// Optional custom logic hook.
    pub post_hook: Option<PostHook>,
}

#[allow(dead_code)]
#[allow(clippy::too_many_lines)]
/// The top level function which declares and retrieves the spec for a given interaction.
pub(crate) fn spec_for(kind: InteractionKind) -> InteractionSpec {
    use IndexSlice::{Range, Single};
    use IoPort::{Inputs, Outputs};
    use Predicate::{Bit, LtConst};
    match kind {
        InteractionKind::Program => InteractionSpec {
            is_io: true,
            on_send: DirectionSpec { port: Inputs, range_slice_predicates: &[] },
            on_recv: DirectionSpec { port: Outputs, range_slice_predicates: &[] },
            post_hook: None,
        },
        InteractionKind::ShaExtend => InteractionSpec {
            is_io: true,
            on_send: DirectionSpec {
                port: Inputs,
                range_slice_predicates: &[(Range { start: 2, end: 5 }, LtConst(65536))],
            },
            on_recv: DirectionSpec {
                port: Outputs,
                range_slice_predicates: &[(Range { start: 2, end: 5 }, LtConst(65536))],
            },
            post_hook: None,
        },
        InteractionKind::Memory => InteractionSpec {
            is_io: true,
            on_send: DirectionSpec {
                port: Inputs,
                range_slice_predicates: &[(Range { start: 5, end: usize::MAX }, LtConst(65536))],
            },
            on_recv: DirectionSpec {
                port: Outputs,
                range_slice_predicates: &[(Range { start: 5, end: usize::MAX }, LtConst(65536))],
            },
            post_hook: None,
        },

        InteractionKind::Byte => todo!(),
        InteractionKind::State => InteractionSpec {
            is_io: true,
            on_send: DirectionSpec {
                port: Outputs,
                range_slice_predicates: &[
                    (Single(3), LtConst(65537)),
                    (Range { start: 4, end: 6 }, LtConst(65536)),
                ],
            },
            on_recv: DirectionSpec {
                port: Inputs,
                range_slice_predicates: &[
                    (Single(3), LtConst(65537)),
                    (Range { start: 4, end: 6 }, LtConst(65536)),
                ],
            },
            post_hook: None,
        },
        InteractionKind::Syscall => InteractionSpec {
            is_io: true,
            on_send: DirectionSpec {
                port: Outputs,
                range_slice_predicates: &[(Range { start: 2, end: 8 }, LtConst(65536))],
            },
            on_recv: DirectionSpec {
                port: Outputs,
                range_slice_predicates: &[(Range { start: 2, end: 8 }, LtConst(65536))],
            },
            post_hook: None,
        },
        InteractionKind::ShaCompress => InteractionSpec {
            is_io: true,
            on_send: DirectionSpec {
                port: Outputs,
                range_slice_predicates: &[
                    (Range { start: 2, end: 8 }, LtConst(65536)),
                    (Range { start: 9, end: 25 }, LtConst(65536)),
                ],
            },
            on_recv: DirectionSpec {
                port: Inputs,
                range_slice_predicates: &[
                    (Range { start: 2, end: 8 }, LtConst(65536)),
                    (Range { start: 9, end: 25 }, LtConst(65536)),
                ],
            },
            post_hook: None,
        },
        InteractionKind::Global => InteractionSpec {
            is_io: true,
            on_send: DirectionSpec {
                port: Outputs,
                range_slice_predicates: &[
                    (Single(0), LtConst(16777216)),
                    (Single(7), LtConst(65536)),
                    (Range { start: 9, end: 11 }, Bit),
                    (Single(10), LtConst(64)),
                ],
            },
            on_recv: DirectionSpec {
                port: Inputs,
                range_slice_predicates: &[
                    (Single(0), LtConst(16777216)),
                    (Single(7), LtConst(65536)),
                    (Range { start: 9, end: 11 }, Bit),
                    (Single(10), LtConst(64)),
                ],
            },
            post_hook: None,
        },
        InteractionKind::Keccak => InteractionSpec {
            is_io: true,
            on_send: DirectionSpec {
                port: Outputs,
                range_slice_predicates: &[
                    (Range { start: 2, end: 5 }, LtConst(65536)),
                    (Range { start: 6, end: usize::MAX }, LtConst(65536)),
                ],
            },
            on_recv: DirectionSpec {
                port: Inputs,
                range_slice_predicates: &[
                    (Range { start: 2, end: 8 }, LtConst(65536)),
                    (Range { start: 9, end: 25 }, LtConst(65536)),
                ],
            },
            post_hook: None,
        },
        InteractionKind::GlobalAccumulation
        | InteractionKind::InstructionDecode
        | InteractionKind::InstructionFetch
        | InteractionKind::PageProt
        | InteractionKind::PageProtAccess
        | InteractionKind::PageProtGlobalFinalizeControl
        | InteractionKind::PageProtGlobalInitControl => todo!(),
        InteractionKind::MemoryGlobalInitControl | InteractionKind::MemoryGlobalFinalizeControl => {
            InteractionSpec {
                is_io: true,
                on_send: DirectionSpec {
                    port: Inputs,
                    range_slice_predicates: &[
                        (Range { start: 1, end: 4 }, LtConst(65536)),
                        (Single(4), Bit),
                    ],
                },
                on_recv: DirectionSpec {
                    port: Outputs,
                    range_slice_predicates: &[
                        (Range { start: 1, end: 4 }, LtConst(65536)),
                        (Single(4), Bit),
                    ],
                },
                post_hook: None,
            }
        }
    }
}
