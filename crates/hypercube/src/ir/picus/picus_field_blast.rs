use std::{fmt, str::FromStr};

use crate::air::PicusInfo;

/// ``BlastKey`` defines a variable or set of variables to be field blasted.
/// We say that a variable x is field blasted during the translation to Picus if 1) x has range [l,
/// h],
/// 2) we extract a different Picus module for every value of x in [l, h]. Field blasting is
///    expected
/// to stack combinatorially so if we field blast x over range [0, 1] and y over range [0, 1] then
/// we will extract 4 modules corresponding to (x, y) \in {(0, 0), (0, 1), (1, 0), (1, 1)}. For this
/// to be sound, we will need the field blasted variables to be deterministic and actually have that
/// range (or have a subset of the range). So we extract a special Picus module where the blasted
/// variables are outputs and we prove their ranges correspond to the blast ranges.
#[derive(Clone, Debug)]
pub enum BlastKey {
    /// Bind a single column by human name (must resolve to exactly one column).
    Name(String),
    /// Bind a single column by global index (e.g., Main index).
    Index(usize),
    /// Bind a *slice* of columns that belong to the same base name.
    /// The indices [start..=end] are *local* positions within that base’s column vector.
    Slice {
        /// slice name should correspond to struct field name
        name: String,
        /// if struct field spans several columns then start, end
        /// denote a slice over the columns
        start: usize,
        /// end is inclusive in the column slice
        end: usize,
    },
}

/// Field blast spec associates each key with an interval in the range [lo, hi]
#[derive(Clone, Debug)]
pub struct FieldBlastSpec {
    /// blast key
    pub key: BlastKey,
    /// lower bound for range
    pub lo: usize,
    /// upper bound for range
    pub hi: usize,
}

impl fmt::Display for BlastKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlastKey::Name(s) => write!(f, "{s}"),
            BlastKey::Index(i) => write!(f, "#{i}"),
            BlastKey::Slice { name, start, end } => write!(f, "{name}[{start}:{end}]"),
        }
    }
}

impl FromStr for FieldBlastSpec {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Expect LHS=RHS where RHS is lo..=hi
        let (lhs, rhs) =
            s.split_once('=').ok_or_else(|| "field-blast: expected KEY=lo..=hi".to_string())?;
        let key = parse_key(lhs.trim())?;
        let (lo, hi) = parse_inclusive_range(rhs.trim())?;
        if lo > hi {
            return Err("field-blast: lo must be <= hi".to_string());
        }
        Ok(FieldBlastSpec { key, lo, hi })
    }
}

fn parse_key(s: &str) -> Result<BlastKey, String> {
    let s = s.trim();

    // #index form
    if let Some(rest) = s.strip_prefix('#') {
        let idx: usize = rest.parse().map_err(|_| format!("bad index in key: {s}"))?;
        return Ok(BlastKey::Index(idx));
    }

    // Numeric-only LHS (e.g., "42=…") is also accepted as an index.
    if s.chars().all(|c| c.is_ascii_digit()) {
        let idx: usize = s.parse().map_err(|_| format!("bad index in key: {s}"))?;
        return Ok(BlastKey::Index(idx));
    }

    // name[a:b] slice form
    if let Some((name, rest)) = s.split_once('[') {
        let rest = rest.strip_suffix(']').ok_or_else(|| "missing closing ']'".to_string())?;
        let (a, b) =
            rest.split_once(':').ok_or_else(|| "slice must be a:b (inclusive)".to_string())?;
        let start: usize = a.trim().parse().map_err(|_| "bad slice start".to_string())?;
        let end: usize = b.trim().parse().map_err(|_| "bad slice end".to_string())?;
        if start > end {
            return Err("slice start must be <= end".to_string());
        }
        return Ok(BlastKey::Slice { name: name.trim().to_string(), start, end });
    }

    // Plain name
    if s.is_empty() {
        return Err("empty field-blast name".to_string());
    }
    Ok(BlastKey::Name(s.to_string()))
}

fn parse_inclusive_range(s: &str) -> Result<(usize, usize), String> {
    // lo..=hi (inclusive)
    let parts: Vec<&str> = s.split("..=").collect();
    if parts.len() != 2 {
        return Err("expected inclusive range lo..=hi".to_string());
    }
    let lo = parse_int(parts[0].trim())?;
    let hi = parse_int(parts[1].trim())?;
    Ok((lo, hi))
}

fn parse_int(s: &str) -> Result<usize, String> {
    // decimal; adjust if you ever want hex
    s.parse::<usize>().map_err(|_| format!("bad integer: {s}"))
}

/// One “dimension” in the cartesian product: a single concrete column index,
/// a friendly label for naming, and the inclusive value range to enumerate.
#[derive(Debug, Clone)]
pub struct BlastDim {
    /// col in the air to blast
    pub col: usize,
    /// lower bound
    pub lo: usize,
    /// upper bound
    pub hi: usize,
}

/// A blast assignment is a mapping of every column being blasted to a value
#[derive(Clone, Debug, Default)]
pub struct BlastAssignment(pub Vec<(usize, usize)>);

impl BlastAssignment {
    /// Allocates a ``BlastAssignment``
    fn new(capacity: usize) -> Self {
        BlastAssignment(Vec::with_capacity(capacity))
    }

    /// Checks if `BlastAssignment` is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    /// Returns the compact name/suffix for this assignment (e.g., "`3_7_10_2`").
    #[must_use]
    pub fn name(&self) -> String {
        self.to_string()
    }
    /// Convenience for building a module name with this assignment appended.
    /// If empty, returns `base.to_owned()` unchanged.
    #[must_use]
    pub fn append_to(&self, base: &str) -> String {
        if self.0.is_empty() {
            base.to_owned()
        } else {
            let mut s = String::with_capacity(base.len() + 1 + self.name().len());
            s.push_str(base);
            s.push('_');
            s.push_str(&self.name());
            s
        }
    }
}

impl fmt::Display for BlastAssignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, &(col, val)) in self.0.iter().enumerate() {
            if i > 0 {
                f.write_str("_")?;
            }
            write!(f, "{col}_{val}")?;
        }
        Ok(())
    }
}

/// Resolve CLI specs → a flat list of (col, label, range) “dimensions”.
#[must_use]
pub fn resolve_blast_dims(info: &PicusInfo, specs: &[FieldBlastSpec]) -> Vec<BlastDim> {
    // PicusInfo gives you (name, low_idx, high_idx) for each group.
    // Build a quick lookup: name -> [low..=high].
    let mut by_name = std::collections::BTreeMap::<String, (usize, usize)>::new();
    for (name, low, high) in &info.field_map {
        by_name.insert(name.clone(), (*low, *high));
    }

    let mut out = Vec::new();
    for spec in specs {
        match &spec.key {
            BlastKey::Slice { name, start, end } => {
                assert!(*start < *end);
                let cols =
                    by_name.get(name).unwrap_or_else(|| panic!("Unknown field-blast name: {name}"));
                if *start >= cols.1 - cols.0 {
                    panic!("incorrect field blast config. Start of slice should lie within spanned cols: {start}, {}", cols.1 - cols.0)
                }
                if *end >= cols.1 - cols.0 {
                    panic!("incorrect field blast config. End of slice should lie within spanned cols: {end}, {}", cols.1 - cols.0)
                }
                for i in 0..=(*end - *start) {
                    out.push(BlastDim { col: cols.0 + i, lo: spec.lo, hi: spec.hi });
                }
            }
            BlastKey::Index(col) => {
                out.push(BlastDim { col: *col, lo: spec.lo, hi: spec.hi });
            }
            BlastKey::Name(ident) => {
                let cols = by_name
                    .get(ident)
                    .unwrap_or_else(|| panic!("Unknown field-blast name: {ident}"));
                if cols.0 != cols.1 - 1 {
                    panic!(
                        "bitblasting name that spans multiple columns: {ident} -- {}:{}",
                        cols.0, cols.1
                    )
                }
                // Restrict the group to the requested slice:
                let col = cols.0;
                out.push(BlastDim { col, lo: spec.lo, hi: spec.hi });
            }
        }
    }
    out
}

/// Build the full cartesian product as `Vec<Vec<(col, value)>>`.
/// Returns one row per combination; each row has length = ``dims.len()``.
/// NOTE: This allocates all combinations so be very very careful field blasting.
#[must_use]
pub fn cartesian_product_usize(dims: &[BlastDim]) -> Vec<BlastAssignment> {
    fn dfs(i: usize, dims: &[BlastDim], cur: &mut BlastAssignment, out: &mut Vec<BlastAssignment>) {
        if i == dims.len() {
            out.push(cur.clone());
            return;
        }
        let d = &dims[i];
        debug_assert!(d.lo <= d.hi, "invalid range: lo > hi at dim {i}");
        for v in d.lo..=d.hi {
            cur.0.push((d.col, v));
            dfs(i + 1, dims, cur, out);
            cur.0.pop();
        }
    }

    if dims.is_empty() {
        return vec![];
    }

    let mut out = Vec::new();
    let mut cur = BlastAssignment::new(dims.len());
    dfs(0, dims, &mut cur, &mut out);
    out
}
