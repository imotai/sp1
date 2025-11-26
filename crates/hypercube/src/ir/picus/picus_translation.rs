use sp1_primitives::SP1Field;
/// Short alias for the BabyBear-parameterized IR references used by Succinct.
type KoalaBearExpr = ExprRef<SP1Field>;

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fmt::Write,
    path::PathBuf,
};

use crate::{
    air::{AirInteraction, PicusInfo},
    ir::{
        cartesian_product_usize, current_modulus,
        expr_impl::{Expr, ExprExt},
        op,
        picus::picus_interaction_specs::{
            expand_indices, per_index_ports, spec_for, Direction, InteractionSpec, IoPort,
        },
        resolve_blast_dims, set_field_modulus, Ast, BlastDim, ExprRef, FieldBlastSpec, Func,
        FuncDecl, IrVar, OpExpr, PicusArg, PicusCall, PicusConstraint, PicusExpr, PicusModule,
        PicusProgram, Shape,
    },
    InteractionKind,
};

/// PCL module name used for byte-level boolean ops (XOR/AND/OR).
const BYTEOP_MOD: &str = "byteop";

/// Heuristic threshold for when we introduce a temporary to cap expression size.
const MAX_EXPR_SIZE: usize = 100;

/// ``KoalaBear`` modulus as a u32 (used e.g. for `pow_mod` in some byte ops).
const KOALABEAR: u32 = 0x7f000001;

// ---------- data carriers ----------

/// Data computed at the *call site* for a modular call.
/// Contains the specialized module name, concrete input arguments,
/// the concrete outputs allocated at the caller, and any equality constraints
/// needed to tie caller expressions to callee outputs that must be variables.
struct CallSite {
    mod_name: String,
    call_args: Vec<PicusExpr>,
    call_results: Vec<PicusExpr>,
    caller_constraints: Vec<PicusConstraint>,
}

/// Signature of the specialized callee we are about to materialize:
/// the formal inputs and outputs that will appear in the callee module header.
struct CalleeSignature {
    formal_inputs: Vec<PicusExpr>,
    formal_outputs: Vec<PicusExpr>,
}

/// Instructions for wiring results between caller and callee:
/// - `extra_call_results` are additional outputs to record at the *caller* call site
/// - `extra_module_outputs` are additional outputs to declare on the *callee*
/// - `env_updates` seeds the callee's environment with bindings for formal return positions
struct ResultWiring {
    extra_call_results: Vec<PicusExpr>,
    extra_module_outputs: Vec<PicusExpr>,
    env_updates: Vec<(KoalaBearExpr, PicusExpr)>,
}

/// Small trait so we can push constraints into either a `PicusModule` or any other sink later.
trait ConstraintSink {
    fn add_constraint(&mut self, c: PicusConstraint);
}

impl ConstraintSink for PicusModule {
    #[inline]
    fn add_constraint(&mut self, c: PicusConstraint) {
        self.constraints.push(c);
    }
}

/// Per-translation *environment* that maps Succinct `ExprRef`s in a given scope
/// (operation/module name) to their corresponding `PicusExpr`.
///
/// Responsibilities:
/// - Maintain scope-local bindings: (scope, IR expr) -> `PicusExpr`
/// - Store human-readable names for indices (for deterministic pretty printing)
/// - Mint fresh variables when needed
/// - Provide translation helpers (`translate`, `translate_truncated`)
#[derive(Debug, Default, Clone)]
pub struct Environment {
    // scope/op/module name -> local bindings
    scopes: HashMap<String, HashMap<KoalaBearExpr, PicusExpr>>,
    names: BTreeMap<String, BTreeMap<usize, String>>,
    // fresh counter
    fresh_id: usize,
}

impl Environment {
    fn new() -> Self {
        Self { scopes: HashMap::new(), names: BTreeMap::new(), fresh_id: 0 }
    }

    /// Create an environment with a single pre-seeded scope and its index→name map.
    fn with_names(scope: &str, name_map: BTreeMap<usize, String>) -> Self {
        let mut scopes = HashMap::new();
        scopes.insert(scope.to_string(), HashMap::new());
        let mut names = BTreeMap::new();
        names.insert(scope.to_string(), name_map);
        Self { scopes, names, fresh_id: 0 }
    }

    #[inline]
    fn bind(&mut self, scope: &str, k: KoalaBearExpr, v: PicusExpr) {
        self.scopes.entry(scope.to_string()).or_default().insert(k, v);
    }

    #[inline]
    #[allow(dead_code)]
    fn bind_many<I>(&mut self, scope: &str, it: I)
    where
        I: IntoIterator<Item = (KoalaBearExpr, PicusExpr)>,
    {
        let m = self.scopes.entry(scope.to_string()).or_default();
        for (k, v) in it {
            m.insert(k, v);
        }
    }

    #[inline]
    fn lookup(&self, scope: &str, k: &KoalaBearExpr) -> Option<&PicusExpr> {
        self.scopes.get(scope).and_then(|m| m.get(k))
    }

    /// WARNING: Will panic if k is not in scope
    fn get(&self, scope: &str, k: &KoalaBearExpr) -> &PicusExpr {
        self.lookup(scope, k).unwrap()
    }

    /// Replace every binding value equal to `old_val` with `new_val` across **all scopes**.
    ///
    /// Used by `assert_zero` to propagate equalities like `x = c` or `x = 0`
    fn replace_everywhere(&mut self, old_val: &PicusExpr, new_val: &PicusExpr) {
        for m in self.scopes.values_mut() {
            for v in m.values_mut() {
                if *v == *old_val {
                    *v = new_val.clone();
                }
            }
        }
    }

    #[inline]
    fn fresh(&mut self) -> PicusExpr {
        let id = self.fresh_id;
        self.fresh_id += 1;
        PicusExpr::Var("fresh".to_string(), id)
    }

    #[inline]
    /// Create a named variable.
    fn named(name: &str, expr_id: usize) -> PicusExpr {
        PicusExpr::Var(name.to_string(), expr_id)
    }

    /// Resolve a Succinct expression to a `PicusExpr` in a scope, allocating a named or fresh var
    /// on first reference. Constants are parsed into `Integer`s.
    ///
    /// Panics if given an internal `ExprRef::Expr` that has not been bound yet (translation
    /// order bug).
    fn translate(&mut self, scope: &str, succ: &KoalaBearExpr) -> PicusExpr {
        use IrVar::{Constant, InputArg, Main, OutputArg, Preprocessed, Public};
        match succ {
            ExprRef::IrVar(Constant(c)) => PicusExpr::Const(c.to_string().parse::<u64>().unwrap()),
            ExprRef::IrVar(InputArg(v) | OutputArg(v) | Main(v) | Preprocessed(v) | Public(v)) => {
                if let Some(p) = self.lookup(scope, succ) {
                    return p.clone();
                }
                let var = if let Some(name) = self.names.get(scope).and_then(|m| m.get(v)) {
                    Self::named(name, *v)
                } else {
                    self.fresh()
                };
                self.bind(scope, *succ, var.clone());
                var
            }
            ExprRef::Expr(_) => {
                self.lookup(scope, succ).expect("internal expr must be bound before use").clone()
            }
        }
    }

    /// Same as `extract`, but if the expression is too large, introduce a fresh tmp and emit `tmp =
    /// expr`.
    fn translate_truncated(
        &mut self,
        scope: &str,
        succ: &KoalaBearExpr,
        max_size: usize,
        sink: &mut impl ConstraintSink,
    ) -> PicusExpr {
        let pe = self.translate(scope, succ);
        if pe.size() > max_size {
            let tmp = self.fresh();
            sink.add_constraint(PicusConstraint::new_equality(tmp.clone(), pe));
            tmp
        } else {
            pe
        }
    }
}

/// Translates a single Succinct module (op) or Chip into a Picus module.
/// Acts over an `Environment`, emitting Picus constraints/calls/IO wires.
/// Handles both inline extraction and modular extraction (specialized callee modules).
pub struct PicusModuleTranslator<'a, Expr, ExprExt> {
    module_name: String,
    picus_module: PicusModule,
    succinct_circuit: &'a Ast<Expr, ExprExt>,
    succinct_modules: &'a BTreeMap<String, Func<Expr, ExprExt>>,
    env: Environment,
    extract_modularly: bool,
    ops_to_inline: HashSet<String>,
    no_apply_summaries: bool,
}

impl<'a> PicusModuleTranslator<'a, Expr, ExprExt> {
    // NOTE: happy to refactor this but it isn't clear what a better grouping is for now.
    // perhaps a factory style constructor would bypass the too_many_arguments warning
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    /// Constructs a new `PicusModuleTranslator`
    pub fn new(
        op_name: String,
        picus_module_name: &str,
        ast: &'a Ast<Expr, ExprExt>,
        modular_extraction: bool,
        succinct_modules: &'a BTreeMap<String, Func<Expr, ExprExt>>,
        env: Environment,
        ops_to_inline: &HashSet<String>,
        no_apply_summaries: bool,
    ) -> Self {
        let picus_module = PicusModule::new(picus_module_name.to_string());
        Self {
            module_name: op_name,
            picus_module,
            succinct_circuit: ast,
            succinct_modules,
            env,
            extract_modularly: modular_extraction,
            ops_to_inline: ops_to_inline.clone(),
            no_apply_summaries,
        }
    }

    /// Top-level driver: walk every operation of the scope and translate it.
    fn translate(&mut self, translated_picus_modules: &mut BTreeMap<String, PicusModule>) {
        let op_name = self.module_name.clone();
        let operations = self.succinct_circuit.operations.clone();
        for operation in &operations {
            self.translate_operation(&op_name, operation, translated_picus_modules);
        }
    }

    /// Inline-translate another Succinct module (switch scope temporarily).
    fn translate_succinct_module(
        &mut self,
        op_name: &str,
        operations: &Vec<OpExpr<Expr, ExprExt>>,
        translated_picus_modules: &mut BTreeMap<String, PicusModule>,
    ) {
        let cur_op_name = self.module_name.clone();
        self.module_name = op_name.to_string();
        for operation in operations {
            self.translate_operation(op_name, operation, translated_picus_modules);
        }
        self.module_name = cur_op_name;
    }

    /// Whether a given operation name has a specialized summary rule (faster than inlining).
    fn should_summarize_operation(&self, module_name: &str) -> bool {
        !self.no_apply_summaries
            && (module_name == "IsZeroWordOperation"
                || module_name == "IsEqualWordOperation"
                || module_name == "U16toU8OperationSafe"
                || module_name == "U16MSBOperation"
                || module_name == "LtOperationUnsigned")
    }

    /// Apply the relevant summary rule for an operation (when enabled).
    fn summarize_operation(
        &mut self,
        target_module_name: &String,
        actual_args: &[PicusExpr],
        actual_out_exprs: &[KoalaBearExpr],
    ) {
        if target_module_name == "IsEqualWordOperation" {
            self.summarize_is_equal_word(actual_args);
        } else if target_module_name == "IsZeroWordOperation" {
            self.summarize_is_zero_word(actual_args);
        } else if target_module_name == "U16toU8OperationSafe" {
            self.summarize_u16_to_u8_safe(actual_args, actual_out_exprs);
        } else if target_module_name == "U16MSBOperation" {
            self.summarize_u16_msb_operation(actual_args);
        } else if target_module_name == "LtOperationUnsigned" {
            self.summarize_lt_operation_unsigned(actual_args);
        }
    }

    /// Summary for equality of two 4-limb words with a boolean result bit.
    /// Adds `res ∈ {0,1}` and `res = 1 <=> (limbs equal)`.
    fn summarize_is_equal_word(&mut self, actual_args: &[PicusExpr]) {
        let is_real = actual_args[actual_args.len() - 1].clone();
        if !matches!(is_real, PicusExpr::Const(_)) {
            panic!("selector for equal word is not constant: {is_real}");
        }
        if let PicusExpr::Const(c) = is_real {
            if c == 0 {
                return;
            }
        }
        let eq_constraint = (0..4)
            .map(|i| {
                PicusConstraint::new_equality(actual_args[i].clone(), actual_args[i + 4].clone())
            })
            .reduce(|acc, eq| PicusConstraint::And(Box::new(acc), Box::new(eq)))
            .expect("expected at least one element");
        let res = &actual_args[actual_args.len() - 2];
        self.picus_module
            .constraints
            .push(PicusConstraint::Eq(Box::new(res.clone() * (res.clone() - PicusExpr::Const(1)))));
        self.picus_module.constraints.push(PicusConstraint::Iff(
            Box::new(PicusConstraint::new_equality(res.clone(), PicusExpr::Const(1))),
            Box::new(eq_constraint),
        ));
    }

    /// Summary for zero-check of a 4-limb word with a boolean result bit.
    fn summarize_is_zero_word(&mut self, actual_args: &[PicusExpr]) {
        let is_real = actual_args[actual_args.len() - 1].clone();
        if !matches!(is_real, PicusExpr::Const(_)) {
            panic!("selector for is zero word is not constant: {is_real}");
        }
        if let PicusExpr::Const(c) = is_real {
            if c == 0 {
                return;
            }
        }
        let eq_constraint = (0..4)
            .map(|i| PicusConstraint::new_equality(actual_args[i].clone(), PicusExpr::Const(0)))
            .reduce(|acc, eq| PicusConstraint::And(Box::new(acc), Box::new(eq)))
            .expect("expected at least one element");
        let res = &actual_args[actual_args.len() - 2];
        self.picus_module
            .constraints
            .push(PicusConstraint::Eq(Box::new(res.clone() * (res.clone() - PicusExpr::Const(1)))));
        self.picus_module.constraints.push(PicusConstraint::Iff(
            Box::new(PicusConstraint::new_equality(res.clone(), PicusExpr::Const(1))),
            Box::new(eq_constraint),
        ));
    }

    /// Summary for safe `u16 -> (u8,u8)` split; generates range checks and linear relation
    /// when the selector is active.
    fn summarize_u16_to_u8_safe(
        &mut self,
        actual_args: &[PicusExpr],
        actual_out_exprs: &[KoalaBearExpr],
    ) {
        let is_real = actual_args[actual_args.len() - 1].clone();
        if !matches!(is_real, PicusExpr::Const(_)) {
            panic!("selector for u16 to u8 is not constant: {is_real}");
        }

        assert!(actual_out_exprs.len() == 8);
        for i in 0..4 {
            let low_byte = self.env.fresh();
            let high_byte = self.env.fresh();
            self.env.bind(&self.module_name, actual_out_exprs[2 * i], low_byte.clone());
            self.env.bind(&self.module_name, actual_out_exprs[2 * i + 1], high_byte.clone());
            if let PicusExpr::Const(c) = is_real.clone() {
                if c == 0 {
                    continue;
                }
            }
            // Add the constraints here only if the selector is non zero
            self.picus_module.constraints.push(PicusConstraint::Lt(
                Box::new(low_byte.clone()),
                Box::new(PicusExpr::Const(256)),
            ));
            self.picus_module.constraints.push(PicusConstraint::Lt(
                Box::new(high_byte.clone()),
                Box::new(PicusExpr::Const(256)),
            ));
            self.picus_module.constraints.push(PicusConstraint::new_equality(
                actual_args[i].clone(),
                low_byte.clone() + high_byte.clone() * PicusExpr::Const(256),
            ));
        }
    }

    /// Summary for `u16_msb(a, msb)`: range-check low15 and tie `a = 32768*msb + low15`.
    fn summarize_u16_msb_operation(&mut self, actual_args: &[PicusExpr]) {
        let is_real = actual_args[actual_args.len() - 1].clone();
        if !matches!(is_real, PicusExpr::Const(_)) {
            panic!("selector for u16msb is not constant: {is_real}");
        }
        if let PicusExpr::Const(c) = is_real {
            if c == 0 {
                return;
            }
        }
        let low15 = self.env.fresh();
        self.picus_module.constraints.push(PicusConstraint::Eq(Box::new(
            actual_args[1].clone() * (actual_args[1].clone() - PicusExpr::Const(1)),
        )));
        self.picus_module
            .constraints
            .push(PicusConstraint::Lt(Box::new(low15.clone()), Box::new(PicusExpr::Const(32768))));
        self.picus_module.constraints.push(PicusConstraint::new_equality(
            actual_args[0].clone(),
            actual_args[1].clone() * PicusExpr::Const(32768) + low15,
        ));
    }

    /// Summary for unsigned `<` on 4-limb words, with boolean result.
    fn summarize_lt_operation_unsigned(&mut self, actual_args: &[PicusExpr]) {
        let is_real = actual_args[actual_args.len() - 1].clone();
        if !matches!(is_real, PicusExpr::Const(_)) {
            panic!("selector for lt operation unsigned is not constant: {is_real}");
        }
        if let PicusExpr::Const(c) = is_real.clone() {
            if c == 0 {
                return;
            }
        }
        let res_bit = actual_args[8].clone();
        let mut eq_conjunct: Option<PicusConstraint> = None;
        let mut disjuncts: Option<PicusConstraint> = None;
        for i in (0..4).rev() {
            let left_limbi = actual_args[i].clone();
            let right_limbi = actual_args[4 + i].clone();
            let lt =
                PicusConstraint::Lt(Box::new(left_limbi.clone()), Box::new(right_limbi.clone()));
            let eq = PicusConstraint::new_equality(left_limbi.clone(), right_limbi.clone());

            if let Some(conj) = eq_conjunct {
                let disjunct = PicusConstraint::And(Box::new(conj.clone()), Box::new(lt.clone()));
                if let Some(disj) = disjuncts {
                    disjuncts = Some(PicusConstraint::Or(Box::new(disj), Box::new(disjunct)));
                } else {
                    disjuncts = Some(disjunct);
                }
                eq_conjunct = Some(PicusConstraint::And(Box::new(conj), Box::new(eq.clone())));
            } else {
                disjuncts = Some(lt.clone());
                eq_conjunct = Some(eq.clone());
            }
        }

        self.picus_module.constraints.push(PicusConstraint::Eq(Box::new(
            res_bit.clone() * (res_bit.clone() - PicusExpr::Const(1)),
        )));
        self.picus_module.constraints.push(PicusConstraint::Iff(
            Box::new(PicusConstraint::new_equality(res_bit.clone(), PicusExpr::Const(1))),
            Box::new(disjuncts.unwrap()),
        ));
    }

    /// (Optional) small helper for your MUL inverse threshold check.
    fn is_big_const(c: u64) -> bool {
        // keep your policy (consider lifting BABYBEAR out later)
        c > (KOALABEAR as u64) / 2
    }

    /// Translate a single IR operation in the current scope.
    fn translate_operation(
        &mut self,
        op_name: &str,
        operation: &OpExpr<Expr, ExprExt>,
        extracted_picus_modules: &mut BTreeMap<String, PicusModule>,
    ) {
        match operation {
            OpExpr::Assign(a, b) => {
                let p = self.env.translate(op_name, b);
                self.env.bind(op_name, *a, p);
            }

            OpExpr::BinOp(op, l, a, b) => {
                self.translate_binop(op_name, l, *op, a, b);
            }

            OpExpr::AssertZero(e) => {
                self.translate_assert_zero(op_name, e);
            }

            OpExpr::Neg(l, r) => {
                let pr = self.env.translate(op_name, r);
                self.env.bind(op_name, *l, -pr);
            }

            OpExpr::Call(fdecl) => {
                self.translate_call(op_name, fdecl, extracted_picus_modules);
            }

            OpExpr::Send(a, _) => {
                self.translate_interaction(op_name, Direction::Send, a, extracted_picus_modules);
            }

            OpExpr::Receive(a, _) => {
                self.translate_interaction(op_name, Direction::Receive, a, extracted_picus_modules);
            }

            _ => {}
        }
    }

    fn materialize_interaction_values(
        &mut self,
        op_name: &str,
        require_vars: bool,
        values: &[KoalaBearExpr],
    ) -> Vec<PicusExpr> {
        let mut picus_values: Vec<PicusExpr> = Vec::new();
        for value in values {
            let picus_expr = self.env.translate(op_name, value);
            if require_vars && !matches!(picus_expr.clone(), PicusExpr::Var(_, _)) {
                let fresh_var = self.env.fresh();
                picus_values.push(fresh_var.clone());
                self.picus_module
                    .constraints
                    .push(PicusConstraint::new_equality(fresh_var, picus_expr));
            } else {
                picus_values.push(picus_expr);
            }
        }
        picus_values
    }

    fn translate_interaction(
        &mut self,
        op_name: &str,
        dir: Direction,
        air: &AirInteraction<KoalaBearExpr>,
        extracted_picus_modules: &mut BTreeMap<String, PicusModule>,
    ) {
        // Byte requires special case handling
        if matches!(air.kind, InteractionKind::Byte) && matches!(dir, Direction::Send) {
            self.translate_byte_interaction(op_name, air, extracted_picus_modules);
            return;
        }

        let spec = spec_for(air.kind);
        self.translate_spec(op_name, dir, &spec, air);
    }

    fn translate_spec(
        &mut self,
        op_name: &str,
        dir: Direction,
        spec: &InteractionSpec,
        air: &AirInteraction<KoalaBearExpr>,
    ) {
        if !spec.is_io {
            return;
        }

        // IO positions must be variables.
        let values = self.materialize_interaction_values(
            op_name,
            /* require_vars= */ true,
            &air.values,
        );

        // Pick the direction spec & build the per-index port map (inputs/outputs/neither).
        let dspec = match dir {
            Direction::Send => &spec.on_send,
            Direction::Receive => &spec.on_recv,
        };
        let per_idx_port = per_index_ports(dspec.port, values.len());
        // Wire values according to the per-index port map (preserving original order).
        for (i, v) in values.iter().cloned().enumerate() {
            match per_idx_port[i] {
                IoPort::Inputs => self.picus_module.add_input(v),
                IoPort::Outputs => self.picus_module.add_output(v),
                IoPort::Neither => { /* don’t wire */ }
                IoPort::Mixed { .. } => unreachable!("Mixed is expanded to per-index ports"),
            }
        }

        // Apply declarative predicates to selected indices; bucket = pre vs post:
        // - If an index is wired to Inputs => preconditions (`constraints`)
        // - If wired to Outputs         => postconditions
        // - If Neither: fall back to direction (Send→pre, Receive→post)
        for &(slice, ref pred) in dspec.range_slice_predicates {
            for i in expand_indices(std::slice::from_ref(&slice), values.len()) {
                let val = values[i].clone();
                let bucket: &mut Vec<PicusConstraint> = match per_idx_port[i] {
                    IoPort::Inputs => &mut self.picus_module.constraints,
                    IoPort::Outputs => &mut self.picus_module.postconditions,
                    IoPort::Mixed { .. } | IoPort::Neither => unreachable!(),
                };
                bucket.push(pred.build_picus_pred(&val));
            }
        }

        // Custom logic for interactions. Might move the byte interaction handling to the hook.
        if let Some(hook) = spec.post_hook {
            hook(self, &values, matches!(dir, Direction::Send));
        }
    }

    /// Handle `assert e = 0`:
    /// - Emit equality constraint
    /// - Perform common substitution patterns into the environment (x := c or x := 0)
    #[inline]
    fn translate_assert_zero(&mut self, op_name: &str, e: &KoalaBearExpr) {
        let pe = self.env.translate(op_name, e);

        // Propagate env substitutions for the common cases:
        //   x - c = 0 || c - x = 0  =>  x := c everywhere
        //   v     = 0  =>  v := 0 everywhere
        match &pe {
            PicusExpr::Sub(x, y)
                if matches!(**x, PicusExpr::Var(..)) && matches!(**y, PicusExpr::Const(_)) =>
            {
                self.env.replace_everywhere(&x.clone(), &y.clone());
            }
            PicusExpr::Sub(x, y)
                if matches!(**y, PicusExpr::Var(..)) && matches!(**x, PicusExpr::Const(_)) =>
            {
                self.env.replace_everywhere(&y.clone(), &x.clone());
            }
            PicusExpr::Var(..) => {
                self.env.replace_everywhere(&pe, &PicusExpr::Const(0));
            }
            _ => {}
        }

        self.picus_module.constraints.push(PicusConstraint::Eq(Box::new(pe)));
    }

    #[inline]
    fn translate_binop(
        &mut self,
        op_name: &str,
        l: &KoalaBearExpr,
        bop: op::BinOp,
        a: &KoalaBearExpr,
        b: &KoalaBearExpr,
    ) {
        let pa = self.env.translate_truncated(op_name, a, MAX_EXPR_SIZE, &mut self.picus_module);
        let pb = self.env.translate_truncated(op_name, b, MAX_EXPR_SIZE, &mut self.picus_module);

        match bop {
            op::BinOp::Add => {
                self.env.bind(op_name, *l, pa + pb);
            }
            op::BinOp::Sub => {
                // Prefer the specialized patterns if they match.
                self.env.bind(op_name, *l, pa - pb);
            }
            op::BinOp::Mul => {
                // “Inverse trick” if one side is a large constant.
                if let PicusExpr::Const(ref c) = pa {
                    if Self::is_big_const(*c) {
                        self.rewrite_mul_inverse(op_name, *c, l, pb);
                        return;
                    }
                }
                if let PicusExpr::Const(ref c) = pb {
                    if Self::is_big_const(*c) {
                        self.rewrite_mul_inverse(op_name, *c, l, pa);
                        return;
                    }
                }
                self.env.bind(op_name, *l, pa * pb);
            }
        }
    }

    /// Handle function calls:
    /// - Decide between inline vs modular translation (or summary)
    /// - Seed callee environment
    /// - Wire outputs back to caller environment
    fn translate_call(
        &mut self,
        op_name: &str,
        fdecl: &FuncDecl<Expr, ExprExt>,
        extracted_picus_modules: &mut BTreeMap<String, PicusModule>,
    ) {
        let (actual_args, actual_outs) = Self::walk_func_decl(fdecl);
        let func = self.succinct_modules.get(&fdecl.name).expect("callee function must exist");
        let (formal_args, formal_outs) = Self::walk_func_decl(&func.decl);

        assert_eq!(actual_args.len(), formal_args.len(), "arity mismatch for call {}", fdecl.name);

        // Map actual → formal (callee scope), and collect actual arg Picus exprs.
        let mut actual_picus = Vec::with_capacity(actual_args.len());
        for (formal, actual) in formal_args.iter().zip(&actual_args) {
            let p = self.env.translate(op_name, actual);
            actual_picus.push(p.clone());
            self.env.bind(&fdecl.name, *formal, p.clone());
        }

        let summarize = self.should_summarize_operation(&fdecl.name);
        let modular =
            self.extract_modularly && !summarize && !self.ops_to_inline.contains(&fdecl.name);

        if modular {
            self.translate_succinct_call_modularly(
                func,
                &actual_args,
                &formal_args,
                &actual_outs,
                &formal_outs,
                extracted_picus_modules,
            );
            return;
        }

        if summarize {
            self.summarize_operation(&func.decl.name, &actual_picus, &actual_outs);
            return;
        }

        // Inline extraction.
        self.translate_succinct_module(&fdecl.name, &func.body.operations, extracted_picus_modules);

        // Bind callee formal outs back to caller actual out slots.
        for (act_out, form_out) in actual_outs.iter().zip(&formal_outs) {
            let p = self.env.get(&fdecl.name, form_out);
            self.env.bind(op_name, *act_out, p.clone());
        }
    }

    // Modular translation of a call:
    /// 1) Build a *call site* (specialize name, collect args/results, caller constraints)
    /// 2) Build a *callee environment* and *signature*
    /// 3) Wire formal outputs to actual outputs and emit the call
    /// 4) Extract the specialized callee module once
    fn translate_succinct_call_modularly(
        &mut self,
        func: &Func<Expr, ExprExt>,
        actual_exprs: &[KoalaBearExpr],
        formal_exprs: &[KoalaBearExpr],
        actual_out_exprs: &[KoalaBearExpr],
        formal_out: &[KoalaBearExpr],
        extracted_picus_modules: &mut BTreeMap<String, PicusModule>,
    ) {
        let caller_scope = &self.module_name.clone(); // same scope you used elsewhere

        // Phase 1: call site
        let CallSite { mod_name, call_args, mut call_results, caller_constraints } =
            self.build_call_site(func, actual_exprs, caller_scope);
        self.picus_module.constraints.extend(caller_constraints);

        // Phase 2: setup the callee environment (env + signature)
        let (mut callee_env, mut callee_sig) =
            self.build_callee_env_and_signature(func, actual_exprs, formal_exprs, caller_scope);

        // Phase 3: results
        let ResultWiring { extra_call_results, extra_module_outputs, env_updates } =
            self.wire_results(actual_out_exprs, formal_out, caller_scope);

        call_results.extend(extra_call_results);
        callee_sig.formal_outputs.extend(extra_module_outputs);
        for (k, v) in env_updates {
            callee_env.bind(&func.decl.name, k, v);
        }

        // Emit the call at the caller
        self.picus_module.calls.push(PicusCall {
            mod_name: mod_name.clone(),
            outputs: call_results,
            inputs: call_args,
        });

        // Extract the specialized callee module once
        if !extracted_picus_modules.contains_key(&mod_name) {
            let mut module_extractor = PicusModuleTranslator::new(
                func.decl.name.clone(), // callee op_name
                &mod_name,              // specialized module name
                &func.body,             // callee body
                self.extract_modularly,
                self.succinct_modules,
                callee_env,
                &self.ops_to_inline,
                self.no_apply_summaries,
            );

            // seed bindings and signature
            module_extractor.picus_module.inputs.extend(callee_sig.formal_inputs);
            module_extractor.picus_module.outputs.extend(callee_sig.formal_outputs);

            module_extractor.translate(extracted_picus_modules);
            extracted_picus_modules.insert(mod_name, module_extractor.picus_module.clone());
        }
    }

    // -------- Phase 1: specialize name & collect call-site args/results/constraints --------

    /// Build the call site for a modular call:
    /// - Specialize the module name by appending `_i_const` for constant actuals on input ports
    /// - For output ports passed as arguments, allocate an output var and assert equality
    fn build_call_site(
        &mut self,
        func: &Func<Expr, ExprExt>,
        actual_exprs: &[KoalaBearExpr],
        caller_scope: &str,
    ) -> CallSite {
        let mut mod_name = func.decl.name.clone();
        let mut call_args = Vec::new();
        let mut call_results = Vec::new();
        let mut caller_constraints = Vec::new();

        let mut idx = 0usize;
        for (inp, attr, shape) in &func.decl.input {
            let w = shape.width();
            for i in idx..idx + w {
                let actual_picus = self.env.translate(caller_scope, &actual_exprs[i]);
                match attr.picus {
                    PicusArg::Input => {
                        if let PicusExpr::Const(c) = &actual_picus {
                            // specialize module name on constant actuals
                            write!(&mut mod_name, "_{i}_{c}").unwrap();
                        } else {
                            call_args.push(actual_picus);
                        }
                    }
                    // The tricky part is that some of the input arguments logically correspond to
                    // outputs of the translated module. As such, we have to denote those arguments
                    // as picus outputs. As such, these outputs should be merged with the results of
                    // the call when translating to Picus.
                    PicusArg::Output => {
                        // Caller allocates an out var and ties it to the actual expr.
                        let out_var = self.env.fresh();
                        // Create a constraint o - e = 0 where e is the expression passed to the
                        // call in the IR and o is a variable because Picus requires outputs and
                        // inputs to be variables.
                        caller_constraints
                            .push(PicusConstraint::new_equality(out_var.clone(), actual_picus));
                        call_results.push(out_var);
                    }
                    PicusArg::Unknown => {
                        panic!(
                            "Modular extraction requires all arguments to have Input/Output 
                        annotations. Operation {mod_name} missing annotation on column {inp}"
                        )
                    }
                }
            }
            idx += w;
        }

        CallSite { mod_name, call_args, call_results, caller_constraints }
    }

    // -------- Phase 2: build callee env (formal→expr) and callee signature --------

    /// Build a fresh callee environment (formal→local symbol or constant) and callee signature.
    ///
    /// - Each formal input becomes either a constant (if the actual is constant) or a named input
    ///   var
    /// - Each formal output gets a named output var in the callee signature
    fn build_callee_env_and_signature(
        &mut self,
        func: &Func<Expr, ExprExt>,
        actual_exprs: &[KoalaBearExpr],
        formal_exprs: &[KoalaBearExpr],
        caller_scope: &str,
    ) -> (Environment, CalleeSignature) {
        let mut env = Environment::new();
        let mut formal_inputs = Vec::new();
        let mut formal_outputs = Vec::new();

        let mut idx = 0usize;
        for (inp_name, attr, shape) in &func.decl.input {
            let w = shape.width();
            for i in idx..idx + w {
                let formal_ref = formal_exprs[i];
                let actual_picus = self.env.translate(caller_scope, &actual_exprs[i]);

                // a symbol that names this formal position in the callee
                let vname = format!("{inp_name}_{i}");
                let formal_var = PicusExpr::Var(vname.clone(), i);

                // default binding: formal → its symbol
                env.bind(&func.decl.name, formal_ref, formal_var.clone());

                match attr.picus {
                    PicusArg::Input => {
                        if let PicusExpr::Const(c) = actual_picus {
                            // specialize the binding to a constant
                            env.bind(&func.decl.name, formal_ref, c.into());
                        } else {
                            formal_inputs.push(formal_var);
                        }
                    }
                    PicusArg::Output => {
                        formal_outputs.push(formal_var);
                    }
                    PicusArg::Unknown => {
                        panic!("Modular translation requires all arguments to have input or output tags")
                    }
                }
            }
            idx += w;
        }

        (env, CalleeSignature { formal_inputs, formal_outputs })
    }

    // -------- Phase 3: wire formal returns ↔ actual return exprs --------

    /// Connect callee formal return slots to caller actual return expressions.
    ///
    /// - Allocates formal output vars (callee side) and returns them as additional module outputs
    /// - Returns the caller-side concrete expressions that should appear in the call result list
    /// - Returns environment updates to seed the callee with those formal outputs
    fn wire_results(
        &mut self,
        actual_out_exprs: &[KoalaBearExpr],
        formal_out: &[KoalaBearExpr],
        caller_scope: &str,
    ) -> ResultWiring {
        let mut extra_call_results = Vec::with_capacity(actual_out_exprs.len());
        let mut extra_module_outputs = Vec::with_capacity(actual_out_exprs.len());
        let mut env_updates = Vec::with_capacity(actual_out_exprs.len());

        for (&formal_out_i, &actual_out_i) in formal_out.iter().zip(actual_out_exprs) {
            let formal_out_var = self.env.fresh();
            let actual_out_picus = self.env.get(caller_scope, &actual_out_i);

            env_updates.push((formal_out_i, formal_out_var.clone()));
            extra_call_results.push(actual_out_picus.clone());
            extra_module_outputs.push(formal_out_var);
        }

        ResultWiring { extra_call_results, extra_module_outputs, env_updates }
    }

    fn walk_func_decl(
        func_decl: &FuncDecl<Expr, ExprExt>,
    ) -> (Vec<KoalaBearExpr>, Vec<KoalaBearExpr>) {
        let mut arg_exprs: Vec<KoalaBearExpr> = Vec::new();
        for (_, _, argi) in &func_decl.input {
            let argi_exprs = Self::flatten_shape(argi);
            arg_exprs.extend(argi_exprs);
        }
        let out_exprs = Self::flatten_shape(&func_decl.output);
        (arg_exprs, out_exprs)
    }

    /// Flatten a shape into a linear list of `ExprRef`s.
    fn flatten_shape(shape: &Shape<Expr, ExprExt>) -> Vec<KoalaBearExpr> {
        let mut shape_exprs = Vec::new();
        match shape {
            Shape::Struct(_, fields) => {
                for (_, field_shape) in fields {
                    let field_exprs = Self::flatten_shape(field_shape);
                    shape_exprs.extend(field_exprs);
                }
            }
            Shape::Array(shapes) => {
                for shape in shapes {
                    let elem_exprs = Self::flatten_shape(shape);
                    shape_exprs.extend(elem_exprs);
                }
            }
            Shape::Word(exprs) => {
                shape_exprs.extend_from_slice(exprs);
            }
            Shape::Expr(expr) => {
                shape_exprs.push(*expr);
            }
            Shape::Unit => {}
            _ => {
                todo!()
            }
        }
        shape_exprs
    }

    /// Byte interaction:
    /// - opcode 6: range-check `arg < 2^pow2` (handles constant vs symbolic `pow2`)
    /// - opcode 3: range-check bytes for equality/compare
    /// - opcode 5: decompose a byte into `(msb, low7)`
    /// - opcode 0/1/2: OR/AND/XOR via a generic `byteop` PCL call
    /// - opcode 4: compare two bytes with `op1 < op2`
    fn translate_byte_interaction(
        &mut self,
        module_name: &str,
        interaction: &AirInteraction<KoalaBearExpr>,
        translated_picus_mods: &mut BTreeMap<String, PicusModule>,
    ) {
        let picus_multiplicity = self.env.translate(module_name, &interaction.multiplicity);
        if let PicusExpr::Const(c) = picus_multiplicity {
            if c == 0 {
                return;
            }
        }
        let picus_opcode = self.env.translate(module_name, &interaction.values[0]);
        if let PicusExpr::Const(c) = picus_opcode {
            if c == 6 {
                let arg = self.env.translate(module_name, &interaction.values[1]);
                let pow2 = self.env.translate(module_name, &interaction.values[2]);
                if let PicusExpr::Const(c) = pow2.clone() {
                    let rng = Self::mod_pow(2, c, KOALABEAR as u64);
                    self.picus_module
                        .constraints
                        .push(PicusConstraint::Lt(Box::new(arg), Box::new(rng.into())));
                }
            } else if c == 3 {
                let arg1 = self.env.translate(module_name, &interaction.values[2]);
                let arg2 = self.env.translate(module_name, &interaction.values[3]);

                self.picus_module
                    .constraints
                    .push(PicusConstraint::Lt(Box::new(arg1), Box::new(PicusExpr::Const(256))));
                if !matches!(arg2, PicusExpr::Const(_)) {
                    self.picus_module
                        .constraints
                        .push(PicusConstraint::Lt(Box::new(arg2), Box::new(PicusExpr::Const(256))));
                }
            } else if c == 5 {
                let msb = self.env.translate(module_name, &interaction.values[1]);
                let byte = self.env.translate(module_name, &interaction.values[2]);
                let fresh_picus_var: PicusExpr = self.env.fresh();
                let picus128_const = PicusExpr::Const(128);
                self.picus_module.constraints.push(PicusConstraint::Lt(
                    Box::new(fresh_picus_var.clone()),
                    Box::new(picus128_const.clone()),
                ));
                self.picus_module.constraints.push(PicusConstraint::Eq(Box::new(
                    msb.clone() * (msb.clone() - PicusExpr::Const(1)),
                )));
                let decomp = byte - (msb * picus128_const + fresh_picus_var);
                self.picus_module.constraints.push(PicusConstraint::Eq(Box::new(decomp)));
            } else if c == 2 || c == 0 || c == 1 {
                // XOR, AND, OR
                let output = self.env.translate(module_name, &interaction.values[1]);
                let inp0 = self.env.translate(module_name, &interaction.values[2]);
                let inp1 = self.env.translate(module_name, &interaction.values[3]);

                // we treat the bitwise ops as uninterpreted functions which are deterministic
                let picus_call = PicusCall {
                    mod_name: BYTEOP_MOD.to_string(),
                    outputs: [output.clone()].to_vec(),
                    inputs: [inp0.clone(), inp1.clone()].to_vec(),
                };
                if !translated_picus_mods.contains_key(&BYTEOP_MOD.to_string()) {
                    // generate a dummy picus module for bitwise ops which just applies range
                    // constraints
                    let bytemod = Self::gen_byteop_module(inp0, inp1, output);
                    translated_picus_mods.insert(bytemod.name.clone(), bytemod);
                }
                self.picus_module.calls.push(picus_call);
            } else if c == 4 {
                let op1 = self.env.translate(module_name, &interaction.values[2]);
                let op2 = self.env.translate(module_name, &interaction.values[3]);
                self.picus_module.constraints.push(PicusConstraint::Lt(
                    Box::new(op1.clone()),
                    Box::new(PicusExpr::Const(256)),
                ));
                self.picus_module.constraints.push(PicusConstraint::Lt(
                    Box::new(op2.clone()),
                    Box::new(PicusExpr::Const(256)),
                ));
                self.picus_module
                    .constraints
                    .push(PicusConstraint::new_lt(op1.clone(), 256.into()));
                self.picus_module
                    .constraints
                    .push(PicusConstraint::new_lt(op2.clone(), 256.into()));
                self.picus_module
                    .constraints
                    .push(PicusConstraint::new_lt(op1.clone(), op2.clone()));
            } else {
                panic!("Unhandled byte interaction: {c}");
            }
        }
    }

    /// The byteop module is a dummy picus module which is a standin for byteops
    /// We assume the SP1 lookup arguments are correct and so we just assert the inputs and output
    /// 8-bit values. Since the lookup arguments are assumed to be correct we effectively ignore
    /// the output of the bitop (telling Picus to assume it is deterministic within the module)
    /// When verifying callers of the module, Picus will check if the inputs are deterministic and,
    /// if they are, will assume the output is deterministic
    fn gen_byteop_module(in_l: PicusExpr, in_r: PicusExpr, o: PicusExpr) -> PicusModule {
        let mut byteop_mod = PicusModule::new(BYTEOP_MOD.to_string());
        byteop_mod.inputs.extend_from_slice(&[in_l.clone(), in_r.clone()]);
        byteop_mod.outputs.push(o.clone());
        byteop_mod.constraints.extend_from_slice(&PicusConstraint::in_range(in_l, 0, 255));
        byteop_mod.constraints.extend_from_slice(&PicusConstraint::in_range(in_r, 0, 255));
        byteop_mod.constraints.extend_from_slice(&PicusConstraint::in_range(o.clone(), 0, 255));
        // tell picus to assume the bitwise ops are deterministic so we don't get a spurious cex to
        // determinism.
        byteop_mod.assume_deterministic.push(o);
        byteop_mod
    }

    /// Multiply-by-inverse trick for large constants:
    /// Rewrites `fresh = const * e` to `fresh * inv - e = 0` (or equivalent for substructure),
    /// then binds `expr` to `fresh` in the environment.
    fn rewrite_mul_inverse(
        &mut self,
        module_name: &str,
        inverse: u64,
        expr: &KoalaBearExpr,
        picus_expr: PicusExpr,
    ) {
        let inv = Self::mod_pow(inverse, KOALABEAR as u64 - 2, KOALABEAR as u64);
        let fresh_var = self.env.fresh();
        self.env.bind(module_name, *expr, fresh_var.clone());
        match picus_expr {
            PicusExpr::Sub(l, r) => {
                let c = PicusConstraint::Eq(Box::new(*r + fresh_var * inv - *l));
                self.picus_module.constraints.push(c);
            }
            _ => {
                self.picus_module
                    .constraints
                    .push(PicusConstraint::Eq(Box::new(fresh_var * inv - picus_expr)));
            }
        }
    }

    fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
        if modulus == 1 {
            return 0;
        }
        let mut result = 1u64;
        base %= modulus;
        while exp > 0 {
            if exp % 2 == 1 {
                result = result.wrapping_mul(base) % modulus;
            }
            exp >>= 1;
            base = base.wrapping_mul(base) % modulus;
        }
        result
    }
}

/// End-to-end translator that:
/// - Seeds the global field modulus
/// - Iterates over chip selectors (phases) and builds one Picus module per selector
/// - Optionally performs field blasting (cross product of constant assignments)
/// - Runs `PicusModuleTranslator` to emit constraints, calls, and IO
/// - Writes the picus program to file `output_dir/chip_name.picus`
pub struct SuccinctChipToPicusTranslator<'a, Expr, ExprExt> {
    chip_name: String,
    program: PicusProgram,
    ast: &'a Ast<Expr, ExprExt>,
    succinct_modules: &'a BTreeMap<String, Func<Expr, ExprExt>>,
    picus_info: PicusInfo,
    modular_extraction: bool,
    ops_to_inline: HashSet<String>,
    output_dir: PathBuf,
    field_blast: &'a [FieldBlastSpec],
    no_apply_summaries: bool,
}

impl<'a> SuccinctChipToPicusTranslator<'a, Expr, ExprExt> {
    #[must_use]
    /// Allocates Extractor
    pub fn new(
        chip_name: &str,
        ast: &'a Ast<Expr, ExprExt>,
        succinct_modules: &'a BTreeMap<String, Func<Expr, ExprExt>>,
        picus_info: &PicusInfo,
        cfg: PicusExtractConfig<'a>,
    ) -> Self {
        if set_field_modulus(KOALABEAR.into()).is_err() {
            panic!("Failed to set PCL field modulus: {KOALABEAR}")
        }
        if picus_info.field_map.is_empty() {
            panic!(
                "Picus info is empty. Make sure the `picus_info` method is overriden in the chip"
            )
        }
        SuccinctChipToPicusTranslator {
            chip_name: chip_name.to_string(),
            program: PicusProgram::new(current_modulus().unwrap()),
            ast,
            succinct_modules,
            modular_extraction: cfg.modular_extraction,
            picus_info: picus_info.clone(),
            ops_to_inline: cfg.ops_to_inline.iter().cloned().collect(),
            output_dir: cfg.output_dir,
            field_blast: cfg.field_blast,
            no_apply_summaries: cfg.no_apply_summaries,
        }
    }

    /// Build the pretty-name map and detect an `is_real` column if present.
    fn build_var_names_and_is_real(&self) -> (BTreeMap<usize, String>, Option<usize>) {
        let mut names = BTreeMap::new();
        let mut is_real = None;

        for (field, low, upper) in &self.picus_info.field_map {
            for i in *low..=*upper {
                names.insert(i, field.clone());
            }
            if field == "is_real" {
                is_real = Some(*low);
            }
        }
        (names, is_real)
    }

    /// Seed an `Environment` for exactly one active selector (and `is_real=1` if present).
    fn env_for_selector(
        &self,
        pretty_names: &BTreeMap<usize, String>,
        sel_idx: usize,
        is_real: Option<usize>,
    ) -> Environment {
        let mut env = Environment::with_names(&self.chip_name, pretty_names.clone());

        // enable this selector
        env.bind(&self.chip_name, KoalaBearExpr::IrVar(IrVar::Main(sel_idx)), PicusExpr::Const(1));

        // optionally force is_real = 1
        if let Some(ir) = is_real {
            env.bind(&self.chip_name, KoalaBearExpr::IrVar(IrVar::Main(ir)), PicusExpr::Const(1));
        }

        // disable all other selectors
        for (other_idx, _) in &self.picus_info.selector_indices {
            if *other_idx == sel_idx {
                continue;
            }
            env.bind(
                &self.chip_name,
                KoalaBearExpr::IrVar(IrVar::Main(*other_idx)),
                PicusExpr::Const(0),
            );
        }

        env
    }

    /// Run the module translator with a provided env and return the produced ``PicusModule``.
    fn extract_with_env(
        &self,
        module_name: &str,
        env: Environment,
        translated: &mut BTreeMap<String, PicusModule>,
    ) -> PicusModule {
        let mut t = PicusModuleTranslator::new(
            self.chip_name.clone(),
            module_name,
            self.ast,
            self.modular_extraction,
            self.succinct_modules,
            env,
            &self.ops_to_inline,
            self.no_apply_summaries,
        );
        t.translate(translated);

        // Add manual input annotations to Picus module. Usually this isn't needed since I/O is
        // determined by interactions.
        for (low, upper, _) in &self.picus_info.input_ranges {
            for i in *low..*upper {
                // assumes that inputs are `Main` columns. If not, need a way of determining what
                // the column type is.
                let picus_var = t.env.translate(&self.chip_name, &ExprRef::main(i));
                t.picus_module.add_input(picus_var);
            }
        }

        // Add manual output annotations to the Picus module.
        for (low, upper, _) in &self.picus_info.output_ranges {
            for i in *low..*upper {
                // assumes that outputs are `Main` columns.
                let picus_var = t.env.translate(&self.chip_name, &ExprRef::main(i));
                t.picus_module.add_output(picus_var);
            }
        }

        t.picus_module
    }

    /// Override the base module's outputs to be the blasted columns, and add bounds as
    /// postconditions.
    fn override_base_outputs_with_blast_bounds(
        &self,
        module: &mut PicusModule,
        env: &mut Environment,
        blast_dims: &[BlastDim],
    ) {
        module.outputs.clear();
        module.postconditions.clear();
        for dim in blast_dims {
            // Reuse the same resolver you already use elsewhere
            let v = env.translate(&self.chip_name, &ExprRef::IrVar(IrVar::Main(dim.col)));
            module.outputs.push(v.clone());
            module
                .postconditions
                .push(PicusConstraint::Leq(Box::new(v.clone()), Box::new(dim.hi.into())));
            module.postconditions.push(PicusConstraint::Geq(Box::new(v), Box::new(dim.lo.into())));
        }
    }

    /// Generate the full Picus program:
    /// - build pretty names and detect `is_real` (makes `is_real` a selector)
    /// - for each selector, extract the base (unblasted) module
    /// - if blasting enabled: override base outputs to blasted vars (+bounds) and also extract
    ///   specialized modules
    pub fn generate_picus_program(&mut self) {
        let mut translated = BTreeMap::new();

        // 1) names + optional is_real
        let (var_id_to_name, is_real) = self.build_var_names_and_is_real();

        // 2) blasting setup
        let blast_dims = resolve_blast_dims(&self.picus_info, self.field_blast);
        let blast_assignments = cartesian_product_usize(&blast_dims); // Vec<BlastAssignment>

        let mut selector_indices = self.picus_info.selector_indices.clone();
        if selector_indices.is_empty() {
            if let Some(is_real_col) = is_real {
                selector_indices.push((is_real_col, "is_real".to_string()));
            }
        }
        if selector_indices.is_empty() {
            panic!(
                "No columns marked as selector and no `is_real` column found! \
                Annotate selector columns with #[picus(selector)]."
            );
        }
        // 3) per-selector extraction
        for (sel_idx, selector) in &selector_indices {
            let base_name = format!("{}_{}", self.chip_name, selector);

            // Base env
            let mut base_env = self.env_for_selector(&var_id_to_name, *sel_idx, is_real);

            // Extract base (unblasted) module
            let mut base_mod = self.extract_with_env(&base_name, base_env.clone(), &mut translated);

            // If blasting is enabled, override outputs with blasted columns and add bounds.
            if !blast_assignments.is_empty() {
                self.override_base_outputs_with_blast_bounds(
                    &mut base_mod,
                    &mut base_env,
                    &blast_dims,
                );
            }

            translated.insert(base_name.clone(), base_mod);

            // Specialized modules for each assignment (if any)
            if !blast_assignments.is_empty() {
                for assignment in &blast_assignments {
                    let mut name = base_name.clone();
                    name.push('_');
                    name.push_str(&assignment.name());

                    // Start from a clean selector env, then pin the blasted columns.
                    let mut env = self.env_for_selector(&var_id_to_name, *sel_idx, is_real);
                    for (col, val) in &assignment.0 {
                        env.bind(
                            &self.chip_name,
                            KoalaBearExpr::IrVar(IrVar::Main(*col)),
                            (*val as u64).into(),
                        );
                    }

                    let modu = self.extract_with_env(&name, env, &mut translated);
                    translated.insert(name, modu);
                }
            }
        }

        // 4) assemble & write
        self.program.add_modules(&mut translated);
        let _ =
            self.program.write_to_path(self.output_dir.join(format!("{}.picus", self.chip_name)));
    }
}

/// Configuration knobs for Picus extraction.
///
/// This groups all the optional flags so your constructor doesn’t trip
/// Clippy’s `too_many_arguments`, and makes it easier to pass settings around.
///
/// Lifetimes:
/// - `'a` is the lifetime of the *source* that owns the field-blast specs
#[derive(Debug, Clone)]
pub struct PicusExtractConfig<'a> {
    /// If true, extract callees as separate Picus modules when possible
    /// instead of inlining everything. Usually reduces output size.
    pub modular_extraction: bool,

    /// Operations that must always be inlined even if modular extraction is on.
    /// Useful for tiny helper ops or places where specialization explodes.
    pub ops_to_inline: HashSet<String>,

    /// Where to write the final `chip_name.picus`. The directory will be
    /// created by the caller if needed.
    pub output_dir: PathBuf,

    /// Field-blasting directives. Each spec says which column(s) to fix and
    /// which value range(s) to enumerate. We borrow these so you can keep the
    /// specs inside your CLI args without cloning.
    pub field_blast: &'a [FieldBlastSpec],

    /// Do not replace operations with builtin summaries (if they exist) that have been proven
    /// deterministic offline.
    pub no_apply_summaries: bool,
}

impl Default for PicusExtractConfig<'_> {
    fn default() -> Self {
        Self {
            modular_extraction: false,
            ops_to_inline: HashSet::new(),
            output_dir: PathBuf::from("picus_out"),
            field_blast: &[],
            no_apply_summaries: true,
        }
    }
}
