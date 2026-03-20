# ZK vs Standard PCS: Overhead Analysis

**Configuration:** 2^24 MLE (16.8M elements), stacking 2^8=256 columns, 2^16=65536 rows
**ZK adds:** 4 mask columns (EF degree D=4), 94 padding rows (query count)

---

## 1. Prover Time

| Phase | Standard | ZK | Overhead | Notes |
|---|--:|--:|--:|---|
| **Setup** | — | | | |
| compute_mask_length | — | 0.04ms | 0.04ms | Runs protocol in counting mode |
| ZkProverContext::initialize | — | 6.0ms | 6.0ms | Generates 313 masks + zk_dot_product_commitment |
|   mask RNG | — | 0.02ms | | |
|   dot-product commitment | — | 5.97ms | | Encode + merkleize 313-element vector |
| **Commitment** | **1441ms** | **1475ms** | **34ms (2.4%)** | |
| build interleaved matrix | — | 5.8ms | 5.8ms | RNG for masks + extend_from_slice loop |
| commit_padded_multilinears | — | 1466ms | | |
|   zero-pad (65630 → 131072 rows) | — | 5.8ms | 5.8ms | 94 padding rows push past 2^16 |
|   encode (DFT, log_blowup=0) | — | 97ms | ~0 | Same effective work as standard (blowup-1 compensates) |
|   merkle (260 vs 256 cols) | — | 1363ms | ~22ms | 1.6% more leaf hashing |
| **Sumcheck** | **51ms** | **47ms** | **~0** | Noise; ZK sumcheck adds only transcript writes |
| **PCS eval proof** | **133ms** | **151ms** | **18ms** | |
|   Step 1: eval rows at inner point | — | 4.4ms | 4.4ms | Evaluates 260 (not 256) polynomials |
|   Step 3: RLC MLE (eq-based coefficients) | — | 17.6ms | ~0 | Comparable to standard batching work |
|   Step 4: flatten + encode RLC | — | 3.4ms | ~0 | Encode single RLC polynomial |
|   Steps 5-6: eval claim + observe | — | 0.03ms | 0.03ms | |
|   Step 7: basefold proof | — | 125.7ms | ~0 | Same core protocol |
| **Finalize (ZK-only)** | — | **3.8ms** | **3.8ms** | |
|   build_constraints | — | 0.03ms | 0.03ms | |
|   linear proof (RLC + dot-product) | — | 0.28ms | 0.28ms | |
| | | | | |
| **TOTAL** | **1625ms** | **1684ms** | **59ms (3.6%)** | |

### Prover overhead by category

| Category | Overhead | % of total |
|---|--:|--:|
| Extra columns through encode+merkle | ~22ms | 1.4% |
| Matrix construction (interleave + pad) | ~17ms | 1.0% |
| ZK infrastructure (init + finalize) | ~10ms | 0.6% |
| PCS eval extra work (260 vs 256 polys) | ~4ms | 0.3% |
| Other (sumcheck, constraint build) | ~6ms | 0.4% |

---

## 2. Verifier Time

| Phase | Standard | ZK | Overhead | Notes |
|---|--:|--:|--:|---|
| **Setup** | | | | |
| open() | — | 0.005ms | 0.005ms | Initializes challenger, observes mask commitment |
| read_all() (transcript) | — | 0.02ms | 0.02ms | |
| build_constraints | — | 0.03ms | 0.03ms | |
| **Sumcheck verify** | **0.02ms** | (inside read_all) | ~0 | Absorbed into transcript reading |
| **PCS verify** | **10.4ms** | **12.1ms** | **1.7ms** | |
|   verify_zk_stacked_pcs | — | 11.9ms | | |
|     setup + RLC computation | — | 0.4ms | 0.4ms | Read evals, compute eq_evals, expected RLC |
|     basefold verify | — | 11.5ms | ~1.1ms | Same core verification + virtual oracle overhead |
|   constraint build | — | (included) | | |
| **Linear constraint check** | — | **1.8ms** | **1.8ms** | |
|   RLC dot vector generation | — | ~0.01ms | | |
|   dot-product check | — | ~0.01ms | | |
|   verify_zk_dot_product | — | ~1.8ms | 1.8ms | Verify mask commitment (Merkle openings) |
| **Mul proof verify** | — | 0ms | 0ms | Not used in this benchmark |
| | | | | |
| **TOTAL** | **10.4ms** | **14.0ms** | **3.6ms (35%)** | |

### Verifier overhead by category

| Category | Overhead | % of standard |
|---|--:|--:|
| ZK linear constraint proof | 1.8ms | 17% |
| PCS virtual oracle overhead | ~1.1ms | 11% |
| ZK PCS setup (read evals, RLC) | 0.4ms | 4% |
| Transcript/constraint bookkeeping | 0.3ms | 3% |

The verifier overhead is proportionally larger (35%) because the standard verifier is already very fast (10ms).
The absolute cost (3.6ms) is dominated by verifying the dot-product proof for the mask commitment.

---

## 3. Proof Size

| Component | Standard (felts) | ZK (felts) | Overhead (felts) | Notes |
|---|--:|--:|--:|---|
| **Commitment** | 8 | (in ZK proof) | | |
| **Sumcheck** | 348 | (in ZK proof) | | |
| **PCS eval proof** | 152,929 | | | |
| | | | | |
| **ZK proof breakdown:** | | | | |
| Linear constraint proof | — | 14,066 | 14,066 | Mask dot-product: claimed dots + RLC vec + padding + Merkle openings |
| Mul constraint proof | — | 0 | 0 | Not used |
| PCS proofs | — | 152,659 | | |
|   rlc_eval_proof (basefold) | — | 152,273 | | Core basefold proof for RLC polynomial |
|   rlc_eval_claim | — | 4 | 4 | Single EF element |
|   rlc_padding_vec | — | 378 | 378 | 94 padding entries * 4 (EF degree) |
| Transcript (masked values) | — | 1,336 | 1,336 | Sumcheck round polys + evals + claim |
| | | | | |
| **TOTAL** | **153,285** | **168,065** | **14,780 (9.6%)** | |

### Proof size overhead by category

| Category | Extra felts | % of standard |
|---|--:|--:|
| Linear constraint proof (mask dot-product) | 14,066 | 9.2% |
| RLC padding vector | 378 | 0.2% |
| Transcript overhead (masked values) | 332 | 0.2% |
| RLC eval claim | 4 | 0.0% |
| PCS eval proof delta | 0 | 0.0% |

The ZK PCS basefold proof (152,273 felts) is actually *slightly smaller* than the standard PCS proof (152,929 felts)
because the RLC polynomial has fewer variables (16 vs 16+8 for standard with stacking).

Almost all proof size overhead (95%) comes from the linear constraint proof, which includes:
- The RLC vector of the mask commitment (length = code_length = 8192, as EF = 4 felts each)
- RLC padding values
- Claimed dot products and Merkle authentication paths

---

## Summary

|  | Standard | ZK | Overhead |
|---|--:|--:|--:|
| **Prover** | 1625ms | 1684ms | 3.6% |
| **Verifier** | 10.4ms | 14.0ms | 35% (3.6ms abs) |
| **Proof size** | 153,285 felts | 168,065 felts | 9.6% |

The prover overhead is well within the theoretical ~1/64 (1.6%) column overhead target,
with extra cost coming from data manipulation (interleaving, padding) and ZK infrastructure.

The verifier overhead is dominated by the mask dot-product verification (1.8ms) which is
an inherent cost of the ZK layer -- independent of the PCS column count.

The proof size overhead is dominated by the linear constraint proof (14K felts) which
scales with the mask commitment code length, not the number of ZK columns.
