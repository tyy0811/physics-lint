# PH-NUM-002 external-validation anchor

**Scope separation (read first):** PH-NUM-002 validates the production
rule's ability to **detect observed convergence-order on explicitly
declared manufactured-solution cases**. The expected rate is **case-
specific** and depends on PDE, backend, boundary treatment, and
asymptotic regime. The anchor does **not** certify convergence for
arbitrary PDE / backend / BC triples.

The F2 fixture splits into three scoped cases:

- **Case A — F2 harness-level (authoritative):** `_harness/mms.py`
  `mms_observed_order_fd2` — pure-interior 2nd-order FD Laplacian on
  smooth harmonic fixtures. Measured `p_obs` asymptote → 2.00.
  Textbook-clean observed-order methodology anchor, independent of the
  rule's FD4 stencil.
- **Case B — rule-verdict, FD + non-periodic:** rule `PH-NUM-002.check`
  on boundary-dominated FD4 path. Measured `refinement_rate` asymptote
  → 2.50 (derivation: `4N`-cell 2nd-order boundary band dominates
  `N²`-cell 4th-order interior at `O(h^{2.5})` on 2D).
- **Case C — rule-verdict, saturation floor:** either spectral+periodic
  on a period-compatible harmonic (Liouville: = constant) or
  FD+non-periodic on a harmonic polynomial (2nd-order FD exact). Both
  residuals below `_SATURATION_FLOOR = 1e-11` → rule returns
  `rate = inf PASS`. No algebraic rate asserted.

Plan §20's "SymPy MMS for Laplace/Poisson/heat" narrowed to Laplace-only
F2 with SKIP-path contracts for Poisson / heat (rule is Laplace-only by
V1 scope per `ph_num_002.py:92`). Plan §20's single-tolerance criterion
`p_obs matches expected within 0.1` replaced with per-case tolerance
bands (Case A `±0.1`, Case B `±0.25`, Case C exact `inf`) per the
2026-04-24 user-revised Task 12 contract.

Run:

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-NUM-002/ -v
```

Expected: 21 passed in < 20 s (6 Case A parametrized + 2 Case A log-log
slope + 8 Case B parametrized + 1 Case B asymptote + 2 Case C saturation
+ 2 SKIP contracts).

Pure numpy — no mesh assembly, no torch / scikit-fem.

## Citation stack (three-function-labeled)

See `CITATION.md` for the full function-labeled citation stack. Summary:

- **F1 Mathematical-legitimacy** (Tier 2 theoretical-plus-multi-paper):
  Strikwerda 2004 Ch 10 Lax equivalence (section-level per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠); Roy 2005 *J. Comput. Phys.*
  `p_obs = log₂(e_h / e_{h/2})` formula (DOI 10.1016/j.jcp.2004.10.036);
  Ciarlet 2002 §3.2 Céa's lemma (section-level ⚠); Oberkampf-Roy 2010
  Chs 5–6 (section-level ⚠). Five-step proof-sketch with L²-scaling
  derivation of Case A p=2 and Case B p=2.5 boundary-dominance.
- **F2 Correctness-fixture (harness-level, authoritative)**:
  `external_validation/_harness/mms.py` `mms_observed_order_fd2`.
  Measured `p_obs ∈ [2.0021, 2.0373]` across `{exp(x)cos(y),
  sin(πx)sinh(πy)} × {16→32, …, 256→512}`; asymptote 2.00. Tolerance
  `[1.9, 2.1]`.
- **F2 Rule-verdict contract — boundary-dominated**: rule on
  fd+non-periodic gives `refinement_rate ∈ [2.48, 2.63]` across the
  same fixtures + N pairs; asymptote 2.50; tolerance `[2.3, 2.8]`. Rule
  PASSes at all measured pairs.
- **F2 Rule-verdict contract — saturation floor**: rule on
  spectral+periodic `u=0` or fd+non-periodic `x²−y²` → `rate=inf PASS`.
  No algebraic rate asserted.
- **F3 Borrowed-credibility**: **absent with justification** — no live
  MMS benchmark dataset publishes a reproducible `p_obs` for a specific
  PDE+backend+BC triple matching physics-lint's rule path. Oberkampf-Roy
  and Roy 2005 retained in Mathematical-legitimacy as methodology +
  theoretical framing. Per 2026-04-24 user-revised Task 12 contract:
  "published MMS / verification literature can remain mathematical or
  supplementary context."
- **Supplementary calibration context**: Roy 2005 §4 published
  numerical examples (methodology-level reference, flagged not-a-
  reproduction).

## Liouville scope-truth observation

The rule's docstring (`ph_num_002.py:9-22`) lists three backend / BC
regimes including "fd4 interior-dominated (periodic): ~4 per doubling."
This regime is **structurally unreachable** with any non-constant
harmonic fixture on a periodic grid — Liouville's theorem on the
2-torus forces every periodic harmonic to be constant, and constants
trivially saturate. The rule's shipped behavior on periodic paths is
correct (constants saturate via `_SATURATION_FLOOR`; non-harmonic
periodic inputs correctly WARN at `rate ≈ 0` because residual does not
shrink under refinement). The V1 anchor's Case C covers the reachable
saturation regime; the theoretical `~4 per doubling` claim is retained
in the rule docstring for reference but not validated as an observed
rate in V1. See CITATION.md proof-sketch step 5 and the "Liouville on
T²" note.

## Plan-diffs (18 cumulative across Tier-B execution)

See `test_anchor.py` module docstring for diffs 14–18 (Task 12). Diffs
1–13 are from Tasks 2, 3, 4, 5, 8, 9. Summary of Task 12 diffs:

14. Plan §20 "SymPy MMS for Laplace/Poisson/heat" narrowed to Laplace-
    only F2 (rule V1 scope); Poisson / heat covered only by SKIP-path
    contracts per `ph_num_002.py:92-97`.
15. Plan §20 "p_obs matches expected within 0.1" single-tolerance
    replaced with per-case tolerance bands per 2026-04-24 user-revised
    contract: Case A `±0.1`, Case B `±0.25`, Case C exact `inf`.
16. Plan §20 "three-level vs four-level Richardson extrapolation" not
    exercised (rule's shipped path is simple two-level `log₂` ratio at
    `ph_num_002.py:127`; Richardson extrapolation does not enter the
    rule's emitted quantity). Logged as Supplementary methodology
    reference.
17. Plan §20 "Oberkampf-Roy Chs 5–6 p_obs reproduced" borrowed-
    credibility claim → F3 absent-with-justification (no live MMS
    benchmark dataset; Oberkampf-Roy remains in Mathematical-legitimacy
    + Supplementary calibration context as methodology reference).
    Consistent with 2026-04-24 user-revised F3 contract.
18. Rule docstring claim "fd4 interior-dominated (periodic): ~4 per
    doubling" is structurally unreachable (Liouville on T² forces
    periodic harmonics to constants → saturation). V1 F2 scope
    restricted to Case A (harness 2nd-order MMS), Case B (rule fd+
    non-periodic, boundary-dominated ~2.5), Case C (saturation floor).
    Rule's shipped behavior remains correct; no code change. Logged
    scope-truth observation in CITATION.md.
