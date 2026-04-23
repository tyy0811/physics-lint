# PH-POS-001 — Evans positivity for Poisson and heat

## Function-labeled citation stack

Per complete-v1.0 plan §1.3 three-function labeling. Retrofit landed during
Task 0 Step 7 of the complete-v1.0 plan; Tier-A content preserved with the
function-labeled structure added as the primary organizational layer.

### Mathematical-legitimacy (Tier 3 classical-textbook theorem)

- **Primary (Poisson)**: Evans, L.C. (2010). *Partial Differential Equations*,
  2nd ed. AMS Graduate Studies in Mathematics Vol. 19. ISBN 978-0-8218-4974-3.
  §2.2.3 **Theorem 4** (Strong maximum principle) together with its
  "Positivity" corollary on book p. 27: "if U is connected and u satisfies
  `{-Δu = 0 in U, u = g on ∂U}` with g ≥ 0, then u is positive everywhere in
  U if g is positive somewhere on ∂U." Consumed for the Poisson case with
  `f ≥ 0`, homogeneous Dirichlet BCs.
- **Primary (heat)**: Evans §2.3.3 **Theorem 4** (Strong maximum principle
  for the heat equation): `max_{Ū_T} u = max_{Γ_T} u`, where Γ_T is the
  parabolic boundary.
- **Cross-reference**: Protter, M.H. & Weinberger, H.F. (1999). *Maximum
  Principles in Differential Equations.* Dover. ISBN 978-0-486-41302-3.
  Chapters 2 and 3.
- **Verification status** (per `../_harness/TEXTBOOK_AVAILABILITY.md`): Evans
  §2.2.3 Theorem 4 (with its p. 27 Positivity corollary paragraph) and
  §2.3.3 Theorem 4 are both ✅ primary-source verified (AGH Kraków mirror,
  Tier-A Task 0 Step 17 on 2026-04-20, book pages 27 and 55 respectively).
  Tight theorem-number framing is appropriate.
- **Spec-correction provenance**: the Rev 1.6 design spec's attributions of
  "§2.2.4 Theorem 13" (Poisson) and "§2.3.3 Theorem 8" (heat) are factually
  wrong — §2.2.4 Theorem 13 is the symmetry of the Green's function, and
  Evans restarts theorem numbering per section so "Theorem 8" in §2.3.3
  does not exist. Corrections land in this CITATION.md and
  `../_harness/TEXTBOOK_AVAILABILITY.md`; the committed design spec is
  preserved verbatim at commit `78d4cba` per the plan-vs-committed-state
  drift discipline (`feedback_plan_vs_committed_drift.md`).

### Correctness-fixture (CI-runnable, non-credibility-claiming)

- **Poisson fixture**: `u(x,y) = x(1-x)y(1-y)` on the unit square, 64×64.
  The function is factor-wise non-negative on `[0,1]²` and `-Δu =
  2[y(1-y) + x(1-x)] ≥ 0` satisfies the Positivity-corollary hypothesis;
  the corollary gives `u ≥ 0` throughout.
- **Heat fixture**: `u(x,y,t) = 1 + 0.5 · exp(-8π²t) sin(2πx) sin(2πy)`
  periodic at `t ∈ {0, 0.02, 0.05, 0.1}`, 64×64, time-axis LAST
  (`grid_shape=[64, 64, 4]`, `h=(H, H, 0.02)`, `diffusivity=1.0`). Bounded
  `u ≥ 0.5` at all t since `|sin · sin| ≤ 1` with constant offset 1.0.
- **Negative control**: Poisson fixture with a `-0.8` patch injected on a
  5×5 interior region; tests "pointwise min < floor" branch.
- **Audit record**: `./AUDIT.md` in this directory. PH-POS-001 is a pure
  discrete-predicate rule (pointwise `u.min() >= floor` post-BC-gate); the
  enumerate-the-splits audit (per `feedback_readback_codebase_api_contract.md`
  2026-04-22 extension) was run after the Task 4 PH-RES-001 n=3 escalation
  and came back clean — consistent with the discrete-predicate partition
  (`feedback_deviation_pattern_escalation.md`).

### Borrowed-credibility (external published reproduction layers)

**F3 absent with justification.** PH-POS-001 is a discrete-predicate rule —
the emitted quantity is a verdict (PASS/FAIL) with `raw_value` equal to
`min(u)` when it falls below the positivity floor. No published numerical
reproduction target exists for pointwise positivity verdicts on
author-constructed analytical fixtures; the fixtures are analytically-known-
positive and the rule's verdict is mechanically secure on them, not
borrowed from a peer-reviewed table. Per complete-v1.0 plan §1.2,
F3-absent-is-structural for discrete-predicate rules whose correctness
layer is analytical.

### Supplementary calibration context

(None — no calibration-only references accompany this rule.)

## Citation summary

- **Paper:** Evans, *Partial Differential Equations*.
- **Venue:** AMS Graduate Studies in Mathematics, Vol. 19, 2nd ed. 2010.
- **ISBN:** 978-0-8218-4974-3.
- **Sections:**
  - Section 2.2.3 Theorem 4 (strong maximum principle for harmonic functions)
    together with its "Positivity" corollary on book p. 27 ("if U is
    connected and u satisfies `{-Lap u = 0 in U, u = g on dU}` with
    g >= 0, then u is positive everywhere in U if g is positive somewhere
    on dU"). Consumed here for the Poisson case with f >= 0, homogeneous
    Dirichlet.
  - Section 2.3.3 Theorem 4 (strong maximum principle for the heat
    equation): `max_{Ubar_T} u = max_{Gamma_T} u`, where Gamma_T is the
    parabolic boundary.
- **Cross-reference:** Protter-Weinberger, *Maximum Principles in
  Differential Equations*, Dover 1999, ISBN 978-0-486-41302-3, Chapters 2
  and 3.
- **Pinned value:** verdict-based - PASS on both analytical fixtures, FAIL
  on the injected negative spike.
- **Verification date:** 2026-04-20.
- **Verification protocol:** analytical derivation. For Poisson:
  `u = x(1-x)y(1-y)` is factor-wise non-negative on `[0,1]^2` and
  `-Lap u = 2[y(1-y) + x(1-x)] >= 0` satisfies the Positivity corollary's
  hypothesis (f >= 0 on the interior, g = 0 on dU; the corollary gives
  u >= 0 throughout). For heat:
  `u(x,y,t) = 1 + 0.5 * exp(-8 pi^2 t) sin(2 pi x) sin(2 pi y)`
  has `u >= 0.5` at all `t` (since `|sin * sin| <= 1` and the constant
  offset is 1.0); this is consistent with the heat max principle's
  positivity-preservation corollary (max on the parabolic boundary, with
  non-negative initial and boundary data).

Evans theorem numbers verified at theorem-level precision against the text
(AGH Krakow mirror, pymupdf + Claude vision pass on 2026-04-20) and recorded
verbatim in `../_harness/TEXTBOOK_AVAILABILITY.md`. The Rev 1.6 spec's
attributions of "Section 2.2.4 Theorem 13" (Poisson) and "Section 2.3.3
Theorem 8" (heat) are factually wrong in Evans - Theorem 13 in Section 2.2.4
is the symmetry of the Green's function, and Evans restarts theorem numbering
per section so "Theorem 8" in Section 2.3.3 does not exist. Both corrections
land in this per-rule CITATION.md only; the committed design spec is
preserved verbatim at commit 78d4cba.

## Pre-execution audit

See `AUDIT.md` in this directory. PH-POS-001 is a pure discrete-predicate
rule (pointwise `u.min() >= floor` post-BC-gate); the enumerate-the-splits
audit was run per `feedback_deviation_pattern_escalation.md` after the
n=3 escalation on PH-RES-001 and came back clean.

## Test design

- **Poisson fixture:** `u(x,y) = x(1-x)y(1-y)` on unit square, 64x64.
- **Heat fixture:** `u(x,y,t) = 1 + 0.5 * exp(-8 pi^2 t) sin(2 pi x) sin(2 pi y)`,
  periodic, at `t in {0, 0.02, 0.05, 0.1}`, 64x64. Axis convention: time
  LAST (`grid_shape=[64, 64, 4]`, `h=(H, H, 0.02)`). diffusivity=1.0.
- **Negative control:** Poisson fixture with a `-0.8` patch injected on a
  5x5 interior region.

## Acceptance criteria

- Poisson polynomial PASSES; `min(u) >= 0` on every grid point.
- Heat fixture PASSES at every timestep.
- Injected-spike negative control FAILS with `raw_value < 0`.
- Wall-time < 10 s on CPU.
