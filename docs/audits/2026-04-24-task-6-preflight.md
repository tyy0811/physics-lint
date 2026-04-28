# Task 6 (PH-SYM-003) pre-execution audit — 2026-04-24

Authored before any implementation code, per the user's 2026-04-24
stricter-preflight contract for Task 6 (mathematical preflight gate
before implementation). Lives alongside the Task 0 planning audits
and the 2026-04-24 plan-smoke-precheck.

## 1. Rule-source audit (codebase-API-contract)

**Source.** `src/physics_lint/rules/ph_sym_003.py`.

**V1 emitted quantity.** Per-point L² norm of the single-generator
Lie derivative

```
(L_A f)(x) := d/dθ |_{θ=0} ρ_Y(exp(θA))^{-1} · f( ρ_X(exp(θA)) · x )
```

computed via `torch.autograd.functional.jvp` with tangent `v = 1` and
base `θ = 0`. `A` is the `so(2)` generator `[[0,-1],[1,0]]` acting on
the 2D grid; the rule assumes `ρ_Y = identity` (scalar-invariant
output), so the quantity simplifies to `d/dθ |_0 f(R_θ x)`.

**V1 gates** (emit SKIPPED on failure):

1. `SO2` not declared in `SymmetrySpec` → "SO2 not declared".
2. Dump mode (field is not `CallableField`) → "requires a callable
   model; dump mode provides only a frozen tensor".
3. `grid.shape[-1] != 2` → "requires a 2D spatial grid".
4. `grid.numel() == 0` → "requires a non-empty sampling grid".
5. Grid not origin-centered (`center_offset > 1e-6 * max(grid_max, 1)`)
   → "requires an origin-centered sampling grid".
6. Non-square domain (`abs(lx - ly) / max(lx, ly) > 1e-6`)
   → "requires a square domain".

When all six gates pass, the rule computes `lie_norm` and compares
against a floor via `_tristate`:
- `lie_norm / floor ≤ 10 × tolerance` → `PASS`
- `10 × tolerance < ratio ≤ 100 × tolerance` → `WARN`
- `ratio > 100 × tolerance` → `FAIL`

**V1 scope narrower than F1 mathematical family.** The rule computes
the infinitesimal constraint only; it does not test finite rotations,
non-scalar outputs, or disconnected-component symmetries. Per
`feedback_narrower_estimator_than_theorem.md`, F1 must scope the
anchor claim to the implemented quantity.

## 2. Infinitesimal ↔ finite equivariance (user's revised contract)

Let `G = SO(2)` (connected, compact, abelian Lie group; Lie algebra
`so(2) ≅ R` with single generator `A`). Input representation `ρ_X`,
output representation `ρ_Y`. A map `f : X → Y` is finitely
equivariant iff

```
f(ρ_X(g) x) = ρ_Y(g) f(x)    for all g ∈ SO(2).         (E_finite)
```

**Finite ⇒ infinitesimal (trivial direction).** Differentiate `E_finite`
along the one-parameter subgroup `g(t) = exp(tA)` at `t = 0` (assumes
`f` is smooth):

```
d/dt |_{t=0} f(ρ_X(exp(tA)) x)  =  d/dt |_{t=0} ρ_Y(exp(tA)) f(x)
              i.e.  ρ_X_*(A) · ∇f(x)  =  ρ_Y_*(A) · f(x).        (E_inf)
```

The rule measures violation of `E_inf` (not `E_finite`) with
`ρ_Y = identity`, giving `ρ_X_*(A) · ∇f(x) = 0`, i.e.
`L_A f ≡ 0`. When `f` is genuinely invariant, the rule is 0 up to
roundoff; when `f` is not invariant, the rule is nonzero and
proportional to how badly `f` fails to be SO(2)-invariant.

**Infinitesimal ⇒ finite (delicate direction).** The reverse
implication requires:

- **smoothness** of `f` (so the Taylor expansion to all orders is
  valid);
- **connected group / component** (SO(2) is connected; disconnected
  components would require one generator per component);
- **generator coverage** (for `so(2)` one-dimensional, the single `A`
  suffices; for higher-dim Lie algebras like `so(3)`, all generators
  must vanish on `f`);
- **exact constraint** (the infinitesimal identity holds identically
  in `x`, not only pointwise noisy at measured samples).

Under all four, the identity-component theorem (Hall 2015 §2.5 +
§3.7 section-level; Varadarajan 1984 §2.9–2.10 section-level) gives
`E_finite` on the identity component of `G`. Disconnected-group
claims require a separate argument per component.

**The empirical rule does NOT claim `E_inf ⇒ E_finite` globally.**
It measures a diagnostic: a one-sample-at-origin evaluation of
`L_A f` (after `jvp`) averaged over the grid with a per-point L²
norm. This is strictly narrower than:
(a) full finite equivariance testing,
(b) a proof that any model satisfying the infinitesimal identity is
    also finitely equivariant,
(c) a guarantee that the single-generator evaluation implies
    multi-generator equivariance in higher-dim Lie groups.

## 3. Scalar-invariant restriction

Rule assumes `ρ_Y = identity`. V1 tests scalar-output invariant maps
only. For equivariant maps whose output transforms non-trivially
under `G` (e.g., a 2D vector field that rotates with the input grid),
the rule's computation would compare `f(R_θ x)` at `θ = 0` against
itself, missing the `ρ_Y(R_θ)` factor. The rule would register 0 on
a correctly equivariant vector-field output — a false PASS.

V1 scope consequence: PH-SYM-003 validates scalar-invariance, not
general SO(2)-equivariance. V1.1 could extend to vector-output
equivariance with a representation-tagged output contract.

## 4. F3 executability infrastructure audit

Per `feedback_precheck_f3_executability_category.md` (fifth precheck
category: does V1 actually ship the loader/adapter/cache for each
F3-PRESENT pin?).

**Codebase grep.** `grep -r "rotmnist|RotMNIST|modal\.com|escnn|e3nn|
gruver|lie[_-]deriv" --include="*.py" --include="*.toml" --include=
"*.yml" --include="*.yaml"` returns only:

- `pyproject.toml:97` — "gruver" is in the `codespell.ignore-words-
  list` (dictionary of permitted acronyms), not a dependency.
- `scripts/measure_sym_floors.py` — Tier-A script for C4 floor
  measurement; no RotMNIST / escnn / Modal dependence.
- `src/physics_lint/rules/ph_sym_003.py` — the rule itself.

**`pyproject.toml` optional-dependency groups.** No `equivariance`
group; no escnn or e3nn opt-in dep; no Modal client. Plan §0.1
promised `escnn` and `e3nn` as opt-in dev deps but none have landed
in V1.

**`.github/workflows/`.** `ci.yml`, `external-validation.yml`,
`physics-lint.yml` — none reference RotMNIST, Modal, escnn, or
ImageNet. No `workflow_dispatch` trigger for a pre-release Modal A100
validation.

**Classification: F3-INFRA-GAP.** Plan §14 Task 6 F3 (two-layer
RotMNIST policy + ImageNet-opt-in Gruver) cannot be executed as live
CI reproduction in V1. Resolution per the memory + user's revised F3
contract: **pre-demote F3 → F3-absent + Supplementary calibration
context.**

**Supplementary candidates** (theoretical framing, explicitly NOT
reproduction):
- Cohen-Welling 2016 P4CNN (ICML, arXiv:1602.07576) — canonical
  G-equivariant CNN reference. Published RotMNIST: 2.28%.
- Weiler-Cesa 2019 E(2)-CNN (NeurIPS, arXiv:1911.08251) — steerable
  CNN with published RotMNIST 0.705 ± 0.025%.
- Gruver et al. 2023 (ICLR, arXiv:2210.02984) — LEE metric on
  ImageNet-scale classifiers. Same Lie-derivative structure the rule
  inherits.

These are framing references: the rule's quantity is inspired by
Gruver's LEE and the Cohen / Weiler lineage motivates why
equivariance matters, but no Cohen / Weiler / Gruver reproduction
layer is shipped in V1. Author-measured SO(2) fixture numbers in
CITATION.md are **new** measurements, not reproductions of any of
those papers.

## 5. Borrowed structure from Tasks 1, 5, 7, 11

- **Task 1 retrofit** established Hall + Varadarajan section-level
  framing pattern + `_harness/symmetry.py` rotate_test / reflect_test
  for C4 / Z2.
- **Task 5 (PH-BC-002)** first CRITICAL three-layer (V1 stub SKIP):
  F1 math + F2 harness-authoritative + rule-verdict contract.
- **Task 7 (PH-SYM-004)** CRITICAL three-layer on another SKIP-
  always stub: controlled-operator harness (identity + circular conv
  + Fourier multiplier + coord-dependent negative control). Shares
  `_harness/symmetry.py`.
- **Task 11 (PH-NUM-001)** CRITICAL three-layer on PASS-with-reason
  stub: polynomial exactness + under-integration fixture.

Task 6 inherits Task 7's harness module and adds **continuous SO(2)
Lie-derivative primitives** beyond the discrete C4 / Z2 and discrete
translation cases. Rule-verdict contract exercises a PASS / WARN /
FAIL live-callable path (not all-SKIP) because the rule emits live
values in adapter mode.

## 6. Plan-diffs to log in the Task 6 commit Provenance

Expected plan-diffs (n = 29 cumulative before Task 6; Task 6 is
expected to add roughly 5–7 diffs):

- **Plan §14 two-layer RotMNIST CI policy → F3-absent + Supplementary.**
  Reason: F3-INFRA-GAP (no Modal / RotMNIST / escnn in V1
  codebase). Authorized by user's 2026-04-24 revised F3 contract.
- **Plan §14 ImageNet-opt-in Gruver reproduction → Supplementary
  framing only.** Same reason (no ImageNet loader, no Gruver
  `lie-deriv` integration in V1).
- **Plan §14 EMLP + escnn + e3nn three-cross-library correctness
  fixture → controlled-operator harness (radial positive + coord/
  anisotropic negatives + finite-vs-infinitesimal Case C).** Same
  substitution pattern as Task 7 plan-diff 23. Avoids
  unpinned library dependencies.
- **CRITICAL three-layer pattern applied** (Tasks 5, 7, 11
  precedent); rule-verdict contract layer added to verify live-
  callable PASS / WARN / FAIL paths as well as the five SKIP gates
  from `ph_sym_003.py:36-68`.
- **Narrower-estimator-than-theorem scoping applied**
  (Task 10 precedent). F1 explicitly restricts claim to
  infinitesimal-LEE-diagnostic-of-scalar-invariance, not finite
  equivariance, not multi-generator, not disconnected groups, not
  non-scalar outputs.
- **Hall + Varadarajan inherit TEXTBOOK_AVAILABILITY.md ⚠**
  section-level framing per §6.4.

## 7. Acceptance-gate checklist (user's 2026-04-24 contract)

Task 6 is complete only if:

- [x] F1 proof sketch written before test implementation. (this file
      + CITATION.md F1 section, drafted before any code.)
- [x] Connected-component assumptions explicit. (this file §2.)
- [x] Finite ⇒ infinitesimal and infinitesimal ⇒ finite directions
      separated. (this file §2.)
- [ ] F2 has positive and negative controlled fixtures. (T4.)
- [ ] Finite-difference / small-angle tolerance empirically measured,
      not guessed. (T4 Case C sweep.)
- [ ] Production rule scope stated honestly. (CITATION.md + README;
      no "proves rotation equivariance" language.)
- [ ] RotMNIST / ImageNet / Gruver claims executable, opt-in, or
      demoted. (Demoted per §4.)
- [ ] README / CITATION.md avoid global-equivariance claims.
- [ ] Closeout scripts + full regression pass. (T8.)

## 8. Budget posture

- Budget: 2.2 ED for Task 6; 4.8 ED remaining total (2.2 Task 6 + 1.1
  Task 13 + 1.5 margin).
- Reuse of `_harness/symmetry.py` reduces fresh-code volume (jvp
  primitive + positive / negative controls + Case C scaffolding).
- No Modal / RotMNIST / escnn work → ~1.0 ED saved vs plan §14's
  three-layer RotMNIST + ImageNet estimate; that headroom absorbs
  the F1-first discipline overhead.
- No budget overrun expected.
