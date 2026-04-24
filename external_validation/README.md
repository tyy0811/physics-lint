# External Validation

One validated, CI-wired external anchor per benchmark-anchorable rule. See
`docs/superpowers/specs/2026-04-20-external-validation-design.md` for the
design rationale and anchor-selection methodology, and
`docs/plans/2026-04-22-physics-lint-external-validation-complete.md` for
the complete-v1.0 execution plan.

physics-lint v1.0 provides external-validation anchors for all 18
benchmark-anchorable rules. These anchors combine mathematical-
legitimacy arguments, correctness fixtures, and, where executable in
v1.0, borrowed-credibility reproductions or documented absent-with-
justification cases.

**"Externally anchored" is not "formally proven" or "peer reviewed."**
Each anchor documents the specific mathematical backbone a rule relies
on, verifies the rule's implementation against closed-form or
analytical fixtures, and either reproduces a published numerical
baseline or documents why no directly-comparable baseline exists in
v1.0. Per-rule `CITATION.md` files record the exact distribution.

## 18-of-18 anchor matrix

Per the plan's §1.3 three-function labeling:

- **F1 Mathematical-legitimacy** — the citation stack that makes the
  rule's emitted quantity meaningful.
- **F2 Correctness-fixture** — CI-runnable tests that verify the
  implementation against analytical / closed-form / controlled-
  operator answers.
- **F3 Borrowed-credibility** — live reproduction of a published
  numerical baseline. When F3 is absent, the Borrowed-credibility
  subsection of the per-rule `CITATION.md` records the justification
  and moves any Supplementary calibration context there.

| Rule | Anchor type (M / C / B) | Primary F1 citation | F3 state | Tests | Folder |
|------|-------------------------|---------------------|----------|-------|--------|
| `PH-RES-001` | Fornberg 1988 FD convergence + Bachmayr-Dahmen-Oster 2024 BDO norm-equivalence (Tier 3 + Tier 1) | Fornberg 1988 interior O(h⁴) + BDO residual-norm identity | **present** — Fornberg Table I interior-error reproduction, BDO norm-equivalence two-layer | 12 | [`PH-RES-001/`](PH-RES-001/) |
| `PH-RES-002` | Tier 2 multi-paper — LeVeque 2007 + Griewank-Walther 2008 + Chiu et al. 2022 CAN-PINN | LeVeque consistency-order + Griewank-Walther Ch 3 reverse-mode AD accuracy | **absent with justification** — CAN-PINN cross-check published at framework level; no directly-comparable AD-vs-FD per-point baseline | 5 | [`PH-RES-002/`](PH-RES-002/) |
| `PH-RES-003` | Tier 3 textbook reproduction — Trefethen 2000 Chs 3–4 | Trefethen spectral convergence on periodic `exp(sin x)` | **absent with justification** — Trefethen's convergence demo is a plot, not a table; Boyd 2001 in Supplementary | 7 | [`PH-RES-003/`](PH-RES-003/) |
| `PH-BC-001` | Tier 2 multi-paper — Evans 2010 §5.5 trace theorem + PDEBench bRMSE | Evans trace-theorem + PDEBench Diffusion-sorption / 2D diffusion-reaction | **absent with justification** — PDEBench loader not shipped in v1.0; Supplementary calibration context | 13 | [`PH-BC-001/`](PH-BC-001/) |
| `PH-BC-002` | Tier 1 structural (V1-stub CRITICAL) — Evans 2010 App C.2 Gauss-Green + Gilbarg-Trudinger 2001 §2.4 | Gauss-Green + divergence-theorem structural | **absent by structure** — F1 is the structural identity, F2 harness is authoritative; LeVeque 2002 FVM in Supplementary | 12 | [`PH-BC-002/`](PH-BC-002/) |
| `PH-SYM-001` | Tier 1 structural — Hall 2015 §2.5 + §3.7 (⚠) + Varadarajan 1984 §2.9–2.10 (⚠) + Cohen-Welling 2016 | C₄ structural-equivalence via one-parameter subgroup + identity-component | **absent with justification** — no published baseline directly comparable to the rule's `rotate_test` emitted quantity; Helwig 2023 in Supplementary | 4 | [`PH-SYM-001/`](PH-SYM-001/) |
| `PH-SYM-002` | Tier 1 structural — Hall 2015 §2.5 + §3.7 (⚠) + Varadarajan 1984 §2.9–2.10 (⚠) | Z₂ reflection structural-equivalence | **absent with justification** — analogous to PH-SYM-001; Helwig 2023 in Supplementary | 4 | [`PH-SYM-002/`](PH-SYM-002/) |
| `PH-SYM-003` | Tier 1 structural + Tier 2 — Hall 2015 + Varadarajan 1984 (section-level ⚠) + Kondor-Trivedi 2018 | SO(2) Lie-derivative infinitesimal diagnostic — **scalar-invariant only, not global finite equivariance** | **absent with justification** — plan §14 two-layer RotMNIST + Modal A100 + ImageNet-opt-in Gruver pre-demoted per F3-INFRA-GAP; Cohen-Welling / Weiler-Cesa / Gruver in Supplementary | 24 | [`PH-SYM-003/`](PH-SYM-003/) |
| `PH-SYM-004` | Tier 1 structural (V1-stub CRITICAL) — Kondor-Trivedi 2018 + Li et al. 2021 FNO §2 | Translation equivariance via convolution theorem on periodic grid | **absent by structure** — V1 rule is a SKIP-always stub; F2 harness on controlled operators is authoritative; Helwig 2023 in Supplementary | 36 | [`PH-SYM-004/`](PH-SYM-004/) |
| `PH-CON-001` | Tier 2 multi-paper — Evans 2010 §2.3 + Dafermos 2016 Ch I + Hansen et al. 2024 | Mass-conservation balance-law + ProbConserv CE | **absent with justification** — Hansen ProbConserv loader not shipped in v1.0; Supplementary calibration context | 17 | [`PH-CON-001/`](PH-CON-001/) |
| `PH-CON-002` | Tier 1 structural + Tier 2 — Evans 2010 §2.4.3 wave-energy identity + Strauss 2007 §2.2 + Hairer-Lubich-Wanner 2006 Ch IX | Two-layer analytical wave-energy snapshot | **absent with justification** — no peer-reviewed per-point wave-energy table directly comparable; PDEBench + Hansen in Supplementary | 17 | [`PH-CON-002/`](PH-CON-002/) |
| `PH-CON-003` | Tier 1 structural — Evans 2010 §7.1.2 Theorem 2 parabolic energy estimate | Heat-equation energy dissipation eigenmode reproduction | **absent with justification** — rule's emitted `dE/dt` differs definitionally from available published benchmark columns | 3 | [`PH-CON-003/`](PH-CON-003/) |
| `PH-CON-004` | Tier 2 multi-paper — Verfürth 2013 Chs 1–4 (⚠) + Bangerth-Rannacher 2003 (⚠) + Ainsworth-Oden 2000 (⚠) | Conservation-defect localization (interior-only L-shape singularity) — **narrower estimator than Verfürth's full residual** | **absent with justification** — L-shape effectivity-index values depend on estimator + marker + solver triple; Becker-Rannacher 2001 DWR in Supplementary | 13 | [`PH-CON-004/`](PH-CON-004/) |
| `PH-NUM-001` | Tier 2 multi-paper (V1-stub CRITICAL) — Ciarlet 2002 §4.1 (⚠) + Strang 1972 Variational Crimes (⚠) + Brenner-Scott 2008 §10.3 (⚠) | Quadrature exactness + under-integration harness; V1 rule is a PASS-with-stub-reason pass-through | **absent with justification** — no peer-reviewed paper tabulates `p_obs` for the specific `(p, intorder, MMS)` triples; Ern-Guermond 2021 §8.3 + MOOSE in Supplementary | 20 | [`PH-NUM-001/`](PH-NUM-001/) |
| `PH-NUM-002` | Tier 3 textbook reproduction — Ciarlet 2002 §3.2 Céa + Strikwerda 2004 Ch 10 Lax equivalence | Three-case observed-order per fixture (Laplace scope-truth; Liouville periodic case structurally unreachable) | **absent with justification** — Roy 2005 + Oberkampf-Roy 2010 provide the `p_obs` algorithm; Supplementary calibration context | 22 | [`PH-NUM-002/`](PH-NUM-002/) |
| `PH-POS-001` | Tier 1 structural — Evans 2010 §2.2.3 Theorem 4 Positivity corollary + §2.3.3 Theorem 4 | Poisson + heat maximum-principle reproduction | **absent with justification** — Evans theorems reproduced as structural identity; no directly-comparable numerical baseline | 4 | [`PH-POS-001/`](PH-POS-001/) |
| `PH-POS-002` | Tier 1 structural — Evans 2010 §2.2.3 Theorem 4 | Harmonic strong-maximum-principle reproduction | **absent with justification** — as PH-POS-001 | 4 | [`PH-POS-002/`](PH-POS-002/) |
| `PH-VAR-002` | Tier 2 info-flag — Gopalakrishnan-Sepúlveda 2019 + Ernesti-Wieners 2019 + Henning-Palitta-Simoncini-Urban 2022 + Demkowicz-Gopalakrishnan 2010/2011 | Wave-equation diagnostic contract (info severity); points users to DPG norm-equivalence literature | **absent by structure** — info-flag rule emits no numerical output; Demkowicz-Gopalakrishnan 2025 Acta Numerica DOI 10.1017/S0962492924000102 in Supplementary | 7 | [`PH-VAR-002/`](PH-VAR-002/) |

**Caveats on the 18-of-18 framing.**

- **F3 distribution.** Of the 18 rules, one (PH-RES-001) ships a live
  F3 reproduction; 17 document F3-absent with justification (F3-INFRA-
  GAP for infrastructure-dependent pins, F3-absent-by-structure for
  info-flag / V1-stub / analytical-only rules, or F3-absent because
  no directly-comparable published baseline exists for the rule's
  specific emitted quantity). See each rule's `CITATION.md`
  Borrowed-credibility subsection for the exact reasoning.
- **V1 stubs.** Three rules (PH-BC-002, PH-SYM-004, PH-NUM-001) ship
  as V1 production stubs under the CRITICAL three-layer pattern: F1
  states the mathematical family; F2 harness-level is authoritative
  on controlled fixtures; the rule-verdict contract verifies the
  V1 stub's documented SKIP / PASS-with-reason behavior. V1.1 work
  may upgrade these to full emitted quantities using the same
  harness primitives as regression fixtures.
- **Narrower-estimator-than-theorem scoping.** Two rules (PH-CON-004,
  PH-SYM-003) implement a strict subset of their cited theorem's
  quantity (interior-only hotspot vs full Verfürth estimator for
  PH-CON-004; infinitesimal scalar-invariant SO(2) Lie derivative vs
  global finite multi-output equivariance for PH-SYM-003). F1 in
  both cases explicitly scopes the claim to what the implementation
  can validate; no broader guarantee is inherited.
- **Section-level textbook framing.** Hall 2015, Varadarajan 1984,
  Ciarlet 2002, Trefethen 2000, Verfürth 2013, and several other
  textbooks carry ⚠ status in
  [`_harness/TEXTBOOK_AVAILABILITY.md`](_harness/TEXTBOOK_AVAILABILITY.md)
  per the plan §6.4 primary-source-verification discipline. Per-rule
  `CITATION.md` files use section-level framing for those citations;
  the `scripts/check_theorem_number_framing.py` closeout script
  enforces this mechanically.

## Shared harness

[`_harness/`](_harness/) provides primitives shared across rules:

- `assertions.py` — tolerance-gated assertion helpers.
- `citations.py` — Citation dataclass.
- `fixtures.py` — shared analytical fixtures.
- `mms.py` — method-of-manufactured-solutions H¹-error helper.
- `symmetry.py` — C₄ / Z₂ / translation / SO(2) equivariance
  primitives.
- `aposteriori.py` — a-posteriori-error-estimator helpers for
  PH-CON-004.
- `divergence.py` — divergence-theorem helpers for PH-BC-002.
- `energy.py` — wave-energy analytical snapshots for PH-CON-002.
- `quadrature.py` — quadrature exactness + under-integration
  primitives for PH-NUM-001.
- `trace.py` — trace-theorem helpers for PH-BC-001.
- `TEXTBOOK_AVAILABILITY.md` — primary-source verification status
  per textbook.

## Running locally

Each anchor is pytest-collectable (use `--import-mode=importlib` to
avoid the hyphenated-directory import issue). Example single-rule
run:

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-POS-002/ -v
```

Full 18-rule external-validation suite on CPU:

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/
```

The full suite completes in under 30 s on CPU with no GPU / Modal /
ImageNet / escnn / e3nn / RotMNIST dependency.

## CI

[`.github/workflows/external-validation.yml`](../.github/workflows/external-validation.yml)
runs all 18 anchors as a matrix on every PR to `master` and on pushes
to `master`. No opt-in GPU job ships in v1.0; the plan §14 RotMNIST +
Modal A100 two-layer policy and the ImageNet-opt-in Gruver
reproduction are documented as F3-INFRA-GAP deferrals to V1.x (see
[`../docs/backlog/v1.2.md`](../docs/backlog/v1.2.md)).

## Historical note — external validation surfaced real bugs

External validation during v1.0 development surfaced several rule-
level issues that the 314-test unit suite had not caught, the most
notable of which were:

- **PH-CON-003 primitive correctness bug.** The anchor, reproducing
  Evans §7.1.2 Theorem 2 at its textbook parameters (κ=1, Δt=0.05),
  exposed that the rule's Rev 1.6 `dE/dt` primitive produced
  spurious endpoint artifacts on strictly-dissipative eigenmodes
  with decay range above ~50×. Fixed in `e691dd3` via a forward-
  difference primitive; the anchor now cleanly reproduces the
  textbook `exp(−4π²·0.05) ≈ 0.1389`.
- **Two theorem-number corrections** from the Rev 1.6 design spec
  were applied to per-rule `CITATION.md` files during the 2026-04-20
  verification pass: `§2.2.4 Theorem 13 (positivity)` is actually
  Symmetry of Green's function in Evans; positivity is a corollary
  of §2.2.3 Theorem 4. `§2.3.3 Theorem 8 (heat max principle)` is
  actually §2.3.3 Theorem 4 (Evans restarts theorem numbering per
  section). See
  [`_harness/TEXTBOOK_AVAILABILITY.md`](_harness/TEXTBOOK_AVAILABILITY.md)
  for direct quotes confirming both corrections.
- **PH-RES-001 configuration-dependent norm-equivalence split.** The
  rule emits different norms on periodic + spectral vs non-periodic +
  FD configurations; the Bachmayr-Dahmen-Oster framework's norm-
  equivalence claim holds only on the former. Characterized rather
  than fixed; see [`PH-RES-001/CITATION.md`](PH-RES-001/CITATION.md).

These findings are logged in per-rule `CITATION.md` files under the
"Verification protocol" and "Honest findings" sections.
