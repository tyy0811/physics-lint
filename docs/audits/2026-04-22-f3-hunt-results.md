# F3-hunt audit — Tasks 10 / 11 / 3

Date: 2026-04-23 (audit executed during complete-v1.0 plan Task 0 Step 5).

**Purpose.** Complete-v1.0 plan §8 Task 0 Step 5 F3-hunt: search for directly-
comparable published numerical baselines that could promote F3 (borrowed-
credibility) from ABSENT to PRESENT for Tasks 10 (PH-CON-004), 11 (PH-NUM-001),
3 (PH-RES-003). Tasks were searched in priority order per plan §8: Task 10
priority, Task 11 secondary, Task 3 tertiary.

**Search scope.** Per the plan's literature-pin-vs-hunt distinction, this pass
searches for tabulated numerical baselines the rule could reproduce within CI
budget — not for theorem-level anchors (which are Function 1) or for
calibration-only plots (which are Supplementary calibration context per §1.2).

**Sources inspected per task.** Literature search via WebSearch (adaptive-FEM /
FEM-quadrature / spectral-vs-FD queries), targeted WebFetch of the top
candidate mirrors, and `pymupdf.open` text extraction on every non-paywalled
source that returned binary PDF content.

---

## Task 10 — PH-CON-004 (per-element conservation hotspot / adaptive-FEM
effectivity) F3-hunt — PRIORITY

**Rule semantic target.** PH-CON-004 flags per-element conservation violations
as hotspots for adaptive refinement. The emitted quantity is a residual-based
element-wise indicator `η(τ, u_T)` whose concentration pattern should match
known singularity locations (L-shape re-entrant corner; generic corner-singular
solutions). Plan Task 10 Function 1 anchor: Verfürth 2013 Chs 1–4 +
Bangerth-Rannacher 2003 + Ainsworth-Oden 2000.

**Search paths attempted.**

- WebSearch "adaptive FEM L-shape benchmark effectivity index residual
  estimator Poisson corner singularity table numerical" → returned UCI iFEM
  documentation, Gridap tutorials, Verfürth lecture notes, Long Chen's AFEM
  lecture notes (UCI Math 226 Ch4AFEM.pdf), HAL preprint 03201137.
- WebFetch of the Long Chen UCI Math 226 Ch4AFEM.pdf (797 KB / 17 pages).
  `pymupdf.open` text extraction: zero "effectivity" matches, zero "Table"
  matches, 3 "L-shape" matches but only in descriptive context; the document
  describes the adaptive algorithm and the canonical `u = r^(2/3) sin((2/3)θ)`
  exact solution but does not report tabulated effectivity-index numbers.
- Verfürth 2013 *Adaptive Finite Element Methods Lecture Notes* (Bochum
  AdaptiveFEM.pdf mirror): identified as a lecture-notes version of Verfürth
  2013 *A Posteriori Error Estimation Techniques for Finite Element
  Methods* (Oxford NMSC). The lecture notes cover the theorem development
  but tabulated benchmark effectivity values are specific to particular
  implementations (Bochum's own AFEM code or iFEM), which vary by refinement
  marker, estimator, and solver — so no cross-paper "pinned row" exists in
  the Task 4/8 sense.
- Gridap Poisson AMR tutorial: reports specific MATLAB/Julia code for the
  L-shape benchmark with its own estimator implementation; numerical
  effectivity values are reproducible from that code but are not a published
  numerical baseline in a peer-reviewed sense.

**Disposition: F3-ABSENT with justification.** The L-shape benchmark is a
canonical adaptive-FEM benchmark with a well-defined exact solution, but
effectivity-index values depend on (a) which residual-type estimator is used
(classical residual, recovery-type, equilibrated-flux), (b) which marker is
used (maximum, Dörfler, equidistribution), (c) mesh generation specifics.
Different peer-reviewed sources report different effectivity values on the
"same" benchmark because the precise estimator + marker + solver triple varies.
There is no single-paper reproduction target that physics-lint's PH-CON-004
output could map onto within its measurement framework.

- **Carry-forward to Task 10 CITATION.md:** Function 1 cites Verfürth Thm 1.12
  (residual estimator upper/lower bounds); Function 2 implements the scikit-fem
  Example 22 adaptive-Poisson fixture with physics-lint's emitted per-element
  indicator concentrating at the L-corner within 2 element-layers (binary
  concentration test, not numerical reproduction); **Function 3: ABSENT with
  justification.** Supplementary calibration context: Becker-Rannacher 2001
  Acta Numerica DWR survey + Verfürth 2013 Oxford NMSC Chs 1–4 +
  Ainsworth-Oden 2000 (as pedagogical framing, not reproduction).

---

## Task 11 — PH-NUM-001 (FEM quadrature convergence) F3-hunt — SECONDARY

**Rule semantic target.** PH-NUM-001 checks FEM quadrature order matches
theoretical convergence rate: for polynomial order `p ∈ {1, 2, 3}` with
`intorder ∈ {1, 2, 3, 4}`, measure observed convergence rate on MMS solution
and compare to Ciarlet §4.1 Thms 4.1.2–4.1.6 prediction. Plan Task 11 Function 1
anchor: Ciarlet §4.1 Thms 4.1.2–4.1.6 + Strang 1972 Variational Crimes +
Brenner-Scott §10.3.

**Search paths attempted.**

- WebSearch "FEM quadrature order convergence rate Ciarlet intorder table sin
  cos manufactured solution linear quadratic cubic" → returned
  Ciarlet-Raviart mixed-FEM papers, Babuška-Osborn 1978 (Numerische Mathematik)
  on quadrature-precision requirements, MOOSE FEM-convergence tutorial, FEniCS
  convergence-rates notebook, finite-element.github.io Lecture 5.
- WebSearch "MOOSE framework FEM convergence quadrature order FIRST SECOND
  intorder table polynomial error rate" → confirmed MOOSE implementation of
  Ciarlet-style quadrature selection (default `2p` for Galerkin) and standard
  `p+1` convergence-rate theorems, but no tabulated reproduction target.
- Ciarlet 2002 Classics reprint + Brenner-Scott 2008 + Ern-Guermond 2004/2021:
  these carry the theorems but tabulated error/rate values under varying
  `intorder` are typically *examples within proofs* or illustrative figures,
  not systematic reproduction targets. The canonical "under-integrated FEM
  gives wrong rate" check is textbook-method, not a pinned numerical table.

**Disposition: F3-ABSENT with justification.** The quadrature-convergence
theorems (Ciarlet 4.1.2-6, Strang 1972, Brenner-Scott §10.3) establish the
predicted rate; the rule's emitted quantity is `p_obs` from an h-refinement
sweep, which the theorems predict to match `p+1` for conforming FE with
sufficient quadrature. A reproduction target would require a peer-reviewed paper
that tabulates `p_obs` for the specific `(p, intorder, MMS)` triples physics-
lint exercises — no such paper with a fixed-table row emerged from the hunt.

- **Carry-forward to Task 11 CITATION.md:** Function 1 cites Ciarlet
  Thm 4.1.6 + Strang 1972 + Brenner-Scott §10.3 (theorem-level); Function 2
  implements MMS refinement under varying `intorder` with `p_obs` matching
  predicted `p + 1` within 10 % (the theorem is the comparison target, not a
  paper row); **Function 3: ABSENT with justification.** Supplementary
  calibration context: Ern-Guermond 2021 Finite Elements §8.3 (quadrature
  calibration discussion); MOOSE FEM-convergence tutorial page (for illustrative
  framing, not reproduction).

---

## Task 3 — PH-RES-003 (spectral vs FD residual on periodic grids) F3-hunt —
TERTIARY

**Rule semantic target.** PH-RES-003 checks that spectral-method residual
converges exponentially on analytic periodic functions (rate `O(exp(-cN))`)
while FD converges polynomially (`O(N^{-p})` at stencil order p). Plan Task 3
Function 1 anchor: Trefethen 2000 Chs 3–4 Thm 4.

**Search paths attempted.**

- WebSearch "Trefethen spectral methods in MATLAB Output 5 exp(sin convergence
  table" → returned Trefethen book listings (SIAM, MathWorks, Amazon),
  Trefethen's Oxford web page (spectral.html), and a GitHub mirror
  (dumpinfo/MatlabBookCollection) of the book PDF. The book's famous Program 5
  *plots* spectral convergence for `exp(sin x)` but the canonical way it
  presents results is as a log-error-vs-N plot, not as a tabulated
  reproduction target.
- WebFetch of the HMC Math 165 Trefethen PDF
  (`math.hmc.edu/~dyong/math165/trefethenbook.pdf`, 4 MB / 299 pages).
  `pymupdf.open` text extraction: the PDF is actually Trefethen's 1996 draft
  manuscript *Finite Difference and Spectral Methods for Ordinary and Partial
  Differential Equations* (the pre-cursor never formally published), not
  *Spectral Methods in MATLAB* (2000). Zero "Program 5" / "Output 5" / "exp(sin"
  matches. Wrong book entirely.
- Boyd 2001 *Chebyshev and Fourier Spectral Methods* (Dover reprint free PDF
  at `depts.washington.edu/ph506/Boyd.pdf`): access attempt not completed
  within Task 0 budget; the plan classifies Task 3 as hunt-tertiary specifically
  because low-expected-yield — pressing on Boyd would push Task 0 past the
  1.2 d plan budget.

**Disposition: F3-ABSENT with justification.** The canonical spectral-vs-FD
convergence demonstration (Trefethen's famous `exp(sin x)` plot) is a plot, not
a tabulated reproduction target. Plot-shape resemblance is calibration, not
reproduction under the §1.2 stricter F3 definition. Task 3's plan-level
pre-flagging ("likely outcome is F3-absent; low-expected-yield hunt") is
confirmed.

- **Carry-forward to Task 3 CITATION.md:** Function 1 cites Trefethen 2000
  Chs 3-4 Thm 4 (spectral accuracy for analytic periodic functions);
  Function 2 implements closed-form `exp(sin x)` fixture with spectral
  residual at N ∈ {16, 32, 64} matching exponential-decay fit `R² > 0.99`
  (the theorem is the comparison target, not a paper row); **Function 3:
  ABSENT with justification.** Supplementary calibration context: Canuto-
  Hussaini-Quarteroni-Zang 2006 §2.3 convergence curves (curve-shape
  framing, not reproduction); Trefethen 2000 Program 5 plot (plot-shape
  framing, not reproduction).

---

## Summary

| Task | Rule | F3 disposition | Carry-forward to CITATION.md |
|------|------|---------------|------------------------------|
| 10 | PH-CON-004 | **F3-ABSENT with justification** | Supplementary calibration context: Becker-Rannacher 2001 Acta Numerica DWR + Verfürth 2013 Oxford NMSC + Ainsworth-Oden 2000 |
| 11 | PH-NUM-001 | **F3-ABSENT with justification** | Supplementary calibration context: Ern-Guermond 2021 §8.3 + MOOSE FEM-convergence page |
| 3 | PH-RES-003 | **F3-ABSENT with justification** | Supplementary calibration context: Canuto-Hussaini-Quarteroni-Zang 2006 §2.3 + Trefethen 2000 Program 5 plot |

**Plan-level implications.**

- All three tasks' Function 3 subsections land as ABSENT-with-justification.
  Task 10 is no longer "hunt-priority yields F3-present"; Task 11 is no longer
  "hunt-secondary yields F3-present"; Task 3 was already flagged low-yield.
- CITATION.md Supplementary calibration context subsections are the operative
  locations for the calibration references; per §1.2 they must carry explicit
  "calibration, not reproduction" flags.
- The overall v1.0 F3-coverage distribution (per §0.3 γ-narrative) is
  unaffected by this audit: F3-present rules remain F3-present; F3-absent rules
  are justified. The honesty of the "approximately half of rules" F3-coverage
  framing at the main README level is preserved.

**Audit cost reconciliation.** Plan §8 budgeted Task 0 Step 5 at ~0.3 d of the
total 0.4 d allocated across Steps 4+5 (literature-pin + F3-hunt combined).
Actual execution: within budget; all three hunts settled within their
priority-tier expectation (Task 10 priority settled fastest since the L-shape
implementation-dependence argument is clean; Task 11 secondary settled via the
"no tabulated reproduction target" argument; Task 3 tertiary was pre-flagged
low-yield and confirmed so).

## Update posture

If a later pass surfaces a peer-reviewed tabulated reproduction target (e.g., a
standards-publication effectivity-index table for a specific L-shape estimator
type; a quadrature-convergence table with fixed `(p, intorder, MMS)` triples;
a spectral-vs-FD convergence table on `exp(sin x)` with pinned numerical
values), the disposition in this file upgrades and the affected CITATION.md
gets a tightening-pass commit per the plan-vs-committed-state drift discipline
(§7.4). The file is a local-audit-artifact documenting the Task 0 hunt outcome
against the specific sources the complete-v1.0 plan scheduled, not a claim of
exhaustive literature coverage.
