# physics-lint v1.0 Validation Report

## Abstract

This report records the combined mathematical and engineering validation of
the externally validated rules in physics-lint v1.0. For each rule, it states
what the rule is intended to measure, which theorem or established structure
supports that measurement, which assumptions are required, which claims are
deliberately not made, and — in the integrated rule-by-rule matrix in §24 —
the correctness fixtures, CI evidence, commit provenance, and external
borrowed-credibility status that accompany each shipped rule.

The report integrates four layers:

- **Function 1** — mathematical legitimacy: per-rule derivations, theorem
  citations, assumptions, non-claims (§§4–22).
- **Function 2** — correctness fixtures: harness-level and production-level
  test anchors verifying either that the rule's measured quantity behaves
  as the derivation predicts, or — for v1.0 stub rules — that the shipped
  gate / stub contract behaves as documented (§24, F2 column).
- **Function 3** — borrowed credibility: external-benchmark reproduction
  status, including absent-with-justification markers where no
  directly-comparable published target exists (§24, F3 column).
- **CI evidence and provenance**: the CI status and commit SHAs at which
  each rule's anchor landed (§24, CI status and Commit columns).

The mathematical derivations in Parts I–VII are treated as the Function-1
source of truth; the integrated matrix in Part IX combines them with the
engineering audit layer's evidence. Mathematical claims in this report are
limited to those stated in the per-rule sections; the engineering matrix
supplies evidence about how each rule is exercised, not new mathematical
claims.

---

## 1. Purpose and Scope

### 1.1 Purpose

Explain why the rule metrics used by physics-lint are mathematically
meaningful, under what assumptions, and with what limitations; and record
the engineering evidence (correctness fixtures, CI results, commit
provenance, borrowed-credibility status) that accompanies each rule at
the v1.0 freeze.

### 1.2 Scope

This report covers the v1.0 externally anchored rule set:

- residual rules (§§4–6)
- boundary rules (§§7–8)
- conservation rules (§§9–12)
- positivity rules (§§13–14)
- symmetry and equivariance rules (§§15–18)
- numerical-validation rules (§§19–20)
- meta / diagnostic rules (§§21–22)

The integrated rule-by-rule validation matrix in §24 records the
engineering evidence for the 18 v1.0-shipped rules. PH-VAR-001 is
deferred to v1.2 and documented separately in Appendix D.

### 1.3 Non-goals

This report does not claim any of the following:

- Formal machine-checked proof. The derivations here are at the level
  of a research paper's theoretical section, not a Coq or Lean proof.
- Proof that a user's neural model is physically correct. A rule can
  only detect violations of a property, never confirm global
  correctness.
- Generalization guarantees outside the stated assumptions for each
  rule.
- Correctness of every possible discretization. Each rule is framed
  relative to a specific set of discretization assumptions.
- Equivalence between a low residual value and small solution error,
  unless a cited theorem explicitly supports the equivalence for the
  PDE and norm in question.

### 1.4 Relationship to other validation artifacts

| Artifact | Role |
|---|---|
| `physics_lint_v1_validation_report.md` (this document) | Combined v1.0 validation report (Functions 1 + 2 + 3 + CI / provenance) |
| `external_validation/<RULE>/test_anchor.py` | Executable correctness evidence (Function 2) |
| `external_validation/<RULE>/CITATION.md` | Rule-local citation, function labels, audit trail |
| `TEXTBOOK_AVAILABILITY.md` | Citation-verification status and primary-source chain |
| `TRACEABILITY.md` | Engineering audit traceability |

The mathematical sections (Parts I–VII) are the Function-1 source of
truth. The integrated rule-by-rule validation matrix (§24) records the
F2 / F3 / CI / commit evidence at the v1.0 freeze. New mathematical
claims must be added to the relevant per-rule section, not introduced
in §24 alone.

### 1.5 Release evidence summary

The integrated rule-by-rule validation matrix in §24 records per-rule
evidence; the table below records the global repository / CI evidence
at the v1.0 freeze.

| Field | Value |
|---|---|
| Branch | `external-validation-tier-a` |
| Final commit | `6907699` |
| Pull request | #3 |
| CI status | 28 / 28 checks successful |
| External-validation tests | 268 collected: 267 passed, 1 skipped |
| Unit tests | 283 collected, 283 passed |
| Closeout scripts | 2 / 2 passed; 18 OK lines each |
| Ruff check / format | passed |

These figures are the single point of truth for global CI status; the
§24 CI-status column mirrors the same uniform value per row by
construction (a single matrix run at HEAD `6907699`).

---

## 2. Validation Logic

### 2.1 Rule-to-mathematics chain

For each rule, the trust chain flows in one direction:

```
rule verdict
→ implemented quantity (what the code actually computes)
→ mathematical object (what that quantity is meant to approximate)
→ theorem, identity, or established model (why that object is meaningful)
→ assumptions (when the theorem applies)
→ limitations (what the rule cannot say)
```

A rule is mathematically legitimate when every link in this chain is stated
explicitly and each citation in the chain has been verified against a primary
or canonical secondary source.

### 2.2 Three kinds of mathematical justification

We distinguish three styles of justification that appear across the rule set.

**Type A — Structural-equivalence proof sketch.** Used when the rule metric is
a concrete instance of an established structure (e.g. a group-equivariance
defect, a divergence-theorem identity, an energy functional). The derivation
identifies the mathematical object and shows that the code's quantity is its
discrete realization. Examples: finite group equivariance (PH-SYM-001),
infinitesimal Lie-derivative equivariance (PH-SYM-003), divergence theorem
(PH-BC-002), energy identity for the wave equation (PH-CON-002).

**Type B — Theorem-instance derivation.** Used when the rule checks a known
theorem or standard numerical-analysis result as applied to the user's setup.
Examples: trace theorem (PH-BC-001), spectral convergence (PH-RES-003),
manufactured-solution observed-order analysis (PH-NUM-002), quadrature
consistency (PH-NUM-001).

**Type C — Diagnostic / information-flag justification.** Used when the rule
is not a certificate but a mathematically motivated warning about the limits of
a metric. Example: the hyperbolic residual-norm caveat (PH-VAR-002).

A Type C rule is still mathematically legitimate: its legitimacy lies in
warning the user that the elliptic-style residual-error equivalence may fail,
not in proving a positive statement about the model.

### 2.3 Per-rule template

Each rule section follows this template (condensed where sections would
otherwise repeat):

- **Mathematical claim** — what the rule measures, in one sentence.
- **Implemented quantity** — the quantity physics-lint actually emits.
- **Theoretical anchor** — the theorem, identity, or model, with citation.
- **Assumptions** — domain, regularity, boundary, discretization, representation.
- **Derivation / proof sketch** — the actual mathematics.
- **Mapping to implementation** — how the mathematical quantity corresponds to
  the rule's API and output.
- **What this validates** — the valid claim, stated positively.
- **What this does not validate** — non-claims and scope limits.
- **Report-integration note** — how the mathematical claim is connected to the F2 / F3 evidence summarized in §24.

---

## 3. Notation and Standing Assumptions

### 3.1 Domains and function spaces

We fix the following notation throughout the document.

- $\Omega \subset \mathbb{R}^d$ denotes a bounded open spatial domain, $d \in
  \{1, 2, 3\}$. Where regularity of the boundary matters, $\Omega$ is assumed
  to have a Lipschitz boundary $\partial\Omega$ (which is the standard setting
  for the trace theorem and the divergence theorem).
- $[0, T]$ denotes a finite time interval for evolutionary problems.
- $u : \Omega \to \mathbb{R}$ is a scalar field; $\mathbf{F} : \Omega \to
  \mathbb{R}^d$ is a vector field.
- $R(u)$ denotes the residual of a PDE at $u$: if the PDE is $\mathcal{L}u = f$,
  then $R(u) := \mathcal{L}u - f$.
- $B(u)$ denotes the boundary operator value: for Dirichlet conditions
  $B(u) := u|_{\partial\Omega} - g$; for Neumann conditions $B(u) :=
  (\nabla u \cdot \mathbf{n})|_{\partial\Omega} - g_N$.
- $u_h$ is a numerical approximation computed on a grid or mesh with
  characteristic size $h$; $\Delta t$ is a time step.
- $\mathbf{n}$ is the outward unit normal on $\partial\Omega$.

### 3.2 Norms

All norms used by rules are defined in the standard sense.

- $\|v\|_{L^2(\Omega)}^2 := \int_\Omega |v|^2\, dx$.
- $\|v\|_{L^2(\partial\Omega)}^2 := \int_{\partial\Omega} |v|^2\, ds$.
- $\|v\|_{H^1(\Omega)}^2 := \|v\|_{L^2(\Omega)}^2 + \|\nabla v\|_{L^2(\Omega)}^2$.
- $\|v\|_{H^{-1}(\Omega)}$ is the dual norm on the topological dual of
  $H^1_0(\Omega)$, realized concretely as $\|v\|_{H^{-1}} = \sup_{0 \neq \varphi
  \in H^1_0} \langle v, \varphi \rangle / \|\varphi\|_{H^1_0}$.
- Discrete grid norms use the Riemann-sum approximation with weight $h^d$:
  $\|v_h\|_{\ell^2_h}^2 := h^d \sum_i |v_h(x_i)|^2$. For a smooth field
  $v$, $\|v_h\|_{\ell^2_h} \to \|v\|_{L^2(\Omega)}$ as $h \to 0$.
- Relative residual norm: $\|R(u_h)\| / (\|f\| + \epsilon)$, with $\epsilon$
  a small positive floor to avoid division by zero when $f$ is trivial.
- Energy norms are PDE-specific; they are defined at point of use in
  §§10–11.

### 3.3 Discretization conventions

- **Finite-difference grids.** Uniform Cartesian grids unless stated otherwise.
  Stencil order is the order of the underlying difference approximation (e.g.
  a centered second-difference Laplacian is order $2$).
- **Periodic grids.** Functions are assumed periodic with period equal to the
  domain extent; boundary terms in integration-by-parts identities vanish.
- **Spectral derivatives.** Computed via FFT assuming periodicity and
  sufficient regularity (see §6).
- **Meshes.** Simplicial (triangular in 2D, tetrahedral in 3D) unless stated
  otherwise. For v1.0, mesh-based rules (PH-CON-004, PH-NUM-001/002) assume
  2D triangulations; 3D tetrahedral meshes are out of scope.
- **Quadrature.** Rule-specific; stated where invoked.
- **Floating-point floor.** All norm calculations are subject to a
  floating-point floor at roughly $\|\cdot\| \approx 10^{-14}$ in IEEE 754
  double precision. Any asymptotic claim that depends on values below this
  floor is considered unreliable.

### 3.4 General limitation

Every rule verdict is conditional on the metadata the user supplies to
physics-lint: field type, boundary condition, domain, grid, PDE specification,
and tolerance. The rules check *internal consistency between the computed
quantity and the declared mathematical object*. They do not verify that the
user correctly identified the PDE, the boundary conditions, or the physical
variables.

---

# Part I — Residual Rules

## 4. PH-RES-001 — Residual Norm Legitimacy

### 4.1 Mathematical claim

The residual norm $\|R(u_h)\|$ is meaningful as an error indicator only
relative to a specific PDE class, a specific boundary-condition setup, and a
specific norm. For elliptic problems, a posteriori estimators establish an
equivalence between residual and error. For non-elliptic problems, this
equivalence can fail, and the residual becomes a diagnostic rather than a
certificate.

### 4.2 Theoretical anchor

Classical a posteriori error estimation for elliptic problems, in the
formulation of Verfürth [Verfürth 2013] and Ern-Guermond [Ern-Guermond
2021, *Finite Elements II: Galerkin Approximation, Elliptic and Mixed
PDEs*, Springer TAM 73, Part VII "A posteriori error analysis"],
provides the residual-error equivalence for the dual residual
$r_h \in V'$:

$$
c_1 \|r_h\|_{V'} \leq \|u - u_h\|_{V} \leq c_2 \|r_h\|_{V'},
\tag{4.1}
$$

where $c_1, c_2$ are continuity / coercivity constants depending on the
domain and PDE coefficients. Three residual objects must not be conflated:

- The **strong residual** $R(u_h)(x) := (\mathcal{L} u_h)(x) - f(x)$,
  pointwise on $\Omega$, requires $u_h$ to be sufficiently differentiable
  in the classical sense.
- The **dual residual** $r_h \in V'$, defined by
  $\langle r_h, v \rangle := \ell(v) - a(u_h, v)$ for all $v \in V$.
  Equation (4.1) is a statement about $\|r_h\|_{V'}$.
- The **residual-based estimator** $\eta(u_h)$, an element-assembled
  proxy for $\|r_h\|_{V'}$ (element-interior volume terms plus
  inter-element flux jumps; see §12 for the Poisson-case formula).
  Estimators are related to $\|r_h\|_{V'}$ only through reliability and
  efficiency bounds (see §12); they are not the same object.

### 4.3 Derivation

For an abstract linear PDE $Au = f$ on $\Omega$ with $A: V \to V'$ bounded and
$V$-elliptic:

$$
\langle Au, u \rangle_{V', V} \geq \alpha \|u\|_V^2, \quad \alpha > 0,
\quad \forall u \in V.
\tag{4.2}
$$

Define the residual $r_h \in V'$ by $r_h := f - Au_h$. Then by $V$-ellipticity
applied to $u - u_h$:

$$
\alpha \|u - u_h\|_V^2 \leq \langle A(u - u_h), u - u_h \rangle = \langle r_h,
u - u_h \rangle \leq \|r_h\|_{V'} \|u - u_h\|_V,
\tag{4.3}
$$

so that

$$
\|u - u_h\|_V \leq \alpha^{-1} \|r_h\|_{V'}.
\tag{4.4}
$$

The reverse inequality (lower bound on $\|u - u_h\|_V$ in terms of $\|r_h\|_{V'}$)
uses boundedness of $A$:

$$
\|r_h\|_{V'} = \|A(u - u_h)\|_{V'} \leq \|A\|_{V \to V'} \|u - u_h\|_V.
\tag{4.5}
$$

Together, (4.4) and (4.5) yield the residual-error equivalence up to the
continuous constants. The key point is that **this equivalence requires
$V$-ellipticity**; it does not transfer automatically to parabolic or
hyperbolic problems, which are handled separately (see §22).

### 4.4 Assumptions

- $V$ is a Hilbert space (typically $H^1_0(\Omega)$ for Dirichlet problems,
  $H^1(\Omega)$ with a Lagrange multiplier or quotient for Neumann).
- $A: V \to V'$ is bounded and coercive (elliptic).
- The chosen norm for the residual is the $V'$ dual norm (often approximated
  by an $H^{-1}$ norm, which requires care; see §12).

### 4.5 Implementation mapping

physics-lint's residual rule emits $\|R(u_h)\|_X$ for a user-declared norm
$X$. The mathematical legitimacy of the verdict depends on $X$ being
compatible with the PDE's natural test-space norm. For elliptic problems
with $V = H^1_0$, the natural $X$ is $H^{-1}$ or a suitable discrete
approximation (typically the weighted $\ell^2_h$ norm at the grid level,
which approaches $L^2$ under refinement and is an upper bound for $H^{-1}$).

### 4.6 What this validates

- For elliptic problems **and when the emitted norm $\|\cdot\|_X$ is a
  validated approximation to the natural dual norm $\|\cdot\|_{V'}$** (or
  equivalently stands in a known two-sided bound to it), a small residual
  implies a small $V$-norm error, with constants from (4.4). When the
  emitted norm is a strong $L^2$, pointwise, or grid-weighted residual
  not known to be two-sided-equivalent to $\|\cdot\|_{V'}$, the forward
  direction of (4.1) is not available and the verdict should be read as
  a diagnostic only.

### 4.7 Correctness-fixture specifications

The shipped F2 for PH-RES-001 is the two-path fixture summarized in
§24: a Fornberg-style O(h⁴) interior-residual reproduction plus a
Bachmayr-Dahmen-Oster (BDO) periodic-spectral H⁻¹ norm-equivalence
two-layer. §4.7.1 records the truncation-consistency path; §4.7.2
records the norm-equivalence path. §4.7.3 records the joint non-claims.
Measured values cited below are from the F2 layer pinned at commit
`c2dba1e` (Layer 1 in `c07ba33`, Layer 2 in `e28c493`); see the §24
PH-RES-001 row for full provenance.

#### 4.7.1 Truncation-consistency fixture (Fornberg O(h⁴))

- **PDE.** $-\Delta u = f$ on the unit square $[0,1]^2$ with
  homogeneous Dirichlet boundary data. The Fornberg reproduction is
  evaluated on interior grid points only, so boundary-stencil effects
  do not contaminate the interior convergence rate.
- **Manufactured solution.** $u^*(x,y) = \sin(\pi x)\sin(\pi y)$,
  giving $f = 2\pi^2 \sin(\pi x)\sin(\pi y)$. This solution satisfies
  the homogeneous Dirichlet condition exactly on $\partial[0,1]^2$.
- **Object tested.** The *truncation residual*
  $\tau_h(x_i) := (\mathcal{L}^{\mathrm{FD}}_4 u^*)(x_i) - f(x_i)$,
  i.e. the **fourth-order** Fornberg-stencil discrete operator
  applied to the **exact** manufactured solution at interior grid
  points.
- **Norm.** Discrete $\ell^2_h$ norm on the interior grid points.
- **Grid sequence.** $N \in \{16, 32, 64, 128\}$, refinement
  ratio $r = 2$.
- **Expected rate.** $\|\tau_h\|_{\ell^2_h} = O(h^4)$ for the
  Fornberg interior fourth-order stencil; the live measured slope is
  $p_{\mathrm{obs}} = 3.993$ over the validated refinement range.
- **Acceptance tolerance.** $p_{\mathrm{obs}}$ within $[3.8, 4.2]$
  with $R^2 \geq 0.99$ on the log-log fit.

This fixture is a *consistency* test in the Lax-equivalence sense for
the Fornberg fourth-order interior stencil. It does not by itself
test (4.1); it establishes that the discrete operator converges to
the continuous operator at the claimed order on smooth interior data.

#### 4.7.2 Norm-equivalence fixture (BDO periodic-spectral H⁻¹)

- **Setting.** Periodic spectral discretization on $[0, 2\pi]^d$ with
  Fourier basis (the regime in which Bachmayr-Dahmen-Oster two-sided
  norm equivalence is provable; the non-periodic + finite-difference
  path falls back to plain $L^2$ and is not covered by this fixture).
- **Object tested.** The two-sided ratio $C_{\max} / c_{\min}$, where
  $c_{\min}$ and $C_{\max}$ are the smallest and largest measured
  ratios of the periodic-spectral H⁻¹ residual norm against the
  reference dual-residual norm across the perturbation family.
- **Perturbation family.** $k \in \{1, 2, 3\}$ Fourier-mode
  perturbations $u_h = u^* + \epsilon \sin(k \cdot x)$, sampled at
  multiple $\epsilon$ levels.
- **Acceptance criterion.** $C_{\max} / c_{\min} < 10$, indicating
  bounded two-sided equivalence on this regime; the live measured
  ratio is $4.829$.

This fixture is the actual test of (4.1) on the periodic-spectral
regime: it verifies that the emitted residual norm is two-sided-
equivalent to the dual-residual norm with a bounded constant. It
applies only to periodic + spectral discretizations; the non-periodic
+ FD path is characterized but not norm-equivalence-validated, and
falls back to a plain $L^2$ residual interpretation.

#### 4.7.3 What §4.7.1 and §4.7.2 do not validate jointly

A passing §4.7.1 together with a passing §4.7.2 does **not** establish
that the *emitted strong residual* $\|R(u_h)\|_X$ (for an arbitrary
user-declared norm $X$) is two-sided-equivalent to $\|u - u_h\|_V$
outside the periodic + spectral regime. The non-periodic + finite-
difference path is explicitly out of the §4.7.2 norm-equivalence
guarantee and falls back to plain $L^2$ residual reporting; the §4.6
conditional caveat on the emitted norm therefore remains load-bearing
for users running on non-periodic or non-spectral discretizations.

### 4.8 What this does not validate

- For hyperbolic or parabolic problems, (4.1) does not hold in general; the
  rule's verdict is INFO-level in those regimes (see §22).
- A residual smaller than the discretization error does not imply the
  continuous PDE is well-posed; physics-lint cannot check well-posedness
  of the user's declared PDE.
- Residual $\approx 0$ at a single snapshot does not imply the model will
  satisfy the PDE on held-out inputs.

### 4.9 Report-integration note

Function 2 and Function 3 evidence for the shipped v1.0 rule is recorded
in the PH-RES-001 row of §24. The shipped F2 / F3 evidence combines a
Fornberg-style O($h^4$) interior-residual reproduction (live measured
slope $p_{\mathrm{obs}} = 3.993$, $R^2 \geq 0.99$) with a Bachmayr-
Dahmen-Oster (BDO) periodic-spectral $H^{-1}$ norm-equivalence two-layer
(measured $C_{\max} / c_{\min} = 4.829$). Verfürth 2013 and Ern-Guermond
2021 Vol. II remain the mathematical framework anchors for the elliptic
a posteriori residual-error setting; the truncation-consistency and
norm-equivalence fixture specifications in §4.7.1 and §4.7.2 give the
expanded form of the §24 entry.

---

## 5. PH-RES-002 — AD-vs-FD Residual Cross-Check

### 5.1 Mathematical claim

For sufficiently smooth functions, the residual evaluated by automatic
differentiation and the residual evaluated by finite differences must agree
up to the expected discretization/autodiff error regime. A gap larger than
this regime indicates a bug in one of the two differentiation pathways or a
smoothness assumption that has failed — not a physically meaningful model
property.

### 5.2 Theoretical anchor

Consistency of finite-difference differentiation with exact differentiation
for smooth functions, together with the correctness of algorithmic
differentiation for composition of differentiable operations. This is a
Type C / Type B hybrid: we use a textbook consistency result (finite
differences converge to the exact derivative at rate $O(h^p)$ for smooth
functions) and the standard algorithmic-differentiation correctness result
(AD evaluates the exact derivative of the **executed computational
graph** up to floating-point rounding, for compositions of
differentiable primitives) [Griewank-Walther 2008]. Two qualifications
follow from the computational-graph caveat:

- AD differentiates the actual code path taken at evaluation time.
  Data-dependent branches, non-differentiable primitives (`torch.round`,
  `abs` at zero, `max`/`min` at ties, integer casts), custom kernels
  without registered gradients, mixed-precision casts, and in-place
  operations can make the AD-derivative disagree with the mathematical
  derivative of the analytical PDE expression the user *intends*.
- Consequently, "AD equals the exact derivative" applies to the
  computational graph, not to an idealized analytical expression that
  the graph only approximates. The rule catches bugs at the graph level;
  discrepancies that arise from graph-vs-analytical mismatch show up as
  AD-FD gaps even when both routines are individually correct.

### 5.3 Derivation

Let $u$ be the candidate solution (or predicted field) and let $\mathcal{L}$
be the differential operator of interest. Write:

- $R_{\text{AD}}(u)(x) := (\mathcal{L}^{\text{AD}} u)(x) - f(x)$, where
  $\mathcal{L}^{\text{AD}}$ uses autodiff-evaluated derivatives.
- $R_{\text{FD}}(u_h)(x_i) := (\mathcal{L}^{\text{FD}} u_h)(x_i) - f(x_i)$,
  where $\mathcal{L}^{\text{FD}}$ uses a finite-difference stencil of order
  $p$ on a grid with spacing $h$.

For $u \in C^{p+k}(\Omega)$ (with $k$ the derivative order of $\mathcal{L}$),
Taylor expansion gives

$$
|\mathcal{L}^{\text{FD}} u_h(x_i) - \mathcal{L}u(x_i)|
\leq C_p \|u\|_{C^{p+k}(\Omega)} h^p + O(\epsilon_{\text{mach}} h^{-k}),
\tag{5.1}
$$

where the first term is the truncation error of the stencil and the second is
the catastrophic-cancellation error for $k$-th derivatives at spacing $h$.
The autodiff residual $R_{\text{AD}}$, evaluated on the analytical form of
$u$, is exact up to floating-point rounding on the arithmetic operations
themselves:

$$
|R_{\text{AD}}(u)(x) - R(u)(x)| \leq \epsilon_{\text{AD}}(u, x),
\tag{5.2}
$$

with $\epsilon_{\text{AD}}$ bounded by a problem-dependent constant times
machine epsilon [Griewank-Walther 2008, Ch. 3]. Combining (5.1) and (5.2):

$$
\|R_{\text{AD}}(u) - R_{\text{FD}}(u_h)\|_{\ell^\infty}
\leq C_p \|u\|_{C^{p+k}} h^p + \epsilon_{\text{AD}} + \epsilon_{\text{mach}} h^{-k}.
\tag{5.3}
$$

So the AD-vs-FD gap should decrease like $h^p$ in the asymptotic regime and
then saturate at the floating-point floor.

### 5.4 Assumptions

- $u$ is $C^{p+k}$-smooth on $\Omega$, where $p$ is the FD stencil order and
  $k$ is the order of $\mathcal{L}$. This fails at shocks, kinks, and
  boundaries where the user's model may not be differentiable.
- The AD path evaluates derivatives of primitives that have exact derivative
  rules in the AD system (standard elementary functions and compositions).
- $h$ is chosen so that $h^p$ dominates $\epsilon_{\text{mach}} h^{-k}$; i.e.,
  $h \gtrsim \epsilon_{\text{mach}}^{1/(p+k)}$.

### 5.5 Implementation mapping

The rule emits the pair $(R_{\text{AD}}, R_{\text{FD}})$ and reports
$\|R_{\text{AD}}(u) - R_{\text{FD}}(u_h)\|$ in the same norm used for the
residual check. A verdict of "disagreement beyond expected $O(h^p)$" is
raised when the gap exceeds a user-specified factor times the predicted
bound (5.3).

### 5.6 What this validates

- Agreement between two independent derivative-evaluation routes on a smooth
  test case.
- Detection of implementation bugs in either pathway (e.g. a sign error in
  the autodiff chain rule for a custom op, or a wrong-order FD stencil).

### 5.7 What this does not validate

- This is **not** a test of PDE correctness. Both pathways can agree on a
  wrong answer if the PDE formulation is wrong.
- Disagreement does not imply which pathway is wrong; it is a flag for
  further investigation.
- For non-smooth $u$ (e.g. at a shock), the bound (5.3) does not hold, and
  a large gap is expected rather than anomalous.

### 5.8 Report-integration note

Function 2: a fixture with a known-smooth $u$ and known-correct AD and FD
paths, verifying that the gap decays at rate $h^p$. Function 3: cite
Griewank-Walther 2008 for the AD correctness framework and a standard
numerical-analysis text (e.g. LeVeque 2007) for FD truncation orders.

---

## 6. PH-RES-003 — Spectral-vs-FD Residual Behavior

### 6.1 Mathematical claim

For smooth periodic functions on a periodic domain, spectral differentiation
via the discrete Fourier transform converges faster than any finite-difference
stencil of fixed order — specifically, the spectral error decays super-
algebraically (faster than any polynomial in $N^{-1}$), while finite-
difference errors decay algebraically at $O(h^p)$. This behavior holds until
floating-point saturation, after which any exponential-fit claim is
unreliable.

### 6.2 Theoretical anchor

Classical spectral-convergence theorem for periodic Fourier series: for
$u \in C^\infty_{\text{per}}$, the Fourier truncation error decays faster
than any negative power of $N$ [Trefethen 2000, Ch. 4]. For analytic
periodic $u$, the decay is exponential in $N$.

### 6.3 Derivation

Let $u : [0, 2\pi) \to \mathbb{R}$ be periodic, and let $\hat{u}_k$ denote
its Fourier coefficients. The Fourier series is

$$
u(x) = \sum_{k=-\infty}^{\infty} \hat{u}_k e^{ikx}, \quad
\hat{u}_k = \frac{1}{2\pi} \int_0^{2\pi} u(x) e^{-ikx} dx.
\tag{6.1}
$$

For $u \in C^m_{\text{per}}$, integration by parts $m$ times gives
$|\hat{u}_k| \leq C_m |k|^{-m}$ for $k \neq 0$. Hence for any $m$, the
truncation to $|k| \leq N/2$ satisfies

$$
\left\| u - \sum_{|k| \leq N/2} \hat{u}_k e^{ikx} \right\|_{L^2} \leq C_m N^{-m+1/2},
\tag{6.2}
$$

which is super-algebraic in $N$ (valid for all $m$).

For analytic $u$ in a strip of width $\sigma > 0$ around the real axis,
$|\hat{u}_k| \leq C e^{-\sigma |k|}$, so the truncation error decays
exponentially:

$$
\left\| u - \sum_{|k| \leq N/2} \hat{u}_k e^{ikx} \right\|_{L^2}
\leq C' e^{-\sigma N/2}.
\tag{6.3}
$$

The spectral derivative $\partial_x^s u$ is computed as
$\sum_k (ik)^s \hat{u}_k e^{ikx}$, and inherits the same super-algebraic
or exponential decay in the truncation error. A finite-difference stencil of
order $p$, by contrast, yields truncation error $O(h^p) = O(N^{-p})$ —
algebraic, fixed exponent.

The canonical fixture is $u(x) = \exp(\sin x)$, which is analytic in any
horizontal strip, hence (6.3) applies with exponential decay.

### 6.4 Floating-point floor

Once the computed spectral residual reaches the IEEE 754 double-precision
floor (approximately $10^{-14}$ in relative terms), further refinement does
not reduce the residual — additional modes contribute only floating-point
rounding. Any fitted exponential-decay rate derived from data points below
the floor is meaningless. The rule's asymptotic-regime check must
explicitly exclude floor-saturated points before computing a decay rate.

### 6.5 Assumptions

- Periodic boundary conditions on the domain.
- $u$ is $C^\infty_{\text{per}}$ (for super-algebraic decay) or analytic in a
  strip (for exponential decay).
- $N$ grid points with uniform spacing, FFT-based evaluation.
- Sufficient precision (double-precision is assumed; the floor is at
  $\approx 10^{-14}$).

### 6.6 Implementation mapping

The rule computes the residual on two grid sequences: one via spectral
differentiation, one via a user-specified FD stencil of declared order $p$.
It fits the decay rates and compares them to the theoretical rates. A
verdict of "FD-order mismatch" is raised when the FD rate is not close to
$p$; a verdict of "spectral saturation" is raised when the spectral sequence
flattens at the floor before exponential fitting can complete.

### 6.7 What this validates

- The implementation of the user's declared FD stencil has the order they
  declared, on the test function.
- The spectral pathway behaves like a spectral method on this test function
  (super-algebraic or exponential decay before saturation).

### 6.8 What this does not validate

- Exponential convergence cannot be claimed from data points at the floor;
  the rule must detect and exclude these.
- Super-algebraic convergence on *one* smooth periodic function does not
  guarantee the user's operator will converge super-algebraically on every
  smooth function they feed it.
- Nonperiodic problems require different test functions and different
  truncation-error theorems (Chebyshev, Legendre); PH-RES-003 as stated in
  v1.0 is scoped to periodic domains.

### 6.9 Report-integration note

Function 2: a fixture on $u(x) = \exp(\sin x)$ measuring spectral and FD
residuals at a sequence of $N$ values and verifying the rate. Function 3:
cite Trefethen 2000 *Spectral Methods in MATLAB* (SIAM) for the
spectral-convergence theorem.

---

# Part II — Boundary Rules

## 7. PH-BC-001 — Boundary Trace Consistency

### 7.1 Mathematical claim

Boundary residuals are only meaningful when the boundary trace of the
user's field is well-defined for the function space in which the PDE is
posed. For $H^1(\Omega)$ with Lipschitz $\partial\Omega$, the trace theorem
guarantees a bounded, surjective trace map onto the fractional Sobolev space
$H^{1/2}(\partial\Omega)$; the Dirichlet boundary residual is meaningful in
this setting. Neumann boundary residuals involve normal-derivative traces,
which live in a different space ($H^{-1/2}(\partial\Omega)$) and carry
additional regularity requirements.

### 7.2 Theoretical anchor

Trace theorem for Sobolev spaces on Lipschitz domains: there exists a
bounded linear operator $\gamma_0 : H^1(\Omega) \to H^{1/2}(\partial\Omega)$
extending the pointwise restriction for smooth functions
[Evans 2010, §5.5; Grisvard 1985, Ch. 1]. For Lipschitz domains with $1 \leq
p < \infty$, the trace theorem and its surjectivity onto
$W^{1-1/p, p}(\partial\Omega)$ are standard [Evans 2010 §5.5; Gagliardo 1957].

### 7.3 Derivation

Let $\Omega \subset \mathbb{R}^d$ have Lipschitz boundary $\partial\Omega$.
For $u \in C^1(\overline{\Omega})$, define the pointwise trace
$\gamma_0 u := u|_{\partial\Omega}$. The trace theorem [Evans 2010,
§5.5] states that $\gamma_0$ extends uniquely to a bounded linear
operator

$$
\gamma_0 : H^1(\Omega) \to L^2(\partial\Omega)
\tag{7.1}
$$

with image contained in $H^{1/2}(\partial\Omega)$, and there is a constant
$C_\Omega$ depending only on $\Omega$ such that

$$
\|\gamma_0 u\|_{L^2(\partial\Omega)} \leq C_\Omega \|u\|_{H^1(\Omega)}.
\tag{7.2}
$$

For Dirichlet data $g \in H^{1/2}(\partial\Omega)$, the boundary residual
$B_D(u) := \gamma_0 u - g$ is a well-defined element of $L^2(\partial\Omega)$,
and its norm is meaningful.

For Neumann data, the boundary residual involves a normal-derivative
trace. The correct functional setting depends on the regularity available
for $u$. Four regimes should be distinguished, because they correspond
to different sufficient conditions and different production-scope
contracts:

1. **Classical pointwise regime.** If $u \in C^1(\overline{\Omega})$, the
   pointwise normal derivative $(\nabla u \cdot \mathbf{n})|_{\partial
   \Omega}$ exists in the classical sense and lies in
   $C(\partial\Omega) \subset L^2(\partial\Omega)$. The Neumann residual
   $B_N(u) := \gamma_\nu u - g_N$ is then a well-defined
   $L^2(\partial\Omega)$ element.
2. **Strong Sobolev-trace regime.** If $u \in H^2(\Omega)$ on a Lipschitz
   (or more regular, as needed) boundary, the Sobolev trace theorem
   applied to $\partial_i u \in H^1(\Omega)$ yields a trace of each
   first derivative in $H^{1/2}(\partial\Omega)
   \hookrightarrow L^2(\partial\Omega)$, so the normal-derivative trace
   $\gamma_\nu u$ is a well-defined element of $L^2(\partial\Omega)$ in
   the **Sobolev** sense. This does **not** in general imply a classical
   pointwise normal derivative: in dimensions $d \geq 2$, Sobolev
   embedding gives $H^2(\Omega) \hookrightarrow C^0(\overline{\Omega})$
   (for $d \leq 3$) but not $H^2(\Omega) \hookrightarrow
   C^1(\overline{\Omega})$.
3. **Weak $H(\mathrm{div})$-trace regime.** If $u$ is only in
   $H^1(\Omega)$ but the vector field $\mathbf{q} := \nabla u$ satisfies
   $\mathbf{q} \in H(\mathrm{div}; \Omega)$, there is a well-defined
   weak normal-trace $\mathbf{q}\cdot\mathbf{n} \in
   H^{-1/2}(\partial\Omega)$ via the divergence-theorem duality pairing
   (see §8 for the full Gauss-Green statement). The Neumann residual is
   then interpreted in this duality sense; no pointwise or $L^2$
   interpretation is available without further regularity.
4. **$L^2(\partial\Omega)$-evaluability regime.** If $u_h$ happens to
   admit a pointwise normal derivative on the boundary even though its
   global regularity is only $H^1$ (e.g. piecewise polynomial on a mesh
   aligned with $\partial\Omega$), the Neumann residual can be computed
   directly in $L^2(\partial\Omega)$. This is the regime in which
   discrete FE and FD codes typically emit a Neumann residual.

Regime (4) is the natural operational target for a future Neumann /
flux extension, but it is outside the v1.0 production-validation scope.
v1.0 validates Dirichlet-type trace behavior only. Regime (1) is the
sufficient analytical condition for a classical pointwise
interpretation; regime (2) is the sufficient analytical condition for
a strong Sobolev-trace interpretation; regime (3) is the general weak
setting against which the rule's claim is correct but only diagnostic.

### 7.4 Assumptions

- $\Omega$ has Lipschitz boundary. Rougher boundaries require the full
  Gagliardo-type construction via Sobolev-Slobodeckij spaces.
- For Dirichlet checks: $u \in H^1(\Omega)$; $g \in H^{1/2}(\partial\Omega)$
  (or equivalently, $g$ is the trace of some $H^1$-extension).
- For Neumann checks: $u$ has sufficient regularity for the normal-derivative
  trace to be well-defined, or the check is interpreted in the weak
  duality-pairing sense.

### 7.5 Implementation mapping

The v1.0 production rule validates Dirichlet-type boundary trace
behavior. Given a user-declared Dirichlet boundary condition $u|_{
\partial\Omega} = g_D$ and a numerical solution $u_h$, the rule emits

$$
\|\gamma_0 u_h - g_h\|_{L^2(\partial\Omega)},
$$

computed as a surface-integral discretization on the boundary mesh.
Three fixture branches are validated: zero-on-exact-trace, perturbation
scaling under refinement, and absolute-vs-relative-mode behavior.

The Neumann-trace mathematical content of §7.3 (four trace-evaluability
regimes) is retained for context and for the v1.2 normal-derivative
extension, but is **outside the v1.0 production-validation scope**.
v1.0 does not emit a verdict on Neumann or flux semantics; a separate
normal-derivative path is required for v1.2.

### 7.6 What this validates

- For Dirichlet conditions with $u_h$ regular enough to admit an
  $L^2$-trace: a well-posed measurement of boundary discrepancy, with
  continuity from the bulk norm via (7.2).
- The v1.0 scope boundary that Neumann, Robin, and flux semantics are
  not production-validated.

### 7.7 What this does not validate

- The rule does not verify that the user's choice of BC (Dirichlet vs.
  Neumann vs. Robin) is physically correct.
- For non-Lipschitz domains (cusps, slits), (7.1) does not hold as stated;
  the rule's verdict is conditional on the Lipschitz assumption.
- A small boundary residual does not imply a small bulk PDE residual; (7.2)
  is a one-way bound.

### 7.8 Report-integration note

Function 2: a fixture on a unit square with Dirichlet data $g(x,y) = xy$
and a polynomial $u_h$; verify the boundary residual matches hand
calculation. Function 3: cite Evans 2010 *Partial Differential Equations*,
2nd ed., AMS GSM vol. 19, §5.5 for the trace theorem.

---

## 8. PH-BC-002 — Gauss-Green / Divergence-Theorem Boundary Flux

> **High-risk section.** This rule has a specific scope boundary: the
> underlying theorem applies to general vector fields, but the v1.0
> production rule supports Laplace-type operators only. This section must
> preserve that distinction and not over-claim vector-field generality for
> the shipped rule.

### 8.1 Mathematical claim

For a vector field $\mathbf{F}$ with sufficient regularity on a domain
$\Omega$ with sufficiently regular boundary, the integral of the divergence
over $\Omega$ equals the integral of the outward normal flux over
$\partial\Omega$:

$$
\int_\Omega \nabla \cdot \mathbf{F}\, dx = \int_{\partial\Omega} \mathbf{F}
\cdot \mathbf{n}\, ds.
\tag{8.1}
$$

This is the Gauss-Green / divergence theorem. The harness-level
correctness fixture exercises the general identity; the production rule in
v1.0 applies it only to $\mathbf{F} = \nabla u$ for Laplace-type operators,
giving $\int_\Omega \Delta u\, dx = \int_{\partial\Omega} \partial_\nu u\, ds$.

### 8.2 Theoretical anchor

Divergence theorem for Lipschitz domains, standard reference [Evans 2010
§C.2; Hunter, *Notes on Partial Differential Equations*, divergence
theorem notes]. The
classical version requires $\mathbf{F} \in C^1(\overline{\Omega})$; the
Sobolev version holds for $\mathbf{F} \in H(\text{div}; \Omega) := \{\mathbf{F}
\in L^2(\Omega)^d : \nabla \cdot \mathbf{F} \in L^2(\Omega)\}$, with the
boundary flux interpreted in the $H^{-1/2}(\partial\Omega)$ duality with
$H^{1/2}(\partial\Omega)$.

### 8.3 Derivation

For $\mathbf{F} \in C^1(\overline{\Omega})$ and $\Omega$ with Lipschitz
boundary, the divergence theorem states (8.1). The proof [Evans 2010
§C.2] proceeds by (i) establishing the identity for functions with compact
support in a half-space via Fubini, (ii) extending to general Lipschitz
domains by a partition of unity and change of coordinates that flattens the
boundary locally. Integration by parts follows by applying (8.1) to
$\mathbf{F} = u\mathbf{G}$:

$$
\int_\Omega (\nabla u \cdot \mathbf{G} + u\, \nabla\cdot \mathbf{G})\, dx
= \int_{\partial\Omega} u\, \mathbf{G} \cdot \mathbf{n}\, ds.
\tag{8.2}
$$

**Canonical fixture.** Take $\mathbf{F}(x,y) = (x, y)$ on $\Omega = [0,1]^2$.
Then $\nabla \cdot \mathbf{F} = 2$, so $\int_\Omega \nabla \cdot \mathbf{F}\,
dx = 2 \cdot \text{area}(\Omega) = 2$. The boundary flux: on the right edge
$x = 1$, $\mathbf{n} = (1, 0)$, $\mathbf{F} \cdot \mathbf{n} = 1$, integral
$= 1$; on the top edge $y = 1$, $\mathbf{n} = (0, 1)$, $\mathbf{F} \cdot
\mathbf{n} = 1$, integral $= 1$; left and bottom edges contribute $0$.
Total flux $= 2$. The two sides match exactly.

**Laplace specialization.** For $\mathbf{F} = \nabla u$ with $u \in
H^2(\Omega)$:

$$
\int_\Omega \Delta u\, dx = \int_{\partial\Omega} \partial_\nu u\, ds.
\tag{8.3}
$$

This is the parent identity from which the v1.0 harmonic / Laplace-zero
production path is derived; the nonzero-Poisson branch is outside v1.0
production scope and is handled only as a SKIP / NotImplementedError path.

### 8.4 Assumptions

- $\Omega$ has Lipschitz boundary.
- For the classical form (8.1): $\mathbf{F} \in C^1(\overline{\Omega})$.
- For the Sobolev form: $\mathbf{F} \in H(\text{div}; \Omega)$, with the
  boundary integral interpreted in the duality pairing.
- For the Laplace specialization: $u \in H^2(\Omega)$ (so $\partial_\nu u$
  admits an $L^2$-trace).

### 8.5 Implementation mapping

**Harness layer** (testing infrastructure, Function 2): verifies the
general identity (8.1) on the fixture $\mathbf{F}(x,y) = (x,y)$, $\Omega =
[0,1]^2$.

**Production rule layer** (what physics-lint ships in v1.0): checks the
supported harmonic / Laplace-zero production path derived from (8.3).
The nonzero-Poisson branch is present only as a scoped SKIP /
NotImplementedError path in v1.0. The rule's API accepts a scalar field
$u$ and a declared domain; it does not in v1.0 accept a general vector
field $\mathbf{F}$. On the supported harmonic path, the rule emits
$|\int_\Omega \Delta u_h\, dx - \int_{\partial\Omega} \partial_\nu u_h\, ds|$
computed via the domain and boundary quadratures associated with the user's
grid or mesh; on a harmonic fixture both sides equal zero analytically and
the imbalance is zero to quadrature / floating-point tolerance.

This distinction between harness and production is load-bearing: the
harness exercises the theorem at full generality to catch implementation
bugs in the integrator; the production rule exposes only the Laplace
scope, which is what v1.0 supports end-to-end.

### 8.6 What this validates

- For the supported v1.0 harmonic Laplace fixtures (those satisfying
  $\Delta u = 0$ on a Lipschitz domain): the computed bulk Laplacian
  integral and boundary flux integral both vanish to roundoff, so the
  reported imbalance is zero within quadrature / floating-point
  tolerance.
- Detection of inconsistencies between a user's bulk operator and boundary
  operator (e.g. a missing or duplicated boundary term, a sign error on
  the normal).

### 8.7 What this does not validate

- The v1.0 production rule does **not** claim general vector-field
  production support. It verifies (8.3), not (8.1), in the shipped API.
- Non-Lipschitz boundaries (e.g. fractal or slit domains) are out of scope.
- A successful flux-balance check does not imply the model satisfies the
  Poisson equation pointwise; it checks a single integrated identity.
- The harness fixture verifies integrator correctness, not model
  correctness.

### 8.8 Report-integration note

**Sign convention.** The Gauss-Green / divergence-theorem identity used
here is

$$
\int_\Omega \Delta u\, dx \;=\; \int_{\partial\Omega} \partial_\nu u\, ds,
\tag{8.8a}
$$

where $\partial_\nu u = \nabla u \cdot \mathbf{n}$ is the outward
normal derivative on $\partial\Omega$. The shipped Laplace-imbalance
production rule reports the **difference** of the two sides of (8.8a)
as the imbalance scalar; in the harmonic case both sides equal zero
and the difference is zero to roundoff. Implementations that define
the boundary flux with the opposite sign convention (inward normal,
or a sign-flipped surface integrand) must adjust the reported
imbalance accordingly; the rule does not infer the sign convention
from the flux quantity alone.

**Function 2 (harness layer).** The vector-field fixture
$\mathbf{F}(x, y) = (x, y)$ with analytical
$\int_\Omega \nabla \cdot \mathbf{F}\, dx
= \int_{\partial\Omega} \mathbf{F} \cdot \mathbf{n}\, ds = 2$
on the unit square exercises the full Gauss-Green theorem at the
harness level, where both sides are non-trivial and equal. This
fixture is the harness-level test of the divergence theorem itself,
not the production rule's contract.

**Function 2 (production layer).** The shipped production rule covers
the Laplace case ($\Delta u = 0$) only. Production fixtures use
**harmonic** test functions on the unit square so that both sides of
(8.8a) equal zero and the imbalance scalar is zero to roundoff:

- $u(x, y) = x^2 - y^2$ (real part of $z^2$): $\Delta u = 0$.
- $u(x, y) = xy$ (imaginary part of $z^2$ up to a constant):
  $\Delta u = 0$.
- $u(x, y) = x^5 - 10 x^3 y^2 + 5 x y^4$ (real part of $z^5$):
  $\Delta u = 0$.

A Poisson-style fixture such as $u(x, y) = x^2 + y^2$ has
$\Delta u = 4 \ne 0$ and would exercise the non-zero-Laplacian branch,
which is **out of v1.0 production scope** (`NotImplementedError → SKIP`
path); it is therefore not used as a v1.0 production fixture. The
Poisson arm with non-zero $\Delta u$ is deferred to v1.2.

**Function 3.** Evans 2010, *Partial Differential Equations*, 2nd ed.,
AMS GSM vol. 19, §C.2 for the divergence-theorem statement and the
Lipschitz-domain assumption.

---

# Part III — Conservation Rules

## 9. PH-CON-001 — Mass / Integral Conservation

### 9.1 Mathematical claim

For a conservation law $\partial_t u + \nabla \cdot \mathbf{F}(u) = s$, the
time derivative of the spatial integral of $u$ equals the negative of the
boundary outflux plus the source integral. In particular, for
source-free problems with periodic or no-flux boundary conditions, the
spatial integral $\int_\Omega u\, dx$ is constant in time.

### 9.2 Theoretical anchor

Balance law for hyperbolic and parabolic conservation equations, derived
from the divergence theorem (§8) applied to the PDE [LeVeque 2002
*Finite Volume Methods for Hyperbolic Problems*, Ch. 2].

### 9.3 Derivation

Consider the scalar conservation law

$$
\partial_t u + \nabla \cdot \mathbf{F}(u) = s, \quad x \in \Omega,\ t \in [0, T].
\tag{9.1}
$$

Integrate both sides over $\Omega$ and apply the divergence theorem (§8) to
the flux term:

$$
\int_\Omega \partial_t u\, dx + \int_{\partial\Omega} \mathbf{F}(u) \cdot
\mathbf{n}\, ds = \int_\Omega s\, dx.
\tag{9.2}
$$

Under the standing smoothness assumptions (in particular, $u$ and its time
derivative are sufficiently smooth to swap $\int$ and $\partial_t$), this
becomes

$$
\frac{d}{dt} \int_\Omega u\, dx = - \int_{\partial\Omega} \mathbf{F}(u)
\cdot \mathbf{n}\, ds + \int_\Omega s\, dx.
\tag{9.3}
$$

**Special cases.**

- Periodic boundary conditions: the boundary term vanishes by periodicity of
  $\mathbf{F}(u)$, so $\frac{d}{dt} \int_\Omega u\, dx = \int_\Omega s\, dx$.
  In the source-free case, $\int_\Omega u\, dx$ is constant.
- No-flux (zero Neumann) boundary conditions: $\mathbf{F}(u) \cdot \mathbf{n}
  = 0$ on $\partial\Omega$; same conclusion.
- Dirichlet conditions with known boundary flux: the boundary term is
  explicit and can be checked numerically.

### 9.4 Assumptions

- $u \in C^1([0,T]; L^1(\Omega)) \cap C^0([0,T]; W^{1,1}(\Omega))$, or
  equivalently, sufficient regularity to exchange $\int_\Omega$ and
  $\partial_t$.
- The flux $\mathbf{F}$ and source $s$ are integrable.
- Boundary conditions are declared consistently with the PDE.

### 9.5 Implementation mapping

The rule discretizes both sides of (9.3) and reports the *conservation
drift* $\Delta_{\text{cons}}(t) := \int_\Omega u_h(t)\, dx - \int_\Omega
u_h(0)\, dx - \int_0^t [\text{source} - \text{flux}]\, d\tau$. In the
periodic source-free case, this reduces to the change in the total mass
from time $0$ to time $t$. The verdict compares $|\Delta_{\text{cons}}(t)|$
to a user-specified tolerance.

### 9.6 What this validates

- For periodic / no-flux source-free problems: any deviation from
  $\int u\, dx = \text{const}$ indicates a violation of conservation by
  the model.
- For general problems: the integrated balance law is checked, detecting
  model outputs that fail the divergence-theorem identity applied to the
  PDE.

### 9.7 What this does not validate

- Conservation is a *necessary* condition for a correct conservation-law
  solver, not a sufficient one. A model that predicts uniform-drift in
  time can still conserve mass integrally.
- Pointwise PDE accuracy is not checked by this rule.
- For shock-containing problems, the model may conserve mass integrally
  while producing wrong shock locations.

### 9.8 Report-integration note

Function 2 and Function 3 evidence for the shipped v1.0 rule is
recorded in the PH-CON-001 row of §24. The shipped F2 fixture is
the analytical-snapshot eigenmode listed there
($u = \cos(2\pi x)\cos(2\pi y)\exp(-8\pi^2 \kappa t)$, drift floor
$\sim 10^{-18}$ with $\sim 1000\times$ safety factor). The
periodic-advection bump description in earlier drafts is not part of
the shipped v1.0 validation contract. Credibility-anchor framing
remains LeVeque 2002 Ch. 2 for the balance-law formulation.

---

## 10. PH-CON-002 — Energy Conservation Identity

> **High-risk section.** Energy conservation is delicate: the identity
> holds for specific boundary conditions, and it does not imply accuracy.
> This section must preserve both the correct statement and the careful
> non-claims.

### 10.1 Mathematical claim

For the linear wave equation on a bounded domain with periodic or
homogeneous Dirichlet / Neumann boundary conditions, the energy functional

$$
E(t) := \frac{1}{2} \int_\Omega \left[ u_t^2 + c^2 |\nabla u|^2 \right] dx
\tag{10.1}
$$

is constant in time. Energy drift $|E(t) - E(0)|$ therefore measures the
conservation defect of a numerical solution relative to the continuous
identity.

### 10.2 Theoretical anchor

Energy identity for the linear wave equation, derived by multiplying the
PDE by $u_t$ and integrating by parts [Evans 2010 §2.4.3, §7.2]. Analogous
identities hold for other conservative hyperbolic systems (Klein-Gordon,
Maxwell) and for Hamiltonian ODE systems.

### 10.3 Derivation

Consider the linear wave equation

$$
u_{tt} - c^2 \Delta u = 0, \quad x \in \Omega,\ t \in [0,T],
\tag{10.2}
$$

with $c > 0$ a constant. Multiply (10.2) by $u_t$ and integrate over
$\Omega$:

$$
\int_\Omega u_t u_{tt}\, dx - c^2 \int_\Omega u_t \Delta u\, dx = 0.
\tag{10.3}
$$

The first integral equals $\frac{d}{dt} \int_\Omega \frac{1}{2} u_t^2\, dx$.
Apply integration by parts (8.2) to the second integral with $\mathbf{G} =
\nabla u$:

$$
- \int_\Omega u_t \Delta u\, dx = \int_\Omega \nabla u_t \cdot \nabla u\, dx
- \int_{\partial\Omega} u_t\, \partial_\nu u\, ds.
\tag{10.4}
$$

The bulk term on the right equals $\frac{d}{dt} \int_\Omega \frac{1}{2}
|\nabla u|^2\, dx$. Substituting into (10.3):

$$
\frac{dE}{dt} = c^2 \int_{\partial\Omega} u_t\, \partial_\nu u\, ds.
\tag{10.5}
$$

The boundary term in (10.5) is the only source of energy change. It
**vanishes** under any of the following:

1. *Periodic boundary conditions*: $u_t$ and $\partial_\nu u$ are periodic,
   and the outward normals on opposite faces cancel.
2. *Homogeneous Dirichlet* ($u = 0$ on $\partial\Omega$): then $u_t = 0$ on
   $\partial\Omega$.
3. *Homogeneous Neumann* ($\partial_\nu u = 0$ on $\partial\Omega$).

Under any of these conditions, $dE/dt = 0$ and $E(t) = E(0)$ for all
$t \in [0, T]$.

### 10.4 Assumptions

- $c$ is constant (or the identity is stated in a weighted form for
  variable $c$; see Evans §7.2 for the variable-coefficient case).
- $u \in C^2$ or $u$ is a weak solution with the regularity needed for
  the integration by parts (10.4) to hold.
- Boundary conditions are homogeneous Dirichlet, homogeneous Neumann, or
  periodic. Inhomogeneous or radiative BCs introduce a nonzero boundary
  term that must be accounted for separately.

### 10.5 Implementation mapping

The rule computes $E(t_n)$ at discrete time levels $t_n$ using the
discrete approximations of $u_t^2$ and $|\nabla u|^2$ (the user specifies
the grid and the time-derivative scheme). It reports the *energy drift*
$\max_n |E(t_n) - E(0)| / E(0)$ (or in absolute units when $E(0)$ is
near zero). The verdict compares this drift to a tolerance.

### 10.6 What this validates

- For a conservative wave PDE with homogeneous-or-periodic BCs: large
  energy drift indicates the model is not preserving the continuous
  identity (10.5).
- Detection of dissipative or anti-dissipative numerical behavior (e.g.
  upwind dissipation, or instability that pumps energy).

### 10.7 What this does not validate

- **Energy conservation does not imply accuracy.** A model can conserve
  energy while producing wrong phase (dispersive errors), wrong wavelength,
  or wrong solution values. This is the single most important non-claim
  for this rule.
- For inhomogeneous BCs, radiative BCs, or absorbing boundary layers, the
  identity (10.5) has a nonzero right-hand side that is not checked by
  v1.0; the rule should not be applied in those regimes without user
  acknowledgment.
- For nonlinear wave equations (e.g. $u_{tt} - \Delta u + u^3 = 0$), the
  energy functional changes form (adds a potential term); v1.0 does not
  auto-derive the correct energy for nonlinear systems.
- Energy $\approx$ constant on a test trajectory does not imply
  conservation on held-out trajectories.

### 10.8 Report-integration note

Function 2 and Function 3 evidence for the shipped v1.0 rule is
recorded in the PH-CON-002 row of §24. The shipped F2 uses a two-layer
analytical wave-energy fixture: a harness-authoritative analytical
energy value computed from $(u_t, \nabla u)$ snapshots (roundoff
$\sim 5 \times 10^{-16}$), and a rule-verdict path whose internal
second-order central-difference $u_t$ primitive gives the documented
log-log slope of 1.94. Function 3 is absent with justification:
PDEBench shallow-water reports mass cRMSE and Hansen ProbConserv `CE`
is defined for first-order-in-time integral conservation laws, neither
of which is semantically comparable to the second-order-in-time wave-
energy functional $E[u, u_t] = \tfrac{1}{2}\int(u_t^2 + c^2 |\nabla u|^2)$
checked here. Evans 2010 §2.4.3 remains the mathematical energy-
identity anchor.

---

## 11. PH-CON-003 — Heat Energy-Dissipation Sign

> **Audit note.** Earlier drafts framed PH-CON-003 as a positivity /
> conservation coupled check (combining the integral conservation of
> §9 with the positivity invariant of §13). The shipped v1.0 rule
> instead checks the sign of the heat-equation energy dissipation
> `dE/dt`. This section now matches the shipped rule: the public README
> label "Energy dissipation sign violation" and the F1-side label "Heat
> Energy-Dissipation Sign" describe the same mathematical content. See
> Appendix D for the broader README label reconciliation.

### 11.1 Mathematical claim

For the heat equation $u_t - \Delta u = 0$ on a domain $\Omega$ with
compatible boundary conditions, the $L^2$ energy

$$
E(t) := \tfrac{1}{2} \|u(\cdot, t)\|_{L^2(\Omega)}^2
\tag{11.1}
$$

is non-increasing in time:

$$
\frac{dE}{dt}(t) \leq 0.
\tag{11.2}
$$

The rule emits a verdict when an empirically measured `dE/dt` is
positive beyond a tolerance, which is mathematically inconsistent with
a heat-equation solution under the stated boundary conditions.

### 11.2 Theoretical anchor

Standard parabolic-equation energy identity [Evans 2010 §7.1.2]. For
the heat equation $u_t = \Delta u$, multiplying by $u$ and integrating
gives the energy-dissipation identity (11.4).

### 11.3 Derivation

Multiplying $u_t - \Delta u = 0$ by $u$ and integrating over $\Omega$,

$$
\int_\Omega u\, u_t\, dx - \int_\Omega u\, \Delta u\, dx = 0.
\tag{11.3}
$$

The first term is $\frac{d}{dt} \int_\Omega \tfrac{1}{2} u^2\, dx =
dE/dt$. For the second, integration by parts gives

$$
- \int_\Omega u\, \Delta u\, dx = \int_\Omega |\nabla u|^2\, dx
- \int_{\partial\Omega} u\, \partial_\nu u\, ds.
$$

So

$$
\frac{dE}{dt} = -\int_\Omega |\nabla u|^2\, dx
+ \int_{\partial\Omega} u\, \partial_\nu u\, ds.
\tag{11.4}
$$

The boundary term vanishes under any of the standard compatible
boundary conditions:

- Periodic boundary: traces cancel pairwise.
- Homogeneous Dirichlet ($u|_{\partial\Omega} = 0$): the integrand is
  zero pointwise on $\partial\Omega$.
- Homogeneous Neumann ($\partial_\nu u|_{\partial\Omega} = 0$): the
  integrand is zero pointwise on $\partial\Omega$.

Under any of these regimes,

$$
\frac{dE}{dt} = -\|\nabla u\|_{L^2(\Omega)}^2 \leq 0,
\tag{11.5}
$$

with equality only if $\nabla u \equiv 0$ on $\Omega$, i.e. $u$ is
spatially constant. Under homogeneous Dirichlet boundary conditions
this forces $u \equiv 0$; under periodic or homogeneous Neumann
boundary conditions nonzero constants are compatible steady states.

### 11.4 Implementation mapping

The rule consumes a snapshot trajectory $\{u_h(\cdot, t_n)\}$ and emits
a measured `dE/dt` via a forward-difference primitive applied to (11.1):

$$
\widehat{\frac{dE}{dt}}(t_n) :=
\frac{E(t_{n+1}) - E(t_n)}{t_{n+1} - t_n}.
\tag{11.6}
$$

A WARN/FAIL verdict is raised when the measured value exceeds zero
beyond a positivity tolerance. Earlier revisions of the rule used a
central-difference primitive that produced spurious sign-flip artifacts
on the first and last samples of a trajectory; the central-diff bug was
fixed in commit `e691dd3`, and v1.0 ships the forward-difference form.

### 11.5 What this validates

- Sign-consistency between an empirically measured `dE/dt` and the
  heat-equation energy-dissipation identity (11.5), under the declared
  compatible boundary conditions.

### 11.6 What this does not validate

- Equation (11.5) is specific to the heat equation. The rule applies
  only when the user declares the parabolic / heat-equation PDE class.
  Schemes for other parabolic systems (with reaction terms, anisotropic
  coefficients, or non-self-adjoint structure) require separate
  derivations and are out of v1.0 scope.
- Magnitude of `dE/dt`: the rule checks only the sign. The exact
  per-step energy ratio for the eigenmode fixture (§11.7) is a
  consistency check on the measurement primitive, not a claim about
  the rule's verdict on user models.
- The rule does not validate the integral mass conservation of §9 or
  the positivity invariant of §13; those are independent rules.

### 11.7 Report-integration note

Function 2: an analytical eigenmode fixture
$u(x, y, t) = \sin(\pi x) \sin(\pi y) \exp(-2\pi^2 t)$ on the unit
square with homogeneous Dirichlet boundary conditions. This solution
gives a closed-form per-step energy ratio
$E(t + \Delta t) / E(t) = \exp(-4\pi^2 \Delta t)$, which at
$\Delta t = 0.05$ evaluates to $\exp(-0.2 \pi^2) \approx 0.13888$.
The fixture exercises the forward-difference `dE/dt` primitive against
this analytical decay rate and verifies sign-consistency. Function 3:
no borrowed-credibility reproduction target — the per-step energy
ratio is derivable in closed form from Evans 2010 §7.1.2 directly, so
there is no published numerical baseline to reproduce.
PDEBench has no standalone heat-equation entry; Hansen ProbConserv
Table 1 covers diffusion mass conservation rather than per-step
heat-energy dissipation.

---

## 12. PH-CON-004 — A Posteriori / Mesh-Based Conservation Indicator

### 12.1 Mathematical claim

On a simplicial mesh, residual-based a posteriori error indicators
$\eta_K$ defined element-wise localize regions where the strong residual
and inter-element normal-flux jumps of the numerical solution are
concentrated. These indicators are reliability/efficiency proxies for
the $H^1$ error of an elliptic FEM solution; they are **not**
certificates of local conservation in the sense of a discrete
divergence identity. Hotspots in $\eta_K$ flag locations where the
strong PDE residual and inter-element flux continuity are most
strained; whether this also indicates a discrete local-conservation
violation depends on the specific scheme (e.g. mixed methods enforce
local conservation by construction; standard $H^1$-conforming Lagrange
FEM does not). For v1.0, the rule supports 2D triangulated meshes only;
3D tetrahedral extensions are deferred to v1.2.

### 12.2 Theoretical anchor

A posteriori error estimation for finite-element methods [Verfürth 2013
*A Posteriori Error Estimation Techniques for Finite Element Methods*,
OUP; Ern-Guermond 2021 *Finite Elements II: Galerkin Approximation,
Elliptic and Mixed PDEs*, Springer TAM 73, Part VII "A posteriori error
analysis"]. The residual-based element indicator for the Poisson
equation $-\Delta u = f$ takes the form

$$
\eta_K^2 := h_K^2 \|f + \Delta u_h\|_{L^2(K)}^2 + \frac{1}{2} \sum_{e \subset
\partial K} h_e \| [\![ \partial_\nu u_h ]\!] \|_{L^2(e)}^2,
\tag{12.1}
$$

where $[\![\cdot]\!]$ denotes the jump of the normal derivative across
interior edges, and $K$ is a mesh element with diameter $h_K$.

### 12.3 Derivation

For the model problem $-\Delta u = f$ with $u_h$ the piecewise-linear
finite-element solution on a triangulation $\mathcal{T}_h$:

**Reliability bound** [Verfürth 2013, Ch. 1]:

$$
\|u - u_h\|_{H^1(\Omega)} \leq C_{\text{rel}} \left( \sum_K \eta_K^2
\right)^{1/2},
\tag{12.2}
$$

with $C_{\text{rel}}$ depending on $\Omega$ and the shape regularity of the
mesh.

**Efficiency bound** (local):

$$
\eta_K \leq C_{\text{eff}} \left( \|u - u_h\|_{H^1(\omega_K)} +
\text{oscillation}(f) \right),
\tag{12.3}
$$

where $\omega_K$ is the patch of elements adjacent to $K$ and oscillation
is the local data-oscillation term.

Together, (12.2)–(12.3) say: $\eta_K$ is a *local* proxy for the error
with bounded ratio to the true error in a patch around $K$. When the
model has a locally concentrated strong residual or inter-element
normal-flux jump (e.g. the element-interior PDE residual $f + \Delta
u_h$ is large, or neighboring elements disagree strongly on the normal
derivative), $\eta_K$ grows locally and flags the hotspot. This is a
claim about residual/error localization, not a claim about discrete
local-conservation violation in the sense of a divergence identity; the
two can coincide in mixed methods but not in general
$H^1$-conforming FEM.

### 12.4 Assumptions

- The user's problem is an elliptic second-order PDE posed on a 2D
  Lipschitz domain. 3D generalization to tet meshes follows the same
  framework but is out of v1.0 scope (see §12.7).
- The mesh is shape-regular (a standard FEM assumption; the reliability
  and efficiency constants depend on shape regularity).
- $u_h$ is piecewise linear on $\mathcal{T}_h$, or the indicator is
  adapted appropriately (higher-order extensions are standard but outside
  v1.0 scope).

### 12.5 Implementation mapping

**Theorem family (Function-1 background).** The full Verfürth-style
residual estimator $\eta_K$ as derived in §12.3 combines element-
volume and inter-element facet-jump terms, with an additional source-
data oscillation term, and admits the reliability and efficiency
bounds (12.2)–(12.3) on shape-regular triangulations.

**Harness-level F2.** A controlled L-shape singularity / localized
hotspot fixture demonstrates that an estimator-style indicator
concentrates on the elements adjacent to the singularity, consistent
with the localization behavior predicted by residual-estimator theory.
This harness layer is the part of the rule that is exercised against
the §12.3 mathematics directly.

**Shipped v1.0 production behavior.** The shipped rule emits a
narrower scalar:

$$
\rho := \frac{\max_{K} \int_K (\Delta_{L^2,\mathrm{zero\text{-}trace}} u_h)^2\, dx}{\mathrm{mean}_{K} \int_K (\Delta_{L^2,\mathrm{zero\text{-}trace}} u_h)^2\, dx},
\tag{12.4}
$$

i.e. the ratio of the maximum to the mean of the interior volumetric
$L^2$-projected zero-trace Laplacian term over interior elements
$K \in \mathcal{T}_h$. The shipped scalar:

- includes only the **interior volumetric term** $\|\Delta u_h\|_{L^2(K)}^2$ (with an $L^2$ projection imposing zero boundary trace);
- **does not include** the inter-element facet-jump term that appears in (12.1);
- **does not include** a source-data oscillation term;
- is computed on scikit-fem 2D triangulated meshes only; 3D tetrahedral support is deferred to v1.2.

The shipped scalar (12.4) therefore inherits **none** of the Verfürth
reliability / efficiency guarantees of (12.2)–(12.3) directly. It is
better described as a refinement-invariant residual-hotspot indicator
ratio, calibrated against the L-shape harness fixture (where the
ratio remains near $\sim 1.70$ across uniform refinements), rather
than as a certified a posteriori error estimator.

### 12.6 What this validates

- **Harness layer.** Localization behavior consistent with residual-
  estimator theory on controlled 2D fixtures (the L-shape hotspot
  refinement-invariance check).
- **Production layer.** A conservation-defect / residual-hotspot
  indicator ratio (12.4), not a certified a posteriori error estimator.
  The ratio is refinement-invariant on the harness fixture and useful
  as a relative hotspot diagnostic; it is not a substitute for the
  full $\eta_K$ estimator of (12.1) for users who need
  reliability / efficiency guarantees.

### 12.7 What this does not validate

- **3D tetrahedral meshes are out of scope for v1.0.** The rule does
  not ship with a 3D tet-mesh indicator; this is deferred to v1.2 per
  the plan's scope boundary.
- Higher-order elements, non-Poisson operators (Stokes, elasticity)
  require different indicator formulas not covered by v1.0.
- A small (12.4) ratio does **not** imply a small $L^\infty$, $L^2$,
  or $H^1$ error in a Verfürth-reliability-bound sense, because the
  shipped scalar drops the facet-jump and oscillation terms.
- Heavily anisotropic meshes can degrade shape regularity; the
  harness-layer reliability behavior is documented only on
  shape-regular triangulations.

### 12.8 Report-integration note

Function 2: the v1.0 harness uses an L-shape singularity / hotspot
fixture and verifies that the interior volumetric indicator ratio
$\rho$ from (12.4) remains refinement-invariant at approximately the
documented level (~1.70 in the shipped configuration) across uniform
refinements. This is an a posteriori-inspired localization fixture,
not a full Verfürth estimator reproduction.

Function 3: cite Verfürth 2013 (OUP) and Ern-Guermond 2021 Vol. II
(Springer TAM 73, Part VII) for the full residual-estimator framework.
The shipped v1.0 production scalar is narrower than that framework
(interior volumetric term only, no facet-jump or oscillation terms)
and does not inherit the reliability / efficiency bounds directly.

---

# Part IV — Positivity Rules

## 13. PH-POS-001 — Positivity / Maximum-Principle Framing

### 13.1 Mathematical claim

For PDEs satisfying a maximum principle or invariant-domain property,
negative values or out-of-interval values in a solution that starts inside
the admissible domain signal a physically invalid output.

### 13.2 Theoretical anchor

Maximum principle for elliptic and parabolic operators [Evans 2010 §6.4,
§7.1.4; Protter-Weinberger 1967]; invariant-domain framework for
hyperbolic systems [Guermond-Popov 2016, *Invariant Domains and
First-Order Continuous Finite Element Approximation for Hyperbolic
Systems*, SIAM J. Numer. Anal. **54(4)**, pp. 2466–2489, DOI
10.1137/16M1074291].

### 13.3 Derivation

For the heat equation $u_t - \Delta u = 0$ on a bounded domain with
nonnegative initial data $u_0 \geq 0$ and nonnegative Dirichlet data
$u|_{\partial\Omega} \geq 0$, the parabolic weak maximum principle
[Evans 2010 §7.1.4] gives

$$
\min_{x \in \overline\Omega,\ t \in [0,T]} u(x, t) \geq 0.
\tag{13.1}
$$

More generally, for $\partial_t u + L u = 0$ with $L$ a second-order
elliptic operator (no zeroth-order term that breaks the principle),
nonnegativity of initial and boundary data implies nonnegativity of the
solution on the parabolic cylinder.

For hyperbolic systems, analogous invariant-domain results hold under
specific compatibility assumptions between the flux and the admissible
set [Guermond-Popov 2016 SIAM J. Numer. Anal. 54(4):2466–2489].

### 13.4 Assumptions

- The PDE's operator admits a maximum principle (standard for elliptic
  and parabolic second-order with bounded lower-order coefficients) or an
  invariant-domain property (for specified hyperbolic systems).
- Initial and boundary data are nonnegative.
- For invariant-domain hyperbolic cases, the admissible set is declared
  by the user.

### 13.5 Implementation mapping

The rule emits $\min_{x} u_h(x, t)$ (and equivalently for multiple
components). A verdict is raised when this quantity is less than
$-\varepsilon_{\text{pos}}$, where $\varepsilon_{\text{pos}}$ is a
tolerance allowing for floating-point and discretization-induced small
negatives that do not indicate a real violation.

### 13.6 What this validates

- Positivity (nonnegativity) of the discrete solution for PDEs with a
  maximum principle, up to a user-declared tolerance.

### 13.7 What this does not validate

- Not every PDE obeys a maximum principle; the rule applies only when the
  user's PDE specification declares this property.
- Very small negative values (below $\varepsilon_{\text{pos}}$) may be
  numerical noise and do not imply physical invalidity.
- A positivity-respecting solution can still be numerically wrong in
  other ways (wrong magnitude, wrong location of features).

### 13.8 Report-integration note

Function 2 and Function 3 evidence for the shipped v1.0 rule is
recorded in the PH-POS-001 row of §24. The shipped F2 uses closed-form
positive harmonic / parabolic fixtures whose positivity is analytically
known on the domain interior, plus an Evans-corner negative control
that injects a strong-maximum-principle violation. The shipped rule is
a discrete predicate (verdict + $raw\_value = \min(u)$ when below
floor); fixtures are analytically-known-positive and the verdict is
mechanically secure on them. Function 3 is absent with justification
because Evans theorems are reproduced as a structural identity rather
than as a numerical baseline. Evans 2010 §6.4 / §7.1.4 remains the
mathematical maximum-principle anchor.

---

## 14. PH-POS-002 — Bound / Invariant-Domain Check

### 14.1 Mathematical claim

Some PDE states are constrained to remain inside a physical interval
$[u_{\min}, u_{\max}]$ or a more general invariant domain. For such
states, pointwise violations indicate physically invalid outputs.

### 14.2 Theoretical anchor

Invariant-domain framework [Guermond-Popov 2016, *Invariant Domains and
First-Order Continuous Finite Element Approximation for Hyperbolic
Systems*, SIAM J. Numer. Anal. **54(4)**, pp. 2466–2489,
DOI 10.1137/16M1074291] and maximum-principle extensions to bounded
intervals [Evans 2010 §7.1.4].

### 14.3 Derivation

For a scalar conservation law $\partial_t u + \nabla \cdot \mathbf{F}(u) =
0$ with initial data satisfying $u_0(x) \in [a, b]$ pointwise, the
(entropy) solution $u(\cdot, t)$ satisfies $u(x, t) \in [a, b]$ for all
$t > 0$ [Kruzkov 1970; LeVeque 2002, §11.13]. For linear advection with
smooth data, this reduces to the maximum principle on a bounded interval.

### 14.4 Assumptions

- The user's PDE is declared to have an invariant domain
  $D \subset \mathbb{R}^m$ (for $m$-component states).
- Initial and boundary data lie in $D$.
- The discretization is intended to preserve $D$; v1.0 does not check
  whether the user's scheme is itself invariant-domain preserving.

### 14.5 Implementation mapping

The rule emits $(\min_x u_h, \max_x u_h)$ per component and checks
whether these fall within the declared invariant domain, with tolerance
$\varepsilon_{\text{dom}}$.

### 14.6 What this validates

- Bound violations (pointwise $u_h$ outside $D$) that cannot be explained
  by numerical noise below $\varepsilon_{\text{dom}}$.

### 14.7 What this does not validate

- The rule does **not** verify that the user's declared $D$ is correct
  for the PDE in question.
- Small violations may arise from interpolation or reconstruction at
  sharp features; the tolerance is user-controlled.
- Satisfying the invariant-domain constraint does not imply solution
  accuracy.

### 14.8 Report-integration note

Function 2 and Function 3 evidence for the shipped v1.0 rule is
recorded in the PH-POS-002 row of §24. The shipped F2 uses three
harmonic polynomial fixtures ($x^2 - y^2$, $xy$, $x^3 - 3xy^2$) plus
an interior-overshoot negative control injecting a maximum-principle
violation. The linear-advection-with-step-function description in
earlier drafts is not part of the shipped v1.0 validation contract.
Credibility-anchor framing: Guermond-Popov 2016 invariant-domain
framework (used as background; see Appendix B.1).

---

# Part V — Symmetry and Equivariance Rules

## 15. PH-SYM-001 — Finite Symmetry Equivariance Check

### 15.1 Mathematical claim

A function $f : X \to Y$ is equivariant under a group action when applying
the group action to the input and then the function gives the same result
as applying the function and then the corresponding output action. The
rule measures a finite-sample discrete approximation to this equivariance
defect, for a declared finite group $G$ acting on the input and output.

### 15.2 Theoretical anchor

Group-equivariance framework as used throughout geometric deep learning
[Cohen-Welling 2016 *Group Equivariant Convolutional Networks*, ICML;
Gerken et al. 2023 *Geometric Deep Learning and Equivariant Neural
Networks*, AI Review 56(12):14605–14662]. The structural equivalence
between the rule's metric and the group-equivariance defect is direct:
the rule evaluates the defect on a finite test set.

### 15.3 Derivation

Let $G$ be a group acting on input space $X$ and output space $Y$ through
representations $\rho_X : G \to \text{Aut}(X)$ and $\rho_Y : G \to
\text{Aut}(Y)$. A map $f : X \to Y$ is **$G$-equivariant** if

$$
f(\rho_X(g)\, x) = \rho_Y(g)\, f(x) \quad \text{for all } g \in G,\ x \in X.
\tag{15.1}
$$

Invariance is the special case $\rho_Y \equiv \text{id}$.

For a finite test set $\{x_1, \ldots, x_N\} \subset X$ and a finite group
$G = \{g_1, \ldots, g_{|G|}\}$, the **empirical equivariance defect** is

$$
\mathcal{E}(f; G, \{x_i\}) := \frac{1}{N |G|} \sum_{i=1}^N \sum_{g \in G}
\| f(\rho_X(g)\, x_i) - \rho_Y(g)\, f(x_i) \|_Y^2.
\tag{15.2}
$$

$\mathcal{E} = 0$ iff (15.1) holds on the finite test set; as a model-level
property, $\mathcal{E}$ small on a well-distributed test set is evidence
for (but does not prove) model equivariance.

### 15.4 Assumptions

- $G$ is finite (e.g. $\mathbb{Z}_4$, $D_4$, $\mathbb{Z}_2^3$). Infinite
  groups are handled in PH-SYM-003 / PH-SYM-004.
- $\rho_X, \rho_Y$ are declared by the user.
- The test set $\{x_i\}$ is supplied by the user; the rule reports
  empirical defect only on that set.

### 15.5 Implementation mapping

The rule emits $\mathcal{E}(f; G, \{x_i\})$ from (15.2) together with the
maximum per-sample defect $\max_i \max_g \| f(\rho_X(g)x_i) -
\rho_Y(g)f(x_i)\|$. A verdict is raised when these exceed user
tolerances.

### 15.6 What this validates

- Finite-sample empirical equivariance of the user-declared function
  under the user-declared finite group, on the user-supplied test set.

### 15.7 What this does not validate

- **Finite test samples do not prove global equivariance.** $\mathcal{E}
  = 0$ on $N = 100$ samples is not a proof that $\mathcal{E} = 0$ for
  all $x$.
- Equivariance under $G$ does not imply equivariance under subgroups or
  supergroups of $G$.
- The rule does not check that $\rho_X, \rho_Y$ are in fact
  representations of $G$ (i.e. that $\rho(gh) = \rho(g)\rho(h)$); this is
  the user's responsibility.

### 15.8 Report-integration note

Function 2 and Function 3 evidence for the shipped v1.0 rule is
recorded in the PH-SYM-001 row of §24. The shipped F2 uses
closed-form rotation-equivariant operators (FFT-based Laplace
inverse, identity) on a 2D periodic square grid as $C_4$ structural-
equivalence positive controls, plus a coordinate-dependent CNN as the
non-equivariant negative control. The rotationally-averaged-CNN-on-
rotated-MNIST description in earlier drafts is not part of the
shipped v1.0 validation contract; downstream RotMNIST evaluation
requires the v1.2 escnn / Modal infrastructure described in the §24
PH-SYM-003 row. Credibility-anchor framing: Cohen-Welling 2016 ICML
and Gerken et al. 2023 AI Review remain supplementary equivariance-
literature anchors.

---

## 16. PH-SYM-002 — Symmetry Violation / Transformation Consistency

### 16.1 Mathematical claim

Symmetry violation can be measured by comparing predictions on
transformed inputs against the corresponding transformations of
predictions on the original inputs. The metric is a variant of the
equivariance defect (15.2), exposed as a separate rule for reporting.

### 16.2 Theoretical anchor

Same framework as PH-SYM-001 (Cohen-Welling 2016, Gerken et al. 2023).
The distinction is operational: PH-SYM-001 checks expectation-level
equivariance, PH-SYM-002 exposes per-transform violation profiles.

### 16.3 Derivation

For each transformation $g \in G$, define the per-transform defect

$$
\mathcal{V}_g(f; x_i) := \| f(\rho_X(g)\, x_i) - \rho_Y(g)\, f(x_i) \|_Y.
\tag{16.1}
$$

The rule reports the distribution of $\mathcal{V}_g$ across $g$ and $i$,
allowing the user to identify *which* transformations are most
violated.

### 16.4 Implementation mapping

Same harness as PH-SYM-001; the rule emits the full array
$\{\mathcal{V}_g(f; x_i)\}$ and aggregate statistics (max, mean, quantiles).

### 16.5 What this validates

- Localized identification of which symmetries the model most violates,
  useful for diagnosis.

### 16.6 What this does not validate

- Same non-claims as PH-SYM-001.
- The score is a per-transform empirical diagnostic, not a global proof.

### 16.7 Report-integration note

Function 2 and Function 3 evidence for the shipped v1.0 rule is
recorded in the PH-SYM-002 row of §24. The shipped F2 uses
closed-form mirror / reflection-equivariant operators on a 2D
periodic square grid as $\mathbb{Z}_2$ structural-equivalence
positive controls, plus a non-equivariant negative control. The
"per-group-element defect statistics extension of PH-SYM-001"
description in earlier drafts is documented but is not the shipped
F2. Credibility-anchor framing: same as PH-SYM-001 (see §15.8 and
the §24 PH-SYM-002 row).

---

## 17. PH-SYM-003 — Lie-Derivative / Infinitesimal Equivariance

> **Highest-risk section in the document.** This rule invokes Lie-group
> machinery that is easy to over-claim. The derivation below is written
> with a deliberate direction-of-implication diagram, a named theorem for
> each direction, and an explicit list of what is NOT proved by the rule
> (including, in particular, the Gruver reproduction claim and the
> disconnected-group extension).

### 17.1 Mathematical claim

For a connected matrix Lie group $G$ with Lie algebra $\mathfrak{g}$, and
for smooth representations $\rho_X, \rho_Y$, empirical infinitesimal
equivariance violations (Lie-derivative defects on a set of generators)
are related to finite equivariance over the identity component $G^0$
through the one-parameter subgroup theorem and the continuous-to-smooth
bridge. physics-lint computes an empirical Lie-derivative defect; this
defect is a diagnostic for infinitesimal equivariance at the generators,
which under specified assumptions implies finite equivariance over $G^0$.

### 17.2 Theoretical anchor

Three named theorems anchor the derivation. Hall's two locators are
verified against a physical copy; Kirillov's locator is verified
against a publicly accessible digital course-note copy. See Appendix B.

**[Hall 2015 Theorem 2.14]** (one-parameter subgroup theorem). Let
$A : \mathbb{R} \to \mathrm{GL}(n; \mathbb{C})$ be a one-parameter
subgroup, i.e. $A$ is continuous and $A(s + t) = A(s) A(t)$. Then there
exists a unique matrix $X \in M_n(\mathbb{C})$ such that
$A(t) = \exp(tX)$ for all $t \in \mathbb{R}$. *(Locator verified
against a physical copy of Hall 2nd edition.)*

**[Hall 2015 Corollary 3.50]** (continuous-to-smooth). Every continuous
homomorphism between matrix Lie groups is smooth (indeed,
real-analytic). *(Locator verified against a physical copy of Hall 2nd
edition.)*

**[Kirillov 2008 Corollary 2.9]** (identity-neighborhood generation).
If $G$ is a connected Lie group and $U$ is a neighborhood of the
identity, then $U$ generates $G$. Consequently, once equivariance is
established on a sufficiently small identity neighborhood generated by
one-parameter subgroups, connectedness extends the result to all of
$G^0$ by finite products. *(Source: Kirillov, *Introduction to Lie
Groups and Lie Algebras*, Cambridge Studies in Advanced Mathematics
vol. 113; publicly accessible course-note copy confirms the corollary
statement at §2.4.)*

Earlier drafts of this document cited Varadarajan GTM 102 §2.9–2.10
for the identity-component generation step. Because that locator has
not been confirmed against a physical copy of Varadarajan, the
load-bearing anchor has been replaced with Kirillov Corollary 2.9,
which states exactly the needed result with a verifiable locator.
Varadarajan remains available as background metadata in Appendix B.1
but is no longer load-bearing for the derivation.

### 17.3 Derivation

Let $G$ be a connected matrix Lie group acting on $X, Y$ via smooth
representations $\rho_X, \rho_Y$. Let $\mathfrak{g}$ denote the Lie
algebra of $G$, and for each generator $A \in \mathfrak{g}$ define the
one-parameter subgroup $\gamma_A(t) := \exp(tA) \in G$. By Hall Theorem
2.14, $\gamma_A$ is the unique one-parameter group with tangent $A$.

**Finite equivariance $\Rightarrow$ infinitesimal constraint.** Suppose $f : X \to Y$
is equivariant under $G$:

$$
f(\rho_X(\exp(tA))\, x) = \rho_Y(\exp(tA))\, f(x), \quad \forall t \in
\mathbb{R},\ \forall A \in \mathfrak{g}.
\tag{17.1}
$$

Differentiating at $t = 0$ (using the smoothness of $\rho_X, \rho_Y$
guaranteed by Hall Corollary 3.50 and the assumed smoothness of $f$),
the coordinate-safe form of the infinitesimal constraint is

$$
Df_x\!\left(\xi^A_X(x)\right) = \xi^A_Y(f(x)),
\tag{17.2a}
$$

where for each $A \in \mathfrak{g}$ the **fundamental vector field**
induced by $A$ is

$$
\xi^A_X(x) := \tfrac{d}{dt}\big|_{t=0}\, \rho_X(\exp(tA))\, x,
\qquad
\xi^A_Y(y) := \tfrac{d}{dt}\big|_{t=0}\, \rho_Y(\exp(tA))\, y,
\tag{17.2b}
$$

and $Df_x$ denotes the differential of $f$ at $x$, viewed as a linear
map between the relevant tangent spaces. Equation (17.2a) is intrinsic:
it does not require identifying tangent vectors with ambient vectors,
and it extends directly to the case where $X, Y$ are smooth manifolds
on which $G$ acts smoothly (not just linear representation spaces).

**Linear-representation specialization.** When $X, Y$ are
finite-dimensional vector spaces and $\rho_X, \rho_Y$ are linear
representations, the fundamental vector fields take the linear form
$\xi^A_X(x) = d\rho_X(A)\, x$ and $\xi^A_Y(y) = d\rho_Y(A)\, y$, where
$d\rho$ denotes the derived representation of $\mathfrak{g}$. In this
case (17.2a) reads

$$
Df_x\,\bigl(d\rho_X(A)\, x\bigr) = d\rho_Y(A)\, f(x),
\tag{17.2c}
$$

which is the form physics-lint evaluates numerically.

Equivalently, defining the Lie derivative of $f$ along $A$:

$$
\mathcal{L}_A f (x) := \frac{d}{dt}\Big|_{t=0}
\left[ \rho_Y(\exp(-tA))\, f(\rho_X(\exp(tA))\, x) \right] = 0.
\tag{17.3}
$$

Equation (17.3) is the **infinitesimal equivariance constraint**: the
derivative of the "equivariance defect curve" $t \mapsto \rho_Y(\exp
(-tA))\, f(\rho_X(\exp(tA))\, x)$ vanishes at $t=0$ for all $A \in
\mathfrak{g}$.

**Infinitesimal constraint $\Rightarrow$ finite equivariance over $G^0$.** The reverse
direction is more delicate and requires the connectedness / generation
argument. Suppose (17.3) holds for all $A \in \mathfrak{g}$ and $x \in X$
(not just at a finite set of samples). Define
$\Phi(t) := \rho_Y(\exp(-tA))\, f(\rho_X(\exp(tA))\, x)$. Then
$\Phi'(t) = \rho_Y(\exp(-tA))\, \mathcal{L}_A f (\rho_X(\exp(tA))\, x)$;
if $\mathcal{L}_A f = 0$ identically, then $\Phi'(t) \equiv 0$, so
$\Phi(t) = \Phi(0) = f(x)$ for all $t$. This gives

$$
f(\rho_X(\exp(tA))\, x) = \rho_Y(\exp(tA))\, f(x),
\tag{17.4}
$$

i.e. equivariance along the one-parameter subgroup $\exp(t\mathfrak{g})$.
By Kirillov Corollary 2.9, any neighborhood of the identity generates
a connected Lie group. Since the exponential map is a local
diffeomorphism near $0 \in \mathfrak{g}$, sufficiently small elements
of the form $\exp(tA)$ form an identity neighborhood in $G^0$.
Therefore finite products of such exponential elements generate $G^0$,
so equivariance on one-parameter subgroups — obtained by iterating
(17.4) — extends to all of $G^0$.

### 17.4 Direction of implication — summary diagram

$$
\begin{array}{l}
\text{Finite equivariance on } G \text{ (i.e. 15.1 holds for all } g \in G\text{)} \\
\quad\Downarrow \quad [\text{differentiation at } t=0] \\
\text{Infinitesimal constraint (17.3) at every } A \in \mathfrak{g},\ x \in X \\
\quad\Updownarrow \quad [\text{requires } G \text{ connected, } \rho \text{ smooth,}\\
\qquad\qquad \text{constraint holding identically — not just at samples}] \\
\text{Finite equivariance on the identity component } G^0 \\
\quad\not\Downarrow \quad [\text{does NOT extend to } G \setminus G^0 \text{ for disconnected } G]
\end{array}
$$

Two gaps in this diagram are load-bearing for the rule's non-claims:

- The downward equivalence requires the infinitesimal constraint to hold
  **identically**, not just at a finite set of generators on a finite set of
  samples. The rule checks the latter empirically and thus provides a
  diagnostic, not a proof.
- For disconnected $G$ (e.g. $O(n) = SO(n) \cup (\text{reflections})$),
  equivariance over $G^0 = SO(n)$ does not imply equivariance over the
  other connected components. A user who wants full $O(n)$ equivariance
  must either (a) explicitly test the reflection generators in addition to
  the Lie-algebra generators, or (b) use a finite-symmetry check
  (PH-SYM-001).

### 17.5 Assumptions

- $G$ is a connected matrix Lie group with Lie algebra $\mathfrak{g}$.
- $\rho_X, \rho_Y$ are smooth representations. By Hall Corollary 3.50,
  continuous representations of matrix Lie groups are automatically
  smooth, so this reduces to continuity.
- $f$ is differentiable at the test points.
- The user supplies a spanning set of generators $\{A_1, \ldots, A_k\}$
  for $\mathfrak{g}$.

### 17.6 Implementation mapping

The rule computes the empirical infinitesimal equivariance defect at a
finite set of generators $\{A_j\}$ and test points $\{x_i\}$:

$$
\widehat{\mathcal{L}}(f; \{A_j\}, \{x_i\}) := \frac{1}{N k}
\sum_i \sum_j \| \mathcal{L}_{A_j} f(x_i) \|_Y^2,
\tag{17.5}
$$

where $\mathcal{L}_{A_j} f(x_i)$ is estimated either by autodiff through
the one-parameter curve $t \mapsto \rho_Y(\exp(-tA_j))\, f(\rho_X(\exp
(tA_j))\, x_i)$ at $t = 0$, or by a small-$t$ finite difference with a
symmetric stencil. All group-element arguments are written in
$\exp(tA)$ form; the shorthand $\rho_X(tA)$ is avoided because it
conflates the group-level action $\rho_X(\exp(tA))$ with the
Lie-algebra-level action $d\rho_X(tA) = t\, d\rho_X(A)$, which are only
equal after differentiation at $t=0$.

### 17.7 What this validates

- Empirical infinitesimal equivariance at the declared generators and
  test points.
- Detection of gross violations of Lie-algebra equivariance that would
  propagate to finite-equivariance failures over $G^0$.

### 17.8 What this does not validate

The non-claims here are extensive and load-bearing. physics-lint **does
not** claim any of the following from a passing PH-SYM-003 verdict.

- **Global equivariance for disconnected groups from Lie-algebra checks
  alone.** Only $G^0$-equivariance is implied by the implication chain
  of §17.4, and only under the identically-holding-constraint assumption.
- **Exact finite equivariance from finite sampled generators.** The rule
  samples generators and test points; a passing empirical check does not
  prove (17.3) holds identically.
- **Equivalence in the absence of smoothness and representation
  assumptions.** Without Hall Corollary 3.50, the continuous-to-smooth
  bridge is not available, and the derivation of (17.2) from (17.1)
  fails.
- **Reproduction of Gruver et al. 2022/2023 ImageNet results** unless the
  opt-in ImageNet CI job has actually been run successfully; see the
  plan's env-var-gated opt-in policy. The rule's core correctness does
  not depend on that reproduction, but any documentation that cites a
  passing Gruver reproduction must be separately gated on the opt-in
  pipeline completing.

### 17.9 Report-integration note

**Function 2 — primary v1.0 SO(2) LEE fixtures.** The shipped F2
foregrounds the SO(2) Lie-derivative diagnostic via:

- **Radial scalar-invariant positive controls.** Test functions of the
  form $f(x, y) = g(x^2 + y^2)$ for smooth $g : \mathbb{R}_{\geq 0}
  \to \mathbb{R}$ are exactly $SO(2)$-invariant: applying the rotation
  generator $A = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$ to the
  input gives a tangent vector orthogonal to $\nabla f$ at every point,
  so $\mathcal{L}_A f \equiv 0$ analytically. The rule's JVP-based
  Lie-derivative diagnostic should report
  $\widehat{\mathcal{L}}(f) \approx 0$ at floating-point floor.
- **Coordinate / anisotropic negative controls.** Coordinate-dependent
  test functions (e.g. $f(x, y) = x^2$, $f(x, y) = x \cdot y$) and
  anisotropically-weighted operators are non-equivariant under SO(2);
  the rule's diagnostic should report a non-negligible
  $\widehat{\mathcal{L}}$ that scales with the anisotropy parameter.
- **Finite-vs-infinitesimal scaling.** Verifying that the empirical
  finite-equivariance defect on a refining $t$-grid matches
  $t \cdot \widehat{\mathcal{L}} + O(t^2)$, confirming the
  Taylor-expansion direction-of-implication of §17.3.
- **JVP-based Lie-derivative diagnostic.** The rule uses
  `torch.autograd.functional.jvp` to evaluate
  $\mathcal{L}_A f(x) = \frac{d}{dt}\big|_{t=0} \rho_Y(\exp(-tA))
  f(\rho_X(\exp(tA)) x)$ at the test points, gated by six
  preconditions (smooth callable, shape compatibility, declared
  symmetry group, etc.) before emitting a verdict.

**Supplementary controlled-operator context.** The FFT-based Laplace
inverse and the identity operator are exactly $C_4$- and reflection-
equivariant on a 2D periodic square grid. They appear in the harness
as cross-checks on the SO(2) generator's discrete-orbit behavior, but
they are not the primary F2 fixtures for the SO(2) LEE diagnostic
because their continuous SO(2) equivariance is broken by the discrete
grid structure (only the $C_4$ subgroup of SO(2) acts exactly on the
grid). Treat them as auxiliary calibration, not as the primary
demonstration of the SO(2) Lie-derivative property.

**Function 3.** Cite Hall 2015 Theorem 2.14 (one-parameter subgroup
theorem, $A(t) = \exp(tX)$) and Corollary 3.50 (continuous
homomorphisms between matrix Lie groups are smooth) for the Lie-group
bridge — theorem locators verified against a physical copy of Hall
2nd edition; Kirillov 2008 *Introduction to Lie Groups and Lie
Algebras*, Corollary 2.9 (any identity neighborhood generates a
connected Lie group), for the identity-component generation step —
locator confirmed in the publicly accessible course-note copy;
Weiler-Forré-Verlinde-Welling *Equivariant and Coordinate Independent
Convolutional Networks* (World Scientific, Progress in Data Science
vol. 1, DOI 10.1142/14143, ISBN 978-981-98-0662-1; publication year
2026 per DBLP) as a forthcoming monograph anchor (background only,
not theorem-level); and Gerken et al. 2023 AI Review for the peer-
reviewed survey. Cohen-Geiger-Weiler 2019 NeurIPS (arXiv:1811.02017)
and Kondor-Trivedi 2018 ICML (arXiv:1802.03690) remain the theorem-
level anchors for the kernel-characterization side of the equivariance
literature, although their specific results are more directly invoked
in PH-SYM-004.

---

## 18. PH-SYM-004 — Shift Equivariance / Convolution Structure

> **High-risk section.** The subtle claim here is the direction
> convolution $\Leftrightarrow$ translation-equivariance, which is a
> theorem only under specific conditions. In v1.0, this mathematics
> is exercised at the harness level on controlled operators. The
> shipped production rule does not evaluate user models for shift
> commutation; it validates only the declared-symmetry / periodicity
> gate-and-skip behavior. Live callable shift-commutation on user
> models is deferred to v1.2.

### 18.1 Mathematical claim

Convolutional and Fourier-operator structures are naturally tied to
translation equivariance under specified domain and regularity
assumptions. For a periodic or infinite-domain setting, convolution with
a fixed kernel is exactly translation-equivariant; conversely, under
natural constraints, any translation-equivariant linear map is of
convolution form.

### 18.2 Theoretical anchor

Two directions anchor the identity:

**Forward direction (convolution $\Rightarrow$ translation equivariance).** Classical
and elementary: direct computation from the definition of convolution.

**Reverse direction (translation equivariance $\Rightarrow$ convolutional structure).**
Under natural constraints and for the action of a compact group,
convolutional structure is not just sufficient but also necessary for
equivariance [Kondor-Trivedi 2018, *On the Generalization of Equivariance
and Convolution in Neural Networks to the Action of Compact Groups*,
ICML Proceedings Vol. 80, pp. 2747–2755; arXiv:1802.03690]. The
general-homogeneous-space version is in [Cohen-Geiger-Weiler 2019,
*A General Theory of Equivariant CNNs on Homogeneous Spaces*, NeurIPS
2019; arXiv:1811.02017].

### 18.3 Derivation

**Forward direction.** Let $T_a$ denote the translation operator
$(T_a u)(x) := u(x - a)$. For a kernel $k$ and convolution $(K u)(x) :=
\int k(x - y) u(y)\, dy$ (assuming the integral converges, e.g. $u, k \in
L^1(\mathbb{R}^d)$ or $u, k$ periodic on a torus):

$$
(K (T_a u))(x) = \int k(x - y) u(y - a)\, dy
= \int k(x - (y' + a)) u(y')\, dy'
= (T_a (K u))(x),
\tag{18.1}
$$

via the substitution $y' = y - a$. So $K T_a = T_a K$, i.e. $K$ commutes
with all translations.

**Reverse direction.** The precise statement depends on the domain:

- On $\mathbb{Z}^d$ (or a finite periodic grid), any linear map $K :
  \ell^2(\mathbb{Z}^d) \to \ell^2(\mathbb{Z}^d)$ that commutes with all
  translations is given by convolution with a kernel $k$ (the
  shift-invariant system, diagonalized by the Fourier transform).
- On a compact group $G$, [Kondor-Trivedi 2018] states:
  given natural constraints (bounded linear, translation-equivariant),
  a feed-forward network is equivariant to the action of a compact group
  $G$ if and only if it is a generalized convolution with an equivariant
  kernel.
- On homogeneous spaces $G/H$, the characterization is via induced
  representations and equivariant kernels [Cohen-Geiger-Weiler 2019].

### 18.4 Assumptions

- Periodic (torus) or infinite-domain setting for the elementary
  forward-direction derivation.
- Compact group action for the Kondor-Trivedi reverse-direction result.
- Linearity and a boundedness / regularity hypothesis for the reverse
  direction (the "natural constraints" in the theorem statement).
- The user declares which translations to test (e.g. a subgroup of a
  uniform grid's translation group).

### 18.5 Implementation mapping

The intended full rule family checks the forward direction empirically
on a user-declared grid: given a model $f$, test points $\{x_i\}$, and
a set of shift vectors $\{a_j\}$ (usually integer grid shifts on a
periodic domain), compute

$$
\mathcal{E}_{\text{shift}}(f) := \frac{1}{N M}
\sum_i \sum_j \| f(T_{a_j} u_i) - T_{a_j} f(u_i) \|^2.
\tag{18.2}
$$

**Harness-level F2.** The harness validates four controlled
translation-equivariant operators (identity, circular convolution
1D / 2D, Fourier multiplier 1D / 2D) and one coordinate-dependent
non-equivariant negative control via `shift_commutation_error`.
Harness-level F2 confirms that the mathematical statement of (18.2)
is sound on the controlled operators.

**Shipped v1.0 production behavior.** The v1.0 production rule is a
**stub**: it `SKIPPED`-always past the declared-symmetry and
periodicity preflight gates, after confirming that the gating logic
correctly recognizes when a shift-commutation check is or is not
applicable. The production rule does not invoke the user's model
on shifted inputs in v1.0; the live-callable adapter mode that
performs `f(roll(x)) == roll(f(x))` on user models is deferred to
v1.2. This means the v1.0 verdict on user-supplied models is always
SKIP, with a reason string indicating the gate that triggered the
skip.

**Scope and broader-family context.** The full Kondor-Trivedi
characterization (compactness of the group action, kernel-form
necessity, etc.) and the broader convolution ⇔ translation-
equivariance theorem are documented in §18 for v1.2 extension; the
v1.0 production verdict is **not** a substantive claim about user
models, only about the gate-and-skip behavior.

### 18.6 What this validates

- **Harness layer:** that the controlled translation-equivariant
  operators (identity, periodic convolution, Fourier multipliers)
  satisfy (18.2) at machine precision on integer-pixel shifts of 2D
  periodic grids, and that the coordinate-dependent negative control
  produces a non-negligible `shift_commutation_error`.
- **Production layer (v1.0 stub):** that the gating logic (declared-
  symmetry and periodicity preflight gates) correctly recognizes when
  a shift-commutation check is or is not applicable, and emits a SKIP
  verdict with a reason string in every applicable case.
- The v1.0 production layer does not assert (18.2) on user-supplied
  models. Live-callable empirical shift-commutation checking is a v1.2
  feature.

### 18.7 What this does not validate

- **The reverse direction (equivariance $\Rightarrow$ convolution) is not checked by
  the rule.** The rule reports a shift-commutation defect; it does not
  decompose the model into convolutional structure.
- For non-periodic, non-infinite-domain settings (bounded domains with
  Dirichlet boundaries), translation equivariance is not exact even for a
  convolution; the rule should not be applied in those regimes as a
  strict equivariance check.
- The v1.0 rule does not test the compact-group generalization beyond
  translations; other compact groups are in scope for PH-SYM-001.
- Empirical shift-equivariance on a finite test set does not prove exact
  equivariance for all inputs.

### 18.8 Report-integration note

**Function 2 — harness layer.** Validates four controlled translation-
equivariant operators (identity, circular convolution 1D / 2D, Fourier
multiplier 1D / 2D) plus a coordinate-dependent non-equivariant
negative control via `shift_commutation_error`. Verifies that (18.2)
is satisfied at machine precision for axis-aligned integer-pixel
shifts on a 2D periodic rectangular grid for the equivariant
operators, and is non-negligible for the negative control.

**Function 2 — production layer.** Verifies that the v1.0 stub's
production-rule behavior is `SKIPPED`-always past the declared-
symmetry and periodicity preflight gates, with the appropriate reason
string. The production rule does not invoke the user's model on
shifted inputs in v1.0; live-callable shift-commutation
(`f(roll(x)) == roll(f(x))` on user models) is deferred to v1.2.

**Function 3.** Cite Kondor-Trivedi 2018 ICML (arXiv:1802.03690) for
the compact-group necessity result and Cohen-Geiger-Weiler 2019
NeurIPS (arXiv:1811.02017) for the homogeneous-space characterization.

---

# Part VI — Numerical Validation Rules

## 19. PH-NUM-001 — Quadrature / Variational-Crime Check

> **High-risk section.** Variational-crime theory is precise and has
> non-trivial prerequisites; the rule's claim must match what the theory
> actually establishes and must not over-claim FEM-solver correctness.

### 19.1 Mathematical claim

Insufficient numerical quadrature can corrupt weak-form residuals and
variational estimators. Under the standard variational-crime framework,
the error between the exact Galerkin solution and the quadrature-perturbed
Galerkin solution is bounded by the consistency error of the quadrature
rule, which is quantifiable per element.

### 19.2 Theoretical anchor

Quadrature consistency / variational-crime theory [Strang & Fix 1973,
*An Analysis of the Finite Element Method*, Prentice-Hall, Ch. 4; Ciarlet
1978, *The Finite Element Method for Elliptic Problems*, North-Holland,
§4.1; Brenner-Scott 2008, *The Mathematical Theory of Finite Element
Methods*, 3rd ed., Springer, §10.3; Ern-Guermond 2021, *Finite Elements
II*, Springer TAM 73, Ch. 27 "Error analysis with variational crimes"
and Ch. 30 "Quadratures"]. The term "variational crime" is due to Strang
and denotes any of: (i) non-conforming elements, (ii) numerical
quadrature replacing exact integration, (iii) boundary / domain
approximation. This rule concerns the quadrature case.

### 19.3 Derivation

Consider the variational problem

$$
\text{Find } u \in V : \quad a(u, v) = \ell(v), \quad \forall v \in V,
\tag{19.1}
$$

with $a(\cdot, \cdot)$ a continuous, coercive bilinear form on $V$ and
$\ell \in V'$. A conforming Galerkin discretization with space $V_h
\subset V$ gives

$$
\text{Find } u_h \in V_h : \quad a(u_h, v_h) = \ell(v_h), \quad \forall
v_h \in V_h.
\tag{19.2}
$$

In practice, $a$ and $\ell$ are evaluated by quadrature, yielding the
perturbed problem

$$
\text{Find } \tilde u_h \in V_h : \quad a_h(\tilde u_h, v_h) = \ell_h(v_h),
\quad \forall v_h \in V_h,
\tag{19.3}
$$

where $a_h, \ell_h$ use the quadrature rule. The **first Strang lemma**
[Strang-Fix 1973 §4.1; Ciarlet 1978 §4.1] gives, under the
standing assumption that $a_h$ is uniformly $V_h$-coercive,

$$
\| u - \tilde u_h \|_V \leq C \inf_{v_h \in V_h} \left[ \|u - v_h\|_V
+ \sup_{0 \ne w_h \in V_h}
\frac{|a(v_h, w_h) - a_h(v_h, w_h)|}{\|w_h\|_V} \right]
+ C \sup_{0 \ne w_h \in V_h}
\frac{|\ell(w_h) - \ell_h(w_h)|}{\|w_h\|_V},
\tag{19.4}
$$

where the bilinear-form consistency error is minimized together with the
approximation error over $v_h \in V_h$, and the linear-form consistency
error is independent of $v_h$. The last two norm-of-difference quantities
are the **consistency errors** of the quadrature rule. If the quadrature
is exact on the integrands arising in $a, \ell$, the consistency errors
vanish and (19.4) reduces to the pure interpolation-error term
$C \inf_{v_h} \|u - v_h\|_V$ (Céa-lemma form). Exact constants and the
precise coercivity constant depend on the discrete space and the chosen
$V$-norm.

**Degree-of-precision requirements are integrand-specific.** For
continuous piecewise-polynomial Lagrange elements $V_h$ of degree $k$ on
affine triangulations, the standard requirements are:

- **Mass-like forms** $\int_K w v\, dx$ with $w, v \in P_k$ produce a
  degree-$2k$ polynomial integrand per element; a quadrature rule exact
  for polynomials of degree $2k$ integrates these exactly.
- **Stiffness-like forms** $\int_K \nabla w \cdot \nabla v\, dx$ with
  $w, v \in P_k$ produce a degree-$2(k-1) = 2k-2$ polynomial integrand
  per element (since each gradient drops one polynomial degree); a
  quadrature rule exact for polynomials of degree $2k-2$ suffices on
  affine elements.
- **Linear forms** $\int_K f\, v\, dx$ with $v \in P_k$ and $f$
  approximated by $f_h \in P_s$ produce a degree-$(k+s)$ integrand.
- **Non-affine (mapped, isoparametric) elements** introduce Jacobian
  factors that raise the effective degree; see Ciarlet 1978 §4.1 for
  the full bookkeeping. Nonlinear coefficients likewise change the
  required degree.

**Concretely** for piecewise-linear FEM ($k=1$) on a triangulation: the
stiffness integrand $\nabla w \cdot \nabla v$ is of degree $2k-2 = 0$
(piecewise constant), and any quadrature rule — including the 1-point
centroid rule — is exact. Mass integrands $w v$ are of degree $2k = 2$;
the 1-point rule is *inconsistent* for these and constitutes a
variational crime. The rule must enforce per-form conditions, not a
single uniform "degree $2k$" threshold.

### 19.4 Assumptions

- $V$ is a Hilbert space and $a$ is continuous and coercive on $V$.
- $V_h$ is a conforming subspace (non-conforming elements are a different
  strand of variational-crime theory, outside v1.0 scope).
- The quadrature rule's consistency error can be bounded per element; in
  practice this requires the integrands to be in a Sobolev space the
  quadrature can resolve.

### 19.5 Implementation mapping

**Intended rule family.** Given a declared quadrature rule with order
`intorder` and an FEM space of degree $k$, the rule verifies that
`intorder` is sufficient for the integrands that arise in the user's
bilinear form. The rule reports violations in terms of consistency
error magnitude, as a per-class lookup-table check against the
degree-of-precision thresholds derived in §19.3.

**Harness-level F2.** The harness validates the Strang-lemma /
quadrature-exactness mathematics directly via three controlled cases:
(A) `degree ≤ intorder`, where the quadrature is exact to roundoff;
(B) deliberately under-integrated cases, where the consistency error
is bounded away from zero; and (C) increasing `intorder` across
$\{2, 4, 6, 8, 10\}$, where the error drops by a factor of $\sim 7.8
\times 10^{12}$, demonstrating the consistency-error decay of (19.4).
Harness-level F2 confirms that the mathematical thresholds derived in
§19.3 hold operationally on the supported integrand classes.

**Shipped v1.0 production behavior.** The v1.0 production rule is a
**stub**: on a `MeshField` input, it emits `PASS`-with-stub-reason
using a pass-through `field.integrate()` baseline and the reason
string `"qorder convergence check is a stub until V1.1"`; on
unsupported field types it emits `SKIP`. The production rule does
**not** in v1.0 inspect arbitrary user-defined bilinear forms
symbolically and does not enforce per-class degree-of-precision
thresholds against user input. Full MMS h-refinement and the
production lookup-table check (Poisson, linear-elasticity, mass-
projection on scikit-fem conforming Lagrange meshes) are deferred
to v1.2.

**Reason-string note.** The production reason string still mentions
`V1.1`. The active public backlog has been consolidated under v1.2;
the literal `V1.1` mention is a string-literal artifact that does
not affect rule correctness and is tracked on the v1.2 cleanup list.

### 19.6 What this validates

- **Harness layer:** that the quadrature-exactness mathematics of
  §19.3 holds operationally — exact integration when `degree ≤
  intorder`, bounded-away-from-zero consistency error under
  deliberate under-integration, and the predicted error decay as
  `intorder` increases.
- **Production layer (v1.0 stub):** that the rule emits
  `PASS`-with-stub-reason on `MeshField` input via the pass-through
  `field.integrate()` baseline, and `SKIP` on unsupported field
  types. The production verdict does not enforce a degree-of-
  precision threshold on user input.
- For the intended full rule family (v1.2): bounded consistency
  error of the quadrature rule against user-declared integrand
  classes via the §19.3 lookup-table thresholds. Out of v1.0
  production scope.

### 19.7 What this does not validate

- **Not a certificate of FEM-solver correctness.** A sufficient
  `intorder` does not imply the user's solver is bug-free; it bounds
  only the quadrature consistency term in (19.4), not the interpolation
  error or the solver's linear-algebra correctness.
- Non-conforming elements (e.g. Crouzeix-Raviart, hybridizable DG) are
  not in v1.0 scope and produce a different consistency analysis.
- The rule checks consistency of the quadrature rule against *declared*
  integrands; if the user mis-declares the integrand, the rule cannot
  detect this.
- For nonlinear problems, the consistency error depends on the current
  iterate; the linear rule in v1.0 is a necessary but not sufficient
  check for nonlinear convergence.

### 19.8 Report-integration note

Function 2 — harness layer. A fixture on a supported scikit-fem
conforming Lagrange mesh that deliberately under-integrates an
integrand class for which the lookup threshold is nontrivial. A plain
$P_1$ Poisson stiffness form $\int_K \nabla w \cdot \nabla v\, dx$
does **not** expose under-order quadrature error on affine
triangulations, because its integrand is degree $2k-2 = 0$ (constant)
and any quadrature rule is exact for it (see §19.3). Two fixture
choices that do expose the error at the harness layer:

- **$P_1$ mass-projection form** $\int_K w v\, dx$, whose integrand
  has degree $2k = 2$ on affine elements; a one-point rule (degree
  $1$) is inconsistent.
- **Poisson linear form** $\int_K f v\, dx$ with polynomial source
  data of declared degree $s > 0$, whose integrand has degree
  $k + s \geq 2$; an `intorder` below the threshold produces a
  detectable consistency error.

Verify at the harness level that reducing `intorder` below the
required threshold produces a detectable consistency error
(quantitatively, the harness measures error drops by $\sim 7.8 \times
10^{12}$ across `intorder` $\in \{2, 4, 6, 8, 10\}$ on the validated
fixtures). The v1.0 production rule **does not catch this on user
input**; production behavior is `PASS`-with-stub-reason on `MeshField`
and `SKIP` on unsupported field types. Production enforcement against
user-declared bilinear forms is deferred to v1.2.

Function 3: cite Strang-Fix 1973, Ciarlet 1978, Brenner-Scott 2008,
and Ern-Guermond 2021 Vol. II for the first Strang lemma and
quadrature / variational-crime framework.

---

## 20. PH-NUM-002 — Manufactured-Solution / Observed-Order Check

### 20.1 Mathematical claim

For a numerical method with theoretical convergence order $p$, the error
in a suitable norm should decay like $O(h^p)$ in the asymptotic regime
(sufficiently small $h$ that higher-order terms are negligible, but not
so small that floating-point rounding dominates). The Method of
Manufactured Solutions (MMS) provides a controlled framework for
computing the observed order $p_{\text{obs}}$ and comparing it to $p$.

### 20.2 Theoretical anchor

Method of Manufactured Solutions [Roache 2002 *Code Verification by the
Method of Manufactured Solutions*, J. Fluids Eng. 124(1); Oberkampf & Roy
2010 *Verification and Validation in Scientific Computing*, Cambridge UP,
ISBN 978-0-521-11360-1, Ch. 6 (Exact and manufactured solutions) and
Ch. 5 (Code order-of-accuracy verification)]. The method chooses a
target analytical $u^*$, computes the source term $f^* := \mathcal{L}
u^*$, and solves the PDE with source $f^*$; the known $u^*$ is the exact
solution, so discretization error is directly computable.

### 20.3 Derivation

Let $u^*$ be a user-chosen manufactured solution, $f^* := \mathcal{L} u^*$
the induced source term, and $u_h$ the numerical solution of the PDE
$\mathcal{L} u = f^*$ on a grid with spacing $h$. The discretization
error is $e_h := u_h - u^*$ measured in a norm $\|\cdot\|$. For a method
with theoretical order $p$:

$$
\| e_h \| = C h^p + O(h^{p+1}) + O(\epsilon_{\text{mach}}),
\tag{20.1}
$$

with $C$ a constant independent of $h$. In the **asymptotic regime** —
$h$ small enough that the first term dominates, but large enough that
floating-point rounding does not dominate — we have

$$
\| e_h \| \approx C h^p.
\tag{20.2}
$$

The **observed order** from a pair of grids $(h, h/r)$ for a refinement
ratio $r$ (typically $r = 2$) is

$$
p_{\text{obs}} := \frac{\log(\|e_h\| / \|e_{h/r}\|)}{\log r}.
\tag{20.3}
$$

In the asymptotic regime, $p_{\text{obs}} \to p$. Deviations of
$p_{\text{obs}}$ from $p$ indicate either (i) not yet in the asymptotic
regime, (ii) saturated by rounding, (iii) a bug in the discretization,
or (iv) an unjustified claim of order $p$.

### 20.4 Per-case rate handling

Rates must be interpreted **per PDE case and per norm**. A second-order
scheme for the Poisson equation may be second-order accurate in $L^2$
but only first-order accurate in $L^\infty$ near a singularity. The
rule's verdict must be parameterized by the declared PDE and norm; a
uniform "order = 2" check across all problems is ill-posed.

### 20.5 Assumptions

- The manufactured solution $u^*$ is sufficiently smooth for the
  theoretical order to apply (e.g. $C^{p+1}$ on $\overline{\Omega}$).
- The boundary conditions are consistent with $u^*$ (the user supplies
  $u^*$ on $\partial\Omega$ as Dirichlet data, or the appropriate
  Neumann/Robin derivative).
- The grid sequence covers the asymptotic regime before saturation.
- The norm is the one for which $p$ is claimed.

### 20.6 Implementation mapping

The rule accepts $u^*$, a grid sequence $\{h_i\}$, a target order $p$,
and a norm. It computes $\|e_{h_i}\|$ for each grid and fits the
observed order $p_{\text{obs}}$ from (20.3). Verdicts:

- *Asymptotic agreement*: $p_{\text{obs}} \approx p$ within tolerance on
  consecutive pairs in the asymptotic range.
- *Not in asymptotic regime*: decay is not yet at the target rate on the
  supplied grids; user should refine further.
- *Saturation*: $\|e_h\|$ has flattened near floating-point floor;
  further refinement will not help.
- *Order mismatch*: asymptotic regime is reached, but $p_{\text{obs}}$
  is consistently below $p$ — the scheme does not achieve its claimed
  order.

### 20.7 What this validates

- For a specific manufactured solution, a specific norm, and a grid
  sequence in the asymptotic regime: the numerical method achieves its
  claimed order of accuracy.

### 20.8 What this does not validate

- **One manufactured solution is not all PDEs.** Passing MMS on a smooth
  $u^* = \sin(x)\sin(y)$ does not validate the method on shocks or
  boundary layers.
- Observed convergence in one norm does not imply convergence in others.
- A non-asymptotic or saturated sequence gives no information about the
  true order; the rule must detect these states and not report a rate.
- The rule does not validate physical correctness of the PDE the user
  is solving; it validates implementation correctness of the solver.

### 20.9 Report-integration note

Function 2 and Function 3 evidence for the shipped v1.0 rule is
recorded in the PH-NUM-002 row of §24. The shipped F2 uses three
scoped cases:

- **Case A — harness-authoritative `mms_observed_order_fd2`** with
  $p_{\mathrm{obs}} \to 2$ on the smooth-data MMS path.
- **Case B — FD non-periodic** with $p_{\mathrm{obs}} \approx 2.50$
  arising from boundary-band scaling on the non-periodic FD path.
- **Case C — saturation floor.** Errors below $10^{-11}$ return
  $\text{rate} = \infty$ to flag floating-point floor reach rather
  than report a spurious finite rate.

The per-norm rate discussion below is the mathematical background
that justifies why §24's case-specific rates are declared per-norm
and per-error-quantity rather than as a single uniform template:

- **Configuration A — FD grid-gradient norm.** Measure the solution
  error in discrete $\ell^2_h$ (gives $p_{\mathrm{target}} = 2$) and
  the gradient error as the discrete $\ell^2_h$ norm of a centered-
  difference approximation to $\nabla u$ applied to the grid values
  of $u_h - u^*$. On smooth grid data with a consistent centered
  stencil, this gradient error is also expected to behave as
  $p_{\mathrm{target}} = 2$.
- **Configuration B — $P_1$ interpolation / FEM-like $H^1$ seminorm.**
  Interpret $u_h$ as a piecewise-linear $P_1$ interpolant of the grid
  values on a triangulation of $[0,1]^2$ and measure
  $\|u - u_h\|_V$ in the continuous $H^1$-seminorm
  $\|\nabla(u - u_h)\|_{L^2(\Omega)}$. Under this interpretation,
  $p_{\mathrm{target}} = 2$ for $L^2$ and $p_{\mathrm{target}} = 1$
  for the $H^1$-seminorm, consistent with the Céa / Aubin-Nitsche
  analysis for $P_1$ Lagrange elements.

The rule emits per-norm target rates that the user declares
explicitly; it does **not** infer a uniform rate across different
error norms or derive the gradient-norm rate from the solution-norm
rate. Function 3: cite Oberkampf-Roy 2010 Ch. 5–6 and Roache 2002.

---

# Part VII — Meta / Diagnostic Rules

## 21. PH-VAR-001 — Deferred v1.2 (Pointer)

PH-VAR-001 (L² Strong-Form Residual / Norm-Selection Warning) is a
Type-C meta-rule deferred from v1.0 production scope. The v1.0
codebase contains only a SARIF severity-mapping fixture referencing
this rule-id (used to verify that the SARIF emitter routes
`info → note` correctly); no `src/physics_lint/rules/ph_var_001.py`
and no per-rule unit test exists at the v1.0 freeze. The full
mathematical derivation, intended primary instance, and v1.2 shipped
contract are documented in **Appendix D** (Deferred v1.2 Mathematical
Notes). This section heading is preserved so that downstream
documents and the rule-to-derivation index in Appendix A retain a
stable §21 anchor.

---

## 22. PH-VAR-002 — Hyperbolic Residual-Norm Caveat

> **High-risk section.** This is explicitly an INFO-level diagnostic, not
> a correctness check. The derivation establishes that elliptic-style
> residual-error equivalence does not transfer unchanged to hyperbolic
> settings.

### 22.1 Mathematical claim

For hyperbolic problems, residual-norm interpretation is more delicate
than for elliptic settings. The elliptic a posteriori framework (§4)
rests on $V$-ellipticity (4.2); for hyperbolic systems, this ellipticity
is absent and the residual-error equivalence (4.1) fails in general.
physics-lint flags residual computations for hyperbolic PDEs at INFO
level rather than over-certifying.

### 22.2 Theoretical anchor

Hyperbolic residual analysis requires tools beyond the Riesz-based
elliptic framework: discontinuous Petrov-Galerkin (DPG) methods
[Demkowicz-Gopalakrishnan 2011 *A Class of Discontinuous Petrov-Galerkin
Methods, Part II: Optimal Test Functions*, Numer. Meth. PDEs 27:70–105;
Demkowicz-Gopalakrishnan 2025 *The Discontinuous Petrov-Galerkin
Method*, Acta Numerica vol. 34, pp. 293–384, DOI
10.1017/S0962492924000102 (Open Access)] reconstruct residual-error
equivalence in a dual norm via optimal test functions. The Acta
Numerica 2025 review explicitly states that the DPG method is
equivalent to a minimum-residual method with residual measured in a
dual norm. For classical hyperbolic systems without DPG machinery, the
elliptic equivalence does not hold and residual norms are better
treated as information flags.

### 22.3 Derivation

The elliptic residual-error equivalence (4.1) fails for hyperbolic
problems because the governing operator is not $V$-elliptic in a suitable
Hilbert norm. For the linear advection equation $\partial_t u + c
\partial_x u = f$, the natural "energy" is $\int u^2 dx$, which is
preserved rather than controlled by the operator; there is no coercivity
estimate analogous to (4.2).

The DPG methodology restores equivalence by constructing test functions
that realize the inf-sup supremum, so that the DPG residual in the
optimal-test-function dual norm is equivalent to the error in the trial
norm [Demkowicz-Gopalakrishnan 2011; Demkowicz-Gopalakrishnan 2025 Acta
Numerica]. However, this requires the DPG framework to be set up; a
naive least-squares residual on an arbitrary hyperbolic scheme does not
automatically inherit this equivalence.

For v1.0, physics-lint does not assume the user has set up a DPG
framework; it treats residual norms on hyperbolic problems as
diagnostics and emits INFO-level verdicts.

### 22.4 Assumptions

- The user has declared the PDE type (hyperbolic vs. parabolic vs.
  elliptic) as part of the rule's metadata.
- No DPG framework is assumed; the rule is conservative.

### 22.5 Implementation mapping

For PDE types declared as hyperbolic, residual-norm outputs are tagged
INFO rather than used as a correctness indicator. The rule does not
refuse to report the number; it reports it with an information flag
explaining the interpretation caveat.

### 22.6 What this validates

- The user is alerted that a small residual in a hyperbolic setting
  does not, without further framework, certify a small error.

### 22.7 What this does not validate

- The rule does not prove or disprove accuracy.
- The rule does not construct a DPG framework for the user; if they
  want residual-error equivalence for a hyperbolic problem, they must
  set up DPG separately.
- Conditions under which a hyperbolic residual norm *does* control
  error (DPG-equipped methods, special norms for specific equations)
  are not automatically detected.

### 22.8 Report-integration note

Function 2 and Function 3 evidence for the shipped v1.0 rule is
recorded in the PH-VAR-002 row of §24. PH-VAR-002 is an info-severity
diagnostic rule with **no numerical fixture**: the v1.0 contract is
that the rule emits info-severity `PASS` with a literature-pointer
reason on a wave-equation `DomainSpec` and `SKIPPED` on every other
PDE kind. The "small-residual-but-large-error" hyperbolic fixture
description in earlier drafts is documented here as the broader
mathematical motivation but is not part of the shipped v1.0
validation contract. Credibility-anchor framing:
Demkowicz-Gopalakrishnan 2025 *The discontinuous Petrov-Galerkin
method* (Acta Numerica 34:293–384, DOI 10.1017/S0962492924000102) as
theoretical framing only; the literature-maturity-gated promotion of
PH-VAR-002 to a numerical-comparison rule is deferred to v1.2.

---

# Part VIII — Cross-Rule Limitations

## 23. Common Mathematical Limitations

The following limitations apply across the full rule set and are stated
once here rather than repeating in each rule section. Each rule's
non-claims list defers to this section where appropriate.

### 23.1 Discretization dependence

Every rule verdict depends on the user's grid, stencil, mesh, quadrature
rule, and derivative backend. A passing verdict on one discretization
does not imply a pass on another. Changes to any of these without
re-running the rule invalidate the verdict.

### 23.2 Metadata dependence

Boundary conditions, PDE type, and physical variable meaning must be
correctly specified by the user. physics-lint verifies internal
consistency between the user's declared PDE and the computed
quantities; it does not verify that the user identified the physics
correctly. A rule that passes on a Poisson formulation may be
meaningless if the user's problem is actually Helmholtz.

### 23.3 Floating-point floors

All norm-based rules are subject to the double-precision IEEE 754 floor
at approximately $10^{-14}$ in relative terms. Convergence claims that
rely on values below this floor — exponential decay rates fitted from
floor-saturated points, or observed-order rates from error sequences
that have flattened — are unreliable. Rules that depend on asymptotic
behavior must detect and exclude saturated points (PH-RES-003 §6.4;
PH-NUM-002 §20.6).

### 23.4 Local versus global checks

Every empirical check is on a finite test set, a finite set of
transformations, or a specific grid sequence. Passing such a check
does not prove universal correctness; it provides evidence at the
sampled configurations. The symmetry rules (§§15–18) and the residual
rules (§§4–6) all carry this limitation.

### 23.5 Residual versus error

A residual is the left-over after substituting a candidate solution
into the PDE; an error is the difference between the candidate and the
true solution. These are equivalent under the $V$-ellipticity
assumption (§4) but not in general. A small residual is strong
evidence for a small error in the elliptic setting only when the
emitted residual norm is tied to the natural dual norm by a validated
residual-error framework; it is only indirect evidence in the
hyperbolic setting (§22).

### 23.6 Declaration honesty

All rules assume the user honestly declares the mathematical object
they are working with (the PDE class, the boundary conditions, the
expected order of a scheme). A rule cannot detect a mis-declared
expectation; it can only detect inconsistency between the declared
expectation and the computed behavior. If the user declares a
first-order scheme but expects second-order convergence, PH-NUM-002
will report a failure that is rooted in the declaration, not the
scheme.

---

# Part IX — Integrated Rule-by-Rule Validation Evidence

## 24. Integrated Rule-by-Rule Validation Matrix

The table below records the integrated mathematical and engineering
validation evidence for the 18 v1.0-shipped rules at the freeze. Each
row gives the correctness fixture (Function 2), the borrowed-credibility
status (Function 3, with absent-with-justification markers where no
directly-comparable published target exists), the CI status, the commit
SHA at which the rule's anchor landed, and the limitation that the rule
explicitly does not validate. CI status is uniform across rows because
PR #3's 28 checks ran as a single matrix at HEAD `6907699`; the column
is preserved per rule per the integration contract. PH-VAR-001 is
deferred to v1.2 and documented in Appendix D; its absence here is
intentional, not a coverage gap.

| Rule | Correctness fixture (F2) | External anchor (F3) | CI status | Commit / provenance | Limitation checked |
|---|---|---|---|---|---|
| PH-RES-001 | Two-path: log-log slope of interior residual on `sin($\pi$x)sin($\pi$y)` MMS at `N $\in$ {16, 32, 64, 128}` (Fornberg O(h⁴) reproduction) plus BDO `C_max / c_min < 10` norm-equivalence two-layer on the `k $\in$ {1, 2, 3}` perturbation family | F3 PRESENT — Fornberg 1988 interior O(h⁴) live reproduction (measured slope 3.993, range `[3.8, 4.2]`, `R² $\geq$ 0.99`) plus Bachmayr-Dahmen-Oster (BDO) periodic-spectral H⁻¹ norm-equivalence two-layer (measured `C_max / c_min` 4.829) | 28-of-28 SUCCESS at PR #3 (HEAD 6907699) | `c2dba1e` — three-function CITATION.md retrofit (F2 measured values pinned in `c07ba33` Layer 1 and `e28c493` Layer 2) | Elliptic-only applicability; emitted norm must approximate $\|\cdot\|_{V'}$. Norm-equivalence holds only on periodic+spectral; non-periodic+FD path falls back to L² (characterized via Layer 1b/2b, not fixed) |
| PH-RES-002 | AD-vs-FD agreement on smooth MMS callable; max interior relative discrepancy ratio shrinks at O(h⁴) on log-log refinement | F3 absent with justification — CAN-PINN family (Chiu, Fuks, Oommen, Karniadakis 2022, CMAME arXiv:2110.14432) reports absolute-error reductions and training-time improvements, not a directly-comparable AD-vs-FD discrepancy at successive grid refinements; moved to Supplementary calibration context | 28-of-28 SUCCESS at PR #3 (HEAD 6907699) | `30baf3e` — Task 2 PH-RES-002 AD-vs-FD residual cross-check | Smoothness assumption; AD differentiates the computational graph. Compares two numerical primitives on a controlled MMS, not against a closed-form truth |
| PH-RES-003 | Closed-form `exp(sin x)` periodic fixture at `N $\in$ {16, 32, 64}` matching exponential-decay fit `R² > 0.99`; Trefethen 2000 Program-5-style spectral-vs-FD residual on periodic grids | F3 absent with justification — Trefethen 2000 Program 5 `exp(sin x)` is a log-error-vs-N plot, not a tabulated reproduction target; Boyd 2001 *Chebyshev and Fourier Spectral Methods* unattempted in the F3-hunt budget; both moved to Supplementary calibration context | 28-of-28 SUCCESS at PR #3 (HEAD 6907699) | `0cedc7b` — Task 3 PH-RES-003 spectral-vs-FD residual | Floor saturation. Trefethen and Boyd carry curve-shape framing only; no published-row reproduction target |
| PH-BC-001 | Three analytic Dirichlet fixtures: zero-on-exact-trace, perturbation-scaling, absolute-vs-relative-mode branch (discrete-L² Dirichlet boundary trace `$\|$$\gamma$(u) − boundary_target$\|$`) | F3 absent with justification (F3-INFRA-GAP) — three semantically-equivalent PDEBench bRMSE rows pinned (Diffusion-sorption Table 5, 2D diffusion-reaction Table 5, 1D Advection Table 6); PDEBench HDF5 dataset loader + adapter not shipped in v1.0; rows preserved in Supplementary calibration; loader path on v1.2 backlog | 28-of-28 SUCCESS at PR #3 (HEAD 6907699) | `6800d6f` — Task 4 PH-BC-001 Dirichlet-trace scoped | v1.0 production validation: Dirichlet-trace only on Lipschitz boundaries. Neumann and flux semantics remain mathematical context (§7.3 four-regime classification) but are out of v1.0 production scope; separate normal-derivative path required for v1.2 |
| PH-BC-002 | CRITICAL three-layer: harness-level Gauss-Green on `F = (x, y)` to roundoff on triangulation + quadrilateralization; rule-verdict on harmonic Laplace fixtures (e.g. `u = x² − y²`, `u = x⁵ − 10x³y² + 5xy⁴`, satisfying $\Delta u = 0$ so the boundary-flux integral vanishes); Poisson arm `NotImplementedError` SKIP path | F3 absent by structure — Gauss-Green reproduction on MMS fixtures is tautological under the theorem's stated preconditions (theorem holds exactly; "reproducing" a published numerical value would be repetition of the same closed-form derivation); LeVeque 2002 FVM §2.1 retained as Supplementary pedagogical framing only | 28-of-28 SUCCESS at PR #3 (HEAD 6907699) | `18312b9` — Task 5 PH-BC-002 Gauss-Green scope-separated | Production scope = Laplace only (harmonic fixtures with $\Delta u = 0$). V1-stub CRITICAL three-layer; production rule (Laplace-imbalance only) is narrower than F1 theorem (full Gauss-Green); Poisson arm with non-zero $\Delta u$ deferred to v1.2 |
| PH-CON-001 | Analytical-snapshot fixture `u = cos(2$\pi$x) cos(2$\pi$y) · exp(−8$\pi$²$\kappa$t)` (zero-mean periodic eigenmode); drift floor `~1e-18`, tolerance `1e-15` with `~1000$\times$` safety factor | F3 absent with justification (F3-INFRA-GAP) — Hansen ProbConserv ANP row `CE = 4.68 $\times$ 10⁻³ ± 0.10` pinned at `docs/audits/2026-04-22-pdebench-hansen-pins.md:166`; `amazon-science/probconserv` checkpoint loader + ANP inference path not shipped in v1.0; row preserved in Supplementary calibration; loader on v1.2 backlog | 28-of-28 SUCCESS at PR #3 (HEAD 6907699) | `1112da3` — Task 8 PH-CON-001 analytical-snapshot | Integrated, not pointwise. Analytical-snapshot only; numerically-evolved time-stepper validation is out of v1.0 scope |
| PH-CON-002 | CRITICAL three-layer: harness-authoritative analytical `E(t)` from `(u_t, $\nabla$u)` snapshots (roundoff `~5e-16`); rule-verdict log-log slope 1.94 from rule's internal 2nd-order-central FD `u_t` primitive | F3 absent with justification — PDEBench shallow-water reports mass `cRMSE` (semantically incompatible with wave-energy); Hansen ProbConserv `CE` is defined for first-order-in-time integral conservation laws, not second-order-in-time wave-energy functionals `E[u, u_t] = ½$\int$(u_t² + c²$\|\nabla$u$\|$²)`; both moved to Supplementary | 28-of-28 SUCCESS at PR #3 (HEAD 6907699) | `26ed3bd` — Task 9 PH-CON-002 wave-energy two-layer analytical | Homogeneous BC only; no accuracy implication. Analytical-snapshot only; not a leapfrog or RK4 time-stepper validation |
| PH-CON-003 | Analytical eigenmode `sin($\pi$x) sin($\pi$y) · exp(−2$\pi$²t)` with closed-form per-step ratio `exp(−0.2$\pi$²) $\approx$ 0.13888` at `$\Delta$t = 0.05`; rule emits `dE/dt` via forward-difference primitive | F3 absent with justification — per-step energy-ratio is derivable from Evans §7.1.2 directly (analytically-known exact value, not borrowed reproduction); PDEBench has no standalone heat equation; Hansen ProbConserv Table 1 covers diffusion mass conservation, not per-step heat-energy dissipation | 28-of-28 SUCCESS at PR #3 (HEAD 6907699) | `1a1c097` — three-function CITATION.md retrofit (forward-difference primitive landed in `e691dd3`; original anchor `f652a3d`) | Heat energy-dissipation sign rule. Forward-difference `dE/dt` measurement primitive; central-diff bug fixed in `e691dd3` |
| PH-CON-004 | L-shape singularity hotspot fixture; `max_K / mean_K` of `$\int$_K ($\Delta$_{L²-proj zero-trace} u)² dx` over interior elements; ratio refinement-invariant at `~1.70` element-layers across uniform refinements | F3 absent with justification — effectivity-index values depend on `(estimator, marker, solver)` triple (classical residual vs recovery-type vs equilibrated-flux $\times$ maximum / Dörfler / equidistribution); no single-paper reproduction target maps onto the rule's narrower interior-volumetric scope; scikit-fem Example 22 not importable from pip-installed package; Becker-Rannacher 2001 DWR retained in Supplementary | 28-of-28 SUCCESS at PR #3 (HEAD 6907699) | `87e8a3e` — Task 10 PH-CON-004 conservation-defect localization | 2D only in v1.0; 3D to v1.2; residual/error localization, not local-conservation certification. Narrower-estimator-than-Verfürth-theorem scoping (interior `$\|\Delta$u$\|$²` only, no facet jumps) |
| PH-POS-001 | Closed-form positive harmonic / parabolic fixtures (analytically-known-positive on the domain interior) plus an Evans-corner negative control that injects a strong-maximum-principle violation | F3 absent with justification — discrete-predicate rule (verdict + `raw_value = min(u)` when below floor); fixtures are analytically-known-positive and verdict is mechanically secure on them; Evans theorems reproduced as structural identity, no numerical baseline to reproduce | 28-of-28 SUCCESS at PR #3 (HEAD 6907699) | `5c6a63b` — three-function CITATION.md retrofit (anchor introduced in `6ec3cc1`) | PDE-dependent applicability. Discrete predicate; `raw_value` is a binary verdict on closed-form fixtures |
| PH-POS-002 | Three harmonic polynomial fixtures (`x² − y²`, `xy`, `x³ − 3xy²`) plus an interior-overshoot negative control injecting a maximum-principle violation | F3 absent with justification — discrete-predicate rule (verdict + `raw_value` = max-overshoot scalar); analogous to PH-POS-001; Evans §2.2.3 reproduced as structural identity, no numerical baseline | 28-of-28 SUCCESS at PR #3 (HEAD 6907699) | `071b44f` — three-function CITATION.md retrofit (anchor introduced in `cdc55eb`) | User declares $D$. Discrete predicate; v1.0 known issue: relative-mode floor underflow on homogeneous-Dirichlet samples (deferred to v1.2) |
| PH-SYM-001 | C₄ structural-equivalence on closed-form rotation-equivariant operators (FFT-based Laplace inverse, identity) on a 2D periodic square grid; non-equivariant (coordinate-dependent CNN) negative control | F3 absent with justification — no published numerical baseline directly comparable to the rule's `rotate_test` emitted quantity; Helwig 2023 ICML Table 3 (G-FNO p4 relative MSE on 2D Navier-Stokes), Weiler-Cesa NeurIPS 2019 (E(2)-CNN on RotMNIST), Cohen-Welling ICML 2016 (P4CNN on RotMNIST) report downstream-task accuracy / relative MSE, not equivariance-error on author-constructed analytic fixtures; retained in Supplementary as scale calibration | 28-of-28 SUCCESS at PR #3 (HEAD 6907699) | `21c0a57` — Task 1 PH-SYM-001 / 002 structural-equivalence retrofit (original anchor `3ba694d`) | Finite sample only. Section-level Hall framing (no tight theorem-number); analytic-operator equivariance-error scope only |
| PH-SYM-002 | Z₂ mirror-image structural-equivariance on closed-form reflection-equivariant operators on a 2D periodic square grid; non-equivariant negative control | F3 absent with justification — analogous to PH-SYM-001; Helwig 2023 ICML Table 1 (G-FNO-p4m on symmetric test set, relative MSE 2.37 ± 0.19), Cohen-Welling 2016, Weiler-Cesa 2019 reflection-related rows are downstream-task metrics, not directly-comparable equivariance-error; retained in Supplementary | 28-of-28 SUCCESS at PR #3 (HEAD 6907699) | `21c0a57` — Task 1 PH-SYM-001 / 002 structural-equivalence retrofit (original anchor `3ba694d`) | Per-transform empirical. Section-level framing; analytic-operator reflection-equivariance-error scope only |
| PH-SYM-003 | CRITICAL three-layer SO(2) Lie-derivative diagnostic: F2 primary fixtures are radial scalar-invariant positive controls (e.g. `f(x, y) = g(x² + y²)`) and coordinate / anisotropic non-equivariant negative controls; finite-vs-infinitesimal scaling check on a refining $t$-grid; rule-verdict via `torch.autograd.functional.jvp`-based Lie-derivative `(L_A f)(x)` against a scalar floor under six gating preconditions. Supplementary controlled-operator context: FFT-Laplace inverse and identity (these are exactly C₄- and reflection-equivariant and serve as cross-checks rather than primary SO(2) LEE fixtures) | F3 absent with justification (F3-INFRA-GAP) — RotMNIST + escnn / e3nn checkpoint + Modal A100 provisioning + ImageNet-opt-in Gruver `lie-deriv` LEE pre-demoted: no `equivariance` optional-dep group in `pyproject.toml`; no Modal / RotMNIST workflow; combined build cost exceeds Task 6 budget; Cohen-Welling, Weiler-Cesa, Gruver retained in Supplementary; loaders on v1.2 backlog. Theory anchors: Hall 2015 Thm 2.14, Cor 3.50 (locators verified); Kirillov 2008 Cor. 2.9 (identity-component generation, locator verified); Weiler et al. (World Scientific, forthcoming 2026, background); Gerken 2023 | 28-of-28 SUCCESS at PR #3 (HEAD 6907699) | `267dfd1` — Task 6 PH-SYM-003 SO(2) LEE diagnostic (CRITICAL three-layer + preflight gate) | No disconnected-group implications; no Gruver claim without opt-in run. V1 narrower than F1: infinitesimal scalar-invariant only; finite multi-output equivariance deferred; adapter-mode-only (no dump-mode coverage) |
| PH-SYM-004 | CRITICAL three-layer: harness-level F2 validates four controlled translation-equivariant operators (identity, circular convolution 1D / 2D, Fourier multiplier 1D / 2D) plus a coordinate-dependent non-equivariant negative control via `shift_commutation_error`; **shipped v1.0 production behavior is `SKIPPED`-always past declared-symmetry + periodicity gates** (the rule confirms it correctly recognizes when to skip; live-callable shift-commutation is deferred to v1.2) | F3 absent with justification — Helwig 2023, Li et al. 2021 FNO Appendix, Kondor-Trivedi 2018 are either downstream-task error metrics, theoretical proofs, or solver-specific RMSE rows; none are directly-comparable equivariance-error reproduction targets for a `SKIPPED`-always V1 stub; F3 contract revised 2026-04-24 to honor the user's "do not force borrowed credibility" rule; Helwig moved to Supplementary | 28-of-28 SUCCESS at PR #3 (HEAD 6907699) | `ae1f9a9` — Task 7 PH-SYM-004 V1-stub translation equivariance | V1 stub: production rule is `SKIPPED`-always past gates and verifies gating + SKIP behavior only. Live-callable adapter mode (`f(roll(x)) == roll(f(x))`) deferred to v1.2 |
| PH-NUM-001 | CRITICAL three-layer via `_harness/quadrature.py`: harness-level F2 validates (A) `degree $\leq$ intorder` exact to roundoff; (B) under-integrated case bounded away from zero; (C) error drops by factor `~7.8e12` across `intorder $\in$ {2, 4, 6, 8, 10}`. **Shipped v1.0 production behavior: `PASS`-with-stub-reason on `MeshField` input** (with pass-through `field.integrate()` baseline and reason `"qorder convergence check is a stub until V1.1"`), `SKIP` on unsupported field types | F3 absent with justification — Ciarlet 2002 §4.1, Strang 1972 Variational Crimes, Brenner-Scott 2008 §10.3 publish quadrature-convergence theorems whose tabulated values are examples-within-proofs or illustrative figures, not systematic `(p, intorder, MMS)` reproduction targets; Ern-Guermond 2021 Vol. II + MOOSE FEM-convergence tutorial retained in Supplementary | 28-of-28 SUCCESS at PR #3 (HEAD 6907699) | `2ae7d28` — Task 11 PH-NUM-001 V1-stub quadrature exactness | V1 stub: production rule is `PASS`-with-stub-reason and validates pass-through behavior only. Production reason string still says `V1.1`; the active public backlog has been consolidated under v1.2. Full MMS h-refinement / production lookup-table check deferred to v1.2 |
| PH-NUM-002 | Three scoped cases: (A) harness-authoritative `mms_observed_order_fd2` `p_obs $\rightarrow$ 2`; (B) FD non-periodic `p_obs $\approx$ 2.50` from boundary-band scaling; (C) saturation floor below `1e-11` returns `rate = inf` | F3 absent with justification — Oberkampf-Roy 2010 Chs 5–6 + Roy 2005 (JCP) supply MMS / `p_obs` algorithm framing only (no reproducible numerical dataset with published `p_obs` for a specific PDE+backend+BC triple); PDEBench / Hansen ProbConserv / NSForCFD report aggregate RMSE / nRMSE rather than `p_obs`; methodology references retained in Mathematical-legitimacy / Supplementary | 28-of-28 SUCCESS at PR #3 (HEAD 6907699) | `84c7163` — Task 12 PH-NUM-002 observed-order three-case | Per PDE case and per norm; rates declared per-norm, not uniform. Periodic harmonics on T² are constants by Liouville (structural unreachability documented as scope-truth) |
| PH-VAR-002 | Diagnostic contract verification only (not a numerical fixture): rule emits info-severity `PASS` with literature-pointer reason on wave-equation `DomainSpec`, `SKIPPED` on every other PDE kind | F3 absent by structure — info-flag rules emit no numerical quantity against the field, so there is nothing to reproduce; Demkowicz-Gopalakrishnan 2025 *The discontinuous Petrov-Galerkin method* (Acta Numerica 34:293–384, DOI 10.1017/S0962492924000102) retained in Supplementary as theoretical framing only | 28-of-28 SUCCESS at PR #3 (HEAD 6907699) | `6907699` — Task 13 v1.0 closeout (PH-VAR-002 anchor introduced in this commit) | INFO-level only. Literature-maturity-gated promotion to numerical-comparison rule deferred to v1.2 |

The 18 rows above cover every rule that is implemented and has an
external-validation matrix shard at the v1.0 freeze. CI status is
identical across rows because PR #3's 28 checks ran as a single matrix
at HEAD `6907699`; per-row variation is not possible by construction.
PH-VAR-001 (deferred to v1.2) is documented in Appendix D.

## 25. Rule-Readiness Criteria

A rule is ready to appear in the v1.0 Validation Report when all of the
following are satisfied.

- Mathematical section exists in this document.
- Assumptions are explicit either in a dedicated Assumptions
  subsection or in the rule's theorem / implementation mapping text.
- Non-claims are explicit (every rule has a "What this does not validate"
  subsection; §23 covers cross-rule limitations).
- Rule-to-implementation mapping is stated (every rule has an
  Implementation mapping subsection).
- `CITATION.md` for the rule has three-function structure (Function 1
  mathematical-legitimacy, Function 2 correctness-fixture, Function 3
  borrowed-credibility).
- Correctness fixture exists or absence is justified on the record.
- External anchor exists or absence is justified on the record.
- CI status is recorded.
- Textbook theorem framing matches verification status per the
  Appendix B verification taxonomy and supplementary-citation list.
  Citations used as book-metadata anchors (B.1) may appear without
  theorem-locator claims; citations used as theorem-locator anchors in
  the body text must either be verified (B.2) or flagged as pending
  (B.3) in the relevant section.
  The authoritative per-citation verification record is
  `external_validation/_harness/TEXTBOOK_AVAILABILITY.md`.
- For the v1.0 freeze, release notes must state the exact shipped
  contract for any rule whose §n.5 Implementation mapping could admit
  multiple scope resolutions. In this revision the two candidates
  (PH-NUM-001 §19.5 and PH-SYM-004 §18.5) have been resolved to
  definitive "The v1.0 production rule ships as ..." statements; those
  statements are now the shipped contract for v1.0, and any deviation
  requires updating this document and §24 before release.

## 26. Integration Provenance

This report is the integrated v1.0 Validation Report. It combines the
mathematical-legitimacy layer (Parts I–VII), the cross-rule limitations
(§23), and the engineering audit layer's evidence (§24's rule-by-rule
matrix). Source artifacts feeding into this report:

- Per-rule mathematical sections (Parts I–VII) of this report are the
  Function-1 source of truth.
- `external_validation/*/test_anchor.py` — Function-2 executable
  correctness fixtures referenced in the §24 F2 column.
- `external_validation/*/CITATION.md` — per-rule citation files with
  three-function labels and audit trail; the F3 column of §24 is
  derived from the borrowed-credibility subsection of these files.
- CI run history at PR #3 (HEAD `6907699`) — the §24 CI status
  column.
- Branch commit log on `external-validation-tier-a` — the §24 commit
  / provenance column.
- `TRACEABILITY.md` — engineering audit traceability cross-reference.

**Editing protocol.** New mathematical claims must first be added to
the appropriate per-rule body section (Parts I–VII) before being
referenced in §24 or any appendix. Conversely, engineering evidence
updates (new CI runs, commit moves, additional F3 reproductions)
update §24 only and do not require body edits unless the underlying
mathematical content changes.

---

## Appendix A — Rule-to-Derivation Index

| Rule | Derivation type | Main mathematical object | Section |
|---|---|---|---|
| PH-RES-001 | Type B: theorem / residual norm | PDE residual, elliptic a posteriori | §4 |
| PH-RES-002 | Type B: consistency derivation | AD vs FD residual agreement | §5 |
| PH-RES-003 | Type B: spectral convergence | Periodic derivative residual | §6 |
| PH-BC-001 | Type B: trace theorem | Boundary trace in $H^{1/2}$ | §7 |
| PH-BC-002 | Type A: structural identity | Gauss-Green theorem | §8 |
| PH-CON-001 | Type A: balance law | Conserved integral | §9 |
| PH-CON-002 | Type A: structural identity | Wave energy functional | §10 |
| PH-CON-003 | Type A: parabolic energy identity | Heat-equation $L^2$-energy dissipation | §11 |
| PH-CON-004 | Type B: estimator-inspired diagnostic | A posteriori-inspired residual-hotspot indicator | §12 |
| PH-POS-001 | Type A: maximum principle | Positivity invariant | §13 |
| PH-POS-002 | Type A: invariant domain | Bound violation | §14 |
| PH-SYM-001 | Type A: structural equivalence | Finite group equivariance | §15 |
| PH-SYM-002 | Type A: structural equivalence | Per-transform defect | §16 |
| PH-SYM-003 | Type A: structural equivalence | Lie-derivative / infinitesimal equivariance | §17 |
| PH-SYM-004 | Type A: structural equivalence | Shift equivariance | §18 |
| PH-NUM-001 | Type B: numerical-analysis theorem | Quadrature consistency / first Strang lemma | §19 |
| PH-NUM-002 | Type B: MMS derivation | Observed convergence order | §20 |
| PH-VAR-001 | Type C: diagnostic | L² strong-form / norm selection | §21 |
| PH-VAR-002 | Type C: diagnostic caveat | Hyperbolic residual norm | §22 |

---

## Appendix B — Citation Verification Notes

The authoritative citation-verification record is
`external_validation/_harness/TEXTBOOK_AVAILABILITY.md`. This appendix
summarizes the verification status of the load-bearing citations using
a four-bucket verification taxonomy, followed by a supplementary
non-load-bearing citation list, that together separate what has
actually been confirmed. Metadata verification (the book exists, at
this ISBN, published by this publisher) is weaker than theorem-locator
verification (this specific theorem is numbered Thm X.Y in a checked
copy of the book). Neither is mathematical-proof verification.

### B.1 Verified metadata

The following citations have been primary-source verified at the
book/article-metadata level (publisher record, DOI, ISBN, series
volume, year, pages) but are not used for specific theorem-locator
claims — or, in the case of Varadarajan, have been moved here from a
previously load-bearing role because the section locator was not
confirmed against a physical copy.

- **Varadarajan GTM 102** (V. S. Varadarajan, *Lie Groups, Lie
  Algebras, and Their Representations*, Springer GTM vol. 102, 1984,
  ISBN 978-0-387-90969-1). Book metadata is primary-verified via
  Springer. The previously cited section locator §2.9–2.10 was not
  physically verified and is **no longer load-bearing** for PH-SYM-003;
  the identity-component generation step is now anchored to Kirillov
  2008 Corollary 2.9 (see B.2). Varadarajan is retained here as
  background only.
- **Weiler-Forré-Verlinde-Welling** (Maurice Weiler, Patrick Forré,
  Erik Verlinde, Max Welling, *Equivariant and Coordinate Independent
  Convolutional Networks: A Gauge Field Theory of Neural Networks*,
  World Scientific, Progress in Data Science vol. 1, DOI 10.1142/14143,
  ISBN 978-981-98-0662-1). Book metadata primary-verified via World
  Scientific and DBLP. DBLP gives the authoritative publication year
  as **2026**; some retailer records list late 2025 advance copies.
  Used in PH-SYM-003 as a forthcoming comprehensive monograph anchor;
  not used as a theorem-level primary source in this document.
- **Gerken-Aronsson-Carlsson-Linander-Ohlsson-Petersson-Persson 2023**
  (Jan E. Gerken et al., *Geometric Deep Learning and Equivariant
  Neural Networks*, Artificial Intelligence Review, vol. 56 no. 12,
  pp. 14605–14662, DOI 10.1007/s10462-023-10502-7). Peer-reviewed
  survey article, published 4 June 2023. Used in PH-SYM-001 and
  PH-SYM-003.
- **Cohen-Geiger-Weiler 2019** (Taco Cohen, Mario Geiger, Maurice
  Weiler, *A General Theory of Equivariant CNNs on Homogeneous
  Spaces*, NeurIPS 2019; arXiv:1811.02017). Metadata verified. Used
  in PH-SYM-003 (backdrop) and PH-SYM-004 (homogeneous-space kernel
  characterization).
- **Kondor-Trivedi 2018** (Risi Kondor, Shubhendu Trivedi, *On the
  Generalization of Equivariance and Convolution in Neural Networks
  to the Action of Compact Groups*, ICML 2018, PMLR vol. 80,
  pp. 2747–2755; arXiv:1802.03690). Metadata verified via PMLR
  directly. The compact-group convolution/equivariance necessity
  claim appears in the paper's abstract. Used in PH-SYM-004 for the
  compact-group necessity direction.
- **Cohen-Welling 2016** (Taco Cohen, Max Welling, *Group Equivariant
  Convolutional Networks*, ICML 2016, PMLR vol. 48, pp. 2990–2999).
  Metadata verified via PMLR. Used in PH-SYM-001, PH-SYM-002.
- **Evans 2010** (Lawrence C. Evans, *Partial Differential
  Equations*, 2nd ed., AMS GSM vol. 19, ISBN 978-0-8218-4974-3).
  Book metadata primary-verified. Standard locators used — §2.4.3
  (wave-equation energy identity, PH-CON-002), §5.5 (trace theorem,
  PH-BC-001), §6.4 and §7.1.4 (maximum principles, PH-POS-001), §C.2
  (divergence theorem, PH-BC-002) — are consistent with the 2nd-edition
  structure.
- **Oberkampf-Roy 2010** (William L. Oberkampf, Christopher J. Roy,
  *Verification and Validation in Scientific Computing*, Cambridge UP,
  ISBN 978-0-521-11360-1). Book metadata primary-verified. Used Ch. 5
  (code order-of-accuracy verification) and Ch. 6 (exact and
  manufactured solutions) in PH-NUM-002.
- **Strang-Fix 1973** (Gilbert Strang, George Fix, *An Analysis of the
  Finite Element Method*, Prentice-Hall, Englewood Cliffs, NJ; 2008
  reprint, Wellesley-Cambridge Press). Ch. 4 for the first Strang
  lemma and variational-crime framework (PH-NUM-001).
- **Ciarlet 1978** (Philippe G. Ciarlet, *The Finite Element Method
  for Elliptic Problems*, North-Holland, Amsterdam; 2002 SIAM reprint
  ISBN 978-0-89871-514-9). §4.1 for quadrature consistency and the
  first Strang lemma framework (PH-NUM-001).
- **Ern-Guermond 2021 Vol. II** (Alexandre Ern, Jean-Luc Guermond,
  *Finite Elements II: Galerkin Approximation, Elliptic and Mixed
  PDEs*, Springer, Texts in Applied Mathematics vol. 73, 2021, DOI
  10.1007/978-3-030-56923-5, ISBN 978-3-030-56922-8). Book metadata
  and table-of-contents primary-verified via Springer and the HAL
  open copy. Volume II explicitly covers Galerkin approximation,
  Strang's lemmas, variational crimes (Ch. 27), quadratures (Ch. 30),
  and a posteriori error analysis (Part VII). Used in PH-RES-001,
  PH-CON-004, PH-NUM-001. Earlier versions of this document cited
  Vol. I (Approximation and Interpolation) for these results; that
  was an error and has been corrected in this revision.
- **Verfürth 2013** (Rüdiger Verfürth, *A Posteriori Error Estimation
  Techniques for Finite Element Methods*, Oxford University Press,
  ISBN 978-0-19-967942-3). Ch. 1 for the reliability and efficiency
  framework (PH-CON-004).
- **Brenner-Scott 2008** (Susanne C. Brenner, L. Ridgway Scott, *The
  Mathematical Theory of Finite Element Methods*, 3rd ed., Springer
  TAM vol. 15, ISBN 978-0-387-75933-3). §10.3 for variational-crime
  theory (PH-NUM-001).
- **LeVeque 2002** (Randall J. LeVeque, *Finite Volume Methods for
  Hyperbolic Problems*, Cambridge UP, ISBN 978-0-521-00924-9). Ch. 2
  for conservative form (PH-CON-001), §11.13 for invariant-domain /
  maximum-principle extension (PH-POS-002).
- **Trefethen 2000** (Lloyd N. Trefethen, *Spectral Methods in
  MATLAB*, SIAM, ISBN 978-0-89871-465-4). Used for spectral residual
  behavior (PH-RES-003).
- **Griewank-Walther 2008** (Andreas Griewank, Andrea Walther,
  *Evaluating Derivatives: Principles and Techniques of Algorithmic
  Differentiation*, 2nd ed., SIAM, ISBN 978-0-89871-659-7). Ch. 3 for
  AD correctness on the executed computational graph (PH-RES-002).
- **Guermond-Popov 2016** (Jean-Luc Guermond, Bojan Popov, *Invariant
  Domains and First-Order Continuous Finite Element Approximation
  for Hyperbolic Systems*, **SIAM Journal on Numerical Analysis
  vol. 54 no. 4, pp. 2466–2489, 2016**, DOI **10.1137/16M1074291**).
  Metadata primary-verified via SIAM. Used as the invariant-domain
  framework anchor in PH-POS-002. Earlier versions of this document
  misquoted this as SIAM JNA 55(6) (which is the 2017
  second-order-scalar-conservation paper by the same authors,
  DOI 10.1137/16M1106560); that was an error and has been corrected.
- **Roache 2002** (Patrick J. Roache, *Code Verification by the
  Method of Manufactured Solutions*, Journal of Fluids Engineering
  vol. 124 no. 1, pp. 4–10, 2002, DOI 10.1115/1.1436090). Used in
  PH-NUM-002.
- **Demkowicz-Gopalakrishnan 2011** (Leszek Demkowicz, Jay
  Gopalakrishnan, *A Class of Discontinuous Petrov-Galerkin Methods,
  Part II: Optimal Test Functions*, Numerical Methods for Partial
  Differential Equations vol. 27 no. 1, pp. 70–105, 2011, DOI
  10.1002/num.20640). Metadata verified. Used in PH-VAR-002.
- **Demkowicz-Gopalakrishnan 2025** (Leszek Demkowicz, Jay
  Gopalakrishnan, *The Discontinuous Petrov-Galerkin Method*, **Acta
  Numerica vol. 34, pp. 293–384, 2025**, DOI
  10.1017/S0962492924000102, Open Access, Cambridge University
  Press, published online 01 July 2025). Review article;
  primary-source verified via Cambridge. The paper states
  explicitly that the DPG method is equivalent to a minimum-residual
  method with residual measured in a dual norm. Used in PH-VAR-002;
  **replaces** the "Demkowicz-Gopalakrishnan 2014 review article"
  citation in earlier drafts (that reference was to the Wiley
  Encyclopedia chapter, ECM2105, and is not the authoritative modern
  review).

### B.2 Verified theorem locator

- **Hall 2015** (Brian C. Hall, *Lie Groups, Lie Algebras, and
  Representations: An Elementary Introduction*, 2nd ed., Springer GTM
  222, ISBN 978-3-319-13466-6, DOI 10.1007/978-3-319-13467-3).
  **Physical-copy theorem-locator check completed by project
  maintainer.**
  - **Theorem 2.14** states the one-parameter subgroup result for
    $A : \mathbb{R} \to \mathrm{GL}(n; \mathbb{C})$: if $A(t)$ is a
    one-parameter subgroup, then there exists a unique matrix $X$ such
    that $A(t) = \exp(tX)$.
  - **Corollary 3.50** states that every continuous homomorphism
    between two matrix Lie groups is smooth.

  These locators are used in PH-SYM-003 for the bridge from
  one-parameter subgroup structure and continuous representations to
  differentiability of the equivariance constraint.
- **Kirillov 2008** (Alexander Kirillov Jr., *Introduction to Lie
  Groups and Lie Algebras*, Cambridge Studies in Advanced Mathematics
  vol. 113, ISBN 978-0-521-88969-8). **Corollary 2.9** states: if
  $G$ is a connected Lie group and $U$ is a neighborhood of the
  identity, then $U$ generates $G$. Locator verified in the publicly
  accessible course-note copy of the book. Used in PH-SYM-003 for the
  identity-component generation step, replacing the previously cited
  Varadarajan GTM 102 §2.9–2.10 locator (which was not physically
  verified and has been moved to B.1 as background-only metadata).

### B.3 Primary-source theorem locator pending

None at this time. Hall 2015 Theorem 2.14 and Corollary 3.50 have
been resolved into B.2 via physical-copy check; Varadarajan GTM 102
§2.9–2.10 is no longer load-bearing (the load-bearing anchor for
identity-component generation is now Kirillov Corollary 2.9, B.2).
Promotion to this bucket would require (i) a new theorem-locator claim
introduced in a later revision and (ii) that claim not yet being
physically verified.

### B.4 Incorrect / replace

The following citation statements appeared in earlier drafts of this
document and have been **corrected** in this revision. They are
listed here for audit traceability.

- "Guermond-Popov 2016 SIAM J. Numer. Anal. 55(6)" — **incorrect**.
  The 55(6)/2017 reference is the authors' *second-order* scalar
  conservation paper (DOI 10.1137/16M1106560). The first-order
  invariant-domain paper referenced in PH-POS-002 is SIAM JNA
  **54(4)**, pp. 2466–2489, 2016, DOI 10.1137/16M1074291. Corrected
  throughout §14 and §24. (Earlier drafts also routed this anchor
  through PH-CON-003; after PH-CON-003 was rewritten as a heat-
  energy-dissipation rule, that linkage is no longer in effect — see
  Appendix D for the README reconciliation.)
- "Ern-Guermond 2021, Vol. I" as anchor for a posteriori analysis,
  variational crimes, quadratures — **incorrect volume**. These
  topics are in Vol. II (Galerkin Approximation, Elliptic and Mixed
  PDEs, Part VI "Galerkin approximation" Ch. 27, Part VI Ch. 30
  quadratures, Part VII "A posteriori error analysis"). Vol. I is
  Approximation and Interpolation; its contents do not include those
  chapters. Corrected throughout §4, §12, §19.
- "Demkowicz-Gopalakrishnan 2014 review article" — **replaced** with
  the peer-reviewed Open Access 2025 Acta Numerica review (DOI
  10.1017/S0962492924000102). The 2014 reference in earlier drafts
  pointed to the Wiley Encyclopedia chapter (ECM2105), which is not
  the authoritative modern review of the DPG methodology.
- "Primary-source verified via Wikipedia" as a verification label
  for Hall's theorem locators — **resolved**. The interim Wikipedia-
  mediated status has been superseded by a physical-copy check of Hall
  Theorem 2.14 and Corollary 3.50, which is now recorded in B.2.
  Wikipedia is no longer cited as a verification source.
- "Varadarajan GTM 102 §2.9–2.10 as load-bearing anchor for
  identity-component generation" — **demoted**. The section locator
  was not confirmed against a physical copy, so Varadarajan can no
  longer carry a theorem-locator claim in this document. The
  load-bearing anchor for identity-component generation has been
  replaced with Kirillov 2008 Corollary 2.9 (B.2), which states
  exactly the needed result with a verifiable locator. Varadarajan is
  retained in B.1 as background metadata only.
- "Weiler-Forré-Verlinde-Welling 2025, published monograph" —
  **updated**. The authoritative DBLP record lists the publication
  year as 2026 (World Scientific, Progress in Data Science vol. 1);
  retailer listings from late 2025 reflect advance copies. The
  citation is retained as a forthcoming monograph anchor, not a
  theorem-level primary source.

### B.5 Supplementary / non-load-bearing citations

The following citations appear as supplementary references or optional
secondary anchors. They are not load-bearing for the rule-to-theorem
chain because the corresponding primary load-bearing anchor is already
listed in B.1 or B.2.

- Grisvard 1985 — supplementary trace / elliptic-regularity reference
  cited in §7.2; Evans 2010 §5.5 is the load-bearing trace-theorem
  anchor.
- Gagliardo 1957 — historical trace-theorem reference cited in §7.2;
  Evans 2010 is the load-bearing modern anchor.
- Hunter, *Notes on Partial Differential Equations* — supplementary
  divergence-theorem reference cited in §8.2; Evans 2010 §C.2 is the
  load-bearing divergence-theorem anchor.
- Protter-Weinberger 1967 — supplementary maximum-principle reference
  cited in §13.2; Evans 2010 §6.4 / §7.1.4 is the load-bearing anchor.
- Kruzkov 1970 — supplementary scalar-conservation-law invariant-bound
  reference cited in §14.3; Guermond-Popov 2016 (SIAM JNA 54(4)) and
  LeVeque 2002 §11.13 are the operational anchors.
- LeVeque 2007 — optional FD truncation-order reference mentioned in
  §5.8; not load-bearing unless adopted in the final Function-3
  citation table.

---

## Appendix C — Glossary

The following terms are used throughout the document with technical
meanings that may differ from colloquial usage.

- **Mathematical-legitimacy anchor** (Function 1). Evidence that a rule
  measures a genuine mathematical property under stated assumptions.
  Supplied by this document.
- **Correctness fixture** (Function 2). An executable test whose
  expected output can be computed analytically or from a known-correct
  reference, demonstrating that the rule's implementation computes the
  intended quantity. Supplied by `external_validation/<rule>/
  test_anchor.py`.
- **Borrowed-credibility anchor** (Function 3). A reproduction, benchmark,
  or external study that independently corroborates the rule's claim
  beyond the unit-test level. Supplied by external benchmarks where
  available.
- **Structural equivalence**. The rule's computed quantity is a
  concrete instance of an established mathematical structure (e.g. a
  group-equivariance defect). Justification via §2.2 Type A.
- **Manufactured solution**. A user-chosen analytical function $u^*$
  used as a controlled test case by computing its induced source term
  and solving the PDE with that source.
- **Residual**. For a PDE $\mathcal{L}u = f$ and candidate $u_h$, the
  quantity $\mathcal{L}u_h - f$. Measured in some norm, it is a
  diagnostic; its equivalence to error depends on the PDE class.
- **Trace**. The boundary restriction of a Sobolev-space function, as
  extended by the trace theorem from classical pointwise restriction.
- **Equivariance**. A function $f$ is $G$-equivariant when applying a
  group action to the input and then $f$ gives the same result as
  applying $f$ and then the action on the output.
- **Lie derivative**. The derivative of a quantity along a
  one-parameter subgroup of a Lie group, evaluated at the identity.
  For equivariance, a vanishing Lie derivative at all generators (under
  connectedness) implies equivariance over the identity component.
- **Energy identity**. A conservation law of the form $dE/dt =
  \text{boundary terms}$, derived by multiplying a PDE by an
  appropriate test function and integrating by parts.
- **Variational crime**. A departure from the exact Galerkin procedure
  via non-conforming elements, numerical quadrature, or domain
  approximation. Characterized by the first and second Strang lemmas.
- **Identity component**. The connected component of a Lie group
  containing the identity element, denoted $G^0$. Generated by any
  neighborhood of the identity (Kirillov 2008 Corollary 2.9).

---

## Appendix D — Deferred v1.2 Notes and README Reconciliation

This appendix collects two kinds of audit-trail content that do not
fit cleanly into the per-rule body sections: (D.1) mathematical notes
for rules that are documented but not implemented in v1.0, and (D.2)
the reconciliation between the public README rule labels and the
Function-1 labels used in this report.

### D.1 Deferred v1.2 Mathematical Notes

#### PH-VAR-001 — L² Strong-Form Residual / Norm-Selection Warning (deferred)

PH-VAR-001 is a Type-C diagnostic meta-rule deferred from v1.0
production scope to v1.2. The v1.0 codebase contains only a SARIF
severity-mapping fixture referencing this rule-id (used to verify
`info → note` SARIF emission); no
`src/physics_lint/rules/ph_var_001.py` and no per-rule unit test
exists at the v1.0 freeze. The design specification classifies it as a
meta-rule alongside PH-NUM-003 and PH-NUM-004, all deferred from v1.0
production scope.

**Mathematical claim.** The intended primary instance of PH-VAR-001
warns when a user interprets an $L^2$ norm of a second-order strong-
form residual as though it were the variationally correct dual
residual norm. More generally, different norms on the same function
space are not equivalent to the error quantity a user may implicitly
assume. The rule is therefore a norm-selection diagnostic: it flags a
mismatch between the declared PDE formulation, the emitted residual
norm, and the norm in which residual-error control is theoretically
justified.

**Theoretical anchor.** Standard fact that $L^2$, $H^1$, $H^{-1}$,
$L^\infty$ and so on give different convergence rates and different
error characterizations. Type-C diagnostic justification (the rule
documents and flags a known interpretive pitfall, rather than checking
a theorem-level identity).

**Derivation.** For elliptic PDEs, the natural norm for $V$-ellipticity
is the $H^1$ norm; measuring error in $L^2$ is a lower bound but gives
a different convergence rate (typically one higher via Aubin-Nitsche
duality). For hyperbolic PDEs, the natural norm is problem-dependent
and often weaker. Reporting error in a norm that does not control the
quantity of interest can mislead, even when the chosen norm is itself
a meaningful seminorm on the function space.

**Intended v1.2 implementation mapping.** The v1.2 deferred contract:
the rule inspects the user's declared residual formulation and norm.
The primary trigger is the use of an $L^2$ residual for a second-order
strong-form formulation where the variationally natural object is a
dual residual, typically measured in an $H^{-1}$-type norm. The
broader lookup-table machinery records natural norms for common PDE
classes and emits a WARNING / INFO-level diagnostic when the declared
norm is not aligned with the formulation.

**What v1.2 will validate.** Surface-level coherence between the
declared PDE formulation, the emitted residual norm, and the norm in
which the residual is being interpreted. Detection of the specific
$L^2$-strong-form / dual-residual mismatch as the primary case.

**What v1.2 will not validate.** This is a warning rule, not a
correctness certificate. The user may have a valid reason for an
unconventional norm choice (for example a specific quantity of
interest); the warning does not imply the choice is wrong. The rule
will continue to be INFO/WARNING-level, user-overridable.

### D.2 README Rule-Label Reconciliation

The public README rule catalog and this report use different levels of
specificity for several rule labels. For v1.0 Function-1 purposes, this
report follows the external-validation rule map. The README remains a
user-facing catalog and should be reconciled in release notes or
updated before public v1.0 release.

| Rule ID | README label | Function-1 label | Resolution |
|---|---|---|---|
| PH-CON-003 | Energy dissipation sign violation | Heat Energy-Dissipation Sign | Earlier draft framed PH-CON-003 as a positivity / conservation coupled check; that framing was replaced (this revision) with the heat-equation energy-dissipation derivation that matches the shipped F2 fixture. The README label "Energy dissipation sign violation" and the Function-1 label "Heat Energy-Dissipation Sign" now describe the same mathematical content. |
| PH-VAR-001 | L² residual on second-order strong-form formulation | L² Strong-Form Residual / Norm-Selection Warning | Deferred from v1.0 to v1.2; full mathematical derivation in §D.1. |
| PH-POS-002 | Maximum principle violation | Bound / Invariant-Domain Check | Accepted as a generalization: invariant-domain checks include maximum-principle-style bounds where applicable. |
| PH-SYM-001 | $C_4$ rotation equivariance violation | Finite Symmetry Equivariance Check | Accepted as a generalization: $C_4$ is the canonical finite-group fixture. |
| PH-SYM-002 | Reflection equivariance violation | Symmetry Violation / Transformation Consistency | Accepted as a reporting/generalization layer over finite reflection defects. |

---

*End of physics-lint v1.0 Validation Report. This report intentionally
includes mathematical derivations (Function 1), engineering correctness
fixtures (Function 2), borrowed-credibility status (Function 3), CI
results, and commit provenance, integrated into a single document. The
mathematical claims are limited to those stated in the per-rule
sections (Parts I–VII); the rule-by-rule integrated validation matrix
in §24 records the engineering audit layer's evidence at the v1.0
freeze.*
