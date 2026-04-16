# physics-lint tradeoffs

Methodology and design decisions that deviated from a plan or
specification, the reasoning behind the deviation, and what future
reviewers or plan authors should take away. Each entry is appended as
deviations occur; entries are not retroactively edited.

---

## 2026-04-15 — MeshField's V1 Laplacian operator is not a pointwise approximation

**Context.** Week 3 Task 5 added a `MeshField` Field-ABC subclass backed
by a scikit-fem `Basis` and DOF vector. The plan scaffolded the Galerkin
Laplacian as `lap = -M⁻¹ K u`, presented as a simple L² projection. A
human code review caught that this formula is not correct and that the
shipped fix (Dirichlet-condensed projection) computes a *different*
quantity from what the method name `.laplacian()` implies.

**What the plan said.** `lap_dofs = -spsolve(M, K @ self._dofs)` —
described as "the Galerkin projection of the continuous Laplacian onto
the FE space."

**Why it was wrong.** Integration by parts against FE test functions
gives `(∇u, ∇v) = (-Δu, v) + ∫_{∂Ω} (∂u/∂n) v dS`. The raw formula
drops the boundary-flux term, which for any field with non-zero normal
derivative on the boundary contributes large values in stiffness-matrix
rows whose basis supports touch `∂Ω`. When those rows are fed through
`M⁻¹`, the mass matrix couples interior and boundary DOFs and smears
the pollution globally — not just at the boundary. Numerical
verification on `u = sin(πx) sin(πy)` at P2 refine=4 showed interior
relative error ~260% for the plan's formula, not a discretization
artifact but a fundamental mismatch.

**What V1 ships.** `MeshField.laplacian_l2_projected_zero_trace()` —
the L² projection of `Δu` onto the zero-trace FE subspace `V_{h,0}`,
computed via `skfem.condense` to hard-pin boundary DOFs of the output
to zero:

    M_II lap_I = -(K u)_I    on interior DOFs
    lap_B      = 0            on boundary DOFs

This operator converges at the expected O(h²) rate on smooth analytical
solutions (refinement test: 4.1e-2 → 1.2e-2 → 3.1e-3 at refine=3/4/5).
For non-smooth inputs the rate may be lower; the docstring documents
this.

**Why the rename.** The Field ABC's `.laplacian()` contract implies
pointwise semantics — `GridField` and `CallableField` both satisfy that
contract via FD/spectral or autograd computation at every point. The V1
FE operator does **not** satisfy the pointwise contract: interior
values are an O(h²) L² projection, and boundary values are
structurally 0 regardless of the true boundary value of `Δu`.
Leaving the V1 operator named `.laplacian()` would silently break the
ABC promise in the most invisible way possible — downstream rules
would consume the result as if it were pointwise Δu and quietly get
wrong answers on any field with non-zero boundary Δu.

`MeshField.laplacian()` therefore raises `NotImplementedError` with a
message pointing at `laplacian_l2_projected_zero_trace()` and
explaining that the rename is a correctness decision, not API
housekeeping. V1.1 may add a true pointwise `.laplacian()` via
superconvergent patch recovery or similar — that work is tracked as a
backlog item and deliberately out of Week-3 scope.

**Downstream rule constraint.** Task 6's `PH-CON-004` must handle the
boundary artifact structurally, not by documentation. V1 excludes
boundary elements from the per-element residual computation entirely.
Interpreting boundary-element residuals as true conservation violation
would be wrong because they reflect the zero-pinning artifact rather
than any property of the input field.

## Takeaway for future plans

**Spikes validate tooling, not formulas.** The plan's defence for
`-M⁻¹ K u` was "this matches the scikit-fem spike at commit 941658d."
The spike was a Week-1 Day-2 viability test: assemble Poisson
stiffness, solve a Dirichlet BVP, verify O(h²) convergence. That
validated scikit-fem's capacity to solve Dirichlet BVPs at the expected
rate. It did **not** validate the independent operation of applying a
Galerkin Laplacian to a non-BVP input. Those are different
computations that happen to reuse some of the same linear-algebra
pieces. Future plans should not cite spike success as evidence that a
formula derived from spike machinery is correct for a different
operation.

**Cite the target operation, not the matrices.** The plan named "the
stiffness matrix `K` and the mass matrix `M`" and claimed the
computation as "a Galerkin projection." Naming the ingredients without
naming the target function space is how this deviation slipped
through: K and M are the right ingredients for many operations; which
operation they're assembling depends on what test space you project
against and how boundary data is handled. V2 plans for mesh-based
rules should explicitly name the target (pointwise Laplacian via SPR,
L² projection onto `V_{h,0}`, H⁻¹ Riesz representative against
`H¹_0`, etc.) and pair the name with a numerical sanity check that
demonstrates the formula works on a test input whose correct answer is
known.

**Distinct mathematical operations deserve distinct method names.**
Method naming is the last line of defence against silent correctness
regressions for consumers who don't read implementation source. If two
FE operations share a function name because they use similar
ingredients, a downstream rule can silently consume the wrong one. For
physics-lint specifically, any operation whose output space,
projection type, or boundary semantics differs from the ABC contract
must get its own method name — not an overload on the ABC method.

**Every non-trivial mathematical operation needs a Day-1 numerical sanity check, not just mesh operations.** This applies to any formula that (a) compares two quantities computed via different paths, (b) defines a norm / operator / projection against a contract a downstream rule must satisfy, or (c) uses a method name implying semantics the code might not actually satisfy. The sanity check belongs in the plan's task text (so the implementer encounters it before writing code), not just in the test file (where it surfaces only after the fact). It should give a concrete input whose correct output is knowable without reading the implementation. The MeshField deviation was caught empirically *after* shipping because the committed test case (`sin(πx) sin(πy)` with zero boundary trace of `Δu`) accidentally masked the artifact; a Day-1 sanity check on `x(1-x)y(1-y)` would have caught it before any code was written. **For all future plans (V1.1 and V2 alike):** every non-trivial operation in the plan pairs with a numerical contract the implementer runs and reports before declaring the task complete.
