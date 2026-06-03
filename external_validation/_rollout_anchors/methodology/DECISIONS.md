# DECISIONS.md — `feature/rollout-anchors` branch

In-flight decision log per design-spec §1.5 / §9 item 4 (snapshot at merge
into `physics-lint/external_validation/_rollout_anchors/docs/DECISIONS.md`
per spec §1.5 option (b)). Target ≥10 entries on this branch by writeup
time per plan §3.3 + §10 item 4; the wider 20-entry target lives at the
`feature/rollout-anchors` + `chore/v1.0-stub-resolution` combined level.

Entries are append-only. Format: ISO-date, title, body, then the commit
SHA(s) the decision is realized in (where applicable).

---

## D0-01 — 2026-05-04 — Branch source: `master`, not `main`

**Question.** Plan §10 item 1 + the executing-agent notes both say "create
`feature/rollout-anchors` from physics-lint's main." physics-lint's actual
default branch is `master` (`git branch --show-current` → `master`); there
is no `main` branch. Branching from a non-existent branch would have
silently created an empty branch.

**Decision.** Branch from `master` (the actual default). User confirmed
before execution: *"Yes, master. v3 names the target branch
(`feature/rollout-anchors`) but not the source; physics-lint's default
being `master` means branch from `master`."*

**Realized.** `feature/rollout-anchors` created at the tip of `master`,
SHA `e50c134` (pre-rollout-anchors).

**Sub-bullet — 2026-05-11 — case-study-02 branch base.** The Phase 1
plan (`methodology/docs/2026-05-11-case-study-02-physicsnemo-mgn-phase-1-plan.md`)
required choosing a base for `feature/case-study-02-physicsnemo-mgn`
depending on PR #9 status. PR #9 (`feature/rung-4c-substrate-class-extension`)
was still **OPEN + MERGEABLE** at the start of Phase 1 execution
(2026-05-11), so the new branch was created off the rung-4c branch
head at SHA `891a13e` (= PR #9 head; tip of
`feature/rung-4c-substrate-class-extension`). Plan dependency: D0-2X
verdicts (Phase 1) layer on top of D0-22 substrate-class taxonomy.

---

## D0-02 — 2026-05-04 — Audit Q1 (Gate A) deferred to Modal session

**Question.** Spec §5.1 / plan §2.1 Audit Q1: try
`MeshField(basis=reconstructed_basis, dofs=node_values_at_t)` on one
PhysicsNeMo NGC sample timestep; verdict feeds Gate A (PASS / PARTIAL /
FAIL). This requires:

1. NVIDIA NGC API key registration + storage in Modal secret (spec §5.2 /
   plan §2.3) — not yet done at Day 0 entry.
2. A downloaded NGC sample timestep — requires (1).
3. The `nvidia-physicsnemo` + `dgl` deps (the `[validation-rollout]`
   extra) installed in a sandbox.

**Decision.** Audit Q1 is deferred to the joint user-and-agent Modal
session that begins Day 1 / Day 2 work. Gate A verdict therefore lands
on Day 2 alongside Gate D (PhysicsNeMo NGC checkpoint usability). Until
then, the cover-letter paragraph variant choice (Appendix A.1 / A.2 /
A.3 / A.4) is also deferred.

This deferral does **not** unblock or block anything Day-0-CPU-only.
Gate A's verdict feeds the *mesh-side* claim shape (cover-letter
paragraph, "what we are NOT claiming"); the particle side (Day 1
LagrangeBench) is gated only on Gate B (passed) + Gate C (Day 1).

**Realized (2026-05-12, CS02 Phase 1 Task 6).** Gate A = **PASS**. The deferred Audit Q1 was fulfilled by `audit_gate_a_pyg_to_meshfield` (Modal, CPU-only): on the first cylinder_flow `test.tfrecord` trajectory (1923 nodes / 3612 triangle cells, domain bbox `x ∈ [0, 1.6]`, `y ∈ [0, 0.41]`), `skfem.MeshTri(p=mesh_pos.T, t=cells.T)` + `skfem.Basis(ElementTriP1())` (scikit-fem 12.0.1) reconstruct cleanly — 1923 P1 DOFs. The PyG→scikit-fem coercion needed only a `.T` axis swap (PyG returns `mesh_pos` as `(n,2)` / `cells` as `(n,3)`; scikit-fem `MeshTri` wants `(2,n)` / `(3,n)`) and a dtype cast (`mesh_pos` float32→float64, `cells` int32→int64) — no Pattern-B surprises; a loud pre-flight assert on `(1923,2)`/`(3612,3)` shapes + dtypes guards that surface. Note the v2.0.0 loader returns PyG `Data`, not DGL graphs, so the mesh comes off the test-split `__getitem__` tuple (`ds[0][0]["mesh_pos"]`, `ds[0][1]`), not `dgl` `ndata`/`edata` — see D0-23 verdict 2/3. Cover-letter consequence: the mesh-side-not-in-scope variant (the Gate-A skeleton's "Appendix A.4" SKIP path) does **not** apply — the mesh side is materializable; the `MeshField` wrapper itself is verified in Task 12. (Which exact Appendix-A letter the PASS path selects is for whoever drafts the cover letter — recorded here only that it is the in-scope one, not the SKIP one.) Full findings: `02-physicsnemo-mgn/preflight/gate_a_audit.json`.

---

## D0-03 — 2026-05-04 — Audit Q2: PH-CON-001 input-domain compatibility

**Question.** Spec §5.1 Audit Q2: read PH-CON-001's source; if its
emitted quantity makes assumptions about the field's domain that
(eventual) NGC vortex-shedding output violates, document the mismatch.

**Outcome (read-only, no NGC access required).** PH-CON-001 at
`src/physics_lint/rules/ph_con_001.py:46-48` requires `spec.pde ==
"heat"`; under any other PDE label it returns SKIPPED with
`"PH-CON-001 heat-only in V1; got <pde>"`. NGC vortex shedding is
**incompressible Navier-Stokes**, not heat — a strict spec mismatch.
Three actionable consequences:

1. The mesh case study cannot directly call PH-CON-001 on
   PhysicsNeMo MGN output via the public CLI without either
   (a) widening PH-CON-001's `pde` accept-list (out of scope per
   spec §1.1 — public-API change), or
   (b) adapting the input by relabelling `spec.pde = "heat"` and
   accepting that the rule's message text will be slightly
   misleading, or
   (c) routing PH-CON-001 through the harness path on the mesh
   side (parallel to the particle harness), accepting the loss of
   the "ran without rule modification on mesh" claim for
   PH-CON-001 specifically.

2. Plan §4.2 step 4 says "PH-CON-001 (mass) on vortex shedding:
   divergence-free check on velocity field." The rule does **not**
   compute a divergence-free check on velocity; it computes a
   mass-balance defect on a heat-equation scalar. The plan's
   wording is imprecise: the *mathematical content* (∫ρ → constant
   under conservative flow) generalises, but the *current rule
   implementation* does not.

3. The cleanest resolution that preserves the spec's "without
   modification" claim narrowly is: route PH-CON-001 through the
   mesh harness on the NS side (option c above), and document
   that PH-CON-001 on NS data is a *different rule kernel* than
   on heat data — same emitted-quantity name, different
   implementation. This is honest and adds another bullet to the
   "what we are NOT claiming" list.

**Decision.** Adopt option (c) for the Day 2 work; flag in
`_rollout_anchors/README.md` "What we are NOT claiming" and reproduce
this entry's reasoning in `_rollout_anchors/02-physicsnemo-mgn/README.md`.
Do **not** modify PH-CON-001's `pde` accept-list (out of scope).

**Realized.** Day 2; flagged here for the Day 2 agent to act on.

---

## D0-04 — 2026-05-04 — Audit Q3: particle-harness model-loading split

**Question.** Spec §5.1 Audit Q3: confirm the particle harness's two
halves work end-to-end on a stub configuration:
- read-only path (PH-CON-* on cached `.npz`, no JAX dependency)
- model-loading path (PH-SYM-* requires JAX/Haiku from
  `[validation-rollout]` extra)

**Outcome (Day-0, structure-only — full validation deferred).** The
read-only path is implemented and Gate-B-validated:
`particle_rollout_adapter.py` exposes `gridify`, `c4_static_defect`,
`reflection_static_defect`, `total_mass`, `kinetic_energy`,
`mass_defect`. None require JAX. All exercised by the Gate B fixture
suite which runs cleanly on the base `pip install physics-lint`
environment (numpy, scipy, scikit-fem, ...).

The model-loading path is **deliberately not implemented** at Day 0
per the executing-agent rule: "no speculative stubs past Gate B" —
bad stubs become ground truth on next read, and the model-loading
path requires:

1. A real LagrangeBench checkpoint to load (Day 1, on Modal).
2. JAX/Haiku running on Modal A100 (hour-2 JAX micro-gate, plan §7).
3. The `[validation-rollout]` pyproject.toml extra defined and
   installable (plan §10 item 9 / spec §9 item 10) — not yet
   committed.

The structure of the split is documented in
`particle_rollout_adapter.py`'s top-of-file docstring and the
"Model-loading path" footer comment block; Day 1 will land the
implementation in a separate commit on the same branch once a
checkpoint is verifiably available.

**Decision.** Day-0 outcome is "structural confirmation only" —
the dual-half split is realized in the module structure and the
read-only half is validated; the model-loading half is gated on
Day 1 conditions and intentionally absent from this commit set.

**Realized.** `30db5fd` (adapter Day-0 surfaces commit).

### D0-04a — 2026-05-04 — Spec §4.2 ambiguity on the particle-path C₄ defect (formal interpretation)

**Surfaced ambiguity.** Spec §4.2 fixture #1 prescribes the
particle-harness path as: *"apply C₄ to particle positions and
velocities, compute ε_C4 directly."* That phrasing reads naturally as
a per-index computation: take each particle's position, apply the
rotation matrix, compare to the corresponding original particle's
position by index, sum the squared differences, normalise. But for an
honest C₄ orbit (4 particles at the corners of a square), per-index
comparison gives ε = O(1) — the rotation *permutes* particle indices,
so the rotated array is a cyclic relabelling of the original, and
``||R x − x|| / ||x||`` picks up the relabelling, not any genuine
symmetry breaking. The spec's expected outcome ("both paths emit
ε_C4 ≤ 10⁻⁶ ... by construction") is therefore inconsistent with the
naïve per-index reading.

**Harness's resolution.** Both paths route through the gridded-density
representation produced by ``gridify`` (Gaussian-bump kernel-density
estimator on the shared periodic 64×64 grid). Both paths then apply
``np.rot90`` to the gridded field and compute the same
denominator-stabilised relative L^2 (``equivariance_error_np``). For
an exactly C₄-invariant configuration constructed on a discrete-rot90
cell-index orbit, the gridified density is itself exactly invariant
under ``np.rot90``, and the resulting ε is ~ machine epsilon (observed:
1.066e-21). For the perturbed fixture, the gridded density picks up
an O(δ/bandwidth) defect, and both paths report the same defect — by
construction of the harness wrapper, they apply identical operations
to the same array.

**Consequence for what Gate B is.** Gate B as run on Day 0 is a
**regression test on the gridify+rot90 pipeline**, not a cross-method
epistemic check. The bit-identical ε = 0.000e+00 result is the
by-design outcome of routing both paths through identical operations;
a non-zero result would indicate a regression in either ``gridify``,
the harness's call into ``np.rot90`` / ``np.flip``, the harness's
``_relative_l2`` denominator-stabilisation, or the public rule's
emission semantics. Reading Gate B's PASS as evidence that "the
harness's particle-side ε agrees with what the rule emits on
genuinely independent computations" is an overclaim; the genuine
cross-method check is the Day 1 model-loading path's per-index ε on
``f(x_0)`` vs ``R^{-1} f(R x_0)``, which is well-defined because
trained-model rollouts preserve particle identity across the
identity-vs-rotated rollout pair.

**Decision.** Document the interpretation explicitly in two places:
SCHEMA.md §4.1 gains a "Reader's note" subsection making the
regression-test framing crisp (rather than burying it in §4.1's
existing "by-design" rationalisation), and this DECISIONS.md entry
records the interpretation for the merge-time snapshot. The spec text
is **not** edited (it is the authoritative source on intent; the
ambiguity is a phrasing issue, not a design issue), but anyone
reading the spec §4.2 ε ≤ 10⁻⁶ expectation should follow the
cross-reference to here.

**Realized.** Doc commit A (DECISIONS.md update) + Doc commit B
(SCHEMA.md §4.1 Reader's note).

---

## D0-05 — 2026-05-04 — Gate B verdict: PASS

**Question.** Run the controlled-fixture validation per spec §4.2 / §6
Gate B; classify against the pre-registered tolerance bands in
`SCHEMA.md` §4.1.

**Outcome.** Gate B run on Day 0 (CPU only, 2026-05-04):

```
=== Gate B verdict record ===
  PH-SYM-001  c4_invariant_4particle    ε_harness_vs_public = 0.000e+00
  PH-SYM-002  c4_invariant_4particle    ε_harness_vs_public = 0.000e+00
  PH-SYM-001  c4_perturbed_4particle    ε_harness_vs_public = 0.000e+00
  PH-SYM-002  c4_perturbed_4particle    ε_harness_vs_public = 0.000e+00
  worst ε = 0.000e+00  ->  Gate B verdict: PASS
```

Worst observed ε = 0 (bit-identical floating-point output). All eight
test functions pass:

- 4 cross-path agreement assertions (≤ 1e-4)
- 1 fixture-construction tolerance check (fixture #1 ε ≤ 1e-6 absolute;
  observed 1.066e-21)
- 1 perturbation linear-response check (doubling delta ~doubles ε)
- 1 perturbation-isolation check (only particle 0 moves)
- 1 orbit-cell construction check (fixture #1 lives on exact
  discrete-rot90 cell-index orbits)

Per spec §6 always-on rule (audit-before-amend): ε = 0 to all digits
is the *expected* outcome under the SCHEMA.md §4.1 by-design
rationalisation — both paths apply identical np.rot90 and relative-L^2
operations to the same gridified field, so bit-identical output is
not a surprise. No measurement audit triggered.

**Decision.** Gate B = PASS. Day 1 LagrangeBench rollouts are
unblocked from this gate (still gated on Modal authentication +
Audit Q1 outcome / Gate A + JAX micro-gate).

PH-CON-001 cross-path comparison was deliberately omitted from Gate B
per the rationale in `mass_conservation_fixture.py` (the public PH-CON-001
rule consumes a heat-equation 3D GridField, not a particle
configuration; cross-path coverage on PH-CON-001 belongs to Day 1+
when a heat-equation MMS fixture or a LagrangeBench rollout produces
the appropriate 3D representation). Flagged for Day 1 follow-up.

**Realized.** `461d09d` (Gate B test + verdict commit).

---

## D0-06 — 2026-05-04 — CI workflow path: repo root, not nested

**Question.** Spec §9 item 6 + plan §10 item 5 reference path
`external_validation/_rollout_anchors/.github/workflows/rollout-anchors.yml`.
GitHub Actions only reads workflows from the repository root's
`.github/workflows/` directory; a nested workflow would never trigger.

**Decision.** The actual workflow lives at the repository root:
`.github/workflows/rollout-anchors.yml`. The misleading nested
`_rollout_anchors/.github/workflows/` skeleton directory created
during initial scaffolding was removed before commit (v).

The spec/plan path is treated as a *documentation organisational
choice* — i.e., "the workflow conceptually belongs to the rollout-
anchors subdirectory". The path text in spec §9 item 6 / plan §10
item 5 is left unchanged (the spec is authoritative on intent, not
on platform plumbing); this DECISIONS.md entry plus the
path-discrepancy note in the workflow file's leading comment are
the two reading aids that prevent future confusion.

**Realized.** `c704534` (workflow skeleton commit).

---

## D0-07 — 2026-05-04 — Day 1 model-loading-path threshold pre-registration

**Question.** Spec §4.4 / SCHEMA.md §4.2 pre-register the trained-model
equivariance band ``ε_rot ≤ 10⁻⁵`` PASS / 10⁻⁵ < ε ≤ 10⁻² APPROXIMATE
/ ε > 10⁻² FAIL for the Day 1 SEGNN-vs-GNS comparison. But the
SCHEMA.md §4.2 prose on Day 0 was written with one foot in the static
fixture computation and one foot in the rollout computation; it does
not explicitly state that the threshold applies to the **per-index**
relative L^2 between ``f(x_0)`` and ``R^{-1} f(R x_0)`` on
trajectory-aligned model output, distinct from any harness-vs-public
agreement check.

This distinction is load-bearing. The Gate B 10⁻⁴ threshold tests
**agreement between two implementations of the same metric**; the
§4.4 / §4.2 trained-model threshold tests **the model's own
equivariance defect**, which is allowed to be non-zero (that is the
point — SEGNN ≪ GNS is the headline). Conflating the two on Day 1
would either mis-classify a genuine equivariance defect as a Gate B
FAIL or hide a Gate B FAIL behind the trained-model band. The two
verdicts answer different questions.

**Pre-registration.** Effective Day 1 — i.e., before any SEGNN / GNS
rollout runs — the trained-model band applies to ``ε_rot`` and
``ε_refl`` computed as **the relative L^2 of f(x_0) and R^{-1} f(R x_0)
matched per particle index across the full trajectory** (or the union
of trajectories when running ``eval.n_trajs > 1``):

| ε_rot or ε_refl (per-index, trajectory-aligned) | Verdict |
|-------------------------------------------------|---------|
| ≤ 10⁻⁵ | PASS — machine-precision equivariance. SEGNN expected. |
| 10⁻⁵ < ε ≤ 10⁻² | APPROXIMATE — flagged, in approximate-equivariance band. GNS expected. |
| > 10⁻² | FAIL — equivariance broken. |

If pilot data (5-trajectory pre-run, plan §6 / spec §8 risk-register
mitigation) shows SEGNN landing at, e.g., 10⁻⁴ instead of 10⁻⁶, the
threshold is **not** silently amended; the divergence is logged here
as a new D0-08+ entry citing the discrepancy explicitly, and the band
may be amended only with the discrepancy reproduced verbatim in the
writeup.

The Gate B 10⁻⁴ band remains in force for the **harness-vs-public-API
agreement check** independently of this entry; the two thresholds are
not interchangeable. SCHEMA.md §4 gains a one-line cross-reference
between §4.1 (Gate B) and §4.2 (trained-model) noting the
"different questions" framing.

**Decision.** Pre-registered above. Day 1's per-index computation
landing in the JAX/Haiku model-loading path of
``particle_rollout_adapter.py`` will assert against this band
explicitly (verdict logging analogous to ``test_record_gate_b_eps_values``
in ``test_harness_vs_public_api.py``, but per-trajectory).

**Realized.** Doc commit A (this entry) + Doc commit B (SCHEMA.md §4
cross-reference). Day 1 implementation of the trained-model assertion
realizes the band in code on a separate commit.

---

## D0-08 — 2026-05-04 — KE-rest skip-with-reason threshold pre-registration

**Question.** ``energy_drift`` and ``dissipation_sign_violation`` in the
particle harness's read-only path normalize by ``KE(0)`` and ``max(KE)``
respectively, with an ``eps = 1e-12`` denominator floor to avoid divide-
by-zero. The eps-floor inflates the emitted defect to an arbitrarily
large finite number when the actual reference KE is below it (e.g., the
synthetic energy-growth fixture starts all particles at rest, so KE(0)
= 0 and ``energy_drift = max(KE)/eps`` is meaningless).

This was surfaced in the Day-0.5 hand-back as an implementation-detail
wart that should be made explicit before Day 1 LagrangeBench rollouts.
If a real LagrangeBench dataset starts with all particles at rest
(unlikely for TGV — nonzero IC by construction — but possible for other
rollouts), discovering this mid-Day-1 forces a mid-execution decision
under time pressure. Pre-handling is the right discipline.

**Decision.** Adopt the **skip-with-reason** pattern (matching physics-
lint's existing PH-CON-001 SKIPPED behaviour on `pde != "heat"`) over
NaN propagation. Skip-with-reason rationale per the user's review:
(a) NaN propagation is a class of bug whose design cost outweighs its
benefit; (b) the cross-stack table can render a SKIP cell with a
footnote citing this entry, which is more legible than a NaN cell;
(c) skip-with-reason matches the existing physics-lint idiom for V1
rules with input-domain restrictions, keeping the harness behaviour
in pattern.

**Threshold (pre-registered, absolute, v1).**

    KE_REST_THRESHOLD = 1e-10   # absolute, in the dataset's natural KE units

- ``energy_drift(rollout)`` skips with reason ``"KE(0) < 1e-10 (rollout
  starts at rest; relative drift undefined)"`` when ``KE(t=0) <
  KE_REST_THRESHOLD``.
- ``dissipation_sign_violation(rollout)`` skips with reason
  ``"max(KE) < 1e-10 (trajectory has no kinetic energy; dissipation
  question undefined)"`` when ``max_t KE(t) < KE_REST_THRESHOLD``.
- ``mass_conservation_defect(rollout)`` is **not** subject to a skip
  threshold: ``M(0) = sum(particle_mass)`` is strictly positive in any
  physical configuration. Symmetry of the return type
  (``HarnessDefect``) is preserved across all three functions for
  downstream SARIF emission consistency.

**Threshold form.** Absolute (1e-10 in the dataset's natural KE units),
not relative-within-rollout (e.g., ``KE(0) < 1e-10 * max(KE)``). The
absolute form is dataset-specific and acknowledged as such; a v1.1
escape hatch may switch to the relative-within-rollout form when
cross-dataset comparison becomes load-bearing. The simpler absolute
form is fine for v1 because (a) physical SPH rollouts have KE(0) of
order unity in their natural units, well above 1e-10; (b) the only
failure mode this threshold catches at v1 is "all particles at rest at
t=0", which is a categorical input-domain mismatch rather than a
calibration gradient. If a real dataset surfaces a KE(0) in the
1e-10 to 1e-6 range, log a new D0-09+ entry citing the discrepancy
and amend the threshold rather than silently shifting it.

**Pre-registration cite.** This entry is pre-registered before any
LagrangeBench rollout runs. If pilot data shows KE(0) within an order
of magnitude of the threshold, the divergence is logged here as a
new D0-09+ entry and the threshold may be amended only with the
discrepancy reproduced verbatim in the writeup — same discipline as
D0-07's per-index threshold.

**Realized.** This entry (DECISIONS.md) + Code commit D (HarnessDefect
dataclass, skip-with-reason in `particle_rollout_adapter.py`, tests
updated) + SCHEMA.md §4 update mirroring this entry.

---

## D0-09 — 2026-05-04 — Mesh FD-noise tolerance pre-registration

**Question.** ``test_uniform_channel_mass_conservation_zero`` in
``test_mesh_read_only_path.py`` asserts ``result.value < 1e-10`` as
the bound on the harness's emitted divergence on a constant-velocity
fixture. The 1e-10 magic number lived only in the test code on the
Day-0.5 commit (``bf55f6d``), not anywhere a drift-guard test could
catch silent amendment — unlike the Gate B 1e-4, the Day 1 per-index
1e-5 / 1e-2, and the KE-rest 1e-10 thresholds, all of which are
pre-registered with corresponding drift-guard tests.

The user surfaced this in the Day-0.5 review pass: an un-pre-registered
mesh-side threshold is the kind of silent miscalibration that's
hardest to catch later, particularly during Day 2's mesh-harness rule
runs where a regression in the FD divergence operator's edge handling
could push the noise floor above 1e-10 without anyone noticing.

**Outcome.** ``np.gradient`` on a 32x16 grid of constant-velocity
data produces edge-stencil artefacts of order 1e-15 (the second-order
forward / backward FD at the boundary picks up ~ machine epsilon
rounding). 1e-10 sits five orders of magnitude above the observed
noise floor — generous enough to absorb implementation-level FD
variation across numpy versions and grid sizes, tight enough to
catch any real divergence violation by a wide margin (the deliberate
violation case at alpha=0.1 produces 0.01-0.5).

**Decision (pre-registered).**

    MESH_FD_NOISE_TOLERANCE = 1e-10   # absolute, dimensionless

This is the upper bound on
``mass_conservation_defect_on_mesh(rollout).value`` for a rollout
whose underlying continuous velocity field is divergence-free (e.g.,
the synthetic uniform channel-flow fixture). Values above the
threshold indicate either (a) the underlying field has genuine
non-zero divergence, or (b) the FD operator's edge handling is
miscalibrated relative to the public ``np.gradient`` semantics this
harness depends on.

The constant is pulled into
``external_validation/_rollout_anchors/_harness/mesh_rollout_adapter.py``
as a module-level named value so the drift-guard test
``test_mesh_fd_noise_tolerance_matches_pre_registration`` can assert
the value matches this entry verbatim. If a future numpy version
changes ``np.gradient``'s edge-stencil behaviour and the noise floor
moves, log a D0-12+ entry citing the discrepancy and amend the
threshold — do not silently shift in code.

**Realized.** This entry + Code commit (named constant in
``mesh_rollout_adapter.py``, drift-guard test in
``test_mesh_read_only_path.py``).

---

## D0-10 — 2026-05-04 — Hour-2 JAX micro-gate escalation pre-registration

**Question.** Plan §7 hour-2 micro-gate fires at hour 2 of Day 1: run
``jax.devices()`` inside the Modal container; FAIL if the call returns
CPU only or errors out. The plan's mitigation text reads: "*Pivot at
hour 2 to either: (a) JAX-CPU read-only mode for PH-CON-* path
(skip PH-SYM-* on Day 1, defer to Day 1.5 if buffer permits); or (b)
Modal-image debugging side-quest (capped at 2h before falling back to
(a)).*" The plan presents (a) and (b) as a runtime choice, not a
pre-registered escalation criterion. JAX-CUDA setup is famously
prone to debugging-rabbit-hole failure modes; choosing (a) vs (b) on
the fly under time pressure is exactly the failure shape this
project's pre-registration discipline exists to prevent.

**Decision (pre-registered).** The hour-2 FAIL escalation is **(a) by
default, no Modal-image debugging without explicit user re-authorization**.

Concretely:

- If ``jax.devices()`` does not return at least one A100 device by
  hour 2 of Day 1, the agent pivots immediately to JAX-CPU read-only
  mode: continue with the synthetic-rollout-validated PH-CON-001/002/003
  read-only path (already landed on the branch as of ``a00433d`` and
  ``bf55f6d``) on cached LagrangeBench rollouts (option (a) of
  D0-02 / spec §1.3 / plan §3.2 step 4 read-only half).
- The PH-SYM-001/002 model-loading path (the JAX/Haiku checkpoint
  inference + rotated-input rollouts pre-registered in D0-07) defers
  to **Day 1.5 if buffer permits, otherwise Day 3 buffer**, with the
  trimmed cross-stack claim sliding from "structural identities held
  for finite-group equivariance + conservation balance" (spec §0
  full headline) to "structural-conservation-balance held; equivariance
  deferred" (Gate C FAIL fallback per plan §7).
- **Modal-image debugging side-quest is OUT OF SCOPE without
  explicit user re-authorization.** The 2h cap in the plan is a
  ceiling, not an authorisation; under this pre-registration the
  agent does not enter the side-quest at all without a fresh
  greenlight on the question "is JAX-CUDA-on-Modal worth N more
  hours of CPU-only work given (a) the synthetic rollouts already
  validate the read-only path, (b) the cover-letter paragraph
  variant under Gate C FAIL is documented in spec Appendix A.4,
  and (c) the project enters round 10+ if we keep going?".

**Why (a) is the default.** Three reasons, in order of weight:

1. **The synthetic-rollout-validated read-only path covers
   PH-CON-001/002/003 cleanly without Modal A100.** Day 0.5 + the
   review-pass commits land this path before Day 1 starts; the only
   thing the JAX side adds is PH-SYM-001/002 model-loading, which is
   one rule pair of seven in scope, not the headline.
2. **Modal-image debugging is the highest-variance failure mode the
   project faces.** Past sessions have lost 4-6 hours to JAX-CUDA
   version mismatches, Modal Image cache misses, and CUDA driver
   incompatibilities — none of which produce useful artefacts for
   the writeup.
3. **The cover-letter Gate C FAIL fallback is pre-written.** Spec
   Appendix A.4 covers the both-harnesses-no-public-mesh case;
   adapting it for the "single-stack-particle-side, conservation-
   only" case is a 30-min writeup edit, not a plan rewrite.

**Realized.** This entry. No code lands until Day 1 hour 2 fires the
gate.

---

## D0-11 — 2026-05-04 — Day 2 hour-1 NGC audit decision criterion pre-registration

**Question.** Day 2 hour-0–1 work installs the PhysicsNeMo container,
downloads the NGC ``modulus_ns_meshgraphnet:v0.1`` checkpoint, and
runs the shipped sample inference. By hour 1 the agent has the
checkpoint's actual output schema in front of it. Two questions
determine whether the mesh-harness work I landed in ``bf55f6d``
applies as-written or needs surgery:

1. **Does ``node_values["velocity"]`` exist in the NGC vortex
   shedding sample output, or is velocity a derived quantity?**
   PhysicsNeMo MGN checkpoints sometimes emit primitive variables
   (density, momentum) and require post-processing to recover
   velocity (v = momentum / density). If velocity is derived, the
   harness's ``_expect_velocity`` precondition lookup misses, and
   ``mass_conservation_defect_on_mesh`` SKIPS — but the
   mass-conservation identity is still well-defined; it just needs
   a reconstruction pass before being fed to the harness.
2. **What is the graph topology — is ``edge_index`` the connectivity
   that scikit-fem ``Basis`` would need for ``MeshField``
   construction (Gate A PASS), or is it DGL-native and topologically
   incompatible with scikit-fem (Gate A PARTIAL → GridField
   resampling, or Gate A FAIL → both-harnesses fallback)?**

The plan's Audit Q1 and Gate A enumerate the verdict logic but do not
pre-register the mesh-harness consequence; without pre-registration,
the agent decides what to do at hour 1 of Day 2 under time pressure
— exactly the failure shape this discipline exists to prevent.

**Decision (pre-registered).** Two-axis decision matrix per the two
questions above:

| Q1: velocity exposure          | Q2: topology                   | Action                                                                     |
|--------------------------------|--------------------------------|-----------------------------------------------------------------------------|
| Direct (``node_values["velocity"]``) | scikit-fem-coercible (Gate A PASS)    | Mesh harness applies as-written; ``mass_conservation_defect_on_mesh`` runs. |
| Direct                         | DGL-native, GridField-resamplable (Gate A PARTIAL) | Resample velocity onto a regular grid (set ``metadata["resampling_applied"] = True`` per SCHEMA.md §2 / D0-03), harness's ``is_regular_grid`` branch fires; cover-letter Appendix A.2 variant. |
| Direct                         | DGL-native, neither (Gate A FAIL) | Harness SKIPS (already landed in ``bf55f6d``); cover-letter Appendix A.4 variant. |
| Derived (``momentum/density``) | scikit-fem-coercible           | Reconstruct velocity in ``run_inference.py``, write back to ``node_values["velocity"]`` before invoking the harness. **Add a 30-min audit budget for the reconstruction step**; if it surfaces edge cases (zero density, missing density field, dimensional mismatch), escalate to user — do NOT improvise. |
| Derived                        | DGL-native, GridField-resamplable | Same as the Direct + PARTIAL row, plus the reconstruction step from the Derived + PASS row. **Cumulative audit budget: 1h.** If it overruns, defer mass conservation to harness-only and route PH-CON-001 through the synthetic-fixture validation only. |
| Derived                        | DGL-native, neither            | Harness SKIPS as in Gate A FAIL row; reconstruction work would not be invoked. |

**Audit budget guardrails.**

- **Total Day 2 hour-1 audit budget: 1h.** If the audit overruns, the
  default action is to scope-trim to whatever the synthetic-rollout
  validation already covers (the uniform-channel and divergence-
  violation fixtures landed in ``bf55f6d``) and document the gap in
  the writeup. Do **not** spend Day 2 hour 2-3 implementing
  reconstruction code under time pressure.
- **What "scikit-fem-coercible" means concretely:** the DGL graph
  must expose either (a) explicit cell connectivity (triangle /
  quad faces enumerable from ``edge_index``) suitable for
  ``skfem.MeshTri`` / ``skfem.MeshQuad`` construction, or (b) a
  documented adapter from PhysicsNeMo's own ``Mesh`` to scikit-fem.
  If neither is available within the 1h audit budget, default to
  PARTIAL.
- **GridField-resampling viability** is established by checking
  whether the cylinder-wake bounding box admits a uniform Cartesian
  resampling at a resolution that preserves the boundary layer
  (resolution ≥ 5 cells across the cylinder diameter). If not,
  default to FAIL.

**Why these audit criteria are pre-registered now, not at Day 2 hour
1.** The audit takes 1h max under good conditions; under bad
conditions (e.g., NGC catalog format change since the v3 plan was
written) it can balloon to 4h+ if the agent keeps "just five more
minutes" trying alternatives. Hard time-cap + default-to-skip is the
discipline that makes mesh-side budget bounded; pre-registering the
matrix here removes the in-session decision burden and lets the
agent execute mechanically.

**Realized.** This entry. Code lands in a separate Day 2 commit on
``feature/rollout-anchors`` once the hour-1 audit returns a verdict.

---

## D0-12 — 2026-05-04 — Pre-registration coverage summary

**Why this entry exists (not a new decision; a durable artifact).** The
preceding eleven entries pre-register every meaningful Day-0-through-Day-2
decision the Modal session can encounter — branch source, deferral
ownership, input-domain compatibility, the read-only / model-loading
split, the Gate B verdict, the CI workflow path, the Day 1 trained-model
band, the KE-rest skip threshold, the mesh FD-noise tolerance, the JAX
hour-2 micro-gate escalation, the Day 2 NGC audit decision matrix.
Reading them in narrative order is the right way to trace *how* the
project arrived at its entry-state for the Modal session; reading them
as a coverage table is the right way to verify *that* every
load-bearing numerical threshold has both a pre-registration cite and
(where the threshold gates a runtime branch) a drift-guard test that
hard-asserts the value matches its pre-registration.

This entry is the coverage table. It is committed before the Modal
session begins so that future-me — and any reviewer of the eventual
writeup's methodology section — can point at one artifact rather than
reconstructing the table from a hand-back message after the fact.

**Coverage table.**

| #     | What it pre-registers / decides                                                                                                                                                       | Pre-reg location                                                              | Drift-guard mechanism                                                                                              |
|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| D0-01 | Branch source: `master`, not `main` (physics-lint default branch is `master`)                                                                                                         | DECISIONS.md D0-01                                                            | n/a — one-shot procedural decision; realized at branch-creation                                                    |
| D0-02 | Audit Q1 / Gate A (MeshField-from-DGL) deferred to joint Modal session                                                                                                                | DECISIONS.md D0-02                                                            | n/a — deferral; verdict lands Day 2 alongside Gate D                                                               |
| D0-03 | PH-CON-001 input-domain mismatch with NS data → route through mesh harness on NS side, do not widen public rule's `pde` accept-list                                                   | DECISIONS.md D0-03 + `02-physicsnemo-mgn/README.md` "What we are NOT claiming" | n/a — routing decision; lands in Day 2 mesh-harness commit set                                                     |
| D0-04 | Particle-harness read-only / model-loading split is structural; Day 0 commits read-only half only (no speculative stubs past Gate B). See also sub-entry D0-04a (Gate B framing).     | DECISIONS.md D0-04 + `particle_rollout_adapter.py` top-of-file docstring      | n/a — structural; read-only half realized via Gate B PASS; model-loading half lands Day 1                          |
| D0-04a | Gate B is a regression test on the gridify+rot90 pipeline, not a cross-method epistemic check; the genuine cross-method check is Day 1's per-index ε on `f(x_0)` vs `R⁻¹ f(R x_0)`    | DECISIONS.md D0-04a + SCHEMA.md §4.1 "Reader's note"                          | n/a — interpretive ruling on spec §4.2; realized via Day 1 model-loading-path implementation                       |
| D0-05 | Gate B harness-vs-public band: ε ≤ 1e-4 PASS                                                                                                                                          | DECISIONS.md D0-05 + SCHEMA.md §4.1                                           | hard assert in `test_harness_vs_public_api.py` (Day 0 verdict: PASS, worst observed ε = 0.000e+00)                 |
| D0-06 | CI workflow lives at repo root `.github/workflows/rollout-anchors.yml`, not nested under `_rollout_anchors/.github/workflows/`                                                        | DECISIONS.md D0-06 + workflow file leading comment                            | n/a — platform-plumbing correction; trigger semantics validated on first `workflow_dispatch` run                   |
| D0-07 | Day 1 trained-model per-index band: ε ≤ 10⁻⁵ PASS / 10⁻⁵ < ε ≤ 10⁻² APPROXIMATE / ε > 10⁻² FAIL on `f(x_0)` vs `R⁻¹ f(R x_0)`, matched per particle index across the full trajectory  | DECISIONS.md D0-07 + SCHEMA.md §4.2 + cross-reference between SCHEMA §4.1/§4.2 | per-trajectory verdict assertion lands in JAX/Haiku model-loading path of `particle_rollout_adapter.py` (Day 1)    |
| D0-08 | KE-rest skip-with-reason threshold: `KE_REST_THRESHOLD = 1e-10` absolute (in dataset's natural KE units); skip-with-reason pattern over NaN propagation                               | DECISIONS.md D0-08 + SCHEMA.md §4.4                                           | `test_ke_rest_threshold_matches_pre_registration` hard-asserts the constant's value matches D0-08 verbatim (PASS)  |
| D0-09 | Mesh FD-noise tolerance: `MESH_FD_NOISE_TOLERANCE = 1e-10` absolute (upper bound on `mass_conservation_defect_on_mesh.value` for divergence-free fields)                              | DECISIONS.md D0-09 + `mesh_rollout_adapter.py` module-level named constant + docstring | `test_mesh_fd_noise_tolerance_matches_pre_registration` hard-asserts the constant's value matches D0-09 verbatim (PASS) |
| D0-10 | Hour-2 JAX micro-gate FAIL escalation: JAX-CPU read-only pivot by default; **no Modal-image debugging side-quest without explicit user re-authorization** (the 2h plan cap is a ceiling, not an authorisation) | DECISIONS.md D0-10                                                            | fires Day 1 hour 2 (`jax.devices()` returns at least one A100, or pivot)                                           |
| D0-11 | Day 2 hour-1 NGC audit decision matrix: 6-cell matrix on (velocity-direct vs derived) × (scikit-fem-coercible vs GridField-resamplable vs neither); **1h hard audit budget**          | DECISIONS.md D0-11                                                            | fires Day 2 hour 1; default-to-skip if budget overruns                                                             |
| D0-13 | GPU-class defaults: T4 (hour-0 micro-gate) / A10G (Day 1 §3.2 step 3 inference) / A100 (OOM fallback only); refines D0-10's "A100" specifier to "any CUDA GPU" (epistemic test is JAX-sees-CUDA, not A100-specifically) | DECISIONS.md D0-13 + `01-lagrangebench/modal_app.py` `gpu=` kwarg               | `test_modal_app_gpu_class_matches_d0_13_pre_registration` hard-asserts the `gpu=` literal in `modal_app.py` matches D0-13's stage-1 default ("T4") |
| D0-14 | Modal Volume layout for rollout-anchors artifacts: `/vol/{checkpoints,datasets,rollouts}/<provider>/...` with `<git_sha>` suffix on rollout filenames for reproducibility + cache invalidation | DECISIONS.md D0-14 + `01-lagrangebench/modal_app.py` Volume mount paths        | layout convention enforced by code paths in the rung-3 function; no separate test (the layout is descriptive of where files land, not a numeric threshold) |
| D0-15 | Rung-3 P0 invocation: SEGNN-10-64 checkpoint on 2D TGV (gdown ID `1llGtakiDmLfarxk6MUAtqj6sLleMQ7RL`); `python main.py mode=infer eval.test=True load_ckp=<path>/best eval.n_rollout_steps=100 eval.n_trajs=20`; config auto-loads from checkpoint per README | DECISIONS.md D0-15 + `01-lagrangebench/modal_app.py` `LAGRANGEBENCH_CHECKPOINT_GDOWN_IDS` constant + invocation arg list | n/a — invocation pattern is descriptive (matches README's documented usage); `eval.n_rollout_steps` and `eval.n_trajs` come from plan §3.2 step 3 / spec §6 and are themselves not numerical thresholds (they're sample-size choices) |

**What the table establishes (and what it does not).** Every numerical
threshold the Modal session will run against is pre-registered with
both a documentation cite and (for thresholds that gate a runtime
branch) a drift-guard test. The drift-guard tests use the same
"hard-assert the constant matches the pre-registration verbatim"
discipline across D0-05, D0-08, and D0-09; the same discipline will be
applied to the D0-07 band on Day 1. The D0-10 and D0-11 escalation
rulings are not numerical thresholds in the same sense — they encode
*decision criteria* under specific failure conditions — but they are
pre-registered as concretely as the numerical bands so that the agent
in-session does not improvise under time pressure.

What the table does **not** establish: it does not certify the *truth*
of the experimental claims (those are gated on Gates A / C / D
verdicts that land Days 1–2), nor does it certify that the
pre-registered bands themselves are well-calibrated for real
LagrangeBench / NGC data (D0-07 and D0-08 explicitly carve out
"if pilot data shows divergence, log a new D0-N+ entry; do not
silently amend"). What it certifies is the *methodology shape* — that
no threshold can be silently shifted in code without a new entry
citing the discrepancy.

**Decision.** Commit this table as a durable artifact. If the Modal
session lands a new threshold (e.g., a Gate D mesh-side band, a
Day 1.5 LagrangeBench cross-rollout consistency band), append a row
here in the same commit that adds the corresponding DECISIONS.md
entry, so the coverage table stays in sync with the coverage shape.

**Realized.** This entry. Future appendages on Day 1+ as new
pre-registrations land.

---

## D0-13 — 2026-05-04 — GPU-class defaults; refinement of D0-10 "A100" → "any CUDA GPU"

**Question.** D0-10 pre-registers the hour-2 micro-gate as "at least
one A100 device". The "A100" specifier is the plan §3.2 step 1 default
GPU choice (mirrored from LagrangeBench paper hardware), but the
load-bearing epistemic claim of D0-10 is "JAX sees a working CUDA
GPU" — the failure mode D0-10 protects against is JAX-CUDA install
brokenness, not A100-vs-other-GPU class mismatch. The strict reading
of D0-10 says any non-A100 device → FAIL → CPU pivot (which is wrong
— a working T4 is not a JAX-CUDA-broken state). The spirit reading
says any CUDA GPU → PASS. Without a refining entry, the agent silently
picks one reading at runtime — exactly the discipline failure D0-10
itself was written to prevent.

**Cost-side context.** Modal GPU class pricing (approx., 2026-05):

| Class           | Memory | Modal $/hr (approx.) | sm_   |
|-----------------|--------|----------------------|-------|
| T4              | 16 GB  | $0.59                | 7.5   |
| L4              | 24 GB  | $0.80                | 8.9   |
| A10G            | 24 GB  | $1.10                | 8.6   |
| A100 (40 GB)    | 40 GB  | $3.40                | 8.0   |
| A100 (80 GB)    | 80 GB  | $4.50                | 8.0   |
| H100            | 80 GB  | $5.00+               | 9.0   |

For the hour-0 micro-gate (~30 sec of GPU time), the absolute spend
differs by pennies. For Day 1 §3.2 step 3 rollout generation (~2 h of
GPU time per the plan), A100 → A10G saves ~$4.60; A100 → T4 saves
~$5.50. Material against the $30 hard cap (plan §2.2).

**Inference-side context.** LagrangeBench rollouts at the in-scope
particle counts are inference-only on small graphs:

- TGV 2D: ~3 000 particles
- Dam break 2D: ~10 000 particles
- TGV 3D (P3, stretch only): ~20 000+ particles

SEGNN/GNS forward-pass memory at these counts is well under 12 GB
even with the SEGNN spherical-harmonics tensor product overhead.
LagrangeBench's paper hardware (A100) was chosen for *training*
throughput, not inference memory; we are loading their checkpoints
and running inference, which is a fraction of training-time memory.

**Decision (pre-registered).**

1. **Refine D0-10's gate threshold.** The hour-2 micro-gate now passes
   if ``jax.devices()`` returns at least one device with
   ``platform == "gpu"`` (any CUDA-compatible Modal GPU class). The
   epistemic test is unchanged from D0-10's spirit. CPU-only return
   → FAIL → JAX-CPU read-only pivot per D0-10 unchanged.

2. **GPU-class defaults by stage.**

   | Stage                                                     | GPU class default | Why                                                                            |
   |-----------------------------------------------------------|-------------------|--------------------------------------------------------------------------------|
   | Hour-0 / hour-2 JAX micro-gate                            | **T4**            | Smoke test only; cheapest CUDA-JAX path; same epistemic content as A100.        |
   | Day 1 §3.2 step 3 rollout generation (SEGNN/GNS inference) | **A10G**          | 24 GB headroom for SEGNN tensor products at TGV2D/Dam2D particle counts; ~3× cheaper than A100. |
   | Day 1 fallback if step 3 OOMs                             | A100 (40 GB)      | Escalation only; engaged on first CUDA OOM during rollout generation.           |
   | Day 2 PhysicsNeMo MGN inference                           | A10G (provisional) | NGC checkpoint memory profile not measured at Day 0; revisit at the D0-11 audit. |

3. **OOM escalation criterion.** "First CUDA OOM during rollout
   generation" means: at the LagrangeBench-default settings
   ``eval.n_trajs = 20``, ``eval.n_rollout_steps = 100``, on any of
   {TGV2D-SEGNN, TGV2D-GNS, Dam2D-SEGNN, Dam2D-GNS} workloads with
   ``pip install -e ".[dev]"`` defaults. If OOM surfaces, switch
   *that workload only* (not all workloads) to A100 and log the switch
   as a sub-entry under this D0-13. Do **not** preemptively switch
   all workloads to A100 on first OOM.

4. **Why this entry and not a silent ``gpu="A100"`` → ``gpu="T4"``
   edit.** Threshold-drift discipline (D0-08, D0-09 precedent) requires
   that any code-side change to a pre-registered value land alongside
   a DECISIONS entry citing the change. D0-10's "A100" specifier is
   exactly the kind of pre-registered value that needed an entry
   rather than a code-only edit; this entry is that artifact.

**Drift-guard.** Code commit on ``feature/rollout-anchors`` lands a
``test_modal_app_gpu_class_matches_d0_13_pre_registration`` test
alongside the ``gpu="T4"`` edit, asserting the literal in
``modal_app.py`` matches D0-13's stage-1 default. Same hard-assert
discipline as ``test_ke_rest_threshold_matches_pre_registration``
(D0-08) and ``test_mesh_fd_noise_tolerance_matches_pre_registration``
(D0-09). When Day 1 step 3 lands the inference function on A10G, a
second drift-guard assertion lands with it.

**Net budget impact.** ~$5 saved on Day 1 inference at expected case
(no OOM); same $30 hard cap retained per plan §2.2 (now with ~17%
more buffer to absorb a Day 2 NGC audit overrun under D0-11's 1h
hard budget).

**Realized.** This entry + Code commit on
``physics-lint feature/rollout-anchors`` changing
``gpu="A100"`` → ``gpu="T4"`` in
``01-lagrangebench/modal_app.py`` and adding the drift-guard test.

---

## D0-14 — 2026-05-04 — Modal Volume layout for rollout-anchors artifacts

**Question.** Rung 3 introduces persistent storage on Modal: pretrained
checkpoints (Google Drive via gdown), datasets (Zenodo via
``download_data.sh``), and generated rollout ``.npz`` files. Without a
pre-registered directory layout, the first artifact dictates the
structure and subsequent artifacts grow organically. That is the
"layout grew organically" failure mode that prevents future
cross-stack queries from being clean ("which rollouts consumed
checkpoint X?", "which adapter SHA produced this ``.npz``?").

**Decision (pre-registered).** Modal Volume
``rollout-anchors-artifacts`` follows the directory layout below.
All paths are absolute under the volume mount point ``/vol``.

```
/vol/
├── checkpoints/
│   ├── lagrangebench/
│   │   ├── segnn_tgv2d/best/         # unzipped from gdown <SEGNN_TGV2D_ID>
│   │   ├── gns_tgv2d/best/           # unzipped from gdown <GNS_TGV2D_ID>
│   │   ├── ...                       # other LagrangeBench pairs as P1+ scales
│   └── physicsnemo/
│       ├── ns_meshgraphnet/          # downloaded via NGC API in Day 2
│       └── ahmed_body_meshgraphnet/  # if Day 2 deferred to FNO/Darcy fallback,
│                                     # this slot reorganises accordingly
├── datasets/
│   └── lagrangebench/
│       └── 2D_TGV_2500_10kevery100/  # downloaded via download_data.sh
│       └── ...                       # other datasets as P1+ scales
└── rollouts/
    ├── lagrangebench/
    │   ├── segnn_tgv2d_<git_sha>.npz
    │   ├── gns_tgv2d_<git_sha>.npz
    │   └── ...
    └── physicsnemo/
        ├── ns_meshgraphnet_<git_sha>.npz
        └── ...
```

**Why ``<git_sha>`` on rollout filenames.** Two reasons:

1. **Reproducibility.** The SARIF artifacts (``lint.sarif`` per plan
   §2.4 implementation tree) reference the rollout filename; the git
   SHA captures *which version of the adapter* generated it. Without
   the suffix, the same filename gets overwritten on regeneration and
   the SARIF↔rollout↔adapter triple is no longer reconstructable from
   the filename alone.
2. **Cache invalidation.** If the rollout is regenerated under a
   different adapter version (e.g., a bug fix in
   ``particle_rollout_adapter.py`` lands), the new file lives
   alongside rather than overwriting; the right version is selectable
   by SHA. Prevents the silent "ran the new code on the old rollout"
   failure mode.

The git SHA is captured at function-invocation time (not at
module-import time) by reading the current physics-lint HEAD inside
the local entrypoint, then passed as a function argument to the
remote function. Short SHA (10 chars) is sufficient for filename
uniqueness within the project's expected commit volume; full SHA
goes in the ``.npz`` metadata block.

**Why volume-persisted from rung 3, not deferred to rung 4.** The
conservative answer (hold off on volume persistence until rung 4
wiring lands) has a real cost: rung 3 then needs to be re-fired (or
the rollout exfiltrated and re-imported) when rung 4 lands, costing
~$0.30+ of A10G time on the second invocation for no methodology
benefit. Volume persistence is also the right shape for the eventual
cached-``.npz`` CI replay infrastructure that plan §10 definition-
of-done specifies; building toward that shape from the start is
cheaper than migrating later.

**Realized.** This entry + ``modal.Volume.from_name(
"rollout-anchors-artifacts", create_if_missing=True)`` in
``01-lagrangebench/modal_app.py`` + rung-3 function code paths that
write to and read from the layout above.

---

## D0-15 — 2026-05-04 — Rung-3 P0 invocation: SEGNN-TGV2D rollout

**Question.** Plan §3.2 step 3 prescribes the production rollout
invocation as:

    python main.py mode=infer \
        load_ckp=checkpoints/segnn_tgv2d/best/ \
        config=configs/segnn_tgv2d.yaml \
        eval.n_rollout_steps=100 eval.n_trajs=20

But upstream investigation reveals slight mismatches with the
LagrangeBench README's documented invocation pattern:

| Plan §3.2 step 3                  | LagrangeBench README                       |
|-----------------------------------|--------------------------------------------|
| ``dataset=tgv2d`` (no underscore) | dataset is ``tgv_2d`` (configs path:        |
|                                   | ``configs/tgv_2d/{base,gns,segnn}.yaml``)   |
| ``config=configs/segnn_tgv2d.yaml`` | omitted; "When loading a saved model with |
|                                   | ``load_ckp`` the config from the checkpoint |
|                                   | is automatically loaded"                   |
| (no eval.test override)           | ``eval.test=True`` to run on test split     |

Without pre-registration, the rung-3 agent decides between the plan's
literal (broken) command and the README's documented (correct) pattern
under time pressure — exactly the failure shape this discipline
exists to prevent.

**Decision (pre-registered).** Rung-3 P0 invocation matches the
LagrangeBench README's documented pattern, with the plan's eval
sample-size overrides preserved:

```
python main.py mode=infer eval.test=True \
    load_ckp=<checkpoint_dir>/best \
    eval.n_rollout_steps=100 \
    eval.infer.n_trajs=20 \
    dataset.src=<volume_dataset_path> \
    dataset.name=tgv2d \
    eval.infer.metrics=[mse,e_kin]
```

The ``dataset.name=tgv2d`` arg was added at first amendment time
(rung-3 first inference attempt at ``bf3741d`` discovered the
need); see "Schema-evolution discovery (amendment 1)" below.

The ``eval.infer.metrics=[mse,e_kin]`` arg (dropping the default
Sinkhorn metric) was added at second amendment time after rung-3's
second inference attempt (``63e33e9``) ran for 199.9 sec inside
``eval_rollout`` and failed on a self-inconsistency in ``ott-jax
0.4.9``; see "Sinkhorn-metric-skip discovery (amendment 2)" below.

The plan's ``eval.n_trajs=20`` was replaced with
``eval.infer.n_trajs=20`` at third amendment time after rung-3's
third inference attempt (``7254193``) hit the 2400-sec subprocess
timeout: current upstream ``rollout.py:351`` reads
``cfg.eval.infer.n_trajs`` for infer mode, NOT the plan's
``cfg.eval.n_trajs`` — making the plan's CLI arg a no-op for our
mode and leaving the default 200 trajectories in effect. See
"Config-key-routing discovery (amendment 3)" below.

Rationale per argument:

- ``mode=infer eval.test=True``: run inference on the test split per
  README's standard pattern. Plan's omission of ``eval.test=True``
  was an oversight; without it, the default eval mode may run on a
  validation slice rather than test.
- ``load_ckp=<path>/best``: matches README; auto-loads config from
  the bundled checkpoint config (the "config that was used to train
  this checkpoint"), avoiding the plan's misnamed ``configs/segnn_tgv2d.yaml``.
- ``eval.n_rollout_steps=100``, ``eval.infer.n_trajs=20``: plan §3.2
  step 3 / spec §6 sample-size pre-registrations preserved verbatim.
  These override whatever defaults the checkpoint config carries.
  Note: the plan's literal ``eval.n_trajs=20`` was replaced with
  ``eval.infer.n_trajs=20`` at amendment 3 because current upstream's
  infer mode reads the nested key (see "Config-key-routing discovery
  (amendment 3)" below); the *value* (20) is the methodology-relevant
  pre-registration and is unchanged.
- ``dataset.src=<volume_dataset_path>``: override to point at the
  volume-persisted dataset path under the D0-14 layout
  (``/vol/datasets/lagrangebench/2D_TGV_2500_10kevery100``), since
  the checkpoint config's relative path
  (``datasets/2D_TGV_2500_10kevery100``) won't resolve correctly
  inside the Modal container's working directory.
- ``dataset.name=tgv2d``: required by current upstream
  ``runner.py:148`` (``dataset_name = cfg.dataset.name``), passed
  verbatim to ``H5Dataset(..., name=dataset_name, ...)``. The publicly
  distributed SEGNN-TGV2D checkpoint's bundled config does not
  include ``dataset.name`` — the older LagrangeBench that trained
  this checkpoint inferred the name from the dataset path via
  ``H5Dataset``'s default ``name=None`` + ``get_dataset_name_from_path``,
  but current HEAD's runner does NOT use that auto-inference and
  requires ``cfg.dataset.name`` to be set before ``H5Dataset`` is
  constructed. The valid name space (per upstream
  ``lagrangebench/data/data.py:23-29`` URL dict + ``data.py:272-290``
  ``get_dataset_name_from_path``) is ``{tgv2d, tgv3d, rpf2d, rpf3d,
  ldc2d, ldc3d, dam2d}`` — no underscore, distinct from the
  ``tgv_2d`` arg ``download_data.sh`` accepts.
- ``eval.infer.metrics=[mse,e_kin]``: drops the Sinkhorn divergence
  metric from the per-rollout evaluation list. Required because
  ``ott-jax 0.4.9`` (the version LagrangeBench's pin pulls) is
  internally inconsistent at this version: ``sinkhorn_divergence``
  in ``ott/tools/sinkhorn_divergence.py`` forwards a
  ``sinkhorn_kwargs`` parameter to ``Geometry.__init__()`` in
  ``ott/geometry/geometry.py``, but ``Geometry.__init__()`` doesn't
  accept that keyword. The default ``eval.infer.metrics`` is
  ``[mse, sinkhorn, e_kin]`` per the SEGNN-TGV2D checkpoint's
  bundled config; this override drops Sinkhorn and keeps the other
  two. This is **non-load-bearing for the physics-lint harness**:
  the harness reads positions, velocities, and KE from the rollout
  ``.npz``; Sinkhorn divergence is a LagrangeBench-internal metric
  for measuring optimal-transport distance between rollout and
  ground-truth distributions, and is not consumed by any of the
  in-scope rules (PH-CON-001/002/003, PH-SYM-001/002, PH-BC-001).
  The methodology cost of dropping it is therefore zero for our
  purposes; the operational cost is one CLI arg.

**Schema-evolution discovery (amendment 1).** ``dataset.name=tgv2d``
was added to the prescribed invocation at first amendment time,
after rung-3's first inference attempt at ``bf3741d`` produced
``OmegaConf ConfigAttributeError: Missing key name`` in 9.5 sec
of inference wall time. Diagnosis (via ``runner.py`` + ``data.py``
upstream inspection): older LagrangeBench checkpoints don't carry
``dataset.name`` because the older runner relied on ``H5Dataset``'s
``name=None``-default auto-inference; current HEAD's runner reads
``cfg.dataset.name`` directly with no fallback. Fix is to provide
the arg explicitly. This is a small upstream-schema-evolution
discovery, not a methodology change; the in-place amendment of
this entry's invocation is the audit trail.

**Sinkhorn-metric-skip discovery (amendment 2).**
``eval.infer.metrics=[mse,e_kin]`` was added to the prescribed
invocation at second amendment time, after rung-3's second
inference attempt at ``63e33e9`` ran for 199.9 sec inside
``eval_rollout`` and failed with
``TypeError: Geometry.__init__() got an unexpected keyword argument
'sinkhorn_kwargs'`` in ``ott-jax 0.4.9``. Diagnosis: ``ott-jax
0.4.9`` is internally self-inconsistent; ``sinkhorn_divergence`` in
``ott/tools/sinkhorn_divergence.py`` forwards
``sinkhorn_kwargs`` to ``Geometry.__init__()`` in
``ott/geometry/geometry.py``, but the latter doesn't accept it.
Pinning ``ott-jax`` to a different version was rejected because
LagrangeBench's pin would force a transitive change to the matched
JAX-CUDA stack (the work landed in physics-lint commit ``91d3ce7``).
The chosen workaround is surgical: drop the Sinkhorn metric from the
per-rollout evaluation list. The ``[mse, e_kin]`` retain provides
sufficient evaluation signal for LagrangeBench's own infer-mode
output; the dropped Sinkhorn is **not consumed** by any in-scope
physics-lint rule.

This is the second amendment to D0-15. The cumulative-patch shape
distribution after this amendment (across all five
upstream-compatibility patches required to reach a clean rung-3
verdict) lives as ``UPSTREAM_COMPAT_PATCHES`` in
``modal_app.py`` and is surfaced in the rung-3 verdict log. As of
this amendment, all five patches sit in the
"non-load-bearing for physics-lint harness" methodology bucket
(three infrastructure, one config, one library-internal). The
discipline going forward: if any subsequent patch enters the
"load-bearing for physics-lint harness" bucket, surface for
higher-level review rather than apply (a)-shape workaround.

**Config-key-routing discovery (amendment 3).**
``eval.infer.n_trajs=20`` replaced ``eval.n_trajs=20`` in the
prescribed invocation at third amendment time, after rung-3's
third inference attempt at ``7254193`` hit the 2400-sec subprocess
timeout (40 min wall) instead of completing in the expected ~10–20
min for 20 trajectories on A10G. Diagnosis (via
``lagrangebench/evaluate/rollout.py:311–394``):

- Line 318: ``cfg_eval_infer: ... = defaults.eval.infer``
- Line 351: ``n_trajs = cfg_eval_infer.n_trajs``

Current upstream's infer mode reads the **nested**
``cfg.eval.infer.n_trajs`` key, NOT the top-level ``cfg.eval.n_trajs``
the plan's literal command sets. The default ``cfg.eval.infer.n_trajs``
in the SEGNN-TGV2D checkpoint's bundled config is ``200``. So the
two prior rung-3 inference attempts (``63e33e9``, ``7254193``) were
running 200 trajectories instead of 20: the third run timed out
because 200 × 100 rollout steps × SEGNN forward-pass × A10G simply
exceeded the 2400-sec subprocess timeout.

The plan §3.2 step 3 / spec §6 pre-registration of ``n_trajs=20``
is the methodology-relevant value and unchanged; only the CLI key
name updates. Same shape as amendment 1's ``dataset.name=tgv2d``:
the plan's literal command was aspirational about config-key paths
that current upstream has reorganised.

This is the sixth upstream-compatibility patch and the second in
the "config" layer (alongside ``dataset.name=tgv2d``). The
cumulative-patch distribution remains entirely in the
non-load-bearing bucket: 3 infrastructure / 2 config / 1
library-internal, all non-load-bearing for physics-lint harness.
``UPSTREAM_COMPAT_PATCHES`` in ``modal_app.py`` is amended to add
this entry.

**Rollout-persistence enablement (amendment 4).**
``eval.rollout_dir`` and ``eval.infer.out_type=pkl`` were added to
the prescribed invocation at fourth amendment time, after rung-3's
PASS verdict at ``9f7df91`` reported ``files_written: 0`` —
LagrangeBench's default ``out_type=none`` computes metrics on
rollouts and discards the rollout tensors themselves, but the
physics-lint harness (rung 4 consumer) needs the trajectories
materialized.

Two CLI args required, at two different config-key paths because
of LagrangeBench's split layout (top-level ``eval.rollout_dir`` per
``rollout.py:319`` reading ``defaults.eval.rollout_dir``; nested
``eval.infer.out_type`` per ``rollout.py:396`` reading
``cfg_eval_infer.out_type``):

    eval.rollout_dir=/vol/rollouts/lagrangebench/segnn_tgv2d_<git_sha>
    eval.infer.out_type=pkl

LagrangeBench's pkl format (``rollout.py:271-297``) is
per-trajectory ``rollout_{j}.pkl`` containing
``{predicted_rollout: (T, N, D), ground_truth_rollout: (T_gt, N, D),
particle_type: (N,)}`` plus a ``metrics{timestamp}.pkl`` with the
eval metrics dict. This is **not** schema-conformant with
SCHEMA.md §1 ``particle_rollout.npz`` — four reconciliations are
required, motivating the per-trajectory conversion module
``external_validation/_rollout_anchors/_harness/lagrangebench_pkl_to_npz.py``
invoked from ``lagrangebench_rollout_p0_segnn_tgv2d`` after the
inference subprocess returns.

**Velocity derivation.** SEGNN outputs positions in this dataset;
velocities are derived by central differences over
``dt = metadata.dt * metadata.write_every`` (matching LagrangeBench's
own factor at ``runner.py:260`` and ``metrics.py:99``); endpoints
use first-order forward / backward differences. The harness's
PH-CON-002 therefore tests whether the positional rollout produces
a kinetic-energy time series consistent with conservation — i.e.,
whether the position dynamics implicitly respect energy
conservation — rather than whether the model directly outputs an
energy-conserving velocity field. This is the stricter test of the
two: a model that outputs an energy-conserving velocity field but
inconsistent positions would pass the velocity-output test and
fail the derived-velocity test, while a model that produces
dynamically-consistent positions necessarily passes both. The
derived-velocity coupling between position and velocity dynamics
through the discretization is therefore not a methodology
limitation but the load-bearing claim: the test asks whether the
model's predictions respect energy conservation when interpreted as
a dynamical system, which is the physically meaningful question.
Documented here for the writeup framing; not a change to the rule
semantics.

**Particle mass — SCHEMA.md drift discovery.** Surfaced during the
rung-3.5 conversion-module work: ``particle_rollout_adapter.load_rollout_npz``
required a ``particle_mass`` field in the npz that SCHEMA.md §1
v1.0/v1.1 didn't document. The drift was already there (the
adapter already raised ``KeyError`` on missing ``particle_mass``);
the rung-3.5 work surfaced it because the conversion module had to
produce a field the source SCHEMA didn't list. Closed by SCHEMA.md
v1.2 bump (additive — adding a field that was already required
makes implicit explicit; no new compatibility break introduced).
For LagrangeBench SPH datasets, conversion populates uniform unit
mass (``np.ones(N, fp64)``); the conservation rules check temporal
*changes* in mass and energy, which are invariant to a global
mass-scale choice, so uniform unit mass is methodologically
equivalent to the dataset's implicit smoothing-length normalization
for these tests. Datasets that *do* carry per-particle mass should
pass it through unchanged; the contract is recorded in SCHEMA.md
§1 v1.2.

**Bounds transpose.** LagrangeBench's ``metadata["bounds"]`` is
``(D, 2)`` per-axis-[min, max] (per ``case.py:133``,
``runner.py:43``, real ``tests/3D_LJ_3_1214every1/metadata.json``);
SCHEMA.md ``domain_box`` is ``(2, D)`` first-row-mins-second-row-
maxes. Transpose at the conversion boundary is the right place to
reconcile. Validation: post-transpose ``shape == (2, D)`` and
``mins < maxes`` elementwise.

**write_every defensive handling.** ``metadata["write_every"]`` is
conditionally present in LagrangeBench dataset metadata.json
(present in production datasets like ``2D_TGV_2500_10kevery100``;
absent in ``tests/3D_LJ_3_1214every1`` tutorial fixture).
Conversion uses ``metadata.get("write_every", 1)`` and records both
the value used AND the source in the rollout metadata dict
(``write_every`` and ``write_every_source ∈ {"dataset", "default"}``)
so future audit-trail reconstruction can distinguish a
dataset-specified dt from a defaulted dt without re-reading the
original metadata.json. Same shape of instrumentation as the
``UPSTREAM_COMPAT_PATCHES`` ledger: record the choice at the
moment it's made.

**Seed pre-registration.** Rung-3 rollouts use ``seed=0`` explicitly
(CLI override of LagrangeBench's default). The rung-4 regeneration
test asserts that re-invoking the rung-3 inference with the same
checkpoint and seed produces a ``particle_rollout_traj*.npz`` with
bit-identical contents-SHA-256, which validates that the conversion
pipeline (LagrangeBench inference → pkl → npz conversion) is
end-to-end deterministic. Without pre-registration, the regeneration
test would either depend on whatever LagrangeBench's default seed
happens to be (fragile to upstream changes) or have to capture-and-
replay the seed (more state to maintain).

**Validation surface — five assertions at conversion time.** The
conversion module fires the following at each rollout, raising
``ValueError`` / ``KeyError`` with the rollout filename in the
message so a failure surfaces both the rule violated and the file
that triggered it:

1. ``particle_mass.shape == (N_particles,)``
2. ``particle_mass.dtype == np.float64``
3. ``domain_box.shape == (2, D)`` post-transpose
4. ``domain_box[0] < domain_box[1]`` elementwise (mins below maxes)
5. ``RolloutMetadata.write_every_source ∈ {"dataset", "default"}``

Cheaper to fail at conversion time than to reconstruct post-hoc
when the harness ``load_rollout_npz`` errors on a malformed npz on
the rung-4 read path.

**Patch ledger — 7th UPSTREAM_COMPAT_PATCHES entry.** The
two-arg-at-split-paths feature toggle is one entry in the
patches-required field, not two. The framing tracks
methodology-relevance (one feature-toggle for "turn on rollout
persistence"), not key-path-syntactic-distinctness. Layer=config,
relevance=non-load-bearing. The cumulative-patch distribution after
amendment 4: 3 infrastructure / 3 config / 1 library-internal / all
seven non-load-bearing for physics-lint harness. The discipline
established at amendment 2 holds: load-bearing-library-internal
patches surface for higher-level review; the other buckets remain
bounded.

The conversion module itself is NOT a patch — it's harness
infrastructure that exists because LagrangeBench's output format
and SCHEMA.md differ legitimately. Different category from
upstream-compatibility workarounds. Reused across rungs (P1
GNS-TGV2D inherits the conversion verbatim; P2/P3 likewise; the
PhysicsNeMo Day 2 equivalent will be a sister module sharing the
metadata-dataclass shape).

**Realized — amendment 4.** ``lagrangebench_pkl_to_npz.py`` module
+ ``test_lagrangebench_pkl_to_npz.py`` (23 tests covering all five
validation assertions + the conversion arithmetic + the
LagrangeBench native pkl shape + the LB-vs-SCHEMA bounds
reconciliation + RolloutMetadata round-trip). SCHEMA.md v1.2 bump
adding ``particle_mass``, ``metadata.write_every``, and
``metadata.write_every_source`` fields. ``modal_app.py`` updates:
(a) ``add_local_file`` of the conversion module into the rung-3
image; (b) three new CLI args in the inference invocation
(``eval.rollout_dir``, ``eval.infer.out_type=pkl``, ``seed=0``);
(c) post-inference conversion step with structured-manifest
failure capture; (d) verdict-printer split into
rung-3-PASS-rung-3.5-PASS / rung-3-PASS-rung-3.5-FAIL /
rung-3-FAIL branches; (e) 7th UPSTREAM_COMPAT_PATCHES entry.

**Checkpoint identity.** SEGNN-10-64 on 2D TGV per LagrangeBench
README pretrained-models table. Google Drive file ID
``1llGtakiDmLfarxk6MUAtqj6sLleMQ7RL``. Reported test-split error
metrics in README: 4.4e-6 / 2.1e-7 / 5.0e-7 (presumably MSE_pos /
MSE_vel / Sinkhorn). These metrics are the upstream-published
reference values that plan §10 item 2 calls out as "5.2 — Reproduce
LagrangeBench-published numbers"; rung 4 will compare against them.

The full LagrangeBench checkpoint catalogue (P0 + P1 + P2 + P3 from
plan §3.1) lives in code as a ``LAGRANGEBENCH_CHECKPOINT_GDOWN_IDS``
dict in ``modal_app.py``. The dict is a code artifact (URLs change
with upstream reorgs and don't belong in DECISIONS.md prose); this
entry pre-registers only the choice of which checkpoint rung 3 P0
targets and the invocation pattern.

**Why pin invocation now, not at rung-3 execution time.** The
audit-budget guardrail at D0-11 (1h hard cap on ad-hoc upstream
investigation) applies here too. Locking the invocation pattern
before scaffolding lets the rung-3 commit reference DECISIONS rather
than improvise; if upstream changes its README pattern, that's a new
DECISIONS entry, not a silent code-side amendment.

**Realized.** This entry + ``LAGRANGEBENCH_CHECKPOINT_GDOWN_IDS``
constant + ``lagrangebench_rollout_p0_segnn_tgv2d`` function in
``01-lagrangebench/modal_app.py``.

---

## D0-16 — 2026-05-04 — Rung-3 + 3.5 P0 PASS verdict capture

**What this captures.** First end-to-end clean run of the full
JAX-CUDA-Modal-LagrangeBench-conversion-validation stack on the
SEGNN-TGV2D production checkpoint. Persists schema-conformant
``particle_rollout.npz`` artifacts (SCHEMA.md §1 v1.2) to the
rollout-anchors-artifacts Modal Volume per the D0-14 layout, with
all five conversion-time validation assertions passing. The entire
infrastructure column from JAX install through harness-readable
artifact has now been verified once on real upstream output, not
synthetic fixtures.

**Verdict.** ``rung 3 + 3.5 P0 verdict: PASS — inference returncode 0;
conversion produced 20 schema-conformant npz file(s).``

**Cumulative SHAs at PASS.**

| Repo                    | Branch                  | SHA                                        |
|-------------------------|-------------------------|--------------------------------------------|
| physics-lint            | feature/rollout-anchors | ``f75e22d`` (10-char ``f75e22d8dd``)        |
| physics-lint-validation | main                    | ``ca67618``                                 |
| LagrangeBench upstream  | (--depth 1 clone)       | ``b880a6c84a93792d2499d2a9b8ba3a077ddf44e2`` |

**Headline manifest.**

    aborted_at_step:        <none>
    inference_returncode:   0
    inference_wall_seconds: 759.5
    conversion_attempted:   True
    conversion_returncode:  0
    converted_npz_count:    20
    inference_seed:         0
    rollout_subdir:         /vol/rollouts/lagrangebench/segnn_tgv2d_f75e22d8dd

**Test-split metrics (FP-identical to prior PASS at 9f7df91).**

    val/mse1:  1.47e-7   val/mse10: 2.87e-6   val/mse20: 1.57e-5
    val/mse5:  7.77e-7   val/mse50: 1.29e-4
    val/e_kin: 5.60e-7   val/loss:  3.42e-4

**Files persisted (41 entries, ~221 MB total).**

| Category                            | Count | Per-file size | Origin                        |
|-------------------------------------|-------|---------------|-------------------------------|
| ``particle_rollout_traj{00..19}.npz`` | 20    | 4.27 MB        | Our SCHEMA.md §1 conversion   |
| ``rollout_{0..19}.pkl``               | 20    | 6.78 MB        | LagrangeBench native (kept for cross-validation) |
| ``metrics2026_05_04_19_54_34.pkl``    | 1     | 71 KB          | LagrangeBench eval metrics    |

**Patch ledger snapshot at PASS.**

    distribution: 3 infrastructure, 3 config, 1 library-internal | 7 non-load-bearing

| #   | Layer            | Relevance         | Ref                  |
|-----|------------------|-------------------|----------------------|
| 1   | infrastructure   | non-load-bearing  | physics-lint 9276596 |
| 2   | infrastructure   | non-load-bearing  | physics-lint 32b6b79 |
| 3   | infrastructure   | non-load-bearing  | physics-lint 91d3ce7 |
| 4   | config           | non-load-bearing  | physics-lint 63e33e9 |
| 5   | library-internal | non-load-bearing  | physics-lint 7254193 |
| 6   | config           | non-load-bearing  | physics-lint 9f7df91 |
| 7   | config           | non-load-bearing  | physics-lint 334cafd |

The cumulative-patch posture remains "stand up a 2026 JAX-CUDA-Modal
stack" rather than "patch around upstream's actual correctness
regressions in our consumed surface" — discipline established at
amendment 2 (load-bearing-library-internal patches surface for
higher-level review) holds; no patches in that bucket.

**Wall-time observation — environmental, not methodological.**
Inference wall: 759.5 sec (~12.7 min) on this run vs 1446 sec
(~24 min) on the prior PASS at 9f7df91. Same A10G GPU class, same
20 trajectories × 100 rollout steps, same SEGNN-TGV2D checkpoint.
Test-split metrics are FP-identical between the two runs, confirming
the rollout is the same trajectory generated faster — attributed to
environmental warm-state effects (probably XLA cache hits on a
recently-used A10G host, or less host contention). Modal's wall-time
is non-deterministic in the small; the methodology contract is
metrics-identical, which is the load-bearing claim and is satisfied.
Logged here for the audit trail; no further investigation warranted.

**Forward agenda for next session.** Two natural opening moves with
bounded cost:

1. **Cross-repo contract spot-check** (~5 min, $0). Pull one
   ``particle_rollout_traj00.npz`` from the Volume to local disk
   (``modal volume get rollout-anchors-artifacts ...``); load it
   via ``particle_rollout_adapter.load_rollout_npz``; sanity-check
   the resulting ``ParticleRollout`` (shapes, dtypes, metadata
   completeness). Optionally invoke the harness defects
   (``mass_conservation_defect``, ``energy_drift``,
   ``dissipation_sign_violation``) on it for a smallest-possible
   rung-4 preview. This is the cross-repo contract verification
   that's been latent throughout — physics-lint's adapter expects
   what physics-lint-validation's conversion produces, and the
   contract has been verified only against synthetic inputs through
   D0-15 amendment 4. Verifying against real LB-derived npz before
   P1 fires means P1's failure modes are scoped to GNS-specific
   issues rather than potentially carrying a P0-conversion bug
   forward.

2. **P1 (GNS-TGV2D) firing** (~25 min, ~$0.50). Sibling Modal
   function ``rollout_p1_gns_tgv2d`` reusing P0's volume, image,
   conversion module, and verdict-printer infrastructure. Marginal
   cost: download GNS checkpoint via ``LAGRANGEBENCH_CHECKPOINT_GDOWN_IDS["gns_tgv2d"]``
   (file ID ``19TO4PaFGcryXOFFKs93IniuPZKEcaJ37``), invoke with
   ``model.name=gns`` (or whatever the LB CLI key is — confirm
   from the bundled config) + ``load_ckp=<gns_ckpt>/best``. Every
   infrastructure piece P1 needs inherits from P0's verified state.

**Realized.** This entry + the verdict log at
``../01-lagrangebench/outputs/verdicts/rung_3.5_20260504_214148_f75e22d.log``
+ the 41 files at ``/vol/rollouts/lagrangebench/segnn_tgv2d_f75e22d8dd/``
on the rollout-anchors-artifacts Modal Volume.

---

## D0-17 — 2026-05-04 — Periodic-aware velocity derivation in rung-3.5 conversion

**Question.** The rung-3.5 P0 PASS at f75e22d8dd (D0-16) produced 20
schema-conformant npzs that load cleanly via ``particle_rollout_adapter.load_rollout_npz``
— so the cross-repo contract holds at the schema level. But the
spot-check for D0-16's forward-agenda item 1 invoked the harness
defects on traj00 and found:

- ``mass_conservation_defect = 0.0``  (correct, mass doesn't depend on velocity)
- ``energy_drift = 0.99998``           (suspicious — claims ~total KE loss)
- ``dissipation_sign_violation = 0.548`` (suspicious — half the timesteps appear to violate dissipation)

Diagnostic: the velocity-magnitude distribution had ``min=-24.8, max=+24.9, std=1.35`` —
physically impossible for TGV2D, where positions are bounded to the unit
square and the canonical Taylor-Green initial condition has
characteristic velocity ``U_0 ≈ 1``. With ``dt = 0.04``, an apparent
velocity of ±25 means a particle traversed the entire domain in a
single timestep, which is wraparound — TGV2D is periodic along both
axes, and a particle moving from position 0.999 → 0.001 (correct
physical forward motion through the periodic boundary) gives
``(0.001 - 0.999) / dt = -25`` under naive central differences.

The conversion module's ``_central_diff_velocities`` did not account
for periodic boundaries. Per traj00 alone: 105 of 106 timesteps had
at least one particle with ``|v| > 5`` (well above any physical TGV
velocity), and individual frames had up to 245 spurious-velocity
particles.

**The asymmetry** between ``mass_conservation_defect = 0.0`` (which
doesn't touch the velocity-derivation path) and ``energy_drift ≈ 1``
(which does) is what makes this a real bug rather than a measurement
artifact: the same npz produces a load-bearing-correct mass result and
a methodologically-meaningless energy result, attributable to one
specific code path.

**Decision.** ``_central_diff_velocities`` accepts ``domain_extent`` and
``periodic_axes`` and applies the minimum-image convention along each
periodic axis. ``convert_rollout_dir`` reads ``periodic_boundary_conditions``
from LagrangeBench dataset metadata.json (defensive default: all-False
per axis if the key is missing), threads it into ``_central_diff_velocities``
for derivation, and stores it in ``RolloutMetadata.periodic_boundary_conditions``
so the npz captures whether positions are wrap-around-aware. Pre-D0-17
npzs on periodic datasets (rung-3.5 PASS at f75e22d8dd) are unusable
for PH-CON-002 / PH-CON-003 and should be regenerated; non-periodic
datasets are unaffected (the correction is no-op when ``periodic_axes``
is all-False).

**Strong verification anchor.** Initial framing was that the
LB-native pkls store positions only (confirmed by inspection —
``rollout_0.pkl`` contains ``predicted_rollout``, ``ground_truth_rollout``,
``particle_type``; no velocity field), so the verification anchor
appeared to be the weaker canonical ``U_0 ≈ 1`` aggregate magnitude
check. But ``ground_truth_rollout`` is the SPH-integrated reference
trajectory from JAX-MD (a known-correct integrator); applying
periodic-corrected central differences to its positions produces:

| Path                                 | v range            | std    |
|--------------------------------------|--------------------|--------|
| Ground truth (SPH ref), periodic     | [-0.9591, +0.9628] | 0.156  |
| SEGNN predicted, periodic            | [-0.9591, +0.9628] | 0.170  |
| SEGNN predicted, naive (current bug) | [-12.50, +12.50]   | 1.316  |

Both periodic-corrected paths match canonical ``U_0 ≈ 1`` to ~4%, decay
profiles agree (GT decays max-|v| from 0.96 at t=0 to 0.006 at t=120;
SEGNN decays to 0.013 at t=100, slightly slower as expected for a
learned approximator). This is a stronger anchor than canonical
``U_0`` alone: it triangulates against a known-correct integrator's
output, not just against an analytical prediction. The fact that the
same correction applied to two independent position sources produces
identical bounded magnitudes confirms both that the correction is
implemented correctly and that the bug applies symmetrically to
predicted and reference positions.

**Forward-flag: PH-SYM rules unaffected.** Verified by grep on
``src/physics_lint/rules/ph_sym_*.py`` and inspection of
``particle_rollout_adapter.gridify``: the harness's gridify function
already applies periodic minimum-image distance when binning particles
to grid cells (``particle_rollout_adapter.py:140-141``), and PH-SYM
rules consume the gridded scalar field rather than raw positions. So
the bug is scoped to velocity-derivation only; no plumbing change
needed in the rules-side code. Any *future* consumer of the npz that
computes distances directly from positions (rather than from
velocities or from gridify output) would need to apply the same
periodic correction — ``metadata.periodic_boundary_conditions`` is in
the npz so they can.

**Framing correction.** Initial draft said "every periodic-boundary
dataset in the LB corpus (TGV2D, RPF2D, LDC2D, etc.)" — incorrect:
LDC2D (lid-driven cavity) is wall-bounded, not periodic. The fix's
metadata-driven detection handles that correctly (LDC2D's
``metadata.periodic_boundary_conditions = [False, False]``, conversion
is no-op for it), but the writeup framing should say "datasets with
``periodic_boundary_conditions=True`` (TGV2D, RPF2D, ...)" rather than
enumerating the corpus inaccurately.

**Test coverage — three additional tests beyond the proposed wraparound case.**

1. ``test_central_diff_periodic_wraparound_corrects_one_step_jump``:
   particle at 0.95 → 0.05 across one step; naive forward-diff gives
   -0.9, periodic gives +0.1; verifies the correction works at the
   forward-diff endpoint specifically.
2. ``test_central_diff_no_op_when_periodic_axes_all_false``: verifies
   that passing ``periodic_axes=[False, False, False]`` is bit-identical
   to omitting the argument. Rules out the case where someone
   accidentally applies the wrap to a non-periodic dataset and
   silently corrupts velocities the other way.
3. ``test_central_diff_periodic_no_op_when_no_boundary_crossing``:
   trajectory that stays in the interior with no boundary crossings —
   naive and periodic-corrected outputs are bit-identical (regression
   guard). The periodic correction's ``round(delta / L)`` is zero
   when ``|delta| < L/2``, so it must be a no-op on interior-only
   trajectories.

Plus 6 more covering the metadata-threading path, RolloutMetadata
extensions, dataset-validation edge cases, and shape validation.
9 new tests total; full suite 80/80 green at fix commit
(physics-lint 8c3d080).

**Schema impact.** SCHEMA.md v1.3 bump (additive: ``metadata.periodic_boundary_conditions``
field). Validation surface grows from 5 assertions (D0-15 amendment 4)
to 6 (added: ``periodic_boundary_conditions`` length must equal D).

**Patches-required ledger.** D0-17 is **not** an
``UPSTREAM_COMPAT_PATCHES`` entry — it's a fix to the harness's own
conversion module, not a workaround for upstream LagrangeBench
behavior. Different category. The patches-required ledger remains at
7 entries (the seven from D0-16's PASS); D0-17 is harness-side
methodology evolution surfaced by the spot-check.

**Why D0-17 (not D0-15 amendment 5).** Amendments to D0-15 have been
about extending the conversion contract within its already-pre-registered
scope (Sinkhorn drop, eval flags, schema bump, seed). Periodic-aware
velocity derivation is a discovered methodology requirement that the
original D0-15 didn't anticipate — surfaced through the spot-check
discipline rather than through a known-pending question. Reads more
cleanly as its own pre-registration ("D0-17: periodic-aware velocity
derivation, motivated by spot-check finding on f75e22d8dd") than as a
fifth amendment to a decision that was already amended four times.

**Realized.** physics-lint commit 8c3d080 (3 files: SCHEMA.md v1.3,
``lagrangebench_pkl_to_npz.py`` periodic-aware ``_central_diff_velocities`` +
``convert_rollout_dir`` threading + ``RolloutMetadata.periodic_boundary_conditions``
field, ``test_lagrangebench_pkl_to_npz.py`` 9 new tests). Re-fire of
rung-3.5 to regenerate the 20 npzs at this commit's git_sha is the
next step (D0-18 will capture the post-D0-17 PASS verdict).

**Amendment 1 (PBC length-3 truncation for stable LB convention).**
The post-D0-17 regen at ``8c3d080397`` PASSed inference (returncode 0,
20 native pkls persisted) but FAILed conversion::

    ValueError: .../2D_TGV_2500_10kevery100/metadata.json
      'periodic_boundary_conditions' must have length D=2;
      got [True, True, True]

The TGV2D production dataset's ``metadata.json`` has explicit
``"dim": 2``, ``"bounds"`` of shape ``(2, 2)``, ``"vel_mean"`` /
``"vel_std"`` / ``"acc_mean"`` / ``"acc_std"`` of length 2 — but
``"periodic_boundary_conditions": [true, true, true]`` (length 3).
Cross-checked against the upstream ``tests/3D_LJ_3_1214every1/metadata.json``
tutorial fixture (3D, dim=3, also length-3 PBC): **the LagrangeBench
upstream convention is "PBC field is always length 3 regardless of
``dim``"**. This is a stable convention, not a one-off dataset bug;
the trailing entries are vestigial axes that the simulator records
as periodic by default and that 2D solvers ignore.

D0-17's strict ``len(pbc) == D`` validation correctly catches the
inconsistency between ``bounds``-shape and PBC-length, but the
inconsistency is upstream-convention rather than a recoverable bug.
Rejecting blocks every LagrangeBench dataset (every dataset on the
production grid has the same shape).

**Decision (amendment 1).** Conversion truncates upstream PBC to the
first D entries when ``len(pbc) > D``, with a sanity check that the
truncated trailing entries are all True (matches LB's
vestigial-axes-always-periodic convention). The npz metadata records
both the post-truncation working vector and the original upstream
vector + a source classification, so future readers can verify the
truncation was sensible without re-reading LB's metadata.json. Same
audit-trail discipline as ``write_every_source`` from D0-15 amendment 4.

Three semantically distinct cases::

  pbc length == D                    -> source = "dataset" (or "default" if missing)
  pbc length > D, trailing all True  -> source = "truncated_from_oversize"
  pbc length > D, any trailing False -> ValueError (sanity-check fail)
  pbc length < D                     -> ValueError (no silent zero-padding)

The trailing-False fail covers the user's "hides upstream bugs"
concern that B (vs A) is exposed to: stable-convention case passes
silently with audit-trail capture; real-upstream-divergence case
fires a hard error naming the violated convention.

**Standalone Modal conversion function.** Added
``lagrangebench_convert_pkls_in_volume`` and the local entrypoint
``convert_pkls_p0_segnn_tgv2d``. CPU-only re-conversion against
existing pkls in the Volume; no GPU re-fire needed. Methodology
motivation: keeps P0 and P1 conversion environments identical (both
Modal-side, same image, same numpy version) so the eventual SEGNN-vs-
GNS cross-stack table doesn't mix conversion environments. Explicit
choice over local-only re-conversion of the 8c3d080397 pkls — the
$0.30 + 13 min savings of skipping Modal didn't justify the
platform-consistency methodology cost; conversion is pure numpy
float64 and metrics agreement should be ≪ all PH-CON thresholds, but
making the choice explicit in the audit trail is cheaper now than
reconstructing it later.

**Schema impact (amendment 1).** SCHEMA.md v1.4 (additive: two new
metadata audit fields ``periodic_boundary_conditions_upstream`` and
``periodic_boundary_conditions_source``). Validation surface grows
from 6 assertions (D0-17) to 8 (added: trailing-True sanity check +
``periodic_boundary_conditions_source`` enum check).

**Test coverage (amendment 1).** 4 new tests beyond D0-17's 9:

1. ``test_pbc_oversize_all_true_truncates_with_audit_fields``:
   length-3 ``[True, True, True]`` on dim-2 dataset truncates to
   ``[True, True]``; ``_upstream`` field captures the original;
   ``_source`` field is ``"truncated_from_oversize"``.
2. ``test_pbc_oversize_with_trailing_false_fires_sanity_check``:
   length-3 ``[True, True, False]`` on dim-2 dataset fires
   ``ValueError`` (the user's "exposes real upstream bugs" coverage).
3. ``test_pbc_length_d_exact_no_truncation``: regression guard —
   length-D PBC must not enter the truncate path; source stays at
   ``"dataset"``.
4. ``test_pbc_length_below_d_rejected_no_zero_padding``: distinct
   from oversize, length < D is a genuine bug.

Plus ``test_pbc_default_path_marks_source_default`` covering the
missing-key case (source = ``"default"``, distinguishing from
``"dataset"`` when the key was present and length-D-exact).

**Realized — amendment 1.** physics-lint commit 5429072 (4 files:
SCHEMA.md v1.4 bump, ``lagrangebench_pkl_to_npz.py`` truncation
logic + audit fields + sanity check, ``test_lagrangebench_pkl_to_npz.py``
4 new tests + 1 replaced (84/84 green), ``modal_app.py`` standalone
conversion function + local entrypoint). Plus physics-lint commit
5857144 (``JAX_PLATFORMS=cpu`` env var fix in standalone conversion
— LB rollout pkls contain ``jnp.ndarray`` references, ``pickle.load``
transitively imports jax, which on a CPU-only Modal instance tries
to init the CUDA backend and raises ``FAILED_PRECONDITION``;
mechanical jax/Modal interop quirk, documented inline). Post-fix
re-fire produced 20 schema-conformant npzs at
``/vol/rollouts/lagrangebench/segnn_tgv2d_8c3d080397/`` with
spot-check confirming periodic correction applied (v range
[-0.994, +0.998] vs canonical TGV2D U_0 ≈ 1) and audit fields
captured (``periodic_boundary_conditions=[True, True]``,
``..._upstream=[True, True, True]``,
``..._source="truncated_from_oversize"``).

---

## D0-18 — 2026-05-04 — PH-CON-002 skip-with-reason for dissipative-by-design systems (harness-layer)

**Question.** The post-D0-17-amendment-1 spot-check on regenerated
SEGNN-TGV2D traj00 produced::

    mass_conservation_defect:    0.0      (PASS — load-bearing correct)
    dissipation_sign_violation:  0.0      (PASS — KE never spuriously increases)
    energy_drift:                0.9999   (?)

The 0.9999 value is the **physically correct measurement** for TGV2D
under the harness's ``energy_drift = max|KE(t) - KE(0)| / |KE(0)|``
formula: KE(0) = 600, KE(end) = 0.07, so ~99.99% of initial KE
dissipated to viscosity over the 100-step rollout (~1 viscous time
scale at TGV2D's Re ≈ 100). The harness's docstring confirms the
intent: *"Zero for conservative rollouts; non-zero and growing for
dissipative rollouts."*

But the rule-semantics interpretation is wrong-by-default for
dissipative systems. PH-CON-002 in physics-lint v1.0 is wired for
the wave equation (``src/physics_lint/rules/ph_con_002.py``):
energy is KE + PE on a wave field (computed via integration-by-parts),
the wave equation conserves total energy, and the tristate floor
classifies drift small = PASS, drift large = FAIL. Applying that
rule semantics to SPH/NS dissipative rollouts produces
"FAIL = expected physics" — exactly the misfire shape PH-CON-002
should not have.

This is methodologically meaningful for the BMW posting's domain:
crash simulation is dissipative; fluid surrogates are dissipative;
most non-toy ML targets are dissipative. **PH-CON-002 misfiring on
dissipative systems isn't a fringe edge case for physics-lint, it's
the primary use case.** A linter that emits FAIL on physically-correct
behavior in the most common use case, with a writeup footnote saying
"ignore those FAILs," is harder to defend than a linter that emits
skip-with-reason and points to ``dissipation_sign_violation`` as the
load-bearing alternative.

**Decision (pre-registered).** Add a skip-with-reason path to the
harness's ``energy_drift`` for dissipative-by-design systems. Same
shape as the existing KE-rest skip-with-reason discipline pre-registered
in DECISIONS.md D0-08: when the trajectory is *known* to be
dissipative-by-design, ``energy_drift`` is methodologically meaningless
as a conservation test and should SKIP rather than emit a numeric
defect that downstream PASS/FAIL interpretation will misread.

**Critical: implementation lives in the harness layer, not in
physics-lint v1.0 core.** Two reasons:

1. **Honest-v1.0-limit precedent.** physics-lint v1.0 has documented
   limits (e.g., PH-BC-001/PH-RES-001 unreliable on homogeneous
   Dirichlet); those limits work as a writeup framing because they
   describe corners of the input space. PH-CON-002 misfiring on
   dissipative systems would be the *primary* input space, not a
   corner. Silent in-core "fix" risks masking the fact that
   physics-lint v1.0 has a real methodology gap on its primary use
   case; explicit harness-layer wrapping demonstrates the gap *and*
   prototypes the v1.1 fix without modifying the shipped behavior.

2. **BMW-narrative argument.** The cross-stack table in the writeup
   reads cleaner if the SARIF reads cleanly. *"PH-CON-002: skip
   (dissipative-by-design); see dissipation_sign_violation = 0.0"*
   is a much better story than *"PH-CON-002: FAIL (energy_drift =
   0.9999); see footnote 7 explaining why this is fine."* Placing
   the skip in the harness layer rather than v1.0 core gives the
   stronger framing: "v1.0 has a documented limit; here's the
   validation work that exposes the limit on real LB output and
   prototypes the v1.1 fix."

physics-lint v1.0 stays as-shipped with its documented limits; the
rollout-anchor validation framework demonstrates the limit and the
path to address it via the harness's ``energy_drift`` extension.

**Heuristic — positive-evidence gate to avoid masking real bugs.**
A heuristic that says "looks dissipative → skip energy_drift" would
silently mask the case where a *supposed-conservative* surrogate
(e.g., a Hamiltonian flow ML model) is *numerically* dissipating due
to a bug — exactly the case PH-CON-002 should catch. The skip
therefore requires **both** positive evidence:

1. ``system_class`` hint says the system is dissipative-by-design
2. ``KE(t)`` is monotone-non-increasing across the rollout

Absent either signal, ``energy_drift`` defaults to its existing
fire-raw-value behavior (i.e., physics-lint v1.0-equivalent semantics).
This way: a supposed-conservative surrogate that's leaking energy
trips fires PH-CON-002 with the raw non-zero drift (because no
``system_class=="dissipative"`` hint); a known-dissipative SPH rollout
that the harness can confirm is monotone-decreasing skips with reason.

The two-half gate is the load-bearing methodology choice. Single-half
(monotone-decreasing alone) would over-skip; single-half (system_class
alone) would silently accept a buggy "dissipative" model that's
*increasing* KE while the metadata claims dissipative. Both halves
required.

**``system_class`` storage — dataset-name mapping initially, metadata
field eventually.** v1 implementation: small dict in the harness module
mapping LagrangeBench dataset names to system_class
(``{"tgv2d": "dissipative", "rpf2d": "dissipative", "ldc2d": "dissipative",
"dam2d": "dissipative"}`` — confirmed against LagrangeBench README:
all five SPH datasets are viscously dissipative; LDC2D (lid-driven
cavity) is wall-bounded forced convection but still dissipative in
the sense that without forcing the flow decays). The harness reads
``rollout.metadata.dataset`` and looks up the system_class.

This is a v1 shortcut. v1.1 promotes ``system_class`` to a proper
metadata field on the npz so non-LB datasets can declare it without
requiring a hardcoded mapping update. The writeup framing names the
v1 shortcut explicitly.

**No P1 block.** P1 (GNS-TGV2D) is unblocked. The energy_drift = 0.9999
is a property of TGV2D physics, not of either model — SEGNN-TGV2D
and GNS-TGV2D will both report essentially the same value because
they're surrogating the same dissipative system. The cross-stack
comparison is informative on ``dissipation_sign_violation``,
``mass_conservation_defect``, position-trajectory deviation, and
KE-decay-curve shape; ``energy_drift`` happens to be one column where
the two stacks agree by physics rather than by accident. Proceed to
P1 fire; D0-18 implementation slots between P1 PASS and rung-4
invocation so the rung-4 SARIF reads with the skip-with-reason filter
active on first invocation.

**Why D0-18 (not a sub-amendment of D0-17).** D0-17 + amendment 1
scope is "conversion module produces methodologically correct
velocities." D0-18 scope is "harness-layer rule-emission semantics
distinguish conservative from dissipative systems." The rule-semantics
question is downstream of the conversion (the harness only sees
correct velocities now; the question is what to *do* with them).
Same scope discriminator as D0-17-vs-D0-15-amendment-5 (within-scope
refinement → amendment, out-of-scope discovery → new D-entry).

**Sequencing.**

1. D0-18 entry committed first (this entry).
2. P1 GNS-TGV2D fires unblocked (parallel with D0-18 implementation;
   ~25 min, ~$0.50).
3. D0-18 option-1 implementation lands in physics-lint
   ``_harness/particle_rollout_adapter.py``: ``system_class`` mapping
   constant + skip-with-reason path in ``energy_drift``. Tests for
   both gate halves (system_class-without-monotonicity → fire raw;
   monotonicity-without-system_class → fire raw; both → skip).
4. Rung-4 invocation against both stacks' npzs with the skip-with-
   reason filter active. SARIF reads cleanly on first invocation.

**Realized.** This entry now.

**Amendment 1 — implementation landing.** physics-lint commit
``d03df3e`` adds the harness-layer skip-with-reason gate per the
design above. ``LAGRANGEBENCH_DATASET_SYSTEM_CLASS`` constant in
``particle_rollout_adapter.py`` maps the seven LB SPH datasets
(TGV2D, RPF2D, LDC2D, DAM2D + 3D analogs) to ``"dissipative"``;
``energy_drift`` adds a SKIP path that fires only when both gate
halves are positive (system_class hint + monotone-non-increasing
KE). 11 new tests in ``test_d0_18_dissipative_skip.py`` cover the
(system_class × monotonicity) 2×2 truth table, the mapping itself,
regression guards (synthetic dataset names — pre-D0-18 fixtures
unaffected), order precedence vs the D0-08 KE-rest skip (more
specific physical condition wins), and the reason-string contract
(SKIP reason names dataset + KE endpoints + signposts
``dissipation_sign_violation`` as the load-bearing alternative for
this system class). Full suite 95/95 green at d03df3e.

The harness-layer placement is preserved: ``src/physics_lint/rules/ph_con_002.py``
is unchanged. The skip-with-reason behavior fires only inside the
rollout-anchor harness, not in the public-API rule path. v1.0
behavior on the public rule is identical to its pre-D0-18 shipped
state; v1.1 promotion of ``system_class`` from hardcoded mapping
to per-rollout metadata field is the natural follow-on.

**Post-merge cleanup TODO (rung 4a, 2026-05-04).** The rung-4a v1.0 docs amendment on physics-lint `master` (commit `514c5af`, see `methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-design.md` §1.3 deferral #4 + the master README's `## v1.0 known limitations` section) embeds a cross-branch qualifier: *"the harness layer (currently on the `feature/rollout-anchors` branch pending merge to master, at `external_validation/_rollout_anchors/_harness/`)"*. **When `feature/rollout-anchors` merges to `master`, this qualifier becomes stale and must be edited out** — the merge resolves the qualifier's premise. The cleanup is a small README edit + a one-line commit on master post-merge; trigger named here so it gets caught at merge PR review rather than living forever as a fossil.

**Realized — post-merge cleanup landed.** PR #6 merged at `68d84c7a` on 2026-05-05. README.md cross-branch qualifier dropped in the same commit as this Realized note; the `physics-lint-validation DECISIONS.md` reference also rewritten to the in-tree `external_validation/_rollout_anchors/methodology/DECISIONS.md` path (the methodology lives in physics-lint post-migration `971b8fc`), and the "(post-merge)" qualifier on the writeup link removed since the writeup is now in master.

---

## Day 0+ status — Modal session unblocked from documentation gates

By end of Day 0 + the post-Gate-B review pass + the Day-0.5 review +
the pre-Modal pre-registration pass + the GPU-class refinement pass +
the rung-3 pre-scaffolding pass (2026-05-04), the load-bearing Gate B
has passed and fifteen DECISIONS.md entries (fourteen decision
entries D0-01 through D0-11 plus D0-13 / D0-14 / D0-15 + one
coverage-summary artifact D0-12) + one sub-entry (D0-04a) are
committed. The Modal session for Days 1–3 is now unblocked from
documentation gates and every meaningful failure mode has a
pre-registered escalation path:

- D0-04a fixes the spec §4.2 ambiguity reading.
- D0-07 pre-registers the Day 1 model-loading-path per-index threshold.
- D0-08 pre-registers the KE-rest skip-with-reason threshold.
- D0-09 pre-registers the mesh FD-noise tolerance.
- D0-10 pre-registers the hour-2 JAX micro-gate escalation
  (JAX-CPU pivot by default; no Modal-image debugging without
  re-authorization).
- D0-11 pre-registers the Day 2 hour-1 NGC audit decision matrix
  (velocity-derived vs node-field × scikit-fem-coercible vs
  DGL-native vs neither, with a 1h hard audit budget).
- D0-13 refines D0-10's GPU specifier ("A100" → "any CUDA GPU") and
  pre-registers stage-by-stage GPU-class defaults (T4 micro-gate /
  A10G Day 1 inference / A100 OOM fallback).
- D0-14 pre-registers the Modal Volume layout for rollout-anchors
  artifacts (checkpoints / datasets / rollouts; ``<git_sha>`` suffix
  on rollout filenames for reproducibility + cache invalidation).
- D0-15 pre-registers the rung-3 P0 invocation pattern (SEGNN-TGV2D
  checkpoint + README's documented ``mode=infer eval.test=True``
  pattern + plan §3.2 step 3 sample-size overrides). Amendment 4
  enables rollout persistence + pkl→npz conversion via the harness
  module ``_harness/lagrangebench_pkl_to_npz.py`` (SCHEMA.md v1.2).
- D0-16 captures the rung-3 + 3.5 P0 PASS verdict: 20
  schema-conformant ``particle_rollout_traj{00..19}.npz`` artifacts
  persisted to Volume at ``/vol/rollouts/lagrangebench/segnn_tgv2d_f75e22d8dd/``;
  patches-required distribution 3 infra / 3 config / 1 lib-internal
  / all 7 non-load-bearing.
- D0-17 surfaces and fixes the periodic-boundary wraparound bug in
  the velocity derivation, discovered via D0-16's forward-agenda
  spot-check. Strong anchor: SPH-integrated ground-truth positions
  give canonical ``U_0 ≈ 1`` under periodic-corrected central
  differences. Pre-D0-17 npzs unusable for PH-CON-002/003 on
  periodic datasets; post-D0-17 fix is no-op for non-periodic.
  SCHEMA.md v1.3 bump. Amendment 1 (post-D0-17 regen FAIL surfaced
  the LB upstream "PBC always length 3 regardless of dim" convention)
  adds truncation-with-audit-trail + standalone Modal conversion
  function for cross-rollout environment consistency. SCHEMA.md
  v1.4 bump.
- D0-18 pre-registers PH-CON-002 skip-with-reason for dissipative-
  by-design systems (harness-layer wrapping, NOT physics-lint v1.0
  core). Discovered via post-amendment-1 spot-check: energy_drift
  = 0.9999 on SEGNN-TGV2D is the physically correct measurement
  of viscous decay but the wrong rule semantics for SPH/NS systems.
  Two-half positive-evidence gate (``system_class`` hint AND
  monotone-non-increasing KE) avoids masking a buggy supposed-
  conservative surrogate. ``system_class`` via dataset-name mapping
  in v1; promotes to metadata field in v1.1.
- D0-03 routes PH-CON-001 through the mesh harness on the NS side.

The remaining gates are GPU-bound and require user-and-agent
Modal-session work:

- **Gate A** (Day 0 Audit Q1): deferred per D0-02; lands Day 2.
- **Hour-2 JAX micro-gate** (Day 1): JAX-on-CUDA on Modal A100.
- **Gate C** (Day 1 hour 4): JAX checkpoint loading + headline
  visibility. Triggers the model-loading half of the particle harness
  per D0-04 with the per-index threshold pre-registered in D0-07.
- **Gate D** (Day 2 hour 4): PhysicsNeMo NGC checkpoint usability.

The current autonomous-block (Day 0.5) is the particle harness
read-only path on synthetic `.npz` rollouts (PH-CON-001/002/003) —
no JAX, no Modal. Per the user's scoping clarification: synthetic
first as a rule-plumbing regression test (~2–3h), pre-recorded
LagrangeBench `.npz` only if available without JAX install.

---

## D0-19 — 2026-05-04 — Harness SARIF result schema (rung 4a pre-registration)

**Question.** Rung 4a will package the 40 npzs from rung 3.5 PASS into
committed SARIF artifacts via the existing `_harness/sarif_emitter.py`.
The artifact's contract — what fields are at the run level vs the result
level, what fields are guaranteed-identical across rows vs allowed to
vary, what the schema_version is, what the renderer can assume — is
load-bearing for the rung 4a writeup's "20 identical fires across both
stacks" claim. That claim is defensible only if the schema enforces the
identity, not if a grep happens to find it on this run's data.

**Decision (pre-registered before any code change).**

The harness SARIF schema gets formal field-level guarantees. Run-level
properties (`runs[0].properties`) carry constants per artifact. Result-
level properties (`runs[0].results[*].properties`) carry per-row data,
explicitly classified as guaranteed-identical-across-rows-within-stack
or may-vary.

Run-level fields (10 total):

- `source` (literal `"rollout-anchor-harness"` — discriminator vs public-API SARIF)
- `harness_sarif_schema_version` (string, e.g., `"1.0"`; renderer asserts on equality)
- `physics_lint_sha_pkl_inference` (sha at which LB CLI ran on Modal to produce pkls)
- `physics_lint_sha_npz_conversion` (sha at which pkl→npz conversion ran)
- `physics_lint_sha_sarif_emission` (sha at which the lint code emitted this SARIF)
- `lagrangebench_sha` (LB upstream sha — the inference engine producing the pkls)
- `checkpoint_id` (LB gdown identifier or symbolic name)
- `model_name` (LB CLI key, e.g., `"segnn"` / `"gns"`)
- `dataset_name` (LB dataset identifier, e.g., `"tgv2d"`)
- `rollout_subdir` (Volume artifact location at npz-genesis time)

The three `physics_lint_sha_*` fields **may be identical** (single-shot
run where inference + conversion + emission collapse to one sha) or
**distinct** (multi-session, as production SEGNN demonstrates: pkl
inference at `8c3d080`, npz conversion at `5857144`, SARIF emission at
post-`d03df3e`). Equality is allowed but never required. Renderer
assertion logic does NOT impose equality across stages.

`harness_sarif_schema_version` co-evolves with `physics_lint_sha_sarif_emission`
by construction (any schema change is a sha change), but is denormalized
into the SARIF for renderer assertion-locality.

Result-level fields:
- `traj_index` (int 0..19; may-vary)
- `npz_filename` (e.g., `"particle_rollout_traj00.npz"`; may-vary)
- `raw_value` (float, when defect emits a value; guaranteed-identical
  iff value is load-bearing-identical, as it is for the four 0.0 cells
  in 4a's data)
- `skip_reason` (string, when defect SKIPs; guaranteed-identical —
  template constant after the energy_drift change below)
- `ke_initial` (float; present only on `harness:energy_drift` SKIP rows; may-vary)
- `ke_final` (float; present only on `harness:energy_drift` SKIP rows; may-vary)

For a fixed (rule, stack), all 20 result rows MUST have identical
`ruleId`, `level`, `message.text`, plus either identical `raw_value` or
identical `skip_reason` (the existing `HarnessDefect` invariant: rule
emits exactly one of the two on every row).

Consumers MAY assert these invariants at render time. The schema makes
them checkable, not mandatory-to-check.

**Energy_drift skip_reason template change (forced by the contract).**

Current emission interpolates per-row varying KE values into the
skip_reason string:

    f"...KE(0)={e0:.3e}, KE(end)={float(e_series[-1]):.3e}..."

This makes skip_reason per-row varying, violating the "guaranteed-identical"
classification. D0-19 mandates a template-constant skip_reason; the
varying values move to dedicated `properties.ke_initial` / `ke_final`
fields on the SARIF row, attached by the new `lint_npz_dir.py` module.

New skip_reason template (interpolates `dataset_name` only, which is
constant per stack):

    f"system_class='dissipative' (dataset={dataset_name!r}); "
    "KE(t) monotone-non-increasing across the rollout; "
    "see properties.ke_initial / ke_final for values; "
    "consult dissipation_sign_violation as load-bearing alternative."

`HarnessDefect` itself stays unchanged (only `value` and `skip_reason`
fields). Other rules don't get `ke_initial` / `ke_final`.

**Schema version:** v1.0 (this entry pins it).

**Forward-flag (v1.x graduation question).** If a future D-entry
proposes graduating D0-18's dissipative-system handling into physics-
lint v1.x core, the SARIF convention divergence (harness-style emits
`level: "note"` for PASS-equivalent rows; public-v1.0 suppresses PASS
rows) must be revisited explicitly: either drop the note-level rows
(matching v1.0 PASS-suppression) or retain them as a deliberate two-
tier convention (harness-derived rules emit informational findings;
pure-v1.0 rules emit only on findings). Recorded here so the question
doesn't get rediscovered cold.

**Realized.** This entry now. SCHEMA.md §3.x extension lands at the
same sequencing position; D0-20 follows immediately as the consumer-
side counterpart. Implementation lands per the 14-step sequence in
`methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-design.md`
§4.

---

---

## D0-20 — 2026-05-04 — Generator-vs-consumer separation architecture (rung 4a pre-registration)

**Question.** D0-19 specifies *what's in* the harness SARIF artifact.
D0-20 specifies *how it's consumed* by the cross-stack writeup table
renderer. Orthogonal pre-registrations: D0-19 is the artifact contract;
D0-20 is the consumption architecture.

**Decision (pre-registered before renderer code).**

1. **Renderer lives in `methodology/tools/`** alongside other methodology
   tooling, not in `_harness/`. Generator (harness + driver under
   `_rollout_anchors/`) and consumer (renderer under `methodology/tools/`)
   communicate only through the SARIF artifact contract. **No Python
   imports cross the subtree boundary.** This makes the renderer
   testable in isolation against synthetic fixtures, and prevents the
   methodology subtree from accruing the harness as an import-time
   dependency.

2. **Schema version is the wire protocol.** Renderer hard-codes an
   `EXPECTED_SCHEMA_VERSION` constant. First action on reading any
   SARIF: assert `runs[0].properties.harness_sarif_schema_version ==
   EXPECTED_SCHEMA_VERSION`. On mismatch, raise `SchemaVersionMismatch`
   — **fail loud, not degraded**. No warnings, no log-and-continue, no
   best-effort coercion. The assertion is the load-bearing primitive
   that lets the cross-subtree separation hold without introducing
   silent version drift.

3. **Source-tag assertion** parallels schema-version: assert
   `runs[0].properties.source == "rollout-anchor-harness"`; raise
   `SourceTagMismatch` on failure. Distinguishes harness SARIF from
   public-API SARIF reaching the renderer by accident.

4. **Run-level field-presence assertions.** All 10 D0-19 run-level
   fields are required on read; missing → raise `MissingRunLevelField`
   naming the missing field. No defaulting.

5. **Test surface — golden + version-mismatch + asymmetric-shas.**
   - Golden: synthetic fixtures → rendered table matches an
     expected_table.md byte-for-byte.
   - Version-mismatch: bumped-version fixture (programmatically derived)
     → SchemaVersionMismatch raises.
   - Source-tag mismatch: wrong-source fixture (programmatically derived)
     → SourceTagMismatch raises.
   - Missing-field: deleted-field fixture (programmatically derived) →
     MissingRunLevelField raises.
   - Asymmetric-shas: feeds renderer fixtures with deliberately-distinct
     three-stage shas; asserts renderer handles them correctly without
     equality assumption (per D0-19's may-be-identical-or-distinct
     contract).
   - Aggregation: "all 20 identical → single cell" detection fires on
     uniform values; falls back to summary stats on non-uniform.

6. **Test fixtures hand-crafted synthetic-but-realistic, NEVER copied
   from production.** Production SARIFs have incidental properties
   (specific shas, paths, run-time-only fields) that have nothing to do
   with the schema contract; copying them couples the test to those
   incidentals. Fixtures use placeholder shas (`synthetic_inference_sha`,
   etc.) and synthetic dataset names. SEGNN fixture defaults to
   asymmetric three-stage shas (matching production); GNS fixture
   defaults to collapsed shas (also matching production). Negative-path
   fixtures (bumped_schema, wrong_source, missing_field) derived
   programmatically at test time from the canonical fixture rather than
   committed as separate files.

7. **Renderer output convention.** Renderer is pure stdin-out: emits
   markdown table to stdout, never writes to the writeup file directly.
   Writeup includes the table via copy-paste, plus a rederivability
   footer with the exact render command + sha at which the table was
   rendered. The footer is what converts copy-paste from "two artifacts
   that happen to agree right now" into "an artifact reproducible from
   the commit-pinned source."

**Reframing note.** Pre-migration, this would have been "cross-repo"
separation (renderer in physics-lint-validation, generator in physics-
lint). Post-migration (`971b8fc`), the methodology subtree lives in
physics-lint alongside the harness; the discipline is intra-repo
subtree separation with the same wire-protocol shape. The cross-repo
URL discipline reduces to sibling-relative paths for in-repo navigation
(commit-pinned GitHub URLs as optional secondary for GitHub-rendered-
markdown readers).

**Realized.** This entry now. Renderer implementation lands per the
14-step sequence in
`methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-design.md`
§4 step 12.

---

## D0-21 — 2026-05-05 — Rung 4b cross-stack equivariance (pre-registration)

**Question.** Rung 4b extends rung 4a's cross-stack conservation result to
equivariance: compute first-step ε under rotation/reflection/translation
on the existing 40 frozen rollouts; emit SARIF at schema_version v1.1;
render a tripartite-grouped cross-stack equivariance table. Multiple
sub-questions (threshold rationale, rule set, persistence, GPU class,
renderer choice, schema bump) need pre-registration before code lands so
the implementation can't drift from the design's load-bearing claims.

**Decision (pre-registered before any code change).**

Composite entry covering rung 4b's design surface. Full rationale lives
in `methodology/docs/2026-05-05-rung-4b-equivariance-design.md`
(committed at `d9a8baa`, branch `feature/rung-4b-equivariance`); this
entry names the load-bearing decisions for searchability in
DECISIONS.md.

1. **Load-bearing claim.** "physics-lint's PH-SYM rule schema, with
   equivariance thresholds set at the float32 numerical-precision floor
   (ε ≤ 10⁻⁵ PASS, 10⁻⁵ < ε ≤ 10⁻² APPROXIMATE, ε > 10⁻² FAIL), runs
   unmodified across SEGNN-TGV2D and GNS-TGV2D rollouts; per-stack ε
   values are emitted in the same SARIF schema as 4a (schema_version
   v1.1) and reported as observed." Frozen at design time. Robust to
   both probe outcomes (SEGNN at floor expected; GNS at floor or in
   APPROXIMATE band — secondary finding wording adapts post-hoc).
2. **Threshold rationale.** Float32 numerical-precision floor (~10⁻⁷/op
   accumulating with depth and conditioning). Architecture-agnostic, no
   cross-paper citation needed at threshold layer. Helwig et al. ICML
   2023 demoted to writeup body context for interpreting observed GNS
   values.
3. **Rule set (γ).** PH-SYM-001 active at θ ∈ {π/2, π, 3π/2} +
   construction-trivial smoke at θ=0; PH-SYM-002 active at y-axis
   reflection through box center; PH-SYM-003 SKIP-with-reason
   ("PBC-square breaks SO(2) symmetry — rotated cell doesn't tile with
   original"); PH-SYM-004 active at t=(L/3, L/7), construction-trivial
   PASS at machine-zero.
4. **Tripartite evidence framing in writeup body and renderer output:**
   architectural-evidence rows / construction-trivial rows /
   substrate-incompatible-SKIP. Prevents the reader from interpreting
   four PASSes as the same kind of evidence.
5. **PH-SYM-003 / PBC-square IS analogous to PH-CON-002 / dissipative**
   under D0-18's skip-with-reason mechanism. Meta-correction: this was
   missed at the brainstorm's non-claim-5 framing, surfaced at design
   pass not mid-execution. Recording the correction makes the
   writeup-framing-baked-into-design discipline self-validating.
6. **Trigger-vs-emission separation for SKIP paths:** rule-specific
   trigger logic ("substrate has periodic boundaries AND rotation angle
   is non-trivial-symmetry of substrate cell") in
   `symmetry_rollout_adapter.py`; shared D0-19 §3.4 emission machinery
   (skip_reason as guaranteed-identical-across-rows) populates the
   SARIF row. Same emission shape as PH-CON-002, different trigger
   source.
7. **Reportable ε.** First-step scalar ε per (rule, transform_param,
   traj_index) in SARIF (load-bearing); ε(t) over T=100 in writeup
   figure (supplementary, not threshold-classified). Compared quantity:
   positions only (matches LagrangeBench's published position-MSE
   metric). Per-particle aggregation: RMS-across-particles
   (ε = sqrt(mean_i ‖R⁻¹ r'_i,1 - r_i,1‖²)).
8. **Pipeline shape (II) with uniform single-tier artifact:** every
   (rule, transform_param, traj_index) tuple persists one ε(t) npz;
   T_steps=1 for non-figure-subset (~hundreds of bytes), T_steps=100
   for figure-subset (~few KB, 6 traces total: 1 angle × 3 trajs ×
   2 stacks). Single schema, single consumer code path
   (`lint_eps_dir` reads `eps_t[0]` always; figure renderer filters
   for `len(eps_t) > 1`). Preserves rung 4a's single-artifact-tier
   structure.
9. **Persistence.** Modal-Volume-only with local mirror (gitignored),
   matching rung 4a's pattern. SARIF + verdicts committed; ε(t) npzs
   not committed. Rotated state intentionally non-persisted (lives in
   Modal-job memory only); recovery via sub-minute re-run.
10. **GPU class A10G** (matched to rung 4a). D0-17 amendment 1
    principle generalized from within-rollout sha consistency to
    GPU-class consistency — at float32-floor measurement scale,
    GPU-class FP behavior differences (TF32, FMA scheduling,
    kernel-fusion) sit at the same order as the threshold (~10⁻⁶–10⁻⁷).
    Matched A10G keeps measurement noise below the floor-vs-architectural
    -error distinction.
11. **SARIF schema_version v1.0 → v1.1 bump.** New rule ids: PH-SYM-001,
    PH-SYM-002, PH-SYM-003, PH-SYM-004. New per-row extra_properties:
    `eps_pos_rms` (None for SKIP), `transform_kind`, `transform_param`.
    v1.0 fields unchanged. Schema_version field on every SARIF row
    enables fail-loud renderer assertion. v1.0 SARIFs unreadable by
    v1.1 renderer; v1.1 SARIFs unreadable by v1.0 renderer.
12. **Sibling renderer choice.** Rung 4b ships
    `methodology/tools/render_eps_table.py`, separate from rung 4a's
    `render_cross_stack_table.py`. Each renderer focused on one schema
    version. Rationale: rung 4b's tripartite-grouped layout differs
    structurally from rung 4a's flat per-rule list; modifying rung 4a's
    renderer risks regressions on v1.0 behavior. Code-duplication is
    bounded; DRY-ification of formatting primitives via
    `methodology/tools/render_lib.py` is a forward-flag for n=3 schema
    versions, not preemptive at n=2.
13. **4-stage provenance for ε(t) npzs.** Generalizes D0-19's 3-stage
    shape: `physics_lint_sha_pkl_inference` and
    `physics_lint_sha_npz_conversion` carried unchanged from rung 4a's
    reference rollouts; `physics_lint_sha_eps_computation` (new)
    recorded at ε(t)-computation time; `physics_lint_sha_sarif_emission`
    recorded at SARIF-emission time (consumer-side). The four shas may
    be identical or distinct.

**Forward flags (recorded for future rungs / case studies):**

- *Float32→float64 threshold-rationale revisit:* the float32 floor
  argument is precision-dependent. Case study 02 (PhysicsNeMo MGN), if
  it runs at float64, needs threshold revisiting (float64 floor
  ~10⁻¹⁶ per op).
- *(rule, substrate) compatibility matrix generalization:* D0-18
  detects PH-CON-002 SKIP via dataset-name + system_class heuristics;
  PH-SYM-003 / PBC-square SKIP triggers on a different (rule,
  substrate) compatibility axis. Future rung could formalize a per-rule
  compatibility matrix in the rule schema rather than per-rule
  heuristic triggers.
- *Sibling-vs-extend renderer choice:* at n=3 schema versions, extract
  shared formatting primitives into `methodology/tools/render_lib.py`.
- *(I)-rejection framing precision:* the full-rotated-state pipeline
  (Option I in the design) was rejected for over-persistence-for-
  unclear-ROI under Modal-Volume cost, *not* commit-bulk. Future
  readers should not interpret it as a commit-bulk rejection.
- *Translation vector "non-grid-commensurate" wording:* L/3 and L/7
  are rational fractions of L; the structural argument doesn't depend
  on Diophantine incommensurability. Captured here for terminology
  precision.

**Realized.** [Filled in post-execution at the rung 4b table writeup
step, naming the merge sha that closes this rung's loop.]

## D0-22 — 2026-05-07 — Rung 4c substrate-class extension to dam-break-2D (pre-registration)

**Status:** pre-registered before code change.
**Predecessor:** D0-21 (rung-4b cross-stack equivariance), D0-18 (dissipative-system skip-with-reason), D0-19 (harness SARIF result schema), D0-08 (KE-rest skip on `energy_drift`).
**Trigger:** rung-4c design pass (`methodology/docs/2026-05-07-rung-4c-substrate-class-extension-design.md`); source review of `_harness/particle_rollout_adapter.py` surfaced that `LAGRANGEBENCH_DATASET_SYSTEM_CLASS["dam2d"]` was preemptively classified as `"dissipative"` during the D0-18 design pass, before any dam-break-2D rollout was measured.

**Decision.** Extend the harness's substrate-detection layer to a second LagrangeBench substrate class — `open-driven-dissipative` — via:

1. **Taxonomy bump.** `LAGRANGEBENCH_DATASET_SYSTEM_CLASS` shifts from implicit-binary (`"dissipative"` or absent → fire-raw) to explicit-tri-state at minimum: `"dissipative"`, `"open-driven-dissipative"`, absent. Future taxonomy entries (e.g., `"strictly-conservative"` for case study 02 anchor) are forward-compatible.

2. **`dam2d` reclassified empirically** from `"dissipative"` to `"open-driven-dissipative"`. Justification: dam-break-2D's KE rises during gravity-loaded fall (gravitational PE → KE conversion), violating the closed-dissipative class's monotone-non-increasing precondition. Empirical verification via 1-traj Modal smoke at pre-flight step 4 (rung-4c design §3.3); the reclassification ships as a TDD-green hypothesis at implementation time, with the smoke at pre-flight step 4 as the gating empirical verification — ABORT-on-smoke-fail triggers a revert of the reclassification commit before any 20-traj fire (pre-flight log records the gate decision).

3. **New SKIP path on `dissipation_sign_violation`** for `system_class == "open-driven-dissipative"`. The rule's strictly-dissipative-or-conservative assumption (docstring-stated zero-for-strictly-dissipative-or-conservative behavior) doesn't apply to open-driven systems where `dE/dt > 0` over a stretch by physics. SKIP-reason template:

   ```
   system_class='open-driven-dissipative' (dataset='<name>'); dE/dt > 0
   over a stretch by physics (gravitational PE → KE conversion); the
   strictly-dissipative-or-conservative assumption underpinning
   dissipation_sign_violation does not apply. See DECISIONS.md D0-22.
   ```

4. **Forward-flag two-tier split** for catalogue-wide reclassification:
   - **Almost certainly misclassified, awaiting empirical probe:** `rpf2d` (reverse Poiseuille, forced flow), `ldc2d` (lid-driven cavity, forced flow), `rpf3d`, `ldc3d` (3D variants).
   - **Likely correctly classified but unverified:** `tgv3d` (3D-TGV inherits 2D-TGV physics; closed dissipative system, KE monotone-non-increasing as turbulent KE decays).
   The two-tier split preserves future-rung actionability: a rung exercising rpf or ldc walks into a known-misclassification (and the empirical probe is its first move); a rung exercising tgv3d treats it as inherited-from-validated.

5. **D0-08 (KE-rest IC SKIP) is NOT absorbed into D0-22.** D0-08 gates on `KE(0) < KE_REST_THRESHOLD`, orthogonal to the `system_class` taxonomy. D0-22 gates on `system_class == "open-driven-dissipative"`. The two SKIP gates are sibling members of a "substrate-detection family" — distinct physical preconditions, distinct SKIP reasons. On dam-break-2D specifically, D0-08 fires on `energy_drift` (start-at-rest IC) and D0-22 fires on `dissipation_sign_violation` (open-driven by physics); both fire on the same rollout but on different rules.

6. **(rule, substrate) compatibility matrix forward-flag from D0-21 cited but not promoted.** D0-21 §forward-flag-2 named "the (rule, substrate) compatibility matrix as a future generalization." Rung 4c demonstrates that pattern's empirical instance (`dissipation_sign_violation × open-driven-dissipative` becomes a new SKIP cell) without yet promoting the matrix to a first-class rule-schema field. Promotion is post-rung-4c work.

**"Classify when you exercise" empirical-classification principle.** Named here as a first-class methodology output of the rung-4 series, generalizing the precedent set by rung-4b's PH-SYM-003 PBC-square-SO(2) substrate-incompatibility SKIP (classified only for the substrate measured, not retrospectively over all (rule, substrate) combinations). Pattern reads as: **substrate properties get verdicts only after empirical probing, never on theoretical intuition alone, even when the theoretical guess is almost certainly correct.** Applies to any future rung exercising a previously-unmeasured substrate.

**Source-review-catches-issue-before-compute pattern (now trilateral).**
1. Rung 4b first-pass math correction (TRAIN_PUSHFORWARD_UNROLLS_LAST=3 conflated +1 target with pushforward count) — caught at LB source review at sha `b880a6c`.
2. Rung 4b first-pass latent figure-sweep failure (`valid.h5` hardcoded `subseq_length=10` vs. dynamic upstream value) — caught at the same source review pass.
3. Rung 4c catalogue-misclassification (`dam2d → "dissipative"` was preemptive and wrong; `rpf2d`/`ldc2d` likely the same) — caught at source review of `particle_rollout_adapter.py` during this design pass.

Three instances at $0 Modal cost. To be elevated as a first-class methodology contribution of the rung-4 series in integrating-README composition rather than three scattered amendments. The pattern reads as: **a source-review pre-flight pass between design and execution catches issues that brainstorm-only and execution-only review miss; the cost is hours of source reading, the saving is multiple GPU runs and methodology errors that would otherwise land in writeups.**

**Forward flags carried into D0-22:**
- Bilateral D0-18 still requires a strictly conservative substrate anchor (case study 02 territory); rung 4c does not exercise this.
- PH-SYM substrate-symmetry-SKIP (gravity-direction-pinning breaks SO(2) on dam-break-2D) → rung 4d.
- PH-BC-particle-wall not in v1.0 rule path → out of scope; plan v2.1 amendment removes "PH-BC (wall)" from §3.1 P1.
- Catalogue-wide reclassification (`rpf2d`/`ldc2d`/`rpf3d`/`ldc3d`) deferred to future rungs; D0-22 records the two-tier split for actionability.

**Implementation:** rung-4c plan (`methodology/docs/2026-05-07-rung-4c-substrate-class-extension-plan.md`). Acceptance criteria in design §8.

**Plan-diff (Task 1 step 1.2 retrofit):** rung-4c plan §Task 1 step 1.2 stated `grep -c "^## D0-" DECISIONS.md` should return 8 ("D0-15 through D0-22"); the actual returns the full file count, which is 22 after this commit. Plan expectation was an off-by-counting-scope error; the commit count check is unchanged in intent (verify the new entry parsed cleanly), only the expected number is updated.

**Plan-diff (Task 1 D0-22 §2 wording reconciliation):** the plan's §2 text said "reclassification commits *after* smoke confirms the rise-then-fall shape, not before." This conflicts with Task 4's commit ordering (which lands the dam2d label flip before Task 7's smoke). Reconciled here as TDD-green hypothesis with smoke as gating verification + ABORT-on-fail revert path; preserves the empirical-justification discipline while letting the test-driven plan order proceed. The substantive contract is unchanged — Task 7 ABORT means the reclassification does not survive in the production fire.

### D0-22 amendment 1 — 2026-05-07 — Extend SKIP gate to `energy_drift` on `open-driven-dissipative`

**Status:** in-rung amendment landed alongside the parent D0-22 entry; mirrors D0-15's amendment-layered pattern (D0-15 amendments 1-4 were all within-rung refinements surfaced during execution).

**Trigger:** rung-4c pre-flight smoke (`preflight/2026-05-07-rung-4c.txt`, Step 5 pipeline smoke) surfaced that `energy_drift` fires a methodologically meaningless raw value of ~2500-2700 on dam-break-2D. Mechanism:
- KE(0)=0.4703 (input window has the column already falling at t=0)
- max(KE)=O(1000) (gravity-loaded fall + dissipation phase)
- `KE_REST_THRESHOLD=1e-10` is an absolute threshold; KE(0)=0.47 clears it by 9 orders of magnitude
- D0-08 KE-rest gate does NOT fire; energy_drift falls through to raw computation
- Raw value: `max|KE(t)-KE(0)|/|KE(0)|` ≈ 2700 — large but uninformative because the KE rise IS the physics, not a model defect

The same strictly-dissipative-or-conservative assumption that D0-22 catches on `dissipation_sign_violation` (the rule assumes `dE/dt ≤ 0` everywhere) ALSO fails for `energy_drift` (the rule assumes `|E(t)-E(0)|` stays small relative to E(0)). Both rules' assumptions fail under the same physics and on the same substrate class.

**Decision.** Extend D0-22's substrate-class dispatch to cover `energy_drift` on `open-driven-dissipative`:

```python
if system_class == "open-driven-dissipative":
    return HarnessDefect(
        value=None,
        skip_reason=(
            f"system_class='open-driven-dissipative' (dataset={dataset_name!r}); "
            "KE grows by orders of magnitude due to physics (gravitational PE → "
            "KE conversion); the strictly-dissipative-or-conservative assumption "
            "underpinning energy_drift does not apply. See DECISIONS.md D0-22 "
            "(amendment 1)."
        ),
    )
```

Gate is purely `system_class`-conditioned (parallel to D0-22's `dissipation_sign_violation` gate; no co-condition on KE shape). D0-08 takes precedence over D0-22 amendment 1 on `energy_drift` (KE-rest IC → D0-08 wins) — verified by `test_energy_drift_d0_08_takes_precedence_over_d0_22_amendment_1`. D0-18 and D0-22 are mutually exclusive on `system_class` value, so order between them is irrelevant.

**Why amendment, not new D-entry.** Within-rung refinement of the principle D0-22 already established. The substrate class (`open-driven-dissipative`), the failing assumption (strictly-dissipative-or-conservative), the SKIP-with-reason mechanism, and the D0-19 §3.4 template-constant invariant are all unchanged from D0-22 §3. The only new fact is that the principle generalizes from `dissipation_sign_violation` to `energy_drift` — an empirical observation the smoke surfaced. Mirrors D0-15 amendment 4 (rollout-persistence enablement surfaced during rung-3 execution) and D0-17 amendment 1 (PBC-truncation convention surfaced during rung-3.5 execution) — both within-rung amendments to existing decisions when in-rung work surfaced the gap.

**Why not Option C (rework D0-08 as relative threshold).** Rejected. D0-08's absolute threshold is validated against rung-4a's TGV2D conservation behavior; cross-cutting it now risks regressing rung 4a. D0-22's substrate-class dispatch is the right surface for "this rule's assumption fails on this substrate class" — exactly what amendment 1 captures.

**Test coverage:** 3 new tests in `test_d0_22_open_driven_skip.py`:
1. `test_energy_drift_skip_when_open_driven_with_rise_then_fall_ke` — positive path
2. `test_energy_drift_d0_08_takes_precedence_over_d0_22_amendment_1` — sibling-gate precedence
3. `test_energy_drift_d0_22_amendment_1_reason_template_constant` — D0-19 §3.4 invariant

Plus the existing 7 tests continue to cover the dissipation_sign_violation gate; total D0-22 coverage = 10 tests.

**SARIF impact (rung-4c artifacts):** the dam-break SARIFs emitted at Task 9 will now show `energy_drift: SKIP D0-22 amendment 1` instead of `energy_drift: raw=2700-ish`. Two distinct skip_reasons within the same rule across stacks (one per class label); the renderer's D0-19 §3.4 template-constant invariant holds within each (rule, stack) pair.

**Methodology contribution this amendment elevates:**
> The substrate-class-extension principle that D0-22 introduced for one rule (`dissipation_sign_violation`) generalizes naturally to OTHER rules with the same failing assumption (`energy_drift`). When an empirical probe surfaces that a rule's assumption fails on a specific substrate class, the dispatch should extend to ALL rules that share the assumption — not just the one the rule's named after.

To be elevated as a first-class methodology output of the rung-4 series in integrating-README composition: substrate-class dispatch is per-substrate-class, not per-rule; the rule-axis enumeration follows from the assumption-axis enumeration.

**"Smoke surfaces honest-limits" pattern (now bilateral within rung 4c).** Smoke caught:
1. Empirical confirmation of dam2d rise-then-fall KE → D0-22 reclassification justified (PRIMARY).
2. D0-08 absolute-threshold misfire on dam-break + energy_drift's strictly-dissipative-or-conservative assumption failure → D0-22 amendment 1 (SECONDARY, fixed in-rung).

The smoke gate's PROCEED authorization is independent of the secondary finding (D0-22's load-bearing claim is unaffected); the amendment is "next-step refinement to close the loop the smoke opened, before the writeup hardens the artifact."

### D0-22 amendment 2 — 2026-05-07 — Rung 4c ships at N=12 trajs (cost-driven N reduction; in-rung methodology-consistency over presentational uniformity)

**Status:** in-rung amendment landed alongside Task 8 production fire; mirrors amendment 1's shape (smoke-discovered drift addressed in-rung rather than ship-with-caveat).

**Trigger:** rung-4c production rollout (Task 8, plan §3.4 `~5 min A10G` per stack at N=20). Empirical timing on dam-break-2D:
- Smoke 1-traj: SEGNN 159.6s, GNS 120.0s (includes JIT + 1 traj inference)
- Production at N=20 (sha e754a4bc2e, SEGNN-dam2d): subprocess timed out at 2400s with 12/20 trajs converted-pending; amortized rate ~200s/traj
- 20-traj projection ~67 min (SEGNN) and ~50 min (GNS) — both exceed the function's 2400s subprocess timeout
- Plan estimate was ~10x optimistic on per-traj inference cost on dam-break (more particles, more neighbour-list work than TGV-2D)

**Decision.** Ship rung 4c at N=12 trajs across both stacks rather than blow the per-rung budget to maintain N=20 cross-rung parity with rung-4a/4b:
1. Convert the 12 SEGNN-dam2d partial pkls already on Volume via `convert_pkls_p1_segnn_dam2d` standalone path (CPU-only, ~$0.05).
2. Fire GNS-dam2d at `n_trajs=12` (matched to SEGNN's effective N, ~15 min A10G, ~$0.21).
3. Update both dam2d rollout functions to `eval.infer.n_trajs=12` canonically (working-tree-only override during the smoke run promoted to permanent setting; future re-fires reproduce the N=12 ship). Inline comments cite this amendment + the preflight log.
4. Production cumulative compute: ~$1.00 (~5x the per-rung $0.20 estimate, ~3% of the ~$30 total-validation-budget ceiling).

**Why amendment, not ship-with-caveat (Option A vs Option B per session log).**
- The rung-4 series has been consistently applying a "smoke-discovered drift gets honest in-rung correction" pattern (rung 4b LB config-load → amendment 2; rung 4b latent figure-sweep → §14.6; rung 4c dam2d preemptive misclassification → caught pre-rung; rung 4c energy_drift methodologically-meaningless raw → D0-22 amendment 1).
- Option A ("re-fire at N=20 with extended timeouts to maintain cross-rung N parity") would have been the first deviation from that pattern — spend more compute to preserve presentational uniformity rather than methodologically address the smoke-discovered finding.
- Option B ("ship at N=12 with honest amendment") fits the established pattern: smoke surfaces a drift, the rung documents the drift, the data ships at the value the budget supports, the structural claims hold at N≥2 and are unaffected.
- Cross-rung presentational cost of N=12 vs N=20: one-character visual on the cross-stack table ("x12 identical" vs "x20 identical"); half-line addition to the cover letter ("rung-4a TGV2D at N=20 trajs; rung-4c dam-break-2D at N=12 trajs (per-traj inference cost ~10x estimate; D0-22 amendment 2)"). Small.
- Methodology-precedent cost of Option A: future rungs inherit "if uniformity is at risk, blow the budget" — a worse precedent than "ship at the data the budget supports, document the drift honestly."

**Why this is "amendment 2" not a new D-entry.** Same shape as amendment 1: in-rung response to a smoke-discovered drift, sharpens the methodology trail, no scope expansion. The substrate-class extension principle D0-22 introduced is unchanged; amendment 2 records a separate but related precedent — when smoke surfaces unexpected per-traj cost, the disciplined response is to ship at supportable N with honest amendment rather than budget-fund presentational uniformity.

**Methodology contribution this amendment elevates:**
> Plan-estimated compute can be ~10x off — a 1-traj smoke gives an early signal but does not fully calibrate per-traj cost on a new substrate (dam-break has more particles than TGV-2D and more neighbour-list overhead). When smoke discovery surfaces this, the rung-4 series's discipline is to ship at the N the budget supports with honest acknowledgment, not to spend more compute to preserve presentational uniformity. The "smoke surfaces honest-limits, in-rung correction is the response" pattern from amendment 1 generalizes from methodology-correctness gaps (extend the SKIP gate) to data-scope gaps (reduce N with documented reasoning).

To be foregrounded in integrating-README composition alongside amendment 1: the "smoke surfaces, rung corrects in-flight, writeup acknowledges honestly" trilateral is a first-class methodology contribution of the rung-4 series.

**Test impact:** none. The amendment is a data-scope decision, not a code-logic change. The 458 existing tests continue to pass. The harness's substrate-class dispatch (D0-22 + amendment 1) is the same regardless of N.

**SARIF impact (rung-4c artifacts at Task 9):** dam-break SARIFs will carry 36 rows each (3 rules × 12 trajs) instead of 60 rows (3 rules × 20 trajs). Renderer's D0-19 §3.4 invariants hold uniformly. Cross-stack identity check: SEGNN-dam2d and GNS-dam2d emit byte-identical structural row distributions at N=12 (verified post-fire via `lint_npz_dir` on both production rollouts at sha e754a4bc2e).

**Provenance shas captured for emit_sarif (Task 9):**
- SEGNN-dam2d: pkl_inference=`e754a4bc2e` (Task 8 fire that timed out at 12 trajs); npz_conversion=`e754a4bc2e` (standalone conversion via `convert_pkls_p1_segnn_dam2d` at same sha)
- GNS-dam2d: pkl_inference=`e754a4bc2e` (Task 8 fire at n_trajs=12); npz_conversion=`e754a4bc2e` (in-fire conversion succeeded)

**Forward-flag (future rungs / case studies):** if a future rung needs N=20 dam-break rollouts (e.g., for rung 4d gravity-direction-pinning PH-SYM-001/002/003 SKIP probe), the timeout strategy needs an explicit refactor: subprocess `timeout=2400` is a per-fire ceiling matched to TGV-2D's per-traj cost; dam-break needs `timeout=5400` minimum + corresponding function `timeout=6000`. Out of scope for rung 4c; recorded here so the timeout-as-substrate-property observation isn't lost.

---

## D0-23 — 2026-05-11 — Case Study 02 Phase 1 audit verdicts (resolved at 2026-05-13; Tasks 1-12 verdicts pinned)

**Status:** open. Skeleton pre-registered before Phase 1 fires; verdicts populate as audit tasks land per `methodology/docs/2026-05-11-case-study-02-physicsnemo-mgn-phase-1-plan.md`.

**Predecessor:** D0-02 (Gate A deferred to Day 2), D0-11 (Day 2 hour 1 NGC audit decision criterion), D0-22 (substrate-class taxonomy).

**Verdicts (populated by Phase 1 tasks):**

1. **BLOCKING-1 (Task 1 + 7):** NGC checkpoint ↔ PhysicsNeMo v2.0.0 state-dict compatibility.
   - Verdict: **key-set match = PASS** (via documented name-remap adapter; 262 expected = 262 actual = 262 common, keys *and* shapes — `model.load_state_dict(remapped, strict=True)` succeeds, re-confirmed on the torch-2.10 image 2026-05-12). **Architecture identity = FALSIFIED by Gate D (verdict 4):** the adapter loads cleanly but the resulting model's one-step velocity prediction on a real cylinder_flow test trajectory is off by ~0.80 (max-abs) — ~40× the normalized signal scale, i.e. garbage, not "a bit inaccurate". So the rename maps key *names* + tensor *shapes* correctly but maps weights to the wrong *slots* (most plausibly a pairwise edge↔node swap or a different interleave order in v2.0.0's `MeshGraphNetProcessor.__init__` than the adapter assumed, or the modulus-era runtime block-application order differing). Per the verdict-1 acceptance-criteria amendment ("architecture identity confirmed by Gate D"), that amendment's confirmation condition is **not met** — see verdict 4 + the Gate-D-failure debug ladder. **[Re-audit update 2026-05-12: the residual was found — the edge-block MLP's *internal* concat order (`[src, dst, edge]` in modulus v0.1.0's `_EdgeBlock` vs `[edge, src, dst]` in v2.0.0's `concat_efeat_pyg`), not an edge↔node swap or an interleave error. The name-remap adapter now also permutes the first-edge-Linear input columns; the ONE pre-registered A10G re-fire dropped `E` 0.80 → 0.0352 (~23×), so the forward is now structurally MGN-on-cylinder_flow ⇒ architecture identity confirmed. The remaining `E = 0.0352` max-abs (Band B) was a tolerance-calibration question, resolved by the verdict-confirmation re-fire: **1-step velocity RMSE = 3.19×10⁻³ ≤ 5×10⁻³** (≈ 1.36× the Pfaff-et-al. CylinderFlow baseline 2.34×10⁻³) ⇒ **Gate D PASS, architecture identity CONFIRMED.** See verdict 4 + "Band-C re-audit result (2026-05-12)" + "Verdict-confirmation re-fire" below.]**
   - NGC artifact: `nvidia/modulus/modulus_ns_meshgraphnet:latest` (catalog only tags `latest`; the plan's initial `:v0.1` guess does not exist).
   - Download URL (direct HTTP, unauthenticated): `https://api.ngc.nvidia.com/v2/models/nvidia/modulus/modulus_ns_meshgraphnet/versions/latest/files/vortex_shedding_mgn.zip`.
   - Zip sha256: `5391bb826448871608722f4e58d476a3205bcb39582a2a42e6a6ea29c5b0f710` (25,994,483 bytes; matches catalog's 24.79 MB).
   - Checkpoint sha256 (`vortex_shedding_mgn/model.pt`): `0153803c8b2c0947948757a2298eec6ef21e8ea28131834963db08f34b4a0726` (28,203,057 bytes).
   - Source: `verify_ngc_checkpoint_state_dict_compat` Modal entrypoint (canonical, runs in pinned image) + `tests/test_ngc_checkpoint_v2_0_0_compat.py` (local, self-skips when physicsnemo absent). Both 262 expected = 262 actual = 262 common after adapter.
   - **Acceptance-criteria amendment** (escalated + user-approved): the plan's "as-shipped state_dict must load directly" becomes "loads via documented name-remap adapter at `_legacy_checkpoint_name_remap.py`; architecture identity confirmed by Gate D's NGC sample reproduction within Phase-1-pinned tolerance." The amendment is recorded here because Pattern A drift surfaced during Task 4 execution.
   - **Rename detail.** The NGC checkpoint was trained against pre-rename modulus (uploaded 2023-05-26). Encoder/decoder rename: `*.mlp.*` → `*.model.*` (22 keys). Processor restructure: parallel `processor.{edge,node}_blocks[K]` ModuleLists became a single interleaved `processor.processor_layers` (240 keys); the new K-th layer is an edge_mlp when K even and node_mlp when K odd, per physicsnemo v2.0.0's `MeshGraphNetProcessor.__init__` which constructs via `chain(*zip(edge_blocks, node_blocks))`. Forward-pass computation is identical; only parameter paths differ — falsifiable by Gate D failure.
   - **Refinement 1: `device_buffer` extra key.** Inspected before adapter authoring per design escalation. Result: `shape=[0], numel=0, dtype=float32` — a 0-element placeholder tensor from the trainer's device-tracking machinery with no learned state. Adapter drops it unconditionally (`_DROP_KEYS = {"device_buffer"}`); dropping cannot cause silent wrong outputs since there is nothing to drop. Audit-trail entrypoint: `inspect_ngc_checkpoint_extras`.
   - **Mechanism drift, captured.** The plan's Task 4 used `ngc registry model download-version` via NGC CLI 3.41.4. That path failed silently with HTTP 403 on the storage backend (consistent with NVIDIA's documented Modulus → PhysicsNeMo deprecation of the old NGC artifact stack; CLI exits 0 even when "Download status: FAILED"). Switched to direct HTTP GET via `urllib.request`; verified unauthenticated via `probe_ngc_direct_urls` (200 OK from both Ahmed-body control and modulus_ns_meshgraphnet across unauth / bearer / apikey).

2. **Loader-contract audit findings (Task 5 part 3):** preflight V1-V18 + the 5 secondary known-unknowns, run against the actual DeepMind cylinder_flow data via `audit_cylinder_flow_loader_contract`.
   - Verdict: **all 19 boolean checks PASS** (V1-V18 + the secondaries). Full report: `02-physicsnemo-mgn/preflight/loader_contract_audit.json` (mirror of the entrypoint's `/vol/datasets/cylinder_flow/loader_contract_audit.json`). Highlights: `meta.json`/`test.tfrecord` shas match the pins above (V1/V2); `field_names = [cells, mesh_pos, node_type, velocity, pressure]`, dtypes `int32`/`float32` all in the supported map (V3/V4); `trajectory_length = 600` (V6); **no `dynamic_varlen` fields** in cylinder_flow (cells/mesh_pos/node_type are `static`, velocity/pressure are `dynamic`) so V7's `length_<k>`-sibling requirement is vacuous; first decoded record = 1923 nodes / 3612 triangle cells, all five fields tiled/reshaped to first-axis 600 (V5/V8/V9/V11/V12); `node_type` unique values `{0, 4, 5, 6}` ⊆ `{0,3,4,5,6}` so the `value-3` one-hot shift lands in `[0,3]` for `one_hot(num_classes=4)` (V16 / secondary 5.7); the test-split dataset constructs from the fitted `edge_stats.json`/`node_stats.json` (V14) with stat dims `edge_*`=3, `velocity_*`=2, `pressure_*`=1, `velocity_diff_*`=2 matching the feature widths (V15); `__getitem__` returns `(graph, cells, rollout_mask)` with `graph` a PyG `Data` carrying `{x, y, edge_index, edge_attr, mesh_pos}`, `graph.x` width 6 = velocity(2)+one_hot(4), `graph.y` width 3 = velocity_diff(2)+pressure(1), `graph.edge_attr` width 3 = disp(2)+disp_norm(1) (V17/V18); `len(ds)=4=num_samples·(num_steps−1)` and `ds[len]` raises (V13); noise is applied only for `split=="train"` (line 127) so the test path is noise-free (secondary 5.3); the stats files are read CWD-relative (the audit stages them in a temp dir and `chdir`s — secondary 5.4); `num_steps=5 ≤ trajectory_length=600` (secondary 5.5); `torch.get_default_dtype()` was `float32` at entry (secondary 5.6). **The DGL→PyG note:** the v2.0.0 loader returns PyG `Data`, not DGL graphs — Gate A (verdict 3 / Task 6) must reconstruct the scikit-fem `MeshTri` from `ds[i][0]["mesh_pos"]` (1923×2) + `ds[i][1]` (cells, 3612×3), not from `dgl` `ndata`/`edata`; the plan's `audit_gate_a_dgl_to_meshfield` skeleton needs rewriting accordingly.
   - Source: `audit_cylinder_flow_loader_contract` Modal entrypoint (CPU-only); citations to `vortex_shedding_dataset.py:<line>` @ `1ca85d65` per preflight `mgn_loader_contract.md` §3.1 + §5.

3. **Gate A verdict (Task 6):** PASS / PARTIAL / FAIL for PyG → MeshField materialization (the plan said "DGL"; the v2.0.0 loader returns PyG `Data` — see verdict 2).
   - Verdict: **PASS** — `skfem.MeshTri(p=mesh_pos.T, t=cells.T)` + `Basis(ElementTriP1())` (scikit-fem 12.0.1) reconstruct cleanly from the first cylinder_flow `test.tfrecord` trajectory (1923 nodes / 3612 triangle cells; domain `x∈[0,1.6]`, `y∈[0,0.41]`); 1923 P1 DOFs. PyG→scikit-fem needed only a `.T` axis swap + a float32→float64 / int32→int64 dtype cast (no Pattern-B coercion surprises; a loud pre-flight assert on the `(1923,2)`/`(3612,3)` shapes + dtypes guards that surface). Also amends D0-02 (Gate A deferred there to "Day 2" — Phase 1 fulfills it).
   - Source: `audit_gate_a_pyg_to_meshfield` Modal entrypoint (CPU-only); full findings `02-physicsnemo-mgn/preflight/gate_a_audit.json`.
   - Benign log noise (recorded so a reviewer doesn't read it as broken CUDA setup): CPU-only entrypoints emit `Warp CUDA error: Could not open libcuda.so` / "Insufficient CUDA driver version" — `warp-lang` probes for a GPU at physicsnemo import time; the dataset + scikit-fem code is CPU-only so it's harmless. Task 7 runs `gpu="A10G"` and will not see these.

4. **NGC sample reproduction (Task 7):** one-step velocity-prediction error vs the cylinder_flow test trajectory's next frame (the NGC zip ships only `model.pt` — no bundled reference output — so "reproduction" = "the checkpoint predicts the data's next frame within the recalibrated tol", not "matches a shipped tensor").
   - **Verdict: PASS** (recalibrated criterion = 1-step velocity RMSE ≤ 5×10⁻³; achieved 3.19×10⁻³ ≈ 1.36× the Pfaff-et-al. CylinderFlow baseline 2.34×10⁻³). Chronology: originally FAIL-hard (Task-7 fire) → bug found + fixed (Band-C re-audit) → recalibrated + confirmed (verdict-confirmation re-fire) — full record below.
   - **[Original Task-7 fire, 2026-05-12]:** **FAIL (hard)** — `audit_ngc_sample_reproduction` (Modal, A10G): NGC checkpoint loaded via the name-remap adapter into `MeshGraphNet(6,3,3)` (defaults: relu, no concat-trick), `load_state_dict(strict=True)` OK; ran 49 one-step predictions (each using the *true* previous frame) over the first cylinder_flow `test.tfrecord` trajectory following `examples/cfd/vortex_shedding_mgn/inference.py`'s exact denorm→re-norm→`model(invar, edge_attr, graph)`→denorm→mask→integrate protocol. **Max-abs velocity error 0.802** over rollout-mask nodes (= over all nodes; per-frame ~0.55–0.80). Tolerance: 10⁻³ — exceeded by ~800×. D0-23 band: **C (cliff, > ~0.1)**. Sanity-checked: the fitted `edge_stats`/`node_stats` are plausible (edge_mean[2]≈0.02 = mesh edge length; velocity_mean[0]≈0.56 = inflow scale), the inference protocol matches `inference.py`, the model loaded keys+shapes — so the residual was a forward-path mismatch the rename didn't cover (verdict 1 amendment). Scope-change decision held pending user confirmation; user chose the bounded Band-C re-audit (Option 1).
   - **[Band-C re-audit, 2026-05-12]:** found the residual — the edge-block MLP's *internal* concat order (`[src, dst, edge]` in modulus v0.1.0's `_EdgeBlock` vs `[edge, src, dst]` in v2.0.0's `concat_efeat_pyg`); see verdict 1 update + "Band-C re-audit result (2026-05-12)" below. Adapter + tests fixed; ONE A10G re-fire → `E` (max-abs) dropped 0.80 → **0.0352** (Band B, per-frame stable ~0.017–0.035) ⇒ architecture identity structurally confirmed.
   - **[Verdict-confirmation re-fire, 2026-05-12, user-directed Option 3]:** instrumented `audit_ngc_sample_reproduction` with RMSE / per-component / quantile metrics (NO adapter change), pre-registered the RMSE verdict bands, fired once (the LAST fire). **1-step velocity RMSE = 3.19×10⁻³ over rollout-mask nodes** (physical units; 3.00×10⁻³ over all nodes; 1.33×10⁻² in velocity-std units), per-component vx=3.71×10⁻³ / vy=2.57×10⁻³ (no concentration ⇒ no residual systematic issue), |err| p50/p90/p99 = 1.22×10⁻³ / 5.03×10⁻³ / 1.12×10⁻², max-abs 3.52×10⁻², over 49 one-step pairs (1706 mask nodes/frame). Pfaff et al. 2020 (MeshGraphNets, ICLR 2021) Table 1 CylinderFlow RMSE-1-step = 2.34 ± 0.12 ×10⁻³, so the reproduction is **~1.36× the published baseline = paper-quality** (the NGC checkpoint is fully trained: `epoch=24`, ~3.7M steps). `RMSE-1 = 3.19×10⁻³ ≤ 5×10⁻³` ⇒ **Gate D PASS** per the pre-registered band. The recalibrated Gate-D criterion *is* "1-step velocity RMSE ≤ 5×10⁻³" — the inherited `10⁻³`-max-abs pass bar is **retired as a category error** (it was a bit-exact-tensor-reproduction tolerance from the LagrangeBench `test_inference_matches_ngc_sample` analog, which shipped a reference tensor; here there is no shipped tensor — the metric bounds the model's *intrinsic* learned one-step accuracy, which is floored by training quality, not float precision). The recalibration was pre-registered (commit before the fire) and anchored to a published number with a stated metric (the paper's own RMSE-1, no max-abs↔RMSE conversion) — so it is not goalpost-moving. Full findings: `02-physicsnemo-mgn/preflight/gate_d_reproduction.json` (updated).

5. **Gate D composite verdict (Task 8):** PASS (checkpoint usable for case study 02) / FAIL-with-FNO-fallback.
   - **Verdict: PASS** = (BLOCKING-1 key-set PASS) ∧ (verdict 4 NGC-reproduction PASS, recalibrated) ⇒ the NGC `modulus_ns_meshgraphnet` checkpoint, loaded via the corrected name-remap adapter (`_legacy_checkpoint_name_remap.py`: encoder/decoder `.mlp.`→`.model.` rename + parallel-`{edge,node}_blocks`→interleaved-`processor_layers` restructure + the edge-MLP first-Linear input-column reorder `[src, dst, edge]`→`[edge, src, dst]`), is **usable for case study 02** — it reproduces the cylinder_flow test trajectory's next frame to ~1.36× the Pfaff-et-al. CylinderFlow RMSE-1 baseline. CS02 Phase 2+ proceeds with the rescued MGN checkpoint; the FNO-on-Darcy fallback (design §3.1.A) is **not** triggered. (The earlier "provisionally FAIL → FNO-on-Darcy" is superseded — it was held precisely because an adapter re-audit could flip verdict 4, which it did.)

6. **Substrate-class empirical verdict (Task 9):** cylinder wake fits `open-driven-dissipative` (default) OR new class label.
   - **Verdict: `open-driven-dissipative` (boundary-driven sub-class) — confirmed; no Pattern-A drift.** `smoke_substrate_class_vortex_shedding` (Modal A10G, `num_steps=400`): ran the GT cylinder_flow test trajectory + a true MGN rollout (corrected name-remap adapter; `inference.py @ 1ca85d65`'s `predict()` protocol) over the same 399-frame window, computed the **three discriminating observables** on each via scikit-fem P1 finite elements on the cylinder_flow mesh (the Gate A path, verdict 3 PASS; an FE-divergence self-test fired first — `∫|∇·v|dV` = 1.4×10⁻¹⁵ for `v=(y,-x)`, = `total_area` 0.6495 for `v=(x,0)` — confirming the P1-divergence wiring before trusting anything). Results (GT; the MGN rollout reproduces all three — see below):
     1. **Incompressibility** `∫|∇·v| dV / ∫ ||∇v||_F dV = 0.049` (stable: max 0.052 over the window) — << 1, i.e. consistent with incompressible NS; the ~5% residual is the data's P1-reinterpolation discretization floor (the COMSOL solver almost certainly uses Taylor–Hood velocity elements, so re-reading its nodal velocities as P1 and differencing introduces this) — *not* a fundamental compressibility, and crucially **the MGN rollout matches it (0.050 ≈ 0.049)**. The PH-CON-001 signal.
     2. **KE budget** `KE(t) = 0.5 ∫ |v|² dV`: dKE/dt is **NOT monotone** (42 sign changes over 358 post-warmup frames), KE oscillates around a steady mean (CV ≈ 0.77%), the lift-proxy spectrum has an extremely sharp shedding peak (prominence ≈ 11600× the median). This **confirms the a priori prediction** (boundary-driven sub-class: dE/dt oscillates around zero, balancing inflow/outflow/dissipation, *not* monotone-rising/falling) ⇒ PH-CON-002 (`energy_drift`) / PH-CON-003 (`dissipation_sign_violation`)'s strictly-dissipative-or-conservative assumption **does** fail on this substrate ⇒ the D0-22-style SKIP-gate dispatch is empirically justified for the mesh side. The PH-CON-002/003 signal.
     3. **Strouhal** `St = f_s · D / U`: `f_s = 3.90 Hz` (the lift-proxy `∫ v_y dV` FFT peak), `D = 0.0912` (cylinder diameter — `diam_x ≈ diam_y`, a circle, center `(0.41, 0.198)` — vertically centred in the `[0, 0.41]` channel), `U_max = 2.22` (centerline inflow speed) ⇒ **St ≈ 0.160** — in the design target band **[0.16, 0.21]** for a cylinder wake at Re ∈ [100, 300] (at its lower edge — consistent with the lower-Re end of the band; the difference from 0.16 is in the 4th decimal, i.e. *at* the target within smoke precision). (With the *bulk*-velocity convention `U_mean = 1.02`, St = 0.347, ~2.17× higher — outside the band; the centerline convention is the one that matches the textbook Strouhal range, as expected for a cylinder on the channel centerline.) The cylinder-wake-specific signature. — *Note: the `ke_osc_freq` reported alongside (0.28 Hz) is a slow-drift artifact of the global KE spectrum, not the shedding-related KE oscillation (~2 f_s); it's a diagnostic field only and does not enter the verdict.*
   - **MGN rollout reproduces the substrate signature** (`mgn_reproduces_substrate_signature = True`): the 399-step rollout has rel_div 0.050 (≈ GT's 0.049), KE not monotone (54 sign changes), and sheds at the *same* frequency (St identical to GT, lift-peak prominence ≈ 1330× — sharp but ~9× less than GT, the expected broadband content from a long rollout). So the checkpoint learned the *dynamics*, not just the one-step map — a strong cross-check on Gate D's PASS.
   - **Routing:** D0-23 verdict 6 = `open-driven-dissipative` ⇒ Task 11 lands `MGN_DATASET_SYSTEM_CLASS` with `vortex_shedding_2d → "open-driven-dissipative"` (parallel to particle-side, per design §2.2; *not* a stack-agnostic refactor). No D-entry amendment needed (no Pattern-A drift — the a priori physical classification held on all three observables; this is a cell-1-shaped confirmation, not a cell-2/4 surprise). Findings: `02-physicsnemo-mgn/preflight/substrate_class_smoke.json` (mirror of the entrypoint's `/vol/datasets/cylinder_flow/substrate_class_smoke.json`).
   - Three discriminating observables (as pre-registered): ∫|∇·v|dV, KE budget dKE/dt, Strouhal St ∈ [0.16, 0.21] for Re ∈ [100, 300].

7. **Persistent-volume decision (Task 9 sub-step):** Modal MGN inference writes to persistent volume? Y/N.
   - **Verdict: Y.** `smoke_substrate_class_vortex_shedding` (and `audit_ngc_sample_reproduction`, `audit_gate_a_pyg_to_meshfield`, etc.) write their findings JSON to `/vol/datasets/cylinder_flow/…` on the `case-study-02-physicsnemo-artifacts` Volume and call `mgn_volume.commit()`. Phase 2's MGN inference entrypoint will do the same for the rollout outputs; the rollout-dir isolation pattern from `inference.py` (`os.chdir(rollout_dir)` — here generalized to the `tempfile.mkdtemp` + stage-stats + `chdir` + `finally: chdir back` pattern these Phase-1 entrypoints already use) carries over.
   - Implication: rollout-dir isolation pattern (round-codex-4) applies (Y) — Phase 2's inference entrypoint isolates its CWD per rollout so the loader's CWD-relative `{edge,node}_stats.json` reads don't collide.

8. **`_expect_velocity` helper key resolution (Task 10):** actual NGC velocity-field key name.
   - Verdict: **`"velocity"`** — the raw `meta["field_names"]` key, and the dataset's internal key (`node_features[g]["velocity"]`). Note: on the PyG `Data` object `__getitem__` returns, velocity is *not* a named attribute — it is the first 2 columns of `graph.x` (`= cat(velocity, one_hot(node_type))`), and the target `velocity_diff` is the first 2 columns of `graph.y` (`= cat(velocity_diff, pressure)`). Task 10's `_expect_velocity` helper should key on `"velocity"` at the record/loader level; if it needs the value off the returned graph it must slice `graph.x[:, :2]` (input) or `graph.y[:, :2]` (one-step target).
   - Source: `audit_cylinder_flow_loader_contract` findings (verdict 2 above) + `vortex_shedding_dataset.py:122-124,165-174` @ `1ca85d65`.

9. **`MGN_DATASET_SYSTEM_CLASS` dispatch (Task 11):** introduced; class label per verdict 6.
   - **Verdict: LANDED at 504e104** (`02-physicsnemo-mgn: Task 11 — MGN_DATASET_SYSTEM_CLASS + substrate-class dispatch on *_on_mesh`). `MGN_DATASET_SYSTEM_CLASS = {"vortex_shedding_2d": "open-driven-dissipative"}` in `_harness/mesh_rollout_adapter.py` (parallel to particle-side `LAGRANGEBENCH_DATASET_SYSTEM_CLASS`, per design §2.2 P0-resolvable Pattern-B response: *duplicated route*, NOT a stack-agnostic refactor). Dispatch wired into both `energy_drift_on_mesh` and `dissipation_sign_violation_on_mesh` — fires AFTER `_expect_velocity` + `is_regular_grid` checks, BEFORE the KE-rest / KE-max gates (substrate-class is the load-bearing assumption; KE gating is moot if the assumption is violated). SKIP message cites D0-22 (particle-side precedent) and D0-23 v9 (mesh-side extension). 4 tests added (entry-pinning + SKIP-on-open-driven for each function + no-dispatch regression guard on unknown dataset name). **Phase-2 readiness note:** dispatch is currently dead code on real NGC cylinder_flow rollouts (framework=pytorch+dgl ⇒ `is_regular_grid=False` ⇒ graph-mesh SKIP fires first); becomes live when Phase 2 lands the DGL→MeshField materialization path that lifts the graph-mesh SKIP. Duplicate-logic-drift risk is *named* per round-codex-4 catalogue, not eliminated; stack-agnostic refactor gated on amendment 1 / case study 03 evidence.

10. **Pre-flight assertions (Task 12):** loader-contract assertions written in `_harness/mesh_rollout_adapter.py`.
    - **Verdict: LANDED at 468de33** (`02-physicsnemo-mgn: Task 12 — _assert_loader_contract_mgn defensive validation`). New helper `_assert_loader_contract_mgn(rollout)` covers the V-entries / known-unknowns that survive into the `MeshRollout` boundary (after V1-V11/V13-V15/V17-V18 are already past at TFRecord-decode time): **KU §5.6** velocity dtype == float32 (NGC checkpoint was trained at default-torch-dtype=float32; fp64 would silently change numerical behavior at `torch.tensor(..., dtype=torch.float)` lift, vortex_shedding_dataset.py:373); **V12 + V18** velocity is 3D `(T, N_nodes, D)`, D ∈ {2, 3}; **KU §5.7 / V16** node_type ∈ {0, 3, 4, 5, 6} (one_hot num_classes=4 bound after value-3 shift, vortex_shedding_dataset.py:363-368); **V-entries on metadata** framework + model present (anchors for v9 dispatch and framework-conditioned SKIP paths). Velocity-absent path is a no-op — `_expect_velocity` owns the informative SKIP wording in that branch. Scope: P0 vortex_shedding_2d only; amendment 1 (Ahmed Body) is the multi-instance trigger that would force generalization (e.g., a dataset-keyed assertion dispatch table). 6 tests added (well-formed-rollout passes + 4 rejection tests + velocity-absent no-op). Full harness suite still 235 passed.

**Phase 1 acceptance criteria (per design §4.1) — all checked:**

- [x] BLOCKING-1 CPU state-dict smoke (verdict 1; commit 9873962).
- [x] NGC checkpoint downloaded; hash pinned (verdict 1 + Task 4).
- [x] Day 2 hour 1 NGC audit findings (verdicts 2 + 8).
- [x] Gate A verdict (verdict 3 + D0-02 amendment; Task 6).
- [x] Gate D verdict (verdict 5; Task 7-8, commit 46fc1ff — RMSE-1 = 3.19e-3 ≈ 1.36× Pfaff et al.).
- [x] `test_inference_matches_ngc_sample` verdict (verdict 4).
- [x] Empirical substrate-class smoke verdict (verdict 6; Task 9, commit 773101e — open-driven-dissipative).
- [x] `MGN_DATASET_SYSTEM_CLASS` + dispatch (verdict 9; Task 11, commit 504e104).
- [x] `_expect_velocity` key resolution (verdict 8; Task 10, commit 82d7f68 — pinned to "velocity").
- [x] Pre-flight assertions (verdict 10; Task 12, commit 468de33; cross-review absorption at 92d315e).
- [x] Persistent-volume decision (verdict 7 — Y; rollout-dir isolation pattern carries to Phase 2).
- [x] Phase 1 boundary cross-review (Tasks 14-15; findings doc at c534307; cross-review absorptions at 92d315e + 34256cb).

**Phase 1 boundary cross-review (round-codex-Phase1, c534307) — findings triaged:**

| # | Finding (1-line) | Severity | Cell | Disposition |
|---|---|---|---|---|
| 1 | dataset metadata load-bearing for v9 dispatch but not required + SCHEMA.md spelling drift ("vortex-shedding-2d" vs "vortex_shedding_2d") | High | 2 | **Absorbed at 92d315e** — `_assert_loader_contract_mgn` now requires `dataset` in metadata; SCHEMA.md canonicalized to underscored spelling. |
| 2 | `_assert_loader_contract_mgn` no-ops on velocity-absent — layered-fail-open with `_expect_velocity`'s SKIP | High | 2 | **Absorbed at 92d315e** — velocity-absent now raises AssertionError (MGN contract violation, not optional SKIP). |
| 3 | `_assert_loader_contract_mgn` not wired into any production path (opt-in helper, no callers outside tests) | Medium | 1 | **Deferred to Phase 2** — design §3.2 + §4.2 already obligate Phase 2's MGN materializer + lint path to wire the helper at the first trusted MGN boundary. Forward-flag added to design successor block (Phase 2 acceptance must include a test that the lint path rejects malformed MGN rollouts). |
| 4 | `load_mesh_rollout_npz` upcasts node_values to fp64 via `dtype=float`, contradicting SCHEMA.md fp32 contract and Task 12's fp32 assertion | Medium | 2 | **Absorbed at 34256cb** — upcast dropped, on-disk dtype preserved; round-trip dtype test added. |
| 5 | Rollout-dir isolation declared as Phase 2 obligation (D0-23 v7), not verified by Phase 1 | Low | 1 | **Deferred to Phase 2** — design §4.2 + Phase 2 acceptance already carry this; Phase 1 entrypoints (audit_ngc_sample_reproduction, smoke_substrate_class_vortex_shedding) demonstrate the pattern; Phase 2 inference entrypoint must reproduce it + carry a smoke assertion that two same-sha retries cannot read each other's CWD-relative stats. |

**Totals:** 0 cell-1-only (Findings 3 and 5 are cell-1-shaped Phase 2 forward-flags), 3 cell-2 (Findings 1+2+4 absorbed in-rung), 0 cell-3, 0 cell-4. **No methodology surprises** — every finding was a layered-fail-open shape predicted by design §2.6, just with new concrete cell-2 instances that Phase 1's executor missed during plan execution. Phase 1 is **clean to close**; Phase 2 inherits two forward-flags (helper-wiring + rollout-dir isolation), both already on its acceptance list.

**Phase 1 data provenance pins (Task 5 prerequisite):**

The V1-V18 loader-contract audit needs real `VortexSheddingDataset` records, which the NGC checkpoint package does **not** ship (the zip contains only `vortex_shedding_mgn/model.pt` — verified via `unzip -l`; no `meta.json`, no `<split>.tfrecord`, no `*_stats.json`). The data is sourced from DeepMind's public MeshGraphNets bucket (Pfaff et al. 2020) — `https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow/` — which `VortexSheddingDataset` reads natively (the physicsnemo MGN datapipe was ported from DeepMind's reference impl). Pulled to Modal Volume `case-study-02-physicsnemo-artifacts` at `/vol/datasets/cylinder_flow/`:

- `meta.json` — 883 bytes — sha256 `2a3e39429a55a0cf47355717cc07f4b292629daeb48a89abd518ea0402033e96`
- `test.tfrecord` — 1,355,376,404 bytes — sha256 `8522932a23da6ccdee996c56158e4b908f7091f0ade11e8acea700be321af8c3`
- `train.tfrecord` — 13,645,805,387 bytes (~13.6 GB; the prior "12.7 GB" estimate was 12.7 GiB) — sha256 `ea63bf985ef131d542b763dded130f242040b9389a59b76f5e2d61a3f40c6ada` — **NOT persisted**: `compute_cylinder_flow_stats` downloads it to *ephemeral local container disk* (never the Volume), fits the two stats files, then `shutil.rmtree`s it in a `finally` block. Sha recorded here for provenance only — there is no on-disk copy to assert against on a re-run (the file is always re-downloaded fresh and the meta.json pin gates the bucket).
- `edge_stats.json` — 164 bytes — sha256 `dd9d0f027eedc787f4ee7a1867c5db03dd8e93048c7c77bf43cdff97e7c81dae` — `{edge_mean: [-1.54e-12, 7.17e-14, 0.019994], edge_std: [0.015899, 0.014761, 0.0084206]}`
- `node_stats.json` — 339 bytes — sha256 `0700ae55c88432eb01d03e6513e748018b0562517d394b3fa0cabefb99bacb99` — `{velocity_mean: [0.564405, 0.000650], velocity_std: [0.571458, 0.145086], velocity_diff_mean: [-3.058e-4, 1.214e-6], velocity_diff_std: [0.021435, 0.020902], pressure_mean: [0.032651], pressure_std: [0.325798]}`

**Stat-fit parameters.** `compute_cylinder_flow_stats` fits the two stats files via `VortexSheddingDataset(split="train", num_samples=1000, num_steps=600, noise_std=0.02)` — the dataset class's own `__init__` defaults, i.e. the **full** DeepMind cylinder_flow train split with DeepMind/modulus-standard input-noise injection. The (1000, 600) match is well-anchored, not arbitrary: the physicsnemo `vortex_shedding_mgn` example README states the dataset is 1000 train / 100 valid / 100 test trajectories, each 600 time steps at dt = 0.01 s (so `num_steps=600` = "use every step", `num_samples=1000` = "use every train trajectory"); and the `num_samples=1000 / num_steps=600 / noise_std=0.02` defaults are stable across `NVIDIA/modulus` v0.2.1 through v0.4.0 — the version range bracketing the NGC checkpoint's 2023-05-26 upload. The v2.0.0 PyG example's `conf/config.yaml` `(num_training_samples=400, num_training_time_steps=300)` is a dev/fast-iteration config, not what the 2023 (DGL-era) checkpoint was trained with. Noise is added to the train-split velocities *before* stats are computed (`vortex_shedding_dataset.py:127-133` @ `1ca85d65`), so `velocity_std` and especially `velocity_diff_std` carry a `~noise_std**2` term — intentional, it is what the checkpoint was normalized against. `torch.manual_seed(0)` makes the noise-injected fit reproducible.

**Gate-D-failure debug order (D0-23 amendment, 2026-05-12).** Task 7's NGC sample reproduction (verdict 4; 10⁻³ on velocity) is the empirical test of whether these stats — and the name-remap adapter — reproduce the checkpoint. If it fails, the cause is *not* primarily the stat-fit parameters (anchored as above). Likelier suspects, in order: **(1)** the name-remap adapter's architecture-identity assumption — the `.mlp.`→`.model.` encoder/decoder rename + the parallel-`{edge,node}_blocks`→interleaved-`processor_layers` restructure (verdict 1); a clean Gate-D FAIL here means the adapter is *wrong, not fixable* → FNO-on-Darcy fallback per design §3.1.A (there is nothing to "debug" — the falsification is the verdict). **(2)** Stats-estimator implementation drift between v2.0.0's PyG `physicsnemo/datapipes/gnn/vortex_shedding_dataset.py` `_get_edge_stats`/`_get_node_stats` and the modulus-era *DGL* version the checkpoint was actually normalized against — fetch the latter from `NVIDIA/modulus` at a ~2023 tag (e.g. `gh api repos/NVIDIA/modulus/contents/modulus/datapipes/gnn/vortex_shedding_dataset.py?ref=v0.2.1`), diff the estimators, recompute with the DGL-era logic if they differ materially. **Note:** physicsnemo's `vortex_shedding_dataset_dgl.py` / `vortex_shedding_mgn_dgl` split is *post-v2.0.0* — there is **no** `_dgl`-suffixed file at `1ca85d65`, so there is no in-repo file pair to diff at our pin; the DGL reference must come from the `modulus` repo. **Also note:** `node_stats` (the reproduction-critical stats — they normalize the velocity *input* and the velocity-diff *target*) are provably graph-representation-independent (means/stds of the raw velocity/pressure fields, no graph traversal), so any DGL↔PyG estimator drift is confined to `edge_stats` (relative-displacement edge features) and would contribute well under 10⁻³ — this suspect is cheap to rule out but low-probability. **(3)** A `(400, 300)` parameter re-fit (the v2.0.0-config value), in case the 2023 checkpoint did use it. **(4)** Noise RNG / float-accumulation order — negligible (<10⁻⁶ on means; exactly `noise_std**2` added to the `velocity_diff` meansqr regardless of seed). **Diagnostic shortcut (load-bearing):** the per-suspect error bounds *route* the diagnosis. `node_stats` is representation-independent, so suspect (2)'s contribution is confined to `edge_stats` (relative-displacement edge features only) and is bounded *well under* 10⁻³ on the reproduced velocity; (4) is <10⁻⁶. Therefore **if Gate D fails by more than ~10⁻³, suspects (2) and (4) are excluded** — neither can account for an error of that size — and the diagnosis routes directly to (1) the adapter architecture or (3) the stat-fit parameters. Only a *near-miss* failure (error just above 10⁻³) keeps (2) live.

**Threshold bands — pre-pinned before Gate D runs (so the routing is not interpreted-during-execution).** Gate D failing means the reproduced-velocity max-abs-error `E > 10⁻³` (below that it PASSes; the bands only apply on a FAIL). Route by which band `E` lands in:
- **Band A — near-miss, `E ∈ (10⁻³, ~5×10⁻³]`:** (1) and (4) are excluded (an architecture mismatch is O(0.1)+; noise RNG is <10⁻⁶). Live suspects: **(2)** edge_stats impl drift (its bound is <10⁻³, but model amplification of the edge-feature normalization + tolerance slack can plausibly push the velocity error into this band) — *primary*: diff the modulus-DGL `_get_edge_stats` and recompute; **(3)** a small `(400,300)`-vs-`(1000,600)` effect — *secondary*.
- **Band B — moderate, `E ∈ (~5×10⁻³, ~0.1]`:** (2) and (4) excluded (neither reaches this magnitude). Live suspects: **(3)** the `(400,300)` parameter re-fit (a ~1% shift in `velocity_std`/`velocity_mean` maps to ~10⁻² error on O(0.5) velocities) — *primary, cheap*: re-fit + re-run; if `E` does not improve materially → **(1)-subtle**: re-audit the adapter's processor `{edge,node}_blocks`→interleaved-`processor_layers` restructure for an off-by-one / wrong interleave order that still yields a finite plausible output; if that re-audit finds a fixable bug, fix + re-run; if not, → FNO-on-Darcy.
- **Band C — cliff, `E > ~0.1`:** "this is a different model" — clean **(1)** architecture FAIL → FNO-on-Darcy fallback fires (design §3.1.A); this is the upper diagnostic exit, no further debugging.

The DGL→PyG migration (suspect 2, the Task 6 PyG-vs-DGL dataset-return-type finding, and the verdict-1 state-dict rename) is a *single* coherent surface, not three independent drift instances.

**Band-C refinement (2026-05-12, after the Task 7 fire — methodology landed honestly, not Band-C reinterpreted post-hoc).** The pre-registered cliff at `E > ~0.1` implicitly bundled two distinct failure modes: **(a)** the NGC checkpoint is a *genuinely different* architecture (different layer count / dims / connectivity) — the name-remap can't even produce a strict-loadable state_dict, or it produces one whose forward is structurally wrong with no fix; and **(b)** the architecture *shape* (layer count, dims, key names) matches — `load_state_dict(strict=True)` succeeds — but the rename maps weights to the wrong *slots*, or the model is fed features in the wrong *column order* (node-input ordering, edge-feature ordering), or its output columns are interpreted in the wrong order. (b) produces cliff-magnitude errors *too* (a swapped weight slot or scrambled feature columns gives garbage, not "a bit off") — but (b) is a *fixable adapter bug*, not a genuine architectural mismatch. The actual Task-7 Gate-D failure (`E = 0.80`, cliff-magnitude, `load_state_dict(strict=True)` OK with keys+shapes matching) is squarely (b). **Refined routing:** (a) → FNO-on-Darcy with no retry (the original Band-C call). (b) → **one bounded adapter re-audit + one retry**, under hard caps so it can't become a sunk-cost spiral: re-audit time-boxed to **1 hour wall-clock** (source-read v2.0.0's `MeshGraphNetProcessor.__init__` `processor_layers` construction + the modulus-era MGN processor + the modulus-era `vortex_shedding_dataset.__getitem__` node/edge/output column orders; verify v2.0.0's order from source *first*, before assuming what modulus did; enumerate which of {processor interleave order, node-input column order, edge-feature column order, output column order} is consistent with the data — *then* fix; if no clear hypothesis emerges within the hour, FNO fires); **exactly one** A10G re-fire of the fixed adapter; then the **post-fix decision branch**: `E < 5×10⁻³` → Gate D **PASS**, proceed (within 5× the inherited 10⁻³ tolerance is firmly pass-range for a reproduction test, no recalibration needed); `E ∈ [5×10⁻³, 5×10⁻²]` → *now* the 10⁻³-vs-actual-MGN-one-step-accuracy question is load-bearing → do the literature dive (DeepMind MGN paper / a known-good baseline on cylinder_flow), recalibrate the tolerance explicitly, and name the post-hoc-recalibration cost in this D-entry (the tolerance was inherited from the LagrangeBench shipped-reference analog; recalibration happened after the fire — the band-result + this protocol document why that isn't goalpost-moving); `E > 5×10⁻²` (or the hour expires without a clear hypothesis) → FNO-on-Darcy fires, decision final, no further adapter iteration.

**Band-C re-audit result (2026-05-12) — bug found, fixed, ONE re-fire done; verdict at the post-fix branch pending recalibration.** The bounded re-audit (1 hr cap; source-read modulus-era vs v2.0.0 first). **Candidates 1-5 from the handoff ruled out by source:** (1) processor *interleave position* — v2.0.0's `MeshGraphNetProcessor.__init__` builds `processor_layers = chain(*zip(edge_blocks, node_blocks)) = [edge_0, node_0, …]` and `forward` runs `for module in segment: edge_features, node_features = module(...)` — i.e. edge-then-node per step with the node block consuming the just-updated edge features — *exactly* modulus v0.1.0's `_GraphProcessor.forward` (`edge_features = edge_blocks[i](…); node_features = node_blocks[i](…)`); the adapter's `edge_blocks.K → processor_layers.2K`, `node_blocks.K → processor_layers.2K+1` is correct. (2,3,4) node-input / edge-feature / output column orders — modulus v0.1.0's `MGNDataset.__getitem__` and v2.0.0's `VortexSheddingDataset.__getitem__` are byte-identical here: node = `cat((velocity, one_hot_node_type))`, edge = `cat((disp, disp_norm))`, target = `cat((velocity_diff, pressure))`, same `_one_hot_encode`/`_get_rollout_mask`. (5) `MeshGraphMLP` internal layer order — `strict=True` passing with identical key sets *and* shapes forces Linears at indices {0,2,4} + LayerNorm at {5} (encoders) in both versions; ruled out. **The actual residual — NOT in the pre-registered candidate list — is the edge-block MLP's *internal* concat order.** modulus v0.1.0's `_EdgeBlock.concat_message_function` builds the per-edge MLP input as `cat((edges.src["x"], edges.dst["x"], edges.data["x"]), dim=1)` = column layout `[src(128), dst(128), edge(128)]` (= DeepMind-meshgraphnets' `tf.concat([sender, receiver, edge])`); physicsnemo v2.0.0's `concat_efeat_pyg` (`physicsnemo/nn/module/gnn_layers/utils.py` @ `1ca85d65`) builds it as `cat((efeat, src, dst), dim=1)` = `[edge(128), src(128), dst(128)]`. So each `processor.edge_blocks.K.edge_mlp.0.weight` (shape `[128, 384]`) has its 384 input columns in modulus order in the checkpoint but v2.0.0 feeds them in a different order — `strict=True` passes either way (shape unchanged; only the *meaning* of the columns differs), which is exactly why this produced a clean-load-then-garbage-forward. The checkpoint is firmly **modulus v0.1.0-era**: it has the *parallel* `processor.{edge,node}_blocks[K]` ModuleLists + `.mlp.`/`.edge_mlp.`/`.node_mlp.` naming (v0.2.0 already restructured to interleaved `processor_layers` + renamed to `MeshEdge/NodeBlock`), and the 2023-05-26 upload sits 3 weeks after the v0.1.0 (2023-05-02) tag. **Fix:** the name-remap adapter now also permutes the first-edge-Linear input columns `[src, dst, edge] → [edge, src, dst]` (only that one weight per edge block — `edge_mlp.{2,4}` are internal `Linear(128,128)`, the node block's `cat((agg, self))` order is identical modulus↔v2.0.0, and the encoders/decoder feed dataset features whose construction is identical); adapter docstring + `tests/test_legacy_checkpoint_name_remap.py` updated (column-reorder, output-equivalence, shape-guard, non-tensor-passthrough tests added — all pass). **ONE A10G re-fire** (`audit_ngc_sample_reproduction`, post-fix adapter): **max-abs velocity error 3.52×10⁻² over rollout-mask nodes** (per-frame ~0.017–0.035, *stable* across all 49 one-step pairs) — was 0.80 → **~23× collapse** from a single source-grounded surgically-scoped weight permutation; the forward now produces physically-meaningful one-step predictions ⇒ **architecture identity (verdict 1) is structurally confirmed** (the forward *is* MGN-on-cylinder_flow, no longer "garbage"). `E = 0.0352` lands in **Band B** `(5×10⁻³, 5×10⁻²]`, so per the pre-registered post-fix branch the verdict now hinges on the **tolerance recalibration**: the inherited `10⁻³` was a *bit-exact-reproduction* tolerance from the LagrangeBench analog (which shipped a reference tensor to match to float precision); here there is no shipped reference tensor — the metric is the model's *intrinsic* one-step prediction error vs the true next frame — so `10⁻³` was the wrong yardstick from the start. **Literature anchor:** Pfaff et al. 2020 (MeshGraphNets, ICLR 2021), Table 1, CylinderFlow: **RMSE-1-step = 2.34 ± 0.12 ×10⁻³** (velocity/momentum; the table's 1-step / rollout-50 (6.3×10⁻³) / rollout-all (40.9×10⁻³) values are in a consistent normalization, ≈ the O(1) physical velocity scale — node_stats: velocity_mean ≈ [0.56, 6.5e-4], velocity_std ≈ [0.57, 0.14]). Our metric is **max-abs** (a tail statistic over 49×1923 ≈ 94k frame-node pairs), not RMSE; max-abs / RMSE for boundary-layer-tailed errors is ≈ 10–15×, under which our 0.0352 max-abs implies RMSE ≈ 0.002–0.0035 ≈ the paper's 2.34×10⁻³ — i.e. **consistent with a correctly-loaded, paper-quality MGN** (the NGC checkpoint is fully trained: `epoch=24`, ~3.7M steps, lr decayed to ~3.5e-6), *but the comparison is indirect* (we measured max-abs, the paper reports RMSE; the conversion factor carries the uncertainty). **Verdict — RESOLVED (user-directed Option 3 → verdict-confirmation re-fire):** rather than recalibrate a *max-abs* tolerance via an uncertain max-abs↔RMSE conversion factor, the verdict was settled by *measuring RMSE directly* (Pfaff et al.'s own metric) in one further measurement-only fire — **1-step velocity RMSE = 3.19×10⁻³ ≤ 5×10⁻³ ⇒ Gate D PASS** (the prediction above — implied RMSE ≈ 0.002–0.0035 ≈ paper — held). See "Verdict-confirmation re-fire" below. Findings: `02-physicsnemo-mgn/preflight/gate_d_reproduction.json` (final; the embedded `gate_d_band` text in the entrypoint is the *pre*-Band-C-refinement routing — the canonical post-fix branch is this paragraph + the Band-C-refinement paragraph above + the verdict-confirmation paragraph below; the entrypoint's `gate_d_rmse_band` is the live verdict basis).

**Verdict-confirmation re-fire — pre-registered BEFORE firing (2026-05-12, user-directed Option 3).** The post-fix `E = 0.0352` max-abs sits in Band B; calling PASS/FAIL off a max-abs metric requires a max-abs↔RMSE conversion factor (≈ 10–15× for boundary-layer-tailed errors, but only ≈ 4.8× = √(2 ln N) for Gaussian-tail errors over N ≈ 94k) whose validity is an empirical property of *this* model on *this* dataset, not decidable a priori. **Resolution: measure RMSE directly** (Pfaff et al.'s metric — no factor conversion) via ONE measurement-only A10G re-fire of `audit_ngc_sample_reproduction`: **NO adapter change** (verdict-confirmation only); add (i) the 1-step velocity RMSE `= sqrt(mean over (frame f∈0..48, node n, component c∈{vx,vy}) of (v_pred[f,n,c] − v_exact[f,n,c])²)` in **physical velocity units** (matching the paper's Table-1 convention — these papers report rollout RMSE on the un-normalized state), over rollout-mask nodes **and** over all nodes; (ii) the same RMSE computed in **normalized** velocity units (`(v − v_exact)/velocity_std` first) as a convention-hedge — physical and normalized differ only by ≈ `velocity_std` ≈ 0.4–0.6×, so if they tell materially different stories vs the paper's number, re-check the convention before finalizing; (iii) per-component physical RMSE (`vx`, `vy` separately) — concentration in one direction would diagnose a residual systematic issue; (iv) the |error| quantiles `p50 / p90 / p99` over mask nodes (the error *distribution*, not just its max). Keep the existing per-frame max-abs. **This is the LAST fire, regardless of result** (the original "exactly one re-fire" cap was a proxy for "no fix iterations beyond one"; a measurement-only re-fire without adapter change is a different category — see the cap-rationale refinement below — but it gets its own hard cap of one). **Pre-registered verdict bands** (on the headline metric = physical 1-step velocity RMSE over rollout-mask nodes; the paper baseline is `2.34 ± 0.12 ×10⁻³`):
- **RMSE-1 ≤ 5×10⁻³** (≲ ~2× paper) → **Gate D PASS.** The recalibrated Gate-D acceptance criterion *becomes* "1-step velocity RMSE ≤ 5×10⁻³" — the cleaner RMSE number IS the verdict basis, the prior `10⁻³`-max-abs criterion is retired as a category error (it was a bit-exact-tensor-repro tolerance, inapplicable absent a shipped reference tensor), and the ~`5×10⁻²` max-abs recalibration becomes moot. Finalize D0-23 v1 = architecture identity CONFIRMED, v4 = PASS, v5 = PASS; proceed to Task 8 (composite verdict) + Task 9 (substrate-class smoke: ∫|∇·v|dV, KE budget, Strouhal St∈[0.16,0.21]; persistent-Volume-write check). Commit the verdict.
- **RMSE-1 ∈ (5×10⁻³, 1.5×10⁻²]** (~2–6× paper) → **marginal — CS02 proceeds with the limitation named.** Architecture identity confirmed (the structural fix landed; ~few× paper RMSE is "below paper baseline but consistent with checkpoint usability", not "broken"). Fold the limitation into the v2.1 methodology trail explicitly — §3.3 "what physics-lint required to fire on this substrate": *"the name-remap adapter reproduces the NGC MGN checkpoint to ~Nx the Pfaff-et-al. 1-step CylinderFlow RMSE; that is the resolution floor of Phase 2's verdict bands on this substrate"* (fill N from the result). v4 = PASS-with-limitation, v5 = PASS-with-limitation; proceed to Tasks 8–9.
- **RMSE-1 > 1.5×10⁻²** (≳ ~6× paper) → **Gate D FAIL → FNO-on-Darcy.** The adapter is structurally correct (the edge-MLP concat fix is verified) but reproduction quality is meaningfully below paper baseline — either un-fixed residual adapter issues or genuine checkpoint↔reproduction drift (stats re-fit, tfrecord re-decode). v4 = FAIL, v5 = FAIL; pivot CS02 Phase 2+ to the FNO-on-Darcy fallback (design §3.1.A); commit "Task 8: Gate D composite FAIL → FNO-on-Darcy".

**Cap-rationale refinement (lands in D0-23 alongside whatever verdict the re-fire produces).** The pre-registered "exactly one A10G re-fire" cap was a *proxy* for "no fix iterations beyond one" — it was guarding against fix→fire→fail→fix sunk-cost spirals. A *verdict-confirmation measurement* (no adapter change, just better instrumentation on the same fixed model) is a different category and is **not** the failure mode the cap was designed to prevent. **Refinement:** re-fire caps apply to *fix iterations* only; verdict-confirmation measurements are permitted as a separate category, each with its own pre-registered hard cap (here: one measurement-only fire, NO adapter change, verdict bands pre-registered before firing). Caps need rationale-based interpretation, not literal interpretation — the literal reading here ("you already used your one re-fire, so FNO fires now even though you found and fixed the bug") would be category-confused. (Generalizes the `feedback_tripwire_pause_before_commit` discipline: a tripwire/cap's *purpose* is the binding constraint, not its surface form.)

**Verdict-confirmation re-fire RESULT (2026-05-12) — Gate D PASS.** `audit_ngc_sample_reproduction` re-fired (the LAST fire; instrumentation only, NO adapter change): **1-step velocity RMSE = 3.19×10⁻³ over rollout-mask nodes** (physical units; 3.00×10⁻³ over all nodes — the masking artifact is minor here; 1.33×10⁻² in velocity-std units — consistent with the physical number under the ≈`velocity_std` scaling, no convention surprise), **per-component vx = 3.71×10⁻³ / vy = 2.57×10⁻³** (comparable ⇒ no direction-concentrated residual systematic issue), **|err| p50 / p90 / p99 = 1.22×10⁻³ / 5.03×10⁻³ / 1.12×10⁻²** (typical error ~10⁻³, a boundary-layer tail to ~10⁻²), **max-abs = 3.52×10⁻²** (= the earlier number; deterministic, no adapter change — so the max-abs/RMSE ratio is ≈11×, in the boundary-layer-tailed regime, vindicating the earlier ~10–15× estimate), over 49 one-step pairs (1706 mask nodes/frame). Pfaff et al. 2020 Table 1 CylinderFlow RMSE-1-step = 2.34 ± 0.12 ×10⁻³ ⇒ the NGC checkpoint reproduces to **≈ 1.36× the published baseline = paper-quality** (it is fully trained: `epoch=24`, ~3.7M steps). `RMSE-1 = 3.19×10⁻³ ≤ 5×10⁻³` ⇒ **Gate D PASS** (the top pre-registered band). **Recalibration landed:** the Gate-D acceptance criterion is now "1-step velocity RMSE ≤ 5×10⁻³" (Pfaff-et-al.'s own metric, no conversion factor); the prior `10⁻³`-max-abs criterion is **retired as a category error** (a bit-exact-tensor-reproduction tolerance inherited from the LagrangeBench shipped-reference analog, inapplicable here — no shipped reference tensor; the metric bounds the model's *intrinsic* learned one-step accuracy, floored by training quality not float precision). Not goalpost-moving: the recalibration was committed *before* the confirming fire, anchored to a published number, expressed in the paper's own metric. **Verdicts finalized:** v1 = architecture identity CONFIRMED; v4 = PASS (recalibrated); v5 = PASS (checkpoint usable; FNO-on-Darcy fallback NOT triggered). **Cap-rationale refinement CONFIRMED in practice:** the "exactly one re-fire" cap was honored *in spirit* — exactly one *fix-iteration* re-fire (the post-fix run), then one *measurement-only* re-fire (no adapter change) under its own pre-registered cap; the literal reading ("you used your one fire, FNO fires now") would have abandoned a paper-quality checkpoint over a category-confused tripwire. The refinement (re-fire caps bind fix iterations; verdict-confirmation measurements are a separate category with their own cap) stands as a methodology entry. Findings: `02-physicsnemo-mgn/preflight/gate_d_reproduction.json` (final). **Next: Task 8 composite-verdict commit (this paragraph + the v5 update *is* it), then Task 9 (1-traj substrate-class smoke + the 3 discriminating observables ∫|∇·v|dV / KE budget / Strouhal St∈[0.16,0.21]; + the persistent-Volume-write check = verdicts 6+7) — a new GPU run, not a re-fire (the cap does not apply).**

**Methodology note — the "shape-conformant load, wrong forward" failure has depth (the third instance).** The Band-C re-audit's bug was *inside* the edge-MLP's concatenation (`[src, dst, edge]` vs `[edge, src, dst]`), three levels below where the pre-registered hypothesis space reached — all five pre-registered candidates assumed the failure boundary was at the **load interface** (key/slot mapping) or the **dataset interface** (feature column order on `graph.x` / `graph.edge_attr` / `graph.y`), both of which are visible at load- or `__getitem__`-time. The actual failure was a **block-internal convention** (the order in which a message-passing block concatenates `src`/`dst`/`edge` features before its first `Linear`), invisible to `load_state_dict(strict=True)` because the concatenated width and the `Linear`'s shape are unchanged — only the *meaning* of the weight matrix's columns differs. This is the **third** distinct "shape matches but semantics differ" failure in this adapter (after the `.mlp.`→`.model.` submodule rename and the parallel-`{edge,node}_blocks`→interleaved-`processor_layers` restructure — both also shape-conformant), so the category clearly has more depth than the original hypothesis space scoped. **Refinement for future checkpoint-adapter work:** when a state-dict adapter produces a shape-conformant `strict=True` load but the forward is wrong, treat **block-internal conventions** — feature-concat order inside message/edge/node blocks, activation-vs-norm placement inside MLPs, src/dst role assignment in message functions — as a *separate audit layer*, distinct from (and below) the load-interface and dataset-interface layers; verify them by reading the *forward* code of both the source-version and target-version blocks, not just the `__init__` (which only reveals shapes). See [[feedback_state_dict_adapter_internal_conventions]].

**Methodology note — the recalibration trail (a template).** The Gate-D tolerance recalibration was made bulletproof (not merely defensible) against the "goalpost-moving" objection by three disciplines worth reusing: (i) the recalibration was **committed before the confirming measurement** (the RMSE bands landed in commit `d4b83d8`; the confirming re-fire's instrumentation in `22de706`; the result in `46fc1ff` — the git order is the audit trail); (ii) the new criterion was **anchored to a published external number in that number's own metric** (Pfaff et al. 2020 Table-1 RMSE-1 = 2.34e-3, used as RMSE-1 directly — no max-abs↔RMSE heuristic conversion whose validity would itself be in question); (iii) the old criterion's retirement was framed as a **category error, not a relaxation** (the inherited `1e-3` was a bit-exact-tensor-reproduction tolerance from the LagrangeBench shipped-reference analog; applied to a quantity floored by *training quality* rather than *float precision*, it was the wrong kind of yardstick, not just too tight). When future work needs a post-hoc acceptance-criterion change: pre-commit the new bands, anchor to an external published metric, and name *why* the old criterion was miscategorised (if it was) rather than just lowering it.

Pre-flight contract: any Phase 1 entrypoint consuming `meta.json` / `test.tfrecord` asserts the on-disk sha matches the pin above before proceeding (catches upstream-bucket drift, which is unlikely for this stable dataset but possible). `compute_cylinder_flow_stats` already does this for `meta.json` before fitting.

**Phase 1 boundary cross-review (Task 14):** Codex pass against verdicts 1-10 + code-absorption (Tasks 10-12). Findings triaged per pattern-C four-cell framework. Cell distributions populated post-cross-review.

**Why pre-registered as skeleton:** mirrors D0-22's pre-registration-before-implementation pattern. Audit-trail discipline: by the time Phase 1 completes, every verdict has a recorded place; no audit finding lands without an entry.

---

## D0-24 — 2026-05-13 — Case Study 02 Phase 2 audit verdicts (open)

Phase 2 fires the MGN inference end-to-end on the canonical cylinder_flow test trajectory selected in Task 1, then lints both the GT trajectory (CPU control arm) and the MGN rollout (Modal A10G) through the mesh harness. Verdicts pre-registered here BEFORE the Phase 2 Modal fires for Tasks 5/6/7 (D0-23 pattern; cap-rationale per [[feedback_cap_rationale_not_literal]]). Task 1's Strouhal-audit findings (refinement 1) are landed; the 7 verdict bands below are open until Task 9 pins them and Task 12's cross-review triage resolves the entry.

**Canonical trajectory (Task 1 result, `f22319d`):** `traj_idx=44`, `St_U_max=0.192` (in design band `[0.16, 0.21]`), `St_U_mean=0.413` (above band — the U-convention ambiguity flagged in the substrate-class smoke; the canonical's in-band hit is on the centerline / U_max convention), `cyl_d=0.135`, `U_max=1.502`, `U_mean=0.699`. Selection criterion = median-Strouhal-in-band, sorted by `strouhal_U_max` for deterministic ordering. See `02-physicsnemo-mgn/preflight/strouhal_test_trajectories.json`.

**Task 1 substrate-variability finding (per the refinement-1 decision tree, "OK with subset in band" branch):** 23/100 test trajectories land in the design band via the either-or `(strouhal_U_mean ∈ [0.16, 0.21]) OR (strouhal_U_max ∈ [0.16, 0.21])` check; 77/100 land out-of-band on the same check; 0 errors in geometry detection. Inspecting the distribution: `strouhal_U_mean` is systematically larger than `strouhal_U_max` across trajectories (consistent with U_mean < U_max, expected for a near-parabolic inflow profile where the mean inflow speed averaged across all inflow nodes is lower than the centerline / max). Most out-of-band trajectories have `strouhal_U_max ∈ [0.16, 0.21]` while `strouhal_U_mean` exceeds 0.21 — i.e. the literature-anchored `[0.16, 0.21]` Strouhal band for cylinder-wake Re ∈ [100, 300] holds on the centerline convention but not the bulk-mean convention. The canonical traj_idx=44 sits inside the band on the centerline convention; the bulk-mean discrepancy is a substrate-variability finding (not a verdict), recorded here per the decision tree's instruction to "record the out-of-band count + indices as a substrate-variability finding." Out-of-band indices are enumerated in `per_trajectory[].in_design_band == false` in the findings JSON.

**Verdict bands (pre-fire commit):**

1. **PH-CON-001 (mass) on the GT trajectory (Task 5):** the harness-FE-on-P1 floor was ≈5% relative divergence per Phase 1 substrate-smoke (`preflight/substrate_class_smoke.json` `incompressibility` row, which was measured on Phase 1's auto-selected GT trajectory — likely trajectory 0 by accident, not 44). Phase 2 bands on the canonical trajectory's GT lint:
   - **≤ 6 %** (10 % wider than the Phase-1 floor): **PASS-control** — the FE+P1 floor is stable across cylinder_flow trajectories.
   - **(6 %, 10 %]**: **MARGINAL** — trajectory-dependent floor; record per-trajectory variability in the Phase-3 writeup but no methodology amendment.
   - **> 10 %**: **FAIL-control** — the harness floor is unstable across trajectories; widen the substrate-smoke design-band claim or audit the FE implementation. Halts Phase 2.

2. **PH-CON-001 (mass) on the MGN rollout (Task 7):** Phase 1 substrate-smoke had ≈5 % on the 399-step MGN rollout (auto-selected trajectory). Phase 2 bands on the canonical trajectory's MGN-rollout lint (`n_rollout_steps = 599` per `inference.py` default; matches `meta["trajectory_length"] - 1 = 599`):
   - **within ±20 % of the GT-trajectory PH-CON-001 value (verdict 1)**: **PASS** — MGN reproduces GT-level mass conservation at the harness floor.
   - **20–50 % above GT**: **MARGINAL** — bounded model error above the floor; named in the Phase-3 writeup but no amendment.
   - **> 50 % above GT**: **FAIL** — MGN's mass-conservation defect exceeds the harness-floor noise; PH-CON-001 has a meaningful surrogate-vs-GT signal. This IS the case-study-02 headline finding if it lands.

3. **PH-CON-002 (energy drift):** D0-23 v9 routes `vortex_shedding_2d` via the open-driven-dissipative dispatch ⇒ SKIP-with-reason expected on both arms (GT and MGN). Bands:
   - **SKIP-with-reason cites D0-22 + D0-23 v9**: **PASS** (dispatch fires as designed; the rule mirror's substrate-class branch lands).
   - **Raw value emitted (dispatch missed)**: **FAIL** — Task 3's `_assert_loader_contract_mgn` should have caught the missing/wrong `metadata["dataset"]` upstream; OR Task 4's MeshField path failed to route through the v9 dispatch.

4. **PH-CON-003 (dissipation_sign_violation):** same dispatch as verdict 3 ⇒ SKIP expected on both arms. Bands as verdict 3.

5. **Loader-contract enforcement (Task 3, F3 absorption):** Task 3 wires `_assert_loader_contract_mgn` into `load_mesh_rollout_npz`; the 4 rejection tests must fail-closed on each of: fp64 velocity, wrong velocity key, missing `metadata["dataset"]`, invalid `node_type` (out of `{0, 3, 4, 5, 6}`). Bands:
   - **All 4 reject with informative `AssertionError`**, AND synthetic-mesh tests still pass (generic `framework="synthetic"` rollouts bypass): **PASS**.
   - **Any rejection fails-open, OR synthetic rollouts subjected to the MGN contract**: **FAIL** — bug in Task 3's wiring or scope-detection logic.

6. **Rollout-dir isolation (Task 6 + Task 7, F5 absorption):** Task 6's Modal entrypoint reproduces the Phase 1 isolation pattern (`tempfile.mkdtemp` → stage `{edge,node}_stats.json` → `os.chdir(rollout_dir)` → `try / finally: os.chdir(old_cwd)`); Task 7's smoke assertion verifies two same-sha retries cannot read each other's CWD-relative stats. Bands:
   - **Smoke assertion fires green on a fresh container**: **PASS**.
   - **Smoke assertion FAILs**: **FAIL** — bug in Task 6's `tempfile`-pattern; fix + re-fire Task 6 (separate fix-iteration A10G fire under its own pre-registered cap per [[feedback_cap_rationale_not_literal]], NOT a verdict-confirmation re-fire).

7. **`inference_run_status` field (Task 8, design §2.5):** Task 8's regression test pins:
   - `gt.sarif`: `inference_run_status = "n/a_gt_control_arm"`.
   - `mgn.sarif`: `inference_run_status = "from_completed_inference"`.
   Bands:
   - **Both SARIFs carry the field at the pinned values**: **PASS**.
   - **Either field missing OR value is a salvage marker (`from_post_oom_salvage` / `from_truncated_inference`)**: **FORWARD-FLAG** — the salvage triggers were design-anticipated but Phase 2's N=1 path is supposed to be uniformly `from_completed_inference`; a salvage outcome is a Phase-3-writeup forward-flag rather than a code-patch (per design §2.5 the field exists precisely so salvage shows up rather than hiding).

**Phase 2 boundary cross-review (Tasks 11-12) — findings triaged.** Codex/GPT-5-CLI adversarial review of CS02 Phase 2 (commits `f22319d..a15b144`) at `docs/2026-05-13-case-study-02-phase-2-cross-review.md` (commit `e358bce`). All 5 findings absorbed in-rung per user directive (initial four-cell triage placed Finding 3 as cell-1 / re-discovery and Finding 5 as cell-4 / novel framing; user promoted both to cell-2 absorption).

| # | Finding (1-line) | Severity | Cell | Disposition |
|---|---|---|---|---|
| 1 | Phase 2 SARIFs advertise harness schema v1.0 but miss required D0-19 run-level fields (`physics_lint_sha_pkl_inference`, `physics_lint_sha_npz_conversion`, `lagrangebench_sha`, `rollout_subdir` on mgn.sarif); renderer fails loud on the committed artifacts | High | 2 | **Absorbed at `a6fbd14`** — lint entrypoints emit all 10 v1.0 fields with CS02 sentinels for LB-specific stages; SARIFs re-emitted at `efcac9c`; renderer's `_assert_run_level` now accepts both. |
| 2 | Single `git_sha` parameter overloaded as both rollout-path key and `physics_lint_sha_sarif_emission` field | Medium | 2 | **Absorbed at `a6fbd14`** — split into `rollout_id`/`rollout_key`/`rollout_subdir` (path key) + `sarif_emission_sha` (run-level field). |
| 3 | Persistent output path `vortex_shedding_<sha>` collides on same-sha re-fires (F5's CWD isolation alone doesn't cover the Volume layer) | Medium | 1 → 2 (per user) | **Absorbed at `a6fbd14`** — new helper `_make_rollout_inference_subdir(rollout_key, run_id)` → `cs02_mgn_inference_<rollout_key>_<run_id>`; 4 CPU tests bind the f-string format + assert uniqueness invariants + regression-guard the legacy segment. |
| 4 | F3 loader-contract scope keyed off brittle `model.startswith("modulus_")` prefix; a future PhysicsNeMo / NVIDIA rebrand would bypass all four assertions | Medium | 2 | **Absorbed at `5ad2cb8`** — explicit `metadata["rollout_contract"]` key (`MGN_ROLLOUT_CONTRACT_P0 = "physicsnemo_mgn_vortex_shedding_p0"`) + `KNOWN_MGN_ROLLOUT_CONTRACTS` allowlist; legacy prefix retained as backward-compat fallback; 3 new tests cover the new dispatch + bypass paths. |
| 5 | Phase 3 scope-qualifier prose overstates representativeness — canonical trajectory ranks 80/100 by `strouhal_U_max` on the full test set; representative of the in-band subset, not the full distribution | Medium | 4 → 2 (per user) | **Absorbed at `5ad2cb8`** — prose rewritten to "representative of the in-band subset under the pre-registered centerline-convention selection rule"; rank (80/100) and in-band range (69-91) named explicitly; full-distribution claim explicitly dropped. |

**Totals:** 0 cell-1 deferred, 5 cell-2 absorbed, 0 cell-3 forward-flagged, 0 cell-4 open. **Phase 2 cross-review closes clean** — every finding landed as a TDD-backed code or prose absorption with a stable commit anchor; no methodology surprises (the High finding was a real artifact bug masked by Phase 1's LB-shaped renderer scoping, now closed by CS02-aware sentinel fields). The Phase 2 boundary cross-review absorption added **15 new tests** (3 F4 + 4 F3 + 8 from F1/F2 path retests) and **6 commits** (`5ad2cb8` + `a6fbd14` + `efcac9c` + this D0-24 close).



**Verdicts pinned (Tasks 5/6/7/8/9 — 2026-05-13).** All 7 PASS; no Pattern-A drift to absorb.

1. **PH-CON-001 GT (Task 5; `4173b32`) → PASS.** `raw_value = 5.857e-02` ⇒ `5.857%`, inside the PASS band `≤ 6 %`. The harness-FE-on-P1 floor on canonical trajectory 44 is consistent with Phase 1 substrate-smoke's `≈ 5 %` on its auto-selected trajectory (trajectory-dependent variability ≈ `0.86 %` between substrate-smoke's pick and the canonical, well within the `≤ 6 %` PASS band). Phase 1's pre-registered floor is stable across the two trajectories observed.

2. **PH-CON-001 MGN (Task 7; `11c7df2`) → PASS.** `raw_value = 5.881e-02` ⇒ `5.881%`, with the GT-vs-MGN gap = `|5.881 - 5.857| / 5.857 = 0.41 %`, well inside the PASS band `±20 %` of GT. **MGN reproduces GT-level mass conservation essentially at the harness-FE-on-P1 floor on the canonical trajectory** — physics-lint's PH-CON-001 doesn't distinguish a hypothetical MGN model-error contribution from the harness-floor contribution at this discretization. This is the floor-bounds-resolution finding the Phase 3 writeup will name (refinement 2 forward-flag): the rule's verdict on GT + MGN at the floor demonstrates "MGN reproduces GT at the floor," NOT "MGN is physically incompressible to 5%." A tighter discretization would be needed to distinguish whether MGN's `5 %` reflects model error or floor error.

3. **PH-CON-002 (energy drift) on both arms → PASS.** Both gt.sarif and mgn.sarif emit SKIP with `skip_reason` citing `"system_class='open-driven-dissipative' (dataset='vortex_shedding_2d')"` + `D0-22`/`D0-23`. The v9 substrate-class dispatch (D0-23 v9) fired as designed on the graph-mesh inputs (lifted to fire-before-topology in Task 4's reorder).

4. **PH-CON-003 (dissipation_sign_violation) on both arms → PASS.** Same dispatch as v3 ⇒ same SKIP with the same reason text on both arms. Uniform behavior across the GT control + MGN-rollout arms confirms the dispatch is substrate-anchored, not arm-anchored.

5. **Loader-contract enforcement (Task 3; `3eb38b4`) → PASS.** All 4 rejection tests fail-closed on the wired `load_mesh_rollout_npz` path: fp64 velocity (on a manually-written NPZ that bypasses save's down-cast), wrong velocity key, missing `metadata["dataset"]`, invalid `node_type`. Synthetic-mesh rollouts (`framework="synthetic"`, model NOT `"modulus_*"`) bypass the MGN contract correctly per scope. 243 → 249 → 252 harness tests across Tasks 3/4/5.

6. **Rollout-dir isolation (Task 6 + Task 7 smoke; `4d35faa` + `11c7df2`) → PASS.** Task 6's production path uses `tempfile.mkdtemp` with a `git-sha-prefixed` name (landed at `/tmp/mgn_rollout_p0_4173b32_x06lz8pq` for this fire); Task 7's 3 CPU smoke tests verify the Python-level invariants the production-path depends on (distinct paths under same prefix; no cross-dir CWD-relative reads; same invariant for `dir=None`). The F5 absorption is closed: two same-sha retries cannot read each other's CWD-relative stats.

7. **`inference_run_status` field (Task 8; `3c028ff`) → PASS.** Both SARIFs carry the field at the design-§2.5-pinned values: `gt.sarif → "n/a_gt_control_arm"` (no inference fire on the control arm); `mgn.sarif → "from_completed_inference"` (Task 6's A10G fire was clean — no salvage). 3-test regression guards pin the values + restrict to the design-§2.5 enumerated set; a salvage-marker value here is a Phase-3 writeup forward-flag, not a code-patch.

**No Pattern-A drift to absorb.** All 7 verdicts landed in their pre-registered PASS bands; no MARGINAL / FAIL routing fired; no D-entry amendment needed. The cross-stack consistency table for Phase 3 populates with `GT(5.857%) + MGN(5.881%)` both at the floor + the v9 substrate-class SKIP on both arms.

**Phase 2 §4.2 acceptance checklist (Task 10; complete except cross-review pending).**

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Modal entrypoint for MGN inference committed; pre-flight assertions cover persistent-volume write path, NGC checkpoint hash verification, rollout output schema, CWD discipline, fp32 default-dtype, split="test". | ☑ | Task 6 (`4d35faa`); `_preflight_mgn_inference_p0` covers all 6 items + `_assert_loader_contract_mgn` belt-and-braces at save time. |
| 2 | Round-codex-4 rollout-dir isolation applied IFF Phase 1 committed persistent volume. | ☑ | Task 6 (`4d35faa`) `tempfile.mkdtemp` + `chdir` + `try / finally`; rollout_dir landed at `/tmp/mgn_rollout_p0_4173b32_x06lz8pq`. |
| 3 | P0 vortex-shedding rollout completed end-to-end on Modal (n_trajs per Phase 1 cost estimate; rung-4c discipline). | ☑ | Task 6 (`4d35faa`); N=1 on canonical traj 44 (refinement 1 + Task 1 Strouhal audit); 599 steps inferred on A10G in one fire. |
| 4 | Per-timestep MeshField/GridField materialization works per Gate A branch. | ☑ | Task 4 (`4debbbf`) wired scikit-fem P1 graph-mesh path into the three `*_on_mesh` rule mirrors; exercised end-to-end in Tasks 5 + 7 (PH-CON-001 = 5.857% / 5.881% via `_fe_divergence_defect_max`). |
| 5 | PH-CON-001 SARIF committed at `02-physicsnemo-mgn/outputs/sarif/`; values within Phase-1-pre-registered mass-conservation drift bound. | ☑ | `gt.sarif` (`4173b32`) + `mgn.sarif` (`11c7df2`); D0-24 v1 = PASS (5.857% ≤ 6%) and v2 = PASS (0.41% gap ≤ 20%). |
| 6 | PH-CON-002 (`energy_drift_on_mesh`) SARIF committed; behavior consistent with `MGN_DATASET_SYSTEM_CLASS` dispatch (SKIP-with-reason or RAW per dispatch). | ☑ | Both SARIFs SKIP with `open-driven-dissipative` reason citing D0-22 + D0-23 (D0-24 v3 = PASS); v9 dispatch fires as designed. |
| 7 | PH-CON-003 (`dissipation_sign_violation_on_mesh`) SARIF committed; SKIP/RAW outcome per dispatch. | ☑ | Same dispatch ⇒ same SKIP on both arms (D0-24 v4 = PASS). |
| 8 | `inference_run_status` field lands per §2.5 (uniformly `from_completed_inference` predicted; salvage triggers fire forward-flag instead). | ☑ | Task 8 (`3c028ff`); 3 regression tests pin the field at design-§2.5 values (`gt → "n/a_gt_control_arm"`, `mgn → "from_completed_inference"`) + restrict to the enumerated set. |
| 9 | Smoke check: rule outputs vs Phase-1-pre-registered tolerances; pattern-A drift absorbed via D-entry amendment if any. | ☑ | Task 9 (`3380742`); all 7 verdicts PASS; no Pattern-A drift to absorb. |
| 10 | Phase 2 boundary cross-review (Codex pass over SARIF + rule outputs) complete; findings triaged. | ☐ | Pending Tasks 11 + 12. |

Plus the two Phase 1 forward-flags Phase 2 inherited:

| # | Forward-flag | Status | Evidence |
|---|---|---|---|
| F3 | `_assert_loader_contract_mgn` wired into `load_mesh_rollout_npz` (the first trusted MGN boundary). | ☑ | Task 3 (`3eb38b4`); 5 tests on the wired path (4 rejection + 1 round-trip + 1 synthetic bypass) all PASS. |
| F5 | Rollout-dir isolation pattern reproduced in the MGN inference entrypoint + smoke assertion that two same-sha retries cannot read each other's CWD-relative stats. | ☑ | Task 6 (`4d35faa`) production-path pattern; Task 7 (`11c7df2`) 3 CPU smoke tests on the Python-level invariants. |

**Phase 3 writeup-framing requirements (refinement 2 forward-flag; lands as Phase 3 writing-plans input).**

The Phase 3 plan must include:

1. **Scope qualifier paragraph** (verbatim, in case-study-02 README's results section): *"Phase 2 results report PH-CON-001 defect on a single cylinder_flow test trajectory (trajectory `M = 44`, Strouhal `St_U_max = 0.192` ∈ design band `[0.16, 0.21]`, cylinder diameter `D = 0.135`, inflow `U_max = 1.502`; Reynolds number not directly captured at Phase 2 — derivable as `Re = U · D / ν` from the cylinder_flow benchmark's `ν` if needed in Phase 3), selected via the Phase 2 pre-fire Strouhal audit (Task 1, 23 / 100 in-band) to be representative of the in-band subset under the pre-registered centerline-convention selection rule (median-`strouhal_U_max` among the 23 in-band trajectories). Trajectory `44` ranks 80th of 100 by `strouhal_U_max` on the full test set (the in-band subset occupies ranks 69-91; out-of-band trajectories sit at lower `strouhal_U_max` values, where the literature-anchored design band would not hold anyway). Coverage-not-statistics framing: physics-lint's value here is rule-firing on a real-world checkpoint, not a distribution over initial conditions. CI-gate threshold derivation from defect-magnitude distributions would require `N > 1` and is deferred (Phase 2 does NOT claim CI-gate calibration or full-distribution representativeness)."*

2. **Floor-bounds-resolution distinction** in §3.3 "what physics-lint did NOT catch": name the harness-FE-on-P1 floor (`≈ 5.8 %` on this trajectory; `≈ 5 %` on Phase 1 substrate-smoke's auto-selected trajectory) explicitly as bounding PH-CON-001's discriminating resolution. The rule's verdict on GT + MGN at `5.857 %` / `5.881 %` (0.41 % gap, within the harness-floor envelope) demonstrates **"MGN reproduces GT at the floor,"** NOT **"MGN is physically incompressible to 5 %."** At this discretization (test-trajectory mesh; `N_nodes = 1787`, `N_cells = 3340`; P1 basis), PH-CON-001 bounds MGN's deviation from GT-equivalence rather than from physical incompressibility. A tighter discretization would distinguish whether the `5 %` reflects model error or floor error. The Phase 3 writeup must hold this distinction explicitly to avoid the over-read.

3. **Substrate-variability finding** (Task 1 result, refinement 1 + decision tree's "OK with subset in band" branch): record that 23 / 100 test trajectories land in the literature-anchored design band `[0.16, 0.21]` on the either-or convention check; 77 / 100 land out-of-band on the same check (mostly with `St_U_mean > 0.21` while `St_U_max ∈ [0.16, 0.21]` — the U-convention ambiguity). The Phase 3 writeup notes this as a substrate-variability characterization, not a methodology failing — the canonical trajectory selection (median-Strouhal-in-band on `strouhal_U_max`) is deterministic and reproducible.

4. **Methodology forward-flag (out of Phase 2 + 3 scope):** `physics-lint-validation-plan-v2.1.2` §1.4 *"Prose scope-qualifiers — claim precision about what evidence demonstrates vs adjacent claims it could be over-read to support."* This is the fourth empirical instance (after round-code-1's three walls — level rendering, file-path location, prose framing): physics-lint's PH-CON-001 verdict on this case study supports the narrow claim "MGN reproduces GT at the harness-FE-on-P1 floor on trajectory 44" but could be over-read to support "MGN is physically incompressible to 5 %" — distinct claims separated by the floor-bounds-resolution distinction. The cumulative pattern is strong enough to formalize as a v2.1.2 methodology entry. Tracked separately, does NOT block Phase 2 / 3 completion.

**Phase 2 COMPLETE (2026-05-13).** All 7 D0-24 verdicts PASS; all 5 cross-review findings absorbed in-rung; §4.2 acceptance checklist closed (criterion 10 cleared at this commit); F3 + F5 forward-flags from Phase 1 closed; the new explicit rollout-contract identifier (Finding 4 absorption) is the going-forward production path with the legacy prefix retained as a backward-compat fallback. Phase 3 (writeup + cross-stack table + v2.1.1 amendment) gets a fresh writing-plans round next; the refinement-2 forward-flag block above carries the scope-qualifier prose + floor-bounds-resolution distinction + the v2.1.2 §1.4 methodology entry into that round.

**Phase 3 boundary cross-review (Task 9-10) — findings triaged.** Codex / GPT-5-CLI adversarial prose review of CS02 Phase 3 writeup (diff scope: commits `0b97c14..4347abb`) at [`docs/2026-05-13-case-study-02-phase-3-cross-review.md`](docs/2026-05-13-case-study-02-phase-3-cross-review.md). 6 findings (2 HIGH, 3 MEDIUM, 1 LOW); all prose-level absorbable — no SARIF data-level finding, no fresh-inference need; the pre-registered ≤1 A10G contingency (~$2) is **unused** and stays available for any later absorption. Findings triaged per the pre-registered Q4a trigger classification: 5 cell-2 absorbed in-rung, 1 cell-1 absorbed in-rung (re-discovery; sharpens existing prose).

| # | Finding | Severity | Cell | Absorption / Defer |
|---|---|---|---|---|
| 1 | Inference-gate test conflates retired `10⁻³` max-abs gate with current RMSE-1 ≤ 5e-3 gate | HIGH | cell-1 | Absorbed at `a42784b` — CS02 README "Inference-gate test" rewritten to name the D0-23 retirement explicitly. |
| 2 | Cross-stack-table Headline says "rule schema unmodified" where README correctly says "structural-identity reapplication" | HIGH | cell-2 | Absorbed at `4d66e44` — cross-stack-table Headline now distinguishes result-row schema reuse (which DOES hold across stacks) from public v1 rule-code reuse (which does NOT for CS02 PH-CON-001); points outward to CS02 README "PH-CON-001 routing" for the routing detail. |
| 3 | Floor-bounds correction lands ≈50 lines after a PASS table that primes a wrong skim-read | MEDIUM | cell-2 | Absorbed at `a42784b` — inline caveat under CS02 README results table cross-references "What physics-lint did NOT catch" §1 for the floor-bounds-resolution distinction. |
| 4 | "representative" word in scope-qualifier rubs against "does NOT claim full-distribution representativeness" disclaimer | MEDIUM | cell-2 | Absorbed at `a42784b` — replaced "to be representative of the in-band subset" with "as the centerline-Strouhal-central member of the in-band subset". |
| 5 | D-entry shorthand ("D0-22 + D0-23 v9", "rung-4a") appears in the results table before its local meaning is introduced | MEDIUM | cell-2 | Absorbed at `a42784b` — one-sentence terminology glossary added immediately above the CS02 README results table; cross-references DECISIONS.md for the full audit trail. |
| 6 | Cross-stack-table "NOT" fence on numeric-comparison priming lives after the model-quality framing has been rhetorically staged | LOW | cell-2 | Absorbed at `4d66e44` — added an upfront "NOT directly comparable as a model-quality measure" sentence to the cross-stack-table Headline, immediately after the numeric contrast. |

**Totals:** 1 cell-1 absorbed (re-discovery; phrasing sharpened), 5 cell-2 absorbed (novel-in-scope, in-rung), 0 cell-3 forward-flagged, 0 cell-4 open. **Phase 3 cross-review closes clean** — every finding landed as a prose absorption with a stable commit anchor; no SARIF re-emission and no GPU fire. The cumulative findings character (ordering / forward-cite hygiene; reader-first cross-reference placement) is a fifth empirical instance of the prose-cross-review-vs-artifact-cross-review distinction's prose-side signature, which v2.1.1 §2.2 names and points outward to.

**Phase 3 §4.3 acceptance checklist.**

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `02-physicsnemo-mgn/README.md`: Rule × checkpoint results table populated (vortex shedding row only) | ☑ | Task 4 (`ba7942a`) — PH-CON-001/002/003 numbers + verbatim scope-qualifier prose; Finding-5 absorption (`a42784b`) added terminology glossary above the table. |
| 2 | Bridge-sentence paragraph drafted | ☑ | Task 5 (`68fe470`) — links CS02 results into rung-4 cross-stack story via v2.1 §1.5 falsification-surface framing. |
| 3 | Cross-stack consistency table populated (vortex shedding row; Ahmed Body row marked "deferred to amendment 1") | ☑ | Task 3 (`79face7`) — new dated artifact `2026-05-13-case-study-02-cross-stack-conservation-table.md` extending rung-4a; Ahmed Body deferral named in "What this table is NOT" §3; Finding-2 + Finding-6 absorptions (`4d66e44`) sharpened the headline to distinguish result-row schema reuse from public v1 rule-code reuse and to fence numeric-comparison priming. |
| 4 | "What physics-lint did NOT catch" section written | ☑ | Task 6 (`486bb86`) — 7 sub-sections covering floor-bounds-resolution + substrate-variability + PH-CON-001 routing + PH-NUM-002 deferral + PH-SYM-* scope + Ahmed Body/PH-RES-001 deferral + v2.1.2 forward-flag; Finding-3 absorption (`a42784b`) added the inline floor-bounds caveat under the results table itself. |
| 5 | Reproducibility section: Modal entrypoints, NGC checkpoint hash, git_sha, image-spec / dependency pinning (criterion label revised from "conda lockfile" at round-codex-2 finding 4 — actual evidence is the Modal image-spec pin, not a conda lockfile; the underlying acceptance is satisfied by either form) | ☑ | Task 7 (`5828980`) — Modal entrypoint table (7 entrypoints) + NGC checkpoint provenance (`ngc_version = latest`, `ckpt_sha256 = 0153803c8b2c0947948757a2298eec6ef21e8ea28131834963db08f34b4a0726`, `physicsnemo_sha = 1ca85d65ac2ce28ea9762910c09a954c08a37140`) + image-spec pin (`modal_app.py` builds the image; no conda lockfile; image-spec sha named); Finding-1 absorption (`a42784b`) corrected the Gate D inference-gate prose (D0-23 retired the original `10⁻³` max-abs criterion; RMSE-1 ≤ 5e-3 is the recalibrated band). |
| 6 | Methodology cross-references to `methodology/README.md` + v2.1.md §1.5 | ☑ | Task 7 (`5828980`) — Cross-references subsection lists all targets (v2.1.md, v2.1.1.md, methodology/README.md, DECISIONS.md, `_harness/SCHEMA.md`). |
| 7 | v2.1.1 amendment landed (per §5.5) | ☑ | Task 11 (`821be50`) — `physics-lint-validation-plan-v2.1.1.md` with 4 items in 2 categories (framework-behavior §1.1+§1.2; specific-finding §2.1+§2.2); Task 12 (`db753ba`) — v2.1.md §1.3 + §1.4 + §3 changelog pointers added (frozen-original-plus-pointer; round-prose-2 forward-flag marked RESOLVED); Task 13 (`d5b0983`) — light self-check landed two link-path corrections (`../DECISIONS.md` + `../../../../README.md#hero-physics-lint-in-ci`). |
| 8 | Phase 3 boundary cross-review complete; findings triaged | ☑ | Task 9 (`36da8b1`) — `2026-05-13-case-study-02-phase-3-cross-review.md` with 6 findings (2 HIGH, 3 MEDIUM, 1 LOW); Task 10 absorptions (`a42784b`, `4d66e44`, `e366dad`) — all 6 findings absorbed in-rung (5 cell-2 + 1 cell-1); A10G contingency unused. |

Plus the CS01 back-port (cross-stack discoverability symmetry per Q3 refinement 2): Task 8 (`4347abb`) — CS01 README points to both rung-4a (frozen) + 2026-05-13 (canonical going forward) cross-stack artifacts, with framing language naming the frozen-original-plus-pointer discipline.

**Phase 3 COMPLETE (2026-05-13).** All 8 §4.3 acceptance criteria met. CS02 README expanded to ~140-line case-internal writeup (3 added sections + 1 absorption-glossary paragraph + 1 inline caveat + 4 single-paragraph absorptions); new dated cross-stack table artifact extends rung-4a via the frozen-original-plus-pointer discipline; v2.1.1 dated artifact lands the four pre-registered §5.5 items in two categories; v2.1.md gains pointer-only additions at §1.3, §1.4, and §3 changelog forward-flag (RESOLVED → v2.1.1 §2.2); CS01 README gets a 3-5 line back-port. Phase 3 boundary cross-review absorbed in-rung per the pre-registered trigger classification (Q4a) — all 6 findings prose-level; A10G contingency unused. The v2.1.2 §1.4 prose-scope-qualifier formalization is the next methodology amendment in the trail; tracked separately, post-Phase-3.

**Phase 3 round-codex-2 (second cross-review pass, artifact-mode) — findings triaged.** After the Phase 3 §4.3 close above committed, a second Codex cross-review fired in artifact-mode (parallel to round-codex-2/-3/-4 on rung-4c PR #9; voice = artifact-mode adversarial-reader auditing committed code + dated artifacts for fail-open paths, data-level invariants, and methodology-trail integrity). Scope: 17-commit Phase 3 diff (`0b97c14..096c2ea`). 6 findings (0 HIGH, 3 MEDIUM, 3 LOW); all prose-or-code level absorbable in-rung; A10G contingency still untouched. Findings triaged per the pre-registered Q4a trigger classification: 5 cell-2 absorbed in-rung, 1 cell-1 absorbed in-rung (re-discovery of round-codex-1 finding 2 with sibling residue in CS02 README's bridge).

| # | Finding | Severity | Cell | Trigger classification | Absorption shape |
|---|---|---|---|---|---|
| 1 | Renderer can silently render empty cross-stack table if all SARIFs filtered as control-arm (no post-filter `if not stacks` guard) | MEDIUM | cell-2 | Code-level absorbable | Absorbed at `b9237b1` — `render_cross_stack_table` raises `MissingRunLevelFieldError` when all SARIFs filtered as control-arm; regression test passes only `gt.sarif` and asserts the raise. |
| 2 | CLI `main()` paths (action='append', positional glob pairing, zero-match, strict-pairing failure) not exercised by tests — only `render_cross_stack_table` API was tested | LOW | cell-2 | Test-coverage absorbable | Absorbed at `b9237b1` — four new `main()`-level tests cover the documented unified invocation, zero-glob default fill, more-globs-than-dirs failure, and zero-match-glob failure. |
| 3 | "rule schema runs unmodified" residue still in CS02 README's bridge sentence (cross-stack artifact itself was fixed at round-codex-1 finding 2; sibling residue remained) | MEDIUM | cell-1 | Prose-level absorbable (re-discovery) | Absorbed at `5ff6832` — bridge sentence corrected to "three-row conservation result-schema + run-level field reuse, NOT public v1 rule-code reuse"; inline PH-CON-001 routing pointer added. |
| 4 | §4.3 acceptance criterion 5 labeled "conda lockfile" but evidence text says "no conda lockfile" — internally self-contradictory | LOW | cell-2 | Prose-level absorbable | Absorbed at `a495162` — criterion 5 relabeled "image-spec / dependency pinning" with inline note that either form satisfies the underlying acceptance. |
| 5 | Prose-cross-review instance counts drift between v2.1.md §3 ("second / third") and v2.1.1.md §2.2 ("fifth / five precedents") — counting unit (review-events vs walls) not declared | LOW | cell-2 | Prose-level absorbable | Absorbed at `a495162` — both files now use "review-events" as the canonical unit; v2.1.1 §2.2 gains an explicit "Counting unit" paragraph; four prose-mode events enumerated (round-prose-1, round-prose-2, round-code-1 prose dimension with three walls as sub-findings, CS02 Phase 3 with six sub-findings). |
| 6 | Phase 3 plan-vs-reality retrofit incomplete — stale `arm == 'control'` literal at L341, "runs unmodified" at L327/482, retired Gate D max-abs at L640 remained in plan template prose | MEDIUM | cell-2 | Prose-level absorbable | Absorbed at `bda0814` — Phase 3 plan gains an "Execution drift note" banner at the top naming all four stale literals + the line numbers + the corrected commit shas; templates preserved verbatim per the plan-as-historical-contract discipline, but the drift is now explicit. |

**Totals:** 1 cell-1 absorbed (re-discovery; sibling residue closed), 5 cell-2 absorbed (4 prose + 1 code + 1 test; novel-in-scope, in-rung), 0 cell-3 forward-flagged, 0 cell-4 open. **Phase 3 round-codex-2 closes clean** — every finding landed as a prose / code / test absorption with a stable commit anchor; no SARIF re-emission, no GPU fire, A10G contingency unused. The round-codex-2 second-pass finding character (audit-trail integrity, code-path fail-open paths after filter, internal contradictions in dated-artifact prose) is what the artifact-mode review is uniquely positioned to catch, parallel to round-codex-3/-4's role on rung-4c PR #9.

**Cumulative cross-review coverage for Case Study 02:**

- Phase 2 round-codex-1 (artifact-mode, pre-Phase-3 close): 5 findings, 5 absorbed.
- Phase 3 round-codex-1 (prose-mode, mid-Phase-3): 6 findings, 6 absorbed.
- Phase 3 round-codex-2 (artifact-mode, post-§4.3-close, pre-merge): 6 findings, 6 absorbed.

Total 17 findings across three review-events; 17 absorbed in-rung; 0 deferred. The triple covers both modes (artifact + prose) plus a same-Phase second pass at the artifact-mode level, satisfying the [[feedback_codex_review_as_task_line_item]] discipline that names cross-review as a planned task with both boundary-review and second-pass coverage.

**Case Study 02 COMPLETE.** Phases 1 + 2 + 3 all closed clean. D0-23 (Phase 1) + D0-24 (Phase 2 + Phase 3) carry the full audit trail. Successor work: amendment 1 (Ahmed Body / PH-BC-001 / steady RANS) per design §5.1; case study 03 candidate (strictly-conservative anchor reassignment per design §5.10 + the v2.1.1 §2.1 falsification flag); cover-letter cross-case-study integration per design §5.6 (now unblocked — both CS01 + CS02 writeups exist).

**Why pre-registered as skeleton:** mirrors D0-22 + D0-23's pre-registration-before-implementation pattern. Audit-trail discipline: by the time Phase 2 fires, every verdict has a recorded place; the routing is not interpreted-during-execution, and any Pattern-A drift (per design §3.1.A) lands as a D-entry amendment with the pre-fire bands as the comparison baseline.

---

## D0-25 — 2026-05-12 — P2 Reverse-Poiseuille-2D substrate-class probe — trilateral substrate-class evidence (pre-registration)

**Renumber note (PR #11 merge, 2026-05-15).** This entry was originally numbered **D0-24** (committed to master via PR #10 before CS02 Phase 2 had merged). PR #11's CS02 Phase 2 verdicts also claimed D0-24, with 94 references across 11 files when PR #11 branched from master. Per the original D0-24 prose explicitly inviting this reconciliation ("merge-time numbering reconciliation is the user's call") and per propagation-footprint asymmetry (renaming RPF2D = 7 textual references; renaming CS02 = 94), this entry was renumbered **D0-24 → D0-25** at merge time. The "Numbering note" paragraph below preserves the historical numbering rationale at original-authoring time. Methodology principle landed: D-entry numbering collisions between parallel branches resolve by propagation-footprint asymmetry, not commit chronology — the entry with the smaller reference footprint yields. If the original entry's prose pre-registered the deferability (as this entry's did), the rename is a pre-registered fallback, not a post-hoc compromise.

**Status:** pre-registered before any RPF2D rollout. Execution decoupled from this pre-registration — see "Execution gate" below. (Verdict slots populate on execution; until then, this entry is the pre-registered intent only.)

**Numbering note (historical, at original authoring).** D0-23 is reserved for the Case Study 02 Phase 1 audit verdicts (pre-registered on `feature/case-study-02-physicsnemo-mgn` at sha `5e7b146`, not yet in master at the time of this entry). D0-24 is the next available number from master's perspective (master had D0-22 as the latest D-entry when this entry was drafted). If CS02 Phase 2 / Phase 3 plans later claim D0-24, the merge-time numbering reconciliation is the user's call (the two sessions run in parallel; branch isolation makes this a git-merge concern, not a coordination protocol — same posture as the round-prose-2 / round-code-1 work). *(Resolved at PR #11 merge per the Renumber note above: this entry became D0-25; CS02 Phase 2 holds D0-24.)*

**Predecessor:** D0-22 (substrate-class taxonomy + the "classify when you exercise" principle + the catalogue-wide two-tier split that lists `rpf2d` as "almost certainly misclassified, awaiting empirical probe"), v2.1 §3.1 P1 (the PH-BC-001 strike precedent for the dam-break-2D row).

**Trigger:** post-`tag:rung-4-closure` review of "what's left for case study 01" surfaced that the LB README's P2 target row — *"P2 | Reverse Poiseuille 2D | SEGNN | `PH-BC-001` (no-slip)"* — has the same problem v2.1 §3.1 P1 caught for the dam-break-2D P1 row: `PH-BC-001` as shipped in physics-lint v1.0 is a **mesh-FEM Dirichlet boundary-trace rule on a unit square**, not an SPH-particle wall rule. The README's P2 headline names a rule that does not apply to LB SPH particle rollouts.

**Decision (pre-registered).**

1. **Strike `PH-BC-001` from the P2 RPF2D headline**, mirroring the v2.1 §3.1 P1 dam-break correction. RPF2D emits no wall-non-penetration / no-slip row. PH-BC-particle-wall as a future SPH rule (PH-BC-NNN with `particle_type == WALL`) remains deferred (post-visa; see D0-22's forward-flags + v2.1 §3.1 P1).

2. **RPF2D substrate class: empirically probe, do not pre-classify.** `LAGRANGEBENCH_DATASET_SYSTEM_CLASS["rpf2d"]` is currently `"dissipative"` (a D0-18-pass preemptive guess; D0-22's two-tier split flags it as "almost certainly misclassified"). Reverse Poiseuille is a **body-force-driven flow** — an anti-symmetric (sign-flipping across the channel midline) body force injects momentum continuously, so the closed-dissipative class's monotone-non-increasing-KE precondition almost certainly fails. Candidate classes (subject to the empirical probe at execution time): `"open-driven-dissipative"` (if the anti-symmetric forcing reads as the same class as dam-break's gravitational drive) or a new `"forced-dissipative"` / `"body-force-driven"` taxonomy entry (if the anti-symmetric-internal-forcing case is distinguishable from dam-break's gravity-loaded-fall-then-dissipate shape). Per "classify when you exercise" (D0-22), the empirical probe — measure KE(t) on a 1-traj smoke, check whether it rises/plateaus/decays — is the **first move of execution**, not a pre-registration commitment.

3. **Expected rule outcomes (subject to empirical probe; same shape as rung-4c dam2d):**
   - `mass_conservation_defect`: raw = 0.0 (×N identical) — trivial-outcome; LB SPH preserves particle count by construction.
   - `energy_drift`: SKIP — driven system; the strictly-dissipative-or-conservative assumption underpinning the rule does not apply to a body-force-driven flow (extends the D0-22-amendment-1 SKIP path to whatever substrate class RPF2D empirically lands in).
   - `dissipation_sign_violation`: SKIP — same substrate-class lookup; `dE/dt > 0` over stretches by physics (the anti-symmetric body force does work on the fluid).
   - i.e., three SKIPs and a trivial-outcome — byte-similar to rung-4c's dam2d cross-stack table.

4. **The load-bearing claim is TRILATERAL substrate-class evidence, not the SARIF artifacts.** Rung 4a/4b exercised TGV2D (`"dissipative"`, closed); rung 4c exercised dam-break-2D (`"open-driven-dissipative"`, gravity-loaded-fall); a rung-5-level RPF2D exercise would add a **third** substrate class (forced-with-anti-symmetric-body-force) under the same harness machinery, with no rule code branching on dataset identity. The trilateral evidence lives in the **substrate-class taxonomy + D-entries** (the harness correctly *chooses not to fire* `energy_drift` / `dissipation_sign_violation` on a third forcing mechanism), **not in the SARIF level distributions** (which would be largely redundant with rung-4c's note-only conservation SARIFs). Do not conflate "marginal new SARIF info" (true — SARIFs are near-identical to rung-4c) with "marginal methodology evidence" (false — the trilateral substrate-class taxonomy entry is the deliverable). SARIF content is what physics-lint *catches*; substrate-class evidence is what physics-lint *correctly chooses not to fire on* — different artifacts, different load-bearing roles.

5. **`N` (trajectory count) TBD at execution.** Per D0-22 amendment 2's "ship at the N the budget supports, document the drift honestly" discipline — RPF2D per-traj cost on SEGNN is not calibrated; a 1-traj smoke calibrates it, and the production fire ships at whatever N the per-rung budget supports. No cross-rung-N-parity commitment (per the round-prose-2 framing: cross-rung N parity is presentation symmetry, not validity for a structural / substrate-class claim).

**Execution gate (decoupled from this pre-registration).** Pre-registering D0-25 creates the *option* to execute RPF2D later without committing the cost (Modal compute ~$0.20–0.50 + design + plan + pre-flight smoke + production fire + a substrate-class-taxonomy D-entry amendment + a paragraph in the trilateral-evidence writeup — lighter than a full case-study-shaped writeup since the SARIF artifacts are largely redundant with rung-4c, but still real). The execution decision is **conditional, gated on Case Study 02 Phase 2 timing**:
- If CS02 Phase 2 stays on the critical path → RPF2D execution **defers to a future case study** (a "case study 03" candidate slot); the trilateral-substrate-class claim becomes writeup-future-work rather than writeup-current-evidence.
- If CS02 Phase 2 closes ahead of schedule → fire RPF2D execution as the buffer-time deliverable.
The conditionality is self-resolving from Phase 2's actual progress; no separate decision point needed beyond observing when Phase 2 closes.

**Forward flags carried into D0-25:**
- Bilateral D0-18 still requires a *strictly conservative* substrate anchor (case study 02 territory — PhysicsNeMo MGN incompressible NS as the candidate); RPF2D does not exercise this (RPF2D is forced/driven, not conservative).
- `ldc2d` / `rpf3d` / `ldc3d` remain in D0-22's "almost certainly misclassified, awaiting empirical probe" tier; a future rung exercising any of them walks into a known-misclassification and the empirical probe is its first move (same posture D0-25 takes for `rpf2d`).
- If RPF2D's empirical probe surfaces that anti-symmetric-internal-forcing is a *distinct* class from dam-break's gravity-drive, the taxonomy bump (adding `"forced-dissipative"` or similar) lands as a D0-25 amendment at execution time, not as a pre-registration commitment here.

**Pattern fit.** D0-25 (the pre-registration) is methodology-trail discipline — pre-register before implement, mirroring D0-22's shape. The PH-BC-001 strike is **pattern A** (source-review-surfaced drift: the README P2 row named a rule that doesn't apply; the response is the honest correction here, mirroring v2.1 §3.1 P1's dam-break correction and continuing the rung-4 series's source-review-catches-issue-before-compute trilateral → now a fourth instance: rung-4b math + rung-4b figure-sweep + rung-4c dam2d-misclassification + rung-5 RPF2D-headline-misclassification, all caught at $0 Modal cost). The execution decision is **conditional**, not a pattern.

**Implementation:** none yet (pre-registration only). A future rung-5 RPF2D plan (if execution fires) follows the rung-4c plan shape: design → implementation plan → pre-flight 5-step checklist → Modal smoke (the substrate-class empirical probe) → production fire at supportable N → SARIF emission → trilateral-evidence writeup → §9 cross-review boundary. The pre-flight smoke's ABORT-on-fail path applies (per the CLAUDE.md pre-flight checklist + D0-22's smoke-gates-reclassification precedent).

---

## D0-26 - 2026-05-18 - P2.1 CS02 multi-trajectory expansion (pre-registration)

**Status:** RESOLVED 2026-05-18. The Decision section was pre-registered
before any P2.1 Modal fire; the Verdict section was filled post-execution
by P2.1 Task 6. See the Amendment note at the end of this entry for two
post-execution corrections caught by Codex review (a transcription error
in Decision 1's index list, and a wording correction in Decision 4).

**Predecessor:** D0-24 (CS02 Phase 2 audit verdicts; the N=1 PH-CON-001
result on canonical trajectory 44).

**Trigger:** roadmap P2.1. The CS02 Phase 2 PH-CON-001 result is reported on
a single cylinder_flow trajectory (traj 44). The README hoists an explicit
"N = 1" fence above the number. P2.1 expands to a 5-trajectory distribution
so the README can lead with a small-N statistical claim.

**Decision (pre-registered).**

1. **Trajectory set: `{88, 48, 44, 38, 60}`.** From the 23 in-band
   trajectories in `preflight/strouhal_test_trajectories.json` sorted by
   `strouhal_U_max`, the members at sorted indices 0, 6, 11, 16, 22 - an
   even spread across the in-band Strouhal range `[0.1601, 0.2069]`, with
   index 11 = the Phase-2 canonical trajectory 44. Selection is
   deterministic and reproducible from the committed audit JSON.
   (Index 16, not 17, identifies traj 38 - see the Amendment note below;
   the pre-registration as first committed read "17".)

2. **N = 5.** Above the roadmap's "3 or more" floor; an odd N gives a clean
   median. If 1-2 trajectories fail irrecoverably, P2.1 still lands at N >= 3
   (the acceptance floor) with the shortfall documented.

3. **Reproduction gate (traj 44, run first).** Trajectory 44 is re-fired
   first because Phase 2 measured it: `gt_raw = 5.857e-02`,
   `mgn_raw = 5.881e-02`. Gate bands:
   - GT `raw_value` within 1e-3 relative of 5.857e-02 (the GT lint is
     deterministic CPU FE - it must reproduce tightly). Outside this band
     => the harness or data pin changed => STOP.
   - MGN `raw_value` within 2e-2 relative of 5.881e-02 (599-step
     autoregressive A10G inference carries float nondeterminism). 2-10%
     drift => investigate before continuing; >10% => STOP.

4. **Per-trajectory band (D0-24 v2 PASS band; MARGINAL/FAIL extended to
   two-sided).** The MGN/GT gap as a percentage of GT: `|gap| <= 20%`
   PASS, `20-50%` MARGINAL, `>50%` FAIL. D0-24 v2 (verdict 2) defined
   PASS two-sided (`±20%`) but MARGINAL/FAIL one-sided (`above GT`),
   because Phase 2's concern was an MGN defect *exceeding* GT. P2.1
   compares multiple trajectories where MGN can deviate from GT in
   either direction, so the MARGINAL/FAIL thresholds here apply to
   `|gap|` (two-sided). This is an extension of D0-24 v2, not a verbatim
   reuse - see the Amendment note below.

5. **Expected outcome.** All 5 trajectories PASS the D0-24 v2 band, and the
   per-trajectory gap stays well inside the harness-FE-on-P1 floor (the gap
   is smaller than the GT defect itself). This would confirm "MGN
   reproduces GT at the floor" generalizes across the in-band subset, not
   just on traj 44. A trajectory-dependent failure (wide gap variance) is a
   reportable finding, not a bug - see Risks.

6. **Out of scope.** The cross-stack conservation table is not touched
   (it reports the canonical-trajectory cell as a schema-uniformity
   artifact; traj 44 stays canonical and Task 2 re-confirms its number).
   P2.2 (PH-BC-001 cylinder no-slip) is a capability-build on a separate
   track. P2.3 (vanilla-GNN baseline) is deferred.

**Verdict.** P2.1 fired the 5-trajectory expansion `{88, 48, 44, 38, 60}`.
Reproduction gate: trajectory 44 re-fired at `gt=5.857e-02`, `mgn=5.881e-02`
(within the pre-registered bands of Phase 2's `5.857e-02 / 5.881e-02` - GT
reproduced exactly, MGN to 1.6e-8 relative). Result: median MGN/GT gap
`-0.36 %` of GT, range `[-1.07 %, +0.41 %]`; all 5 trajectories inside the
harness-FE-on-P1 floor; D0-24 v2 band per trajectory = PASS for all 5. Per
trajectory: 88 `-1.07 %`, 48 `-0.36 %`, 44 `+0.41 %`, 38 `+0.03 %`,
60 `-0.36 %`. Decision 5's expected outcome is CONFIRMED - "MGN reproduces
GT at the floor" generalizes across the in-band subset, not just traj 44.
The CS02 README now leads with the small-N statistical framing.

**Realized in:** 8f52f77, a983af6, 537c14f, 4f9a4a5, 405e78f, 11e5c91.

**Amendment (2026-05-18, post-execution Codex adversarial review).** Two
documentation corrections; neither changes any fired trajectory or any
measured number.

- *Decision 1 index typo.* The pre-registration as first committed
  (8f52f77) read "sorted indices 0, 6, 11, **17**, 22". Re-deriving the
  selection from the committed `strouhal_test_trajectories.json` (23 in-band
  trajectories, sorted ascending by `strouhal_U_max`) shows traj 38 sits at
  sorted index **16**; index 17 is traj 32. The fired set `{88, 48, 44, 38,
  60}` is the intended one - the pre-registration's own `strouhal_U_max`
  column gave `0.2023` for that member, which is traj 38's value (traj 32 is
  `0.2028`). So the trajectory set, the fires, and every result are correct;
  only the written index was wrong. Decision 1 is corrected to "16" above.
- *Decision 4 wording.* The original parenthetical "(D0-24 v2, reused)"
  implied a verbatim reuse of D0-24 verdict 2's band. D0-24 v2's MARGINAL
  and FAIL tiers are one-sided ("above GT"); the P2.1 band applies them to
  `|gap|` (two-sided). Decision 4 is reworded above to state this as an
  *extension*. No P2.1 verdict changes: every trajectory's `|gap|` is
  <= 1.07%, well inside the two-sided PASS band.

---

## D0-27 - 2026-05-19 - P2.2 closes as a structural-degeneracy finding (PH-BC-001 no-slip on masked-wall MGN)

**Predecessor:** roadmap P2.2 + Amendment 3 (the audit closed on the
capability-build branch); D0-26 (P2.1).

**Trigger:** the P2.2 design pass found that a no-slip check, even with the
mesh wall-node capability built, would be degenerate on its only near-term
target. The CS02 cylinder MGN rollout masks boundary nodes during inference
(`mgn_rollout_p0_vortex_shedding`: `v_diff_masked = torch.where(mask2,
pred_i_velo, zeros)` then `v_next = v_diff_masked + invar[:, 0:2]`, where
`invar[:, 0:2]` is the current rollout velocity state), freezing wall nodes at
their step-0 ground-truth value `0`. A no-slip check computes `||v_wall|| ~ 0`
and PASSes by construction, detecting nothing about the surrogate.

**Decision.**

1. **P2.2 closes as a documented structural-degeneracy finding, not a
   capability-build.** PH-BC-001's no-slip check is structurally unreachable
   on a masked-wall MGN rollout - well-defined but predetermined. The planned
   mesh wall-node BC capability-build is retired (Amendment 3 had recorded it
   as the audit's branch; this entry supersedes that with the deeper finding).
2. **The finding is recorded in the CS02 README** ("What physics-lint did NOT
   catch") and **here**; it is deliberately NOT added to the conservation
   cross-stack table, which stays conservation-scoped and golden-tested.
3. **Ahmed Body verification** (`preflight/ahmed_body_protocol_audit.md`):
   the Ahmed Body MeshGraphNet checkpoint, instantiated as plain
   `physicsnemo.models.meshgraphnet.MeshGraphNet` with `output_dim: 4`
   (`conf/model/mgn.yaml:17, :22`; selected by `conf/experiment/ahmed/mgn.yaml`
   via `/model: mgn`), outputs surface pressure and wall shear stress - not a
   body-surface velocity field. `inference.py:149-154` assigns only `p_pred`
   and `wallShearStress_pred`; the experiment's drag coefficient is computed
   downstream from those plus mesh geometry, not predicted by the model. The
   example's separate `AeroGraphNet` variant (`models.py:43-94`, with a
   `c_d_decoder` head) is *not* the model wired by `+experiment=ahmed/mgn` and
   does not affect the verdict (NVIDIA/physicsnemo @
   1ca85d65ac2ce28ea9762910c09a954c08a37140, v2.0.0).
4. **Capability-build disposition:** see the cross-checkpoint scope below.

**Cross-checkpoint scope.** The Ahmed Body MGN outputs surface pressure / wall
shear stress, not a body-surface velocity field, so a no-slip *velocity* BC
check is inapplicable there - degenerate for a different structural reason
than masking. The velocity-BC capability-build has no home among current
targets and is retired from P2.2; any future velocity-BC work requires a new
target and a fresh decision entry. A pressure/force-based surface check would
be a different rule, outside P2.2 and P4.1 scope.

**Realized in:** a2c1bb2, c7c1c95, and this commit.

---

## D0-28 - 2026-05-20 - P2.3 closes as a documented methodology choice (equivariance gap second-architecture question)

**Predecessor:** roadmap P2.3; D0-27 (P2.2 close); D0-21 (rung-4b equivariance pre-registration).

**Trigger:** roadmap §2.3 proposed training a vanilla non-equivariant GNN on TGV2D and running PH-SYM-001 / PH-SYM-002 against it, to "confirm the equivariance gap is not GNS-architecture-specific." The P2.3 design pass surfaced two findings (below) and a structural-empirical-link argument that together make the proposed empirical second data point a poor trade.

**Finding 1 - no published non-equivariant alternative to GNS exists.** Per LagrangeBench's published model zoo (https://github.com/tumaer/lagrangebench, accessed 2026-05-20), the project ships pretrained checkpoints only for GNS (GNS-10-128, across 7 datasets) and SEGNN (SEGNN-10-64, across the same 7 datasets); EGNN, PaiNN, and the linear baseline have model definitions in `lagrangebench/models/` but no published weights. SEGNN, EGNN, and PaiNN are equivariant by design, so GNS is the only non-equivariant model with a downloadable checkpoint. A second non-equivariant architecture would have to be trained.

**Finding 2 - a self-trained model is a structurally weaker evidence class.** The CS01 / CS02 methodology rests on published checkpoints (the F3 borrowed-credibility / published-baseline-reproduction framing from `methodology/docs/2026-05-01-rollout-anchor-extension-design.md` §1.1): the value claim is that physics-lint fires its rule on a real artifact-as-shipped. A vanilla GNN trained by us cannot borrow credibility by construction; the result is in a different and weaker class than the rest of the validation portfolio, and importing it dilutes the published-checkpoint discipline.

**The structural-empirical-link argument.** PH-SYM-001 / PH-SYM-002 measure empirical equivariance: they check `f(g x) ~ g f(x)` on sampled inputs against the model's noise floor (rung-4b found SEGNN monomodally at the float32 floor, ~2.3e-7 to 3.4e-7; GNS bimodally split between APPROXIMATE band ~3.6e-4 to 4.2e-4 and FAIL band quantized at ~0.02). A model with no equivariant inductive bias has no architectural mechanism to force `f(g x) = g f(x)` to noise-floor precision, so under typical training the empirical reading sits well above the noise floor and the rule fires. GNS-as-shipped is one realization of that link; a second non-equivariant trained model would manifest the same link on a different instance.

**Training-regime caveat.** A non-equivariant-by-architecture model trained with full SO(2) data augmentation could in principle approximate equivariance to near-noise-floor accuracy. This is not fatal to the choice: GNS-as-shipped was not trained that way, and the rule fires on it. The argument's scope is "non-equivariant architecture under typical training," not "non-equivariant architecture" in isolation.

**Falsifiability.** A non-equivariant-by-architecture model that, after typical training on TGV2D-like data, produced an equivariance gap at the float32 noise floor would falsify the structural-empirical link. The project knows of no such documented case. The claim is a defeasible empirical generalization grounded in the architectural reason, not an a priori certainty ruled true by definition.

**Decision.** P2.3 closes as a documented methodology choice, not a training run. GNS represents the non-equivariant architecture-plus-training-regime class on a published checkpoint; this entry is the standalone closure document. The rung-4b writeup (`docs/2026-05-07-rung-4b-equivariance-table.md`) is sha-bound by the methodology/README.md `### Adding a new rung writeup` convention and is not edited — the integrating `methodology/README.md` carries the discoverability pointer to this entry for a reader who lands on the rung-4b section first. No training, no model definition, no GPU, no code.

**Out of scope (deliberately not done):** any training run; any new model definition; any new row or column in the rung-4b equivariance table or any cross-stack table; the CS01 LagrangeBench README's stub state (a separate concern, belongs to P3 docs work); anything in CS02 (the equivariance gap is rung-4b / CS01-side, not CS02; CS02 explicitly scopes PH-SYM-* out by design).

**Realized in:** 63e69fe, 4777ad0, and this commit.

**Amendment (2026-05-20, post-execution code review).** Three corrections; none change the decision.

- *Finding 1 catalog citation.* Added the upstream LagrangeBench reference (`github.com/tumaer/lagrangebench`) so the "GNS and SEGNN are the only LagrangeBench checkpoints with downloadable weights" claim is web-verifiable rather than implicit; also noted the linear baseline alongside the four GNN models. The load-bearing claim (a second non-equivariant LagrangeBench model would have to be trained) is unchanged.
- *Finding 2 attribution.* The "borrowed credibility" phrasing was attributed to the CS02 README; the actual source is the F1 / F2 / F3 anchor-class framing at `methodology/docs/2026-05-01-rollout-anchor-extension-design.md` §1.1. Attribution corrected above; the load-bearing claim (the rollout-anchor portfolio rests on published checkpoints, not self-trained baselines) is unchanged.
- *Structural-empirical-link magnitudes.* The original said "rung-4b found SEGNN at ~1e-6, GNS at ~1e-2"; the cited table reports SEGNN monomodally at ~2.3e-7 to 3.4e-7 (well below the PASS band at 1e-5) and GNS bimodally split between APPROXIMATE (~3.6e-4 to 4.2e-4) and FAIL (~0.02) bands. Magnitudes corrected above; the load-bearing point (architectural-equivariance gap between the float32 floor and approximate-by-training reading) is unchanged.

**Amendment (2026-05-20, post-execution Codex adversarial review).** Three corrections; none change the decision.

- *Frozen-writeup convention restored.* The plan as-written inserted a scope note as item 8 of `## 6. What rung 4b is NOT` in `methodology/docs/2026-05-07-rung-4b-equivariance-table.md`. That violates the `methodology/README.md` `### Adding a new rung writeup` rule: "individual writeups stay sha-bound to their snapshot views and are not edited after their rung closes." The precedent (v2.1 §2.5 walking back rung-4c §6 item 8; round-code-1 closing rung-4b §6 item 3) routes all post-closure walk-backs through D-entries plus a `methodology/README.md` pointer, never the frozen writeup itself. The item-8 edit is reverted; this D-entry is the standalone closure document; the integrating README carries the discoverability pointer.
- *Decision-paragraph pointer updated.* Decision paragraph no longer references item 8 (now reverted); it instead names the frozen-writeup convention and the integrating-README pointer route. The load-bearing claim is unchanged.
- *LagrangeBench citation pinned by access date.* Finding 1's URL now reads `https://github.com/tumaer/lagrangebench, accessed 2026-05-20` so a future upstream catalog change does not retroactively make D0-28 look false (mutable-`main`-cite hazard surfaced by Codex; the current content of `main` was independently verified against the same claim by both Claude and Codex).

## D0-29 - 2026-06-03 - v1.2.0 ships the mesh-vector substrate + PH-CON-005 (∇·v public); particle → v1.3.0

**Predecessor:** the 2026-05-22 `docs/backlog/v1.2.md` scope-marker (CLI/Action loader integration for mesh AND particle); narrowed by the 2026-06-01 mesh-vector design spec.

physics-lint v1.2.0 ships the **mesh half** of the post-v1.0 Field-protocol extension and defers the particle half. This entry is the dated decision-of-record for the living-doc corrections (CS02 README, `_rollout_anchors/README.md` §5, CS01 README, the docs-site case studies, `docs/backlog/v1.2.md`, the public README) whose version-pinned "planned for v1.2.0" claims are corrected at the v1.2.0 tag. The dated methodology writeups stay sha-bound (append-only history); this is the standalone closure record per the frozen-writeup convention.

- **New `mesh_vector` field type + rule PH-CON-005.** A `MeshVectorField(Field)` (2D graph-mesh velocity on a scikit-fem `MeshTri`/`ElementTriP1` basis) flows through the public `loader → check` pipeline, and a new emit-only rule **PH-CON-005** reports the ∇·v incompressibility defect (Frobenius-normalized; ports the harness `_fe_divergence_defect_max`). CS02's ∇·v law now runs through the public `physics-lint check` CLI/Action, not only the private harness — reproducing the harness within the pre-registered Gate-B ε ≤ 1e-4 (measured gap 0.0e0 on a controlled fixture).
- **Scope is ∇·v only.** PH-CON-001 itself is unchanged (still `SKIPPED` on `pde != "heat"`); the energy-drift and dissipation-sign NS defects remain harness-side (v1.2.x follow-ons on the same `MeshVectorField`); the scalar `MeshField` loader (`field.type = "mesh"`) is still rejected (follow-on when an FE-mesh surrogate consumer appears); 3D (`MeshTet`) is a follow-on.
- **Particle → v1.3.0.** Particle equivariance is a delicate-direction, model-mode-only rule (per `SCHEMA.md` §4.1 / D0-04a: a per-index `||Rx - x|| / ||x||` is O(1) on any honest C4 orbit because the rotation permutes particle indices; only `||R^-1 f(R x0) - f(x0)|| / ||f(x0)||` on trajectory-aligned model rollouts is well-defined). It earns its own F1 design pass in v1.3.0 rather than riding a coupled release.
- **SARIF surfacing of the emit-only scalar (Option 1).** PH-CON-005 is PASS + `severity=info`, so it never emits a SARIF `run.results` finding (every result renders as a code-scanning alert regardless of `level`). Its scalar surfaces as a `note`-level `toolExecutionNotification` (the non-alerting channel SKIPPED already uses), scoped to PASS + info + a `raw_value` so gating-rule PASSes and PH-VAR-002 (`raw_value=None`) stay silent.
