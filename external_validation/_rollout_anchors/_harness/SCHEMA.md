# `_harness` schemas and tolerances

Authoritative reference for the two `.npz` rollout schemas, the SARIF
property surface emitted by harness-driven rule invocations, and the
pre-registered tolerances that underpin the Gate B (controlled-fixture
validation) and §4.4 (trained-model equivariance) verdicts.

This file is the **single source of truth** for the schema and the
tolerance numbers. The two adapter modules
(`particle_rollout_adapter.py`, `mesh_rollout_adapter.py`) and the
fixture-driven test (`tests/fixtures/test_harness_vs_public_api.py`)
must remain consistent with the values declared here. If a schema field
or a tolerance changes, the change lands here first and the adapters
follow — not the other way around.

---

## 1. `particle_rollout.npz` — LagrangeBench (JAX/Haiku, particles)

Spec §3.2.

```python
{
    "positions":     ndarray,  # (T, N_particles, D)  fp32
    "velocities":    ndarray,  # (T, N_particles, D)  fp32
    "particle_type": ndarray,  # (N_particles,)       int32   fluid/wall/...
    "dt":            float64,
    "domain_box":    ndarray,  # (2, D)               fp64    [[xmin,...], [xmax,...]]
    "metadata": {                                      # serialized via numpy structured array or pickle-as-allow_pickle=True
        "ckpt_hash":         str,   # SHA-256 of checkpoint file
        "ckpt_path":         str,
        "git_sha":           str,   # physics-lint commit at time of generation
        "lagrangebench_sha": str,   # external repo commit
        "dataset":           str,   # "tgv2d" | "dam2d" | "rpf2d" | ...
        "model":             str,   # "segnn" | "gns"
        "seed":              int,
        "framework":         str,   # "jax+haiku"
        "framework_version": str,
    },
}
```

`particle_type` integer codes follow LagrangeBench's convention; the adapter
does not relabel them. `domain_box[0]` is the per-axis minimum, `domain_box[1]`
the maximum.

Both halves of the particle adapter (read-only and model-loading) consume
this schema. The model-loading half additionally consults `ckpt_path` to
re-load the JAX checkpoint for rotated-input rollouts.

## 2. `mesh_rollout.npz` — PhysicsNeMo MGN, or FNO under Gate D fallback

Spec §3.2.

```python
{
    "node_positions": ndarray,                   # (N_nodes, D)         fp32   static (mesh fixed across t)
    "edge_index":     ndarray,                   # (2, N_edges)         int64  (omitted for FNO grid case)
    "node_type":      ndarray,                   # (N_nodes,)           int32  INTERIOR/INFLOW/WALL/...
    "node_values":    dict[str, ndarray],         # per-field, each (T, N_nodes [, D_field]) fp32
    "dt":             float64,
    "metadata": {
        "ckpt_hash":         str,
        "ngc_version":       str,                 # "v0.1" etc.
        "git_sha":           str,
        "dataset":           str,                 # "vortex-shedding-2d" | "ahmed-body" | "darcy"
        "model":             str,
        "framework":         str,                 # "pytorch+dgl" | "pytorch+neuraloperator"
        "framework_version": str,
        "resampling_applied": bool,                # True under Gate A PARTIAL fallback
    },
}
```

`node_values` keys are field names (e.g., `"velocity"`, `"pressure"`,
`"density"`). Each value array's leading axis is time, second is node, and
the optional trailing axis is per-component (D for vector fields).
`edge_index` is omitted from the npz under the FNO-on-Darcy fallback —
consumers must check `metadata.framework` to decide whether to look for
mesh connectivity.

`resampling_applied=True` under Gate A PARTIAL means the mesh adapter
sampled DGL output onto a regular grid before constructing a `GridField`;
the cover-letter paragraph variant A.2 cites this case explicitly.

---

## 3. SARIF properties surface

Both adapters and the public-API mesh path emit SARIF in the same schema
(`runs[].results[].ruleId`, `message`, `locations`, `properties`). The
`properties` object carries the fields below; consumers MUST treat unknown
properties as additive metadata and not error on their presence.

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `source` | enum (literal string) | yes | `"rollout-anchor-harness"` for harness-emitted results, `"physics-lint-public-api"` for results emitted by the public `physics-lint check` CLI. The literal-string namespace is deliberate — see §3.1 below for the namespace rationale. |
| `harness_validation_passed` | bool \| null | yes (harness only) | `True` iff Gate B PASS at the time the result was emitted; `False` under Gate B APPROXIMATE; `null` when not applicable (public-API path). |
| `harness_vs_public_epsilon` | float \| null | conditional | The Gate B ε_harness_vs_public value for the rule × fixture pair the result derives from. `null` for results unrelated to a Gate B fixture. |
| `case_study` | str | yes | `"01-lagrangebench"` \| `"02-physicsnemo-mgn"` \| `"02-fno-darcy"`. |
| `dataset` | str | yes | Mirrors `metadata.dataset` from the source `.npz`. |
| `model` | str | yes | Mirrors `metadata.model` from the source `.npz`. |
| `ckpt_hash` | str | yes | Mirrors `metadata.ckpt_hash` from the source `.npz`. |

### 3.1 `properties.source` namespace

The literal string `"rollout-anchor-harness"` is the value, not a
namespaced URN (e.g., not `"urn:physics-lint:rollout-anchor-harness"`). The
choice tracks two constraints:

1. SARIF v2.1.0's `properties` bag is intentionally schema-free; consumers
   are expected to read by string match, not by namespace dereferencing.
   A literal value matches the existing physics-lint convention in
   `physics_lint.report.RuleResult` and the public-API SARIF emitter.
2. The string is short and grep-friendly. Consumers separating harness-
   from public-API-emitted SARIF use `jq '.runs[].results[] |
   select(.properties.source == "rollout-anchor-harness")"`.

If the SARIF schema convention changes upstream (e.g., physics-lint
public-API SARIF moves to namespaced source strings), update both the
public emitter and `sarif_emitter.py` together; the harness MUST stay in
sync with the public emitter on this field.

---

## 4. Tolerances

§4.1 and §4.2 below are pre-registered tolerances that **answer different
questions**: §4.1 measures *agreement between two implementations of the
same metric* (the harness vs the public rule on a controlled fixture);
§4.2 measures *the trained model's own equivariance defect* on rotated
rollouts. The thresholds are not interchangeable. See §4.1 Reader's note
and `physics-lint-validation/DECISIONS.md` D0-04a / D0-07 for the full
framing.

### 4.1 Gate B — controlled-fixture harness-vs-public-API tolerance

Spec §4.3 / plan §7 Gate B.

| ε_harness_vs_public | Verdict | Action |
|---------------------|---------|--------|
| ≤ 10⁻⁴ | **PASS** | Proceed to Day 1 LagrangeBench rollouts. |
| 10⁻⁴ < ε ≤ 10⁻² | **APPROXIMATE** | Document divergence here and in `_rollout_anchors/README.md` §"What we are NOT claiming". Honest-finding branch. |
| > 10⁻² | **FAIL** | Stop the bus. Apply the 5-minute fixture-sanity-check qualifier (spec §6 Gate B). Do not proceed to Day 1 until the harness is fixed or the writeup pivots to mesh-only. |

The 10⁻⁴ tolerance is **pre-registered** before fixtures are run; it is
not retroactively tuned to match observed values. Methodological-honesty
discipline applied to the validation-of-validation step.

#### Reader's note — what Gate B is and is not

Gate B is a **regression test on the gridify+rot90 pipeline**, not a
cross-method epistemic check. Both the harness path
(`c4_static_defect`, `reflection_static_defect`) and the public-API path
(`ph_sym_001.check`, `ph_sym_002.check`) consume the same gridded scalar
field produced by `gridify` and apply the same `np.rot90` / `np.flip`
transform plus the same denominator-stabilised relative L^2 — so on Day
0's fixtures the two paths emit bit-identical floating-point output by
design. ε_harness_vs_public = 0.000e+00 is the *expected* outcome under
this design, not surprising agreement between independent computations.

The reason Gate B is set up this way is documented in
`physics-lint-validation/DECISIONS.md` D0-04a: spec §4.2's instruction
to "apply C4 to particle positions and velocities, compute ε_C4
directly" reads naturally as a per-index computation, but per-index
``||R x − x|| / ||x||`` gives O(1) on any honest C4 orbit because the
rotation permutes particle indices. Routing both paths through the
gridded-density representation is the harness's resolution: it makes ε
permutation-invariant, agree numerically with the public rule's
emission on the gridded equivalent, and computable on a static fixture
where there is no trained model to apply.

The genuine cross-method check — *does the harness's emitted ε agree
with what the rule would emit on an independent computation?* — is the
Day 1 model-loading path's per-index ε on ``f(x_0)`` vs ``R^{-1} f(R
x_0)``, computed on trained-model rollouts. There the per-index
comparison is well-defined because trained-model rollouts preserve
particle identity across the identity-vs-rotated pair. That check
answers a different question than Gate B (the model's own equivariance
defect, not the harness's faithfulness to the rule), and uses the
threshold structure pre-registered in §4.2.

A reader of Gate B's PASS verdict should therefore *not* read it as
"the harness reproduces the rule's emission on independent
computations" (overclaim) or as "the harness is degenerate"
(under-trust). The correct reading is "the gridify+rot90 pipeline has
no implementation-level regressions, and both the harness and the
public rule produce the same answer when handed the same gridified
input — which is the input the harness produces from a particle
configuration."

### 4.2 Trained-model equivariance band — Day 1 model-loading path

Spec §4.4. *Separate from §4.1.* This band interprets the per-index,
trajectory-aligned ε_rot / ε_refl the harness emits when running
rotated-input rollouts of trained SEGNN / GNS checkpoints (Day 1 work).
Pre-registered in `physics-lint-validation/DECISIONS.md` D0-07; the
per-index, trajectory-aligned framing is what makes the band testable
on actual model output as opposed to the static fixtures of §4.1.

Concretely: with rollout ``f(x_0)`` from initial conditions ``x_0`` and
rollout ``f(R x_0)`` from rotated initial conditions, the harness
computes

    ε_rot(R) = || R^{-1} f(R x_0) − f(x_0) || / max(|| f(x_0) ||, eps)

per-particle-index across the full trajectory (or the union of
trajectories for ``eval.n_trajs > 1``). The same form applies to
``ε_refl`` with ``R`` replaced by the reflection matrix.

| ε_rot or ε_refl (per-index, trajectory-aligned) | Verdict |
|-------------------------------------------------|---------|
| ≤ 10⁻⁵ | PASS — machine-precision equivariance. SEGNN expected. |
| 10⁻⁵ < ε ≤ 10⁻² | APPROXIMATE — flagged, in approximate-equivariance band. GNS expected. |
| > 10⁻² | FAIL — equivariance broken. |

Pre-registered before SEGNN/GNS runs. If pilot data shows SEGNN at, e.g.,
10⁻⁴ instead of 10⁻⁶, the threshold is **not** silently amended; the
divergence is logged in `physics-lint-validation/DECISIONS.md` as a
D0-08+ entry citing the discrepancy explicitly, and the band may be
amended only with the discrepancy reproduced verbatim in the writeup.

### 4.3 §4.2 fixture-construction tolerance distinction

Spec §4.4 closing note. ε_C4 ≤ 10⁻⁶ in fixture #1 (`c4_invariant_4particle`)
is the *fixture-construction* tolerance — the configuration is exactly
C₄-invariant by construction, so observed ε is dominated by floating-point
round-off and lands near machine epsilon. This is **not interchangeable**
with the §4.4 trained-model band (10⁻⁵): the fixture is a tighter sanity
check on harness arithmetic; the trained-model band is a looser empirical
envelope around theoretical equivariance.

The fixture-construction tolerance does NOT gate Gate B. Gate B is gated
by ε_harness_vs_public (§4.1), the *cross-path* difference, not the
absolute-error of either path alone.

---

## 5. Versioning

Schema version: `1.0`. Bumps land here first, with a one-line changelog
below; adapters follow.

- **1.0** (2026-05-04): initial schema, Day 0.
