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
    "particle_mass": ndarray,  # (N_particles,)       fp64
    "dt":            float64,
    "domain_box":    ndarray,  # (2, D)               fp64    [[xmin,...], [xmax,...]]
    "metadata": {                                      # serialized via numpy structured array or pickle-as-allow_pickle=True
        "ckpt_hash":          str,   # SHA-256 of checkpoint directory (sorted-relpath digest of file contents)
        "ckpt_path":          str,
        "git_sha":            str,   # physics-lint commit at time of generation
        "lagrangebench_sha":  str,   # external repo commit
        "dataset":            str,   # "tgv2d" | "dam2d" | "rpf2d" | ...
        "model":              str,   # "segnn" | "gns"
        "seed":               int,
        "framework":          str,   # "jax+haiku"
        "framework_version":  str,
        "write_every":        int,   # rollout dt = dataset.dt * write_every (LagrangeBench convention)
        "write_every_source": str,   # "dataset" if read from dataset metadata.json; "default" if defaulted to 1
        "periodic_boundary_conditions": list[bool],  # length D; True axes get minimum-image distance for derived velocities
        "periodic_boundary_conditions_upstream": list[bool],  # original len-N PBC from LB metadata.json (N may be > D per LB convention)
        "periodic_boundary_conditions_source": str,   # "dataset" | "truncated_from_oversize" | "default"
    },
}
```

`particle_type` integer codes follow LagrangeBench's convention; the adapter
does not relabel them. `domain_box[0]` is the per-axis minimum, `domain_box[1]`
the maximum.

`particle_mass` is per-particle mass. For datasets where per-particle mass
is not specified by the source (e.g., LagrangeBench SPH datasets, where
mass is folded into the smoothing-length normalization), the conversion
populates uniform unit mass. The conservation rules (PH-CON-001, PH-CON-002)
and dissipation rule (PH-CON-003) check temporal *changes* in mass and
energy, which are invariant to a global mass-scale choice; uniform unit
mass is therefore methodologically equivalent to the dataset's implicit
normalization for these tests. Datasets that *do* carry per-particle mass
should pass it through unchanged; the harness consumes whatever mass field
is present.

`metadata.periodic_boundary_conditions` is a length-D boolean list
mirroring LagrangeBench dataset metadata.json's `periodic_boundary_conditions`
key (post-truncation; see below); `True` indicates the axis has
periodic BC and the conversion applied minimum-image distance
correction when deriving velocities from positions. Without this
correction, particles crossing periodic boundaries produce spurious
O(L/dt) velocities under naive central differences (rung-3.5
spot-check on f75e22d8dd surfaced 5+-order-of-magnitude KE inflation
on SEGNN-TGV2D from this exact failure mode; see DECISIONS.md D0-17).
Datasets where the key is missing from metadata.json get all-False
per axis and the conversion is no-op (non-periodic fallback).
Consumers that compute distances directly from positions (rather
than from velocities) should also consult this field — the harness's
`gridify` already does, but any future consumer of the npz that does
e.g. ``np.linalg.norm(p1 - p2)`` on TGV2D would need to apply the
same correction.

`metadata.periodic_boundary_conditions_upstream` and
`metadata.periodic_boundary_conditions_source` exist because
LagrangeBench's stable upstream convention is "PBC field is always
length 3 regardless of `dim`" (verified across 2D TGV2D production
and 3D LJ tutorial fixture). When the upstream PBC vector is longer
than D, the conversion truncates to the first D entries and records
both the original upstream vector and the source classification
(`"truncated_from_oversize"` for the truncate path; `"dataset"` for
length-D-exact; `"default"` when the key was missing). Trailing
truncated entries are sanity-checked to be all True (matches the
upstream vestigial-axes-always-periodic convention); a trailing
False fires a hard error since it would mean either the convention
changed upstream or the dataset metadata is corrupted. See
DECISIONS.md D0-17 amendment 1 for the full reasoning. Pre-amendment-1
npzs (none on Volume; the post-D0-17 regen at 8c3d080397 failed the
length validation before writing any npz) lack these fields.

`metadata.write_every` and `metadata.write_every_source` exist because
LagrangeBench dataset metadata.json conditionally carries `write_every`
(present in production datasets like `2D_TGV_2500_10kevery100`; absent in
the `tests/3D_LJ_3_1214every1` tutorial fixture). The conversion records
both the value used (defaulting to 1 when the key is missing) and the
source of that value, so future audit-trail reconstruction can distinguish
a dataset-specified dt from a defaulted dt without re-reading the original
metadata.json. This is the same shape of instrumentation as the
``UPSTREAM_COMPAT_PATCHES`` ledger in ``01-lagrangebench/modal_app.py``:
record the choice at the moment it's made, not at reconstruction time.

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

### 3.x Harness SARIF result schema (D0-19)

Pre-registered in `physics-lint/external_validation/_rollout_anchors/methodology/DECISIONS.md` D0-19. Schema version: v1.0.

#### Run-level properties (`runs[0].properties`)

All 10 fields REQUIRED. Renderer raises `MissingRunLevelField` on any absent field.

| Field | Type | Description |
|---|---|---|
| `source` | string | Literal `"rollout-anchor-harness"`. Discriminator vs public-API SARIF. |
| `harness_sarif_schema_version` | string | `"1.0"` for this version. Renderer's `EXPECTED_SCHEMA_VERSION` binds on equality. Bumps on any contract change in this section; co-evolves with `physics_lint_sha_sarif_emission` by construction. |
| `physics_lint_sha_pkl_inference` | string | Sha at which the LB CLI ran on Modal to produce pkls. |
| `physics_lint_sha_npz_conversion` | string | Sha at which pkl→npz conversion ran. May equal inference sha (single-shot run) or differ (multi-session). |
| `physics_lint_sha_sarif_emission` | string | Sha at which the lint code emitted this SARIF. May equal the other two or differ. |
| `lagrangebench_sha` | string | LB upstream sha (the inference engine producing the pkls). |
| `checkpoint_id` | string | LB gdown identifier or symbolic name. |
| `model_name` | string | LB CLI key, e.g., `"segnn"` or `"gns"`. |
| `dataset_name` | string | LB dataset identifier, e.g., `"tgv2d"`. |
| `rollout_subdir` | string | Volume artifact location at npz-genesis time, e.g., `"/vol/rollouts/lagrangebench/segnn_tgv2d_<sha>/"`. |

The three `physics_lint_sha_*` fields MAY be identical (single-shot) or distinct (multi-session). The renderer MUST NOT assume equality across stages.

#### Result-level fields (`runs[0].results[*]`)

Standard SARIF fields:

- `ruleId` — one of `"harness:mass_conservation_defect"`, `"harness:energy_drift"`, `"harness:dissipation_sign_violation"` (the `harness:` prefix distinguishes harness rules from public-API rule IDs).
- `level` — `"note"` for both PASS-equivalent values and SKIPs.
- `message.text` — human-readable summary.

Result-level `properties`:

| Field | Guaranteed-identical across trajs (within stack)? | Description |
|---|---|---|
| `traj_index` | NO (may-vary) | Integer 0..(N-1) where N is the number of trajectories in the rollout. |
| `npz_filename` | NO (may-vary) | E.g., `"particle_rollout_traj00.npz"`. |
| `raw_value` | YES iff present AND value happens to be identical across rows | Float. Present iff row is not a SKIP. |
| `skip_reason` | YES (template constant) | String. Present iff row is a SKIP. Template-constant — no per-row value interpolation. |
| `ke_initial` | NO (may-vary) | Float. Present only on `harness:energy_drift` SKIP rows. |
| `ke_final` | NO (may-vary) | Float. Present only on `harness:energy_drift` SKIP rows. |

#### Schema-enforced invariants

For a fixed (rule, stack), all result rows MUST have:

- Identical `ruleId`, `level`, `message.text`.
- Identical `properties.raw_value` if present, OR identical `properties.skip_reason` if present (HarnessDefect emits exactly one of the two on every row).

Consumers MAY assert these invariants at render time. The schema makes them checkable; checking is not mandatory.

#### Energy_drift skip_reason template

Per D0-19, the `harness:energy_drift` SKIP path uses a template-constant skip_reason (no per-row value interpolation). Per-row varying KE values move to `properties.ke_initial` / `properties.ke_final`. See `_harness/particle_rollout_adapter.py:energy_drift` for the canonical template; renderer-side documentation cites that path.

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
D0-09+ entry citing the discrepancy explicitly, and the band may be
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

### 4.4 KE-rest skip-with-reason threshold (read-only path)

Pre-registered in `physics-lint-validation/DECISIONS.md` D0-08. The
particle harness's read-only-path defect functions
(``energy_drift``, ``dissipation_sign_violation``) skip with a string
reason rather than emit a numeric defect when the rollout's reference
KE is below the threshold:

    KE_REST_THRESHOLD = 1e-10   # absolute, in the dataset's natural KE units

- ``energy_drift(rollout)`` SKIPS when ``KE(t=0) < KE_REST_THRESHOLD``:
  the relative drift ``max|KE(t) − KE(0)| / |KE(0)|`` is undefined for
  rollouts that start at rest, and the eps-floored denominator
  otherwise inflates the emitted value to a meaningless large finite
  number.
- ``dissipation_sign_violation(rollout)`` SKIPS when ``max_t KE(t) <
  KE_REST_THRESHOLD``: the trajectory has effectively no kinetic
  energy at any timestep, so the dissipation question is meaningless.
- ``mass_conservation_defect(rollout)`` is **not** subject to a skip
  threshold (M(0) > 0 in any physical configuration); the
  ``HarnessDefect`` return type is preserved for downstream SARIF
  emission symmetry.

Both functions return a ``HarnessDefect`` dataclass that is either
``(value=numeric, skip_reason=None)`` or ``(value=None, skip_reason=str)``;
the constructor enforces exactly-one-set so consumers can branch on
``defect.value is None`` without ambiguity. The SARIF emitter renders
``skip_reason``-set defects as ``result.kind = "informational"``
entries with the reason in ``result.message.text`` — analogous to
physics-lint's ``RuleResult.status = "SKIPPED"`` rendering.

Threshold form is **absolute** (1e-10 in the dataset's natural KE
units), not relative-within-rollout (``KE(0) < 1e-10 * max(KE)``).
The absolute form is dataset-specific and acknowledged as such; a
v1.1 escape hatch may switch to the relative-within-rollout form
when cross-dataset comparison becomes load-bearing. The simpler
absolute form is fine for v1 because (a) physical SPH rollouts have
KE(0) of order unity in their natural units, well above 1e-10; (b)
the only failure mode this catches at v1 is "all particles at rest
at t=0", which is a categorical input-domain mismatch rather than a
calibration gradient.

If pilot data on Day 1 surfaces KE(0) within an order of magnitude
of the threshold, log a new DECISIONS.md D0-09+ entry citing the
discrepancy and amend the threshold — do not silently shift in
code. The test ``test_ke_rest_threshold_matches_pre_registration``
in ``test_read_only_path.py`` enforces this discipline.

---

## 5. Versioning

Schema version: `1.4`. Bumps land here first, with a one-line changelog
below; adapters follow.

- **1.0** (2026-05-04): initial schema, Day 0.
- **1.1** (2026-05-04): §4.4 KE-rest skip-with-reason threshold pre-registered;
  ``HarnessDefect`` polymorphic return type for read-only-path defects
  (DECISIONS.md D0-08).
- **1.2** (2026-05-04): §1 ``particle_mass`` field documented (was already
  required by ``load_rollout_npz`` but undocumented; SCHEMA-vs-code drift
  surfaced and closed during DECISIONS.md D0-15 amendment 4 / rung-3.5
  conversion-module work). ``metadata.write_every`` and
  ``metadata.write_every_source`` added to capture LagrangeBench's
  conditional dt-stride convention with explicit source-of-truth
  instrumentation.
- **1.3** (2026-05-04): ``metadata.periodic_boundary_conditions`` field
  added (DECISIONS.md D0-17). Threaded through from LagrangeBench
  dataset metadata.json so the conversion's minimum-image-distance
  correction for derived velocities is auditable post-hoc, and so any
  future consumer of the npz that computes distances directly from
  positions can apply the same correction. Additive only; existing
  v1.2 npzs read cleanly via ``load_rollout_npz`` (field is consumed
  through the metadata-dict round-trip, not as a top-level npz field).
  Required for any LagrangeBench dataset with periodic BC (TGV2D,
  RPF2D, ...); pre-1.3 npzs on these datasets had spurious wraparound
  velocities and should be regenerated.
- **1.4** (2026-05-04): ``metadata.periodic_boundary_conditions_upstream``
  and ``metadata.periodic_boundary_conditions_source`` fields added
  (DECISIONS.md D0-17 amendment 1). Surface the upstream-vs-truncated
  PBC asymmetry for audit trail; same shape as ``write_every_source``
  from D0-15 amendment 4. Required because LagrangeBench's stable
  upstream convention is "PBC field is always length 3 regardless of
  ``dim``" — the post-D0-17 regen at 8c3d080397 hit this and rung 3.5
  conversion FAILed under v1.3's strict length check. v1.4 truncates
  to D, sanity-checks trailing entries are all True, and records both
  the post-truncation working vector and the original upstream
  vector. Additive only; v1.3 npzs (none persisted) would lack the
  two new fields.
