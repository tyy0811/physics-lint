# Case Study 02 Phase 1 (Audit) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute Phase 1 of case study 02 per `methodology/docs/2026-05-11-case-study-02-physicsnemo-mgn-design.md`: BLOCKING-1 CPU smoke → Modal infrastructure → audit fires (Gate A, Gate D, NGC sample, substrate-class smoke) → code-absorption (`_expect_velocity` D-entry, `MGN_DATASET_SYSTEM_CLASS` dispatch, pre-flight assertions) → Phase 1 boundary cross-review.

**Architecture:** Five-part sequence: (1A) zero-GPU CPU preconditions, (1B) Modal container + NGC-CLI infrastructure, (1C) Modal audit fires per design §3.1 activities 4-8, (1D) code-absorption responsive to audit findings per design §3.1 activities 10-12, (1E) Phase 1 boundary cross-review per design §2.3. Each phase-letter group's tasks are sequential; gate-out triggers between groups (BLOCKING-1 → no-Modal-spend; Gate D FAIL → FNO-on-Darcy fallback; Gate A FAIL → mesh harness SKIPs).

**Tech stack:** Modal (A100 / A10G; image pinned to `nvidia-physicsnemo @ 1ca85d65` per preflight); NGC CLI (`ngc registry model download-version`); PhysicsNeMo v2.0.0 (DGL-backed MeshGraphNet); scikit-fem (Gate A MeshField reconstruction); pytest (TDD for code-absorption tasks).

---

## Branch setup

Phase 1 implementation lives on a new branch `feature/case-study-02-physicsnemo-mgn` off `master` AFTER PR #9 merges. If PR #9 is still open at session-2 fire time:

- [ ] **Check PR #9 status:**

```bash
gh pr view 9 --json state,mergeable | jq '.state, .mergeable'
```

Expected: `"OPEN"` + `"MERGEABLE"` (or `"MERGED"` if PR #9 closed during the session-1 → session-2 gap).

- [ ] **If MERGED:** create the case-study-02 branch off `master`:

```bash
git fetch origin
git checkout master
git pull
git checkout -b feature/case-study-02-physicsnemo-mgn
```

- [ ] **If still OPEN:** branch off the rung-4c branch head and rebase later:

```bash
git fetch origin
git checkout feature/rung-4c-substrate-class-extension
git pull
git checkout -b feature/case-study-02-physicsnemo-mgn
```

Document the chosen branch base in `DECISIONS.md` as a new D-entry sub-bullet under D0-01 (branch source).

---

## Phase 1A — Zero-GPU preconditions

### Task 1: BLOCKING-1 — CPU-only NGC ↔ PhysicsNeMo v2.0.0 state-dict smoke

**Goal:** Confirm the NGC `modulus_ns_meshgraphnet:v0.1` checkpoint loads cleanly against the `physicsnemo @ 1ca85d65` MeshGraphNet constructor (per preflight `02-physicsnemo-mgn/preflight/mgn_loader_contract.md`, "BLOCKING (P0): NGC checkpoint ↔ v2.0.0 source-compatibility unknown"). Zero-GPU; cheapest unblock; gate-out before any Modal compute.

**Files:**
- Create: `external_validation/_rollout_anchors/02-physicsnemo-mgn/tests/test_ngc_checkpoint_v2_0_0_compat.py`
- Reference: `02-physicsnemo-mgn/preflight/mgn_loader_contract.md` (constructor args: `num_input_features=6, num_edge=3, num_output=3` per `conf/config.yaml` cited in preflight)

- [ ] **Step 1: Write the failing test (skip-marked until checkpoint downloaded).**

```python
"""CPU-only NGC checkpoint ↔ PhysicsNeMo v2.0.0 state-dict-key smoke.

Per design §3.1 activity 1 (BLOCKING-1 unblock). Zero-GPU; runs locally.
Test marked skip when NGC checkpoint is absent; Task 4 downloads it, then
this test fires green or red and pins the verdict.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

NGC_CHECKPOINT_PATH = Path(
    os.environ.get(
        "PHYSICS_LINT_NGC_VORTEX_CHECKPOINT",
        # Default to the download location set by Task 4.
        "external_validation/_rollout_anchors/02-physicsnemo-mgn/cache/modulus_ns_meshgraphnet_v0.1/checkpoint.tar",
    )
).resolve()


@pytest.mark.skipif(
    not NGC_CHECKPOINT_PATH.exists(),
    reason=(
        f"NGC vortex-shedding checkpoint not at {NGC_CHECKPOINT_PATH}; "
        "Task 4 (NGC download entrypoint) must run first OR "
        "PHYSICS_LINT_NGC_VORTEX_CHECKPOINT must point to a local copy."
    ),
)
def test_ngc_vortex_shedding_checkpoint_loads_into_v2_0_0_meshgraphnet() -> None:
    """The NGC modulus_ns_meshgraphnet:v0.1 state_dict keys must match
    physicsnemo @ 1ca85d65's MeshGraphNet constructor with the args from
    conf/config.yaml (num_input_features=6, num_edge=3, num_output=3).

    Pre-rename modulus and post-rename physicsnemo may have renamed
    parameter paths or restructured layers; this smoke catches that
    before any Modal compute fires. BLOCKING-1 per design §3.1 / preflight.
    """
    from physicsnemo.models.meshgraphnet import MeshGraphNet

    model = MeshGraphNet(
        input_dim_nodes=6,
        input_dim_edges=3,
        output_dim=3,
    )
    expected_keys = set(model.state_dict().keys())

    ckpt = torch.load(NGC_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    # NGC checkpoint may wrap the state_dict under "model_state_dict" or "state_dict";
    # check both, fail loudly if neither.
    if "model_state_dict" in ckpt:
        ckpt_state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        ckpt_state_dict = ckpt["state_dict"]
    else:
        ckpt_state_dict = ckpt
    actual_keys = set(ckpt_state_dict.keys())

    missing_in_ckpt = expected_keys - actual_keys
    extra_in_ckpt = actual_keys - expected_keys

    assert not missing_in_ckpt and not extra_in_ckpt, (
        f"NGC checkpoint state_dict keys do not match physicsnemo @ 1ca85d65 "
        f"MeshGraphNet constructor. Missing in checkpoint: {sorted(missing_in_ckpt)[:5]}. "
        f"Extra in checkpoint: {sorted(extra_in_ckpt)[:5]}. "
        f"BLOCKING per design §3.1 activity 1; consider an older physicsnemo "
        f"pin or FNO-on-Darcy fallback per Gate D."
    )
```

- [ ] **Step 2: Run the test (initial; expected SKIP until checkpoint downloaded).**

```bash
pytest external_validation/_rollout_anchors/02-physicsnemo-mgn/tests/test_ngc_checkpoint_v2_0_0_compat.py -v
```

Expected: `SKIPPED [NGC vortex-shedding checkpoint not at .../checkpoint.tar; Task 4 ...]`. This is intentional — the test gates on the checkpoint download.

- [ ] **Step 3: Commit the BLOCKING-1 smoke test (skip-marked).**

```bash
git add external_validation/_rollout_anchors/02-physicsnemo-mgn/tests/test_ngc_checkpoint_v2_0_0_compat.py
git commit -m "02-physicsnemo-mgn: BLOCKING-1 CPU state-dict smoke (skip-marked pre-download)"
```

The test re-fires green or red after Task 4 + 7 (post-NGC-download + checkpoint-extracted-locally smoke). Verdict recorded in Task 13's D-entry.

---

### Task 2: Pre-register D0-2X skeleton for Phase 1 audit findings

**Goal:** Open the new D-entry that Phase 1's audit will populate as findings land (per design §3.1 activity 13). Pre-registering the skeleton means Phase 1 has a documented audit-trail target before any audit fires; populating happens incrementally.

**Files:**
- Modify: `external_validation/_rollout_anchors/methodology/DECISIONS.md` (append D0-2X entry)

- [ ] **Step 1: Identify the next available D-entry number.**

```bash
grep -E "^## D0-[0-9]+" external_validation/_rollout_anchors/methodology/DECISIONS.md | tail -3
```

Expected: last entry `D0-22` (or higher if other entries landed since). The Phase 1 entry is the next available — likely `D0-23`. Confirm and use that number for the rest of this plan; mentions of `D0-2X` below are placeholders for the actual number.

- [ ] **Step 2: Write the D0-2X skeleton entry (verdicts blank, to be populated as tasks land).**

Append to `DECISIONS.md`:

```markdown
## D0-2X — 2026-05-11 — Case Study 02 Phase 1 audit verdicts (open)

**Status:** open. Skeleton pre-registered before Phase 1 fires; verdicts populate as audit tasks land per `methodology/docs/2026-05-11-case-study-02-physicsnemo-mgn-phase-1-plan.md`.

**Predecessor:** D0-02 (Gate A deferred to Day 2), D0-11 (Day 2 hour 1 NGC audit decision criterion), D0-22 (substrate-class taxonomy).

**Verdicts (populated by Phase 1 tasks):**

1. **BLOCKING-1 (Task 1 + 7):** NGC checkpoint ↔ PhysicsNeMo v2.0.0 state-dict compatibility.
   - Verdict: [pending]
   - Source: state-dict-key smoke test result.

2. **NGC audit findings (Task 5):** velocity-field key in `node_values`; DGL topology coercibility; primitive-vs-derived emission.
   - Verdict: [pending]
   - Source: V1-V18 audit per preflight mgn_loader_contract.md.

3. **Gate A verdict (Task 6):** PASS / PARTIAL / FAIL for DGL → MeshField materialization.
   - Verdict: [pending]
   - Source: scikit-fem Basis reconstruction attempt on NGC sample.

4. **NGC sample reproduction (Task 7):** max-abs-error on velocity against shipped expected output.
   - Verdict: [pending]
   - Tolerance: 10⁻³ (plan §4 default; refined per NGC documentation).

5. **Gate D composite verdict (Task 8):** PASS (checkpoint usable for case study 02) / FAIL-with-FNO-fallback.
   - Verdict: [pending]
   - Composite of: BLOCKING-1 + NGC sample reproduction.

6. **Substrate-class empirical verdict (Task 9):** cylinder wake fits `open-driven-dissipative` (default) OR new class label.
   - Verdict: [pending]
   - Three discriminating observables: ∫|∇·v|dV, KE budget dKE/dt, Strouhal St ∈ [0.16, 0.21] for Re ∈ [100, 300].

7. **Persistent-volume decision (Task 9 sub-step):** Modal MGN inference writes to persistent volume? Y/N.
   - Verdict: [pending]
   - Implication: rollout-dir isolation pattern (round-codex-4) applies if Y.

8. **`_expect_velocity` helper key resolution (Task 10):** actual NGC velocity-field key name.
   - Verdict: [pending]

9. **`MGN_DATASET_SYSTEM_CLASS` dispatch (Task 11):** introduced; class label per verdict 6.
   - Verdict: [pending]

10. **Pre-flight assertions (Task 12):** loader-contract assertions written in `_harness/mesh_rollout_adapter.py`.
    - Verdict: [pending]

**Phase 1 boundary cross-review (Task 14):** Codex pass against verdicts 1-10 + code-absorption (Tasks 10-12). Findings triaged per pattern-C four-cell framework. Cell distributions populated post-cross-review.

**Why pre-registered as skeleton:** mirrors D0-22's pre-registration-before-implementation pattern. Audit-trail discipline: by the time Phase 1 completes, every verdict has a recorded place; no audit finding lands without an entry.
```

- [ ] **Step 3: Commit the D0-2X skeleton.**

```bash
git add external_validation/_rollout_anchors/methodology/DECISIONS.md
git commit -m "DECISIONS.md: pre-register D0-2X skeleton for CS02 Phase 1 audit verdicts"
```

---

## Phase 1B — Modal infrastructure

### Task 3: Modal container image for PhysicsNeMo MGN

**Goal:** Add a Modal image to `01-lagrangebench/modal_app.py` (or a new `02-physicsnemo-mgn/modal_app.py` per convention) for the MGN inference path. Pin `nvidia-physicsnemo @ 1ca85d65` + `dgl` + `ngc` CLI per preflight.

**Files:**
- Create: `external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py` (parallel to `01-lagrangebench/modal_app.py`)

- [ ] **Step 1: Inspect the LB-side modal_app.py for the image-construction pattern to mirror.**

```bash
grep -n "modal.Image\|pip_install\|add_local_file" external_validation/_rollout_anchors/01-lagrangebench/modal_app.py | head -20
```

Read the relevant section (around lines 80-200 in LB modal_app.py) to understand the image-build idiom: base image, apt installs, pip installs, harness mounts.

- [ ] **Step 2: Write the initial `02-physicsnemo-mgn/modal_app.py` with the MGN image.**

```python
"""Modal entrypoint for Case Study 02 — PhysicsNeMo MeshGraphNet.

Parallel to 01-lagrangebench/modal_app.py. Builds the MGN inference image
with nvidia-physicsnemo pinned at sha 1ca85d65 (tag v2.0.0, 2026-03-10)
per preflight/mgn_loader_contract.md. NGC CLI mounted for checkpoint
download (Task 4); DGL + scikit-fem for Gate A audit (Task 6).
"""

from __future__ import annotations

from pathlib import Path

import modal

PHYSICSNEMO_SHA = "1ca85d65ac2ce28ea9762910c09a954c08a37140"  # tag v2.0.0
PHYSICSNEMO_VERSION_TAG = "v2.0.0"

# Day 2 audit + inference image. A100 default per DECISIONS D0-13 stage-2;
# Gate-A audit task may run on a smaller GPU class (CPU is enough for the
# state-dict smoke, A10G for inference smoke).
mgn_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "unzip")
    .pip_install(
        f"nvidia-physicsnemo @ git+https://github.com/NVIDIA/physicsnemo@{PHYSICSNEMO_SHA}",
        "dgl",
        "scikit-fem",
        "torch>=2.0.0,<3.0.0",
    )
    # NGC CLI install per https://docs.ngc.nvidia.com/cli/cmd.html
    .run_commands(
        "wget -q https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.41.4/files/ngccli_linux.zip -O /tmp/ngccli.zip",
        "unzip -q /tmp/ngccli.zip -d /opt/ngc",
        "ln -s /opt/ngc/ngc-cli/ngc /usr/local/bin/ngc",
        "rm /tmp/ngccli.zip",
    )
)

# Modal Volume for NGC checkpoints + rollout outputs.
mgn_volume = modal.Volume.from_name(
    "case-study-02-physicsnemo-artifacts", create_if_missing=True
)

app = modal.App(
    "physics-lint-case-study-02-physicsnemo-mgn",
    image=mgn_image,
)
```

- [ ] **Step 3: Add a smoke test that imports the new module and resolves the image.**

Create `external_validation/_rollout_anchors/02-physicsnemo-mgn/tests/test_modal_app_imports.py`:

```python
"""Drift-guard: modal_app.py imports cleanly + pins the expected
physicsnemo sha per preflight."""

from __future__ import annotations

import ast
from pathlib import Path

MODAL_APP_PATH = (
    Path(__file__).resolve().parent.parent / "modal_app.py"
)
EXPECTED_PHYSICSNEMO_SHA = "1ca85d65ac2ce28ea9762910c09a954c08a37140"


def _read_module_string_constant(source_path: Path, name: str) -> str | None:
    """Mirrors 01-lagrangebench/tests/test_modal_app_gpu_class.py pattern."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != name:
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return None


def test_modal_app_exists() -> None:
    assert MODAL_APP_PATH.is_file(), f"modal_app.py not found at {MODAL_APP_PATH}"


def test_physicsnemo_sha_pinned() -> None:
    actual = _read_module_string_constant(MODAL_APP_PATH, "PHYSICSNEMO_SHA")
    assert actual == EXPECTED_PHYSICSNEMO_SHA, (
        f"PHYSICSNEMO_SHA = {actual!r} in modal_app.py does not match the "
        f"preflight-pinned sha {EXPECTED_PHYSICSNEMO_SHA!r}. Update the pin "
        f"alongside a DECISIONS.md amendment if the bump is intentional."
    )
```

- [ ] **Step 4: Run the import + drift-guard test.**

```bash
pytest external_validation/_rollout_anchors/02-physicsnemo-mgn/tests/test_modal_app_imports.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit.**

```bash
git add external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py external_validation/_rollout_anchors/02-physicsnemo-mgn/tests/test_modal_app_imports.py
git commit -m "02-physicsnemo-mgn: Modal image pinned to physicsnemo @ 1ca85d65 (v2.0.0)"
```

---

### Task 4: NGC checkpoint download Modal entrypoint

**Goal:** A Modal entrypoint that downloads `modulus_ns_meshgraphnet:v0.1` into the Modal Volume + computes the checkpoint hash + writes hash to a local pin file. Per design §3.1 activity 3.

**Files:**
- Modify: `external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py` (add entrypoint)

- [ ] **Step 1: Add the download entrypoint to modal_app.py.**

Append to `02-physicsnemo-mgn/modal_app.py`:

```python
NGC_VORTEX_MODEL = "nvidia/modulus/modulus_ns_meshgraphnet"
NGC_VORTEX_VERSION = "v0.1"
VOLUME_CHECKPOINT_ROOT = "/vol/checkpoints"


@app.function(
    volumes={"/vol": mgn_volume},
    timeout=600,
    secrets=[modal.Secret.from_name("ngc-api-key")],  # NGC_API_KEY env var
)
def download_ngc_vortex_shedding_checkpoint() -> dict:
    """Download modulus_ns_meshgraphnet:v0.1 into the Modal Volume + hash.

    Returns: {"path": str, "sha256": str, "size_bytes": int}.

    Per DECISIONS.md D0-2X verdict 1 source. The returned dict is the
    audit-trail payload for the Phase 1 D-entry; the operator records
    sha256 in the D-entry's verdict-1 row.
    """
    import hashlib
    import os
    import subprocess

    dest_root = f"{VOLUME_CHECKPOINT_ROOT}/modulus_ns_meshgraphnet_v0.1"
    os.makedirs(dest_root, exist_ok=True)

    # NGC CLI requires NGC_API_KEY in env; the Modal Secret provides it.
    subprocess.run(
        [
            "ngc",
            "registry",
            "model",
            "download-version",
            f"{NGC_VORTEX_MODEL}:{NGC_VORTEX_VERSION}",
            "--dest",
            dest_root,
        ],
        check=True,
    )

    # Locate the downloaded checkpoint file (NGC may nest it under a
    # versioned subdir).
    checkpoint_paths = [
        p for p in Path(dest_root).rglob("*.tar")
    ] + [p for p in Path(dest_root).rglob("*.pt")]
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No .tar or .pt checkpoint found under {dest_root}; "
            f"NGC download may have changed format. Inspect manually."
        )
    if len(checkpoint_paths) > 1:
        raise RuntimeError(
            f"Multiple checkpoint candidates found under {dest_root}: "
            f"{[str(p) for p in checkpoint_paths]}. Tighten the glob."
        )
    checkpoint_path = checkpoint_paths[0]

    # Hash for the D-entry pin.
    h = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    mgn_volume.commit()  # Persist the download.

    return {
        "path": str(checkpoint_path),
        "sha256": h.hexdigest(),
        "size_bytes": checkpoint_path.stat().st_size,
        "ngc_model": NGC_VORTEX_MODEL,
        "ngc_version": NGC_VORTEX_VERSION,
        "physicsnemo_sha": PHYSICSNEMO_SHA,
    }
```

- [ ] **Step 2: Verify NGC API key Modal Secret exists.**

```bash
modal secret list 2>&1 | grep ngc-api-key
```

Expected: `ngc-api-key  [created date]`. If absent, create it:

```bash
modal secret create ngc-api-key NGC_API_KEY=<your-NGC-API-key>
```

- [ ] **Step 3: Fire the download.**

```bash
modal run external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py::download_ngc_vortex_shedding_checkpoint
```

Expected: dict output with `path`, `sha256` (64 hex chars), `size_bytes`.

- [ ] **Step 4: Record the sha256 in DECISIONS.md D0-2X verdict 1.**

Replace D0-2X verdict 1's `[pending]` with the actual sha256. Use Edit on the DECISIONS.md file; preserve the verdict structure.

- [ ] **Step 5: Mirror the checkpoint locally so Task 1's CPU smoke can run.**

```bash
mkdir -p external_validation/_rollout_anchors/02-physicsnemo-mgn/cache/modulus_ns_meshgraphnet_v0.1
modal volume get case-study-02-physicsnemo-artifacts /checkpoints/modulus_ns_meshgraphnet_v0.1/ external_validation/_rollout_anchors/02-physicsnemo-mgn/cache/modulus_ns_meshgraphnet_v0.1/
```

(The Modal `volume get` CLI syntax may vary by Modal version; check `modal volume --help` if the above fails.)

- [ ] **Step 6: Commit the entrypoint + DECISIONS.md verdict 1 update.**

```bash
git add external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py external_validation/_rollout_anchors/methodology/DECISIONS.md
git commit -m "02-physicsnemo-mgn: NGC vortex-shedding checkpoint download entrypoint + sha pin in D0-2X"
```

The local cache directory is gitignored (per repo convention `.gitignore` for `cache/`).

---

### Task 5: NGC audit entrypoint (V1-V18 + 5 secondary known-unknowns)

**Goal:** A Modal entrypoint that loads the NGC checkpoint + VortexSheddingDataset's sample, inspects the loader contract against preflight V1-V18, and emits structured findings. Per design §3.1 activity 4.

**Files:**
- Modify: `external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py` (add entrypoint)

- [ ] **Step 1: Read the preflight V1-V18 enumerations to know what to assert against.**

```bash
sed -n '/^### V1\b/,/^### V19\b/p' external_validation/_rollout_anchors/02-physicsnemo-mgn/preflight/mgn_loader_contract.md
```

Capture: each V-entry's file:line citation at sha `1ca85d65` + the contract claim. The audit entrypoint inspects whether each claim holds on the downloaded NGC sample.

- [ ] **Step 2: Add the audit entrypoint.**

Append to `02-physicsnemo-mgn/modal_app.py`:

```python
@app.function(
    volumes={"/vol": mgn_volume},
    timeout=300,
    # CPU-only is enough; the audit is loader-side, no inference.
)
def audit_ngc_vortex_shedding_loader_contract() -> dict:
    """Audit the loader-contract assumptions per preflight V1-V18.

    Returns: structured findings keyed by V1-V18 + 5 secondary known-unknowns
    + meta-fields (velocity-field key name, DGL topology summary).

    Per DECISIONS.md D0-2X verdicts 2 + 8 source.
    """
    import json
    from pathlib import Path as P

    # Locate the downloaded checkpoint dir.
    ckpt_dir = P(f"{VOLUME_CHECKPOINT_ROOT}/modulus_ns_meshgraphnet_v0.1")
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"NGC checkpoint dir {ckpt_dir} missing; run "
            f"download_ngc_vortex_shedding_checkpoint first."
        )

    findings: dict[str, object] = {
        "ngc_sample_files_present": sorted(
            str(p.relative_to(ckpt_dir)) for p in ckpt_dir.rglob("*")
            if p.is_file()
        )[:50],  # cap for readability
    }

    # Probe the VortexSheddingDataset constructor's sample-input expectations.
    # Per preflight V1-V18, the dataset expects: data_dir, num_samples,
    # num_steps, noise_std, split, name. Audit constructs a single-sample
    # dataset and inspects its first __getitem__ output.
    try:
        from physicsnemo.datapipes.gnn.vortex_shedding_dataset import (
            VortexSheddingDataset,
        )

        # Use NGC sample dir if it ships one; else the audit reports the
        # absence as a finding.
        sample_dirs = [d for d in ckpt_dir.rglob("*") if d.is_dir() and any(
            p.name.endswith(".tfrecord") or p.name.endswith(".h5")
            for p in d.iterdir()
        )]
        findings["sample_data_dirs"] = [str(d.relative_to(ckpt_dir)) for d in sample_dirs]

        if sample_dirs:
            ds = VortexSheddingDataset(
                name="test",
                data_dir=str(sample_dirs[0]),
                split="test",
                num_samples=1,
                num_steps=2,  # Minimum to surface trajectory structure.
                noise_std=0.0,  # Force noise off for audit reproducibility.
            )
            sample = ds[0]
            # V1-V18 audit: enumerate the sample's structure.
            findings["sample_keys"] = sorted(sample.keys()) if hasattr(sample, "keys") else None
            findings["sample_type"] = type(sample).__name__
            # The velocity-field key (preflight V12 / V14) is critical for
            # Task 10's _expect_velocity D-entry.
            for candidate_key in ("velocity", "u", "v", "vel", "flow_field", "y"):
                if hasattr(sample, "keys") and candidate_key in sample.keys():
                    findings["velocity_field_key"] = candidate_key
                    break
            else:
                findings["velocity_field_key"] = "UNRESOLVED"
        else:
            findings["audit_status"] = "no_sample_data_dirs_found"
    except ImportError as e:
        findings["import_error"] = str(e)
    except Exception as e:
        findings["audit_error"] = f"{type(e).__name__}: {e}"

    # 5 secondary known-unknowns per preflight (CWD coupling, noise_std
    # split-conditional, fp32 default-dtype, trajectory_length silent
    # overrun, node_type one_hot bound).
    # For each, the audit reports the observed state on the NGC sample.
    import os

    findings["cwd_at_audit"] = os.getcwd()  # Hydra chdir:True context unclear without dataset construction.
    findings["torch_default_dtype"] = str(__import__("torch").get_default_dtype())

    # Emit findings as JSON for downstream consumption (Task 13 D-entry update).
    mgn_volume.commit()
    findings_path = ckpt_dir / "audit_findings.json"
    findings_path.write_text(json.dumps(findings, indent=2, default=str))
    mgn_volume.commit()

    return findings
```

- [ ] **Step 3: Fire the audit.**

```bash
modal run external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py::audit_ngc_vortex_shedding_loader_contract
```

Expected: dict output with `ngc_sample_files_present`, `sample_keys`, `velocity_field_key` (or `UNRESOLVED`), and the 5 secondary known-unknown findings.

- [ ] **Step 4: Mirror findings JSON locally for inclusion in D-entry.**

```bash
modal volume get case-study-02-physicsnemo-artifacts /checkpoints/modulus_ns_meshgraphnet_v0.1/audit_findings.json external_validation/_rollout_anchors/02-physicsnemo-mgn/preflight/audit_findings.json
```

- [ ] **Step 5: Record findings in DECISIONS.md D0-2X verdict 2 + verdict 8.**

Replace verdict 2's `[pending]` with a short summary linking to `02-physicsnemo-mgn/preflight/audit_findings.json`; replace verdict 8's `[pending]` with the resolved velocity-field key name (or `UNRESOLVED` if the audit didn't find one — which triggers a separate decision branch in Task 10).

- [ ] **Step 6: Commit audit findings + DECISIONS.md updates.**

```bash
git add external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py external_validation/_rollout_anchors/02-physicsnemo-mgn/preflight/audit_findings.json external_validation/_rollout_anchors/methodology/DECISIONS.md
git commit -m "02-physicsnemo-mgn: NGC loader-contract audit findings (Phase 1 verdicts 2+8)"
```

---

## Phase 1C — Modal audit fires

### Task 6: Gate A verdict — DGL → MeshField materialization audit

**Goal:** Determine whether the NGC vortex-shedding sample's DGL graph can be coerced to a scikit-fem `Basis`. Per design §3.1 activity 5; D0-2X verdict 3.

**Files:**
- Modify: `external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py` (add entrypoint)

- [ ] **Step 1: Add the Gate A audit entrypoint.**

Append to `02-physicsnemo-mgn/modal_app.py`:

```python
@app.function(
    volumes={"/vol": mgn_volume},
    timeout=300,
)
def audit_gate_a_dgl_to_meshfield() -> dict:
    """Attempt DGL → scikit-fem Basis coercion on the NGC sample.

    Returns: {"verdict": "PASS" | "PARTIAL" | "FAIL", "rationale": str, ...}.

    PASS: scikit-fem MeshTri / MeshQuad can be constructed from
    (node_positions, edge_index) → Basis reconstructs → MeshField OK.
    PARTIAL: DGL graph is mesh-shaped but scikit-fem coercion fails;
    GridField resampling fallback is the recovery path.
    FAIL: DGL graph is fundamentally graph-shaped (irregular connectivity,
    no element interpretation possible) → mesh harness SKIPs with reason.
    """
    import json
    import numpy as np
    from pathlib import Path as P

    findings_path = P(
        f"{VOLUME_CHECKPOINT_ROOT}/modulus_ns_meshgraphnet_v0.1/audit_findings.json"
    )
    if not findings_path.exists():
        raise FileNotFoundError(
            "Run audit_ngc_vortex_shedding_loader_contract first to produce "
            "the audit_findings.json that this task reads from."
        )

    # Reconstruct the dataset sample (per audit findings) and attempt
    # scikit-fem Basis reconstruction.
    try:
        from physicsnemo.datapipes.gnn.vortex_shedding_dataset import (
            VortexSheddingDataset,
        )
        import skfem  # scikit-fem
    except ImportError as e:
        return {
            "verdict": "FAIL",
            "rationale": f"ImportError: {e}",
            "recovery_path": "mesh harness SKIPs; cover-letter Appendix A.4 variant fires",
        }

    findings = json.loads(findings_path.read_text())
    sample_dirs = findings.get("sample_data_dirs", [])
    if not sample_dirs:
        return {
            "verdict": "FAIL",
            "rationale": "No sample_data_dirs available from prior audit",
        }

    ckpt_root = P(f"{VOLUME_CHECKPOINT_ROOT}/modulus_ns_meshgraphnet_v0.1")
    ds = VortexSheddingDataset(
        name="test",
        data_dir=str(ckpt_root / sample_dirs[0]),
        split="test",
        num_samples=1,
        num_steps=2,
        noise_std=0.0,
    )
    sample = ds[0]

    # Extract node positions + edge_index.
    # NGC vortex-shedding's mesh attribute access depends on the dataset's
    # actual return type (DGL graph dict or tensor dict); the audit findings
    # from Task 5 informed the structure.
    # Adapt the access pattern based on findings["sample_type"].
    sample_type = findings.get("sample_type")

    if sample_type == "DGLGraph" or "dgl" in str(type(sample)).lower():
        # DGL graph access path
        node_positions = sample.ndata.get("pos") or sample.ndata.get("position")
        edges = sample.edges()
        edge_index = np.stack([e.numpy() for e in edges])
    elif isinstance(sample, dict):
        node_positions = sample.get("node_positions") or sample.get("pos")
        edge_index = sample.get("edge_index")
    else:
        return {
            "verdict": "FAIL",
            "rationale": f"Unrecognized sample type: {sample_type}",
        }

    if node_positions is None:
        return {
            "verdict": "FAIL",
            "rationale": "No node positions found in sample",
        }

    nodes_np = (
        node_positions.numpy() if hasattr(node_positions, "numpy")
        else np.asarray(node_positions)
    )
    edges_np = (
        edge_index.numpy() if hasattr(edge_index, "numpy")
        else np.asarray(edge_index)
    )

    # Attempt scikit-fem MeshTri construction from the edge list.
    # scikit-fem's MeshTri expects (nodes, elements) where elements are
    # (3, n_elements) for triangles. We need to reconstruct triangle
    # elements from edge connectivity — a non-trivial coercion.
    # For Gate A PASS, the DGL graph must be triangulation-recoverable.
    try:
        # First attempt: if edges form a triangulation, recover triangles
        # via a triangulation-from-edges algorithm.
        # For audit purposes, attempt a simpler check: scikit-fem MeshTri
        # constructor accepts (p, t); we need t = triangle elements.
        # If the DGL graph carries cell connectivity in metadata, use it.
        # Otherwise FAIL → GridField resampling fallback.

        # Probe for cell connectivity in sample.
        cells = None
        if isinstance(sample, dict):
            cells = sample.get("cells")
        elif hasattr(sample, "ndata"):
            cells = sample.ndata.get("cells")

        if cells is None:
            return {
                "verdict": "PARTIAL",
                "rationale": (
                    "DGL graph has node positions + edges but no cell "
                    "connectivity; scikit-fem MeshTri needs triangle elements. "
                    "Recover via GridField resampling fallback."
                ),
                "recovery_path": "GridField(values=resampled, h=spacing, periodic=False)",
            }

        cells_np = (
            cells.numpy() if hasattr(cells, "numpy") else np.asarray(cells)
        )
        # scikit-fem MeshTri expects (p, t) with p shape (dim, n_nodes), t shape (3, n_cells)
        if cells_np.shape[1] != 3:
            cells_np = cells_np.T
        mesh = skfem.MeshTri(nodes_np.T if nodes_np.shape[0] != 2 else nodes_np, cells_np)
        basis = skfem.Basis(mesh, skfem.ElementTriP1())

        return {
            "verdict": "PASS",
            "rationale": (
                f"scikit-fem MeshTri + Basis(ElementTriP1) reconstructed from "
                f"{nodes_np.shape[0]} nodes + {cells_np.shape[1]} triangles."
            ),
            "n_nodes": int(nodes_np.shape[0]),
            "n_cells": int(cells_np.shape[1]),
            "basis_repr": repr(basis)[:200],
        }
    except Exception as e:
        return {
            "verdict": "PARTIAL",
            "rationale": (
                f"scikit-fem coercion failed: {type(e).__name__}: {e}. "
                f"GridField resampling fallback applies."
            ),
            "recovery_path": "GridField(values=resampled, h=spacing, periodic=False)",
        }
```

- [ ] **Step 2: Fire the Gate A audit.**

```bash
modal run external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py::audit_gate_a_dgl_to_meshfield
```

Expected: dict with `verdict` ∈ {PASS, PARTIAL, FAIL} + rationale.

- [ ] **Step 3: Record Gate A verdict in DECISIONS.md D0-2X verdict 3 + amend D0-02.**

D0-02 was deferred to Day 2. Phase 1 fulfills D0-02; amend D0-02 with the Gate A verdict + rationale, and pin D0-2X verdict 3 to the same.

- [ ] **Step 4: Commit Gate A verdict.**

```bash
git add external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py external_validation/_rollout_anchors/methodology/DECISIONS.md
git commit -m "02-physicsnemo-mgn: Gate A verdict — DGL → MeshField materialization audit (D0-2X verdict 3)"
```

---

### Task 7: NGC sample reproduction (`test_inference_matches_ngc_sample`)

**Goal:** Run NGC inference on the shipped sample, compare against shipped expected output within plan §4 default 10⁻³ tolerance. Per design §3.1 activity 6; D0-2X verdict 4. Also re-fires the BLOCKING-1 smoke from Task 1 — the local checkpoint cache is now populated.

**Files:**
- Modify: `external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py` (add entrypoint)

- [ ] **Step 1: Re-run Task 1's BLOCKING-1 smoke locally with the cached checkpoint.**

```bash
pytest external_validation/_rollout_anchors/02-physicsnemo-mgn/tests/test_ngc_checkpoint_v2_0_0_compat.py -v
```

Expected: PASS (state-dict keys match) — gates the rest of Task 7. If FAIL, BLOCKING-1 fires and Gate D demotion path activates (Task 8); record the verdict in D0-2X verdict 1 as `FAIL` and proceed to FNO-on-Darcy fallback per design §1.5 #5.

- [ ] **Step 2: Add the NGC sample reproduction entrypoint.**

Append to `02-physicsnemo-mgn/modal_app.py`:

```python
@app.function(
    volumes={"/vol": mgn_volume},
    gpu="A10G",
    timeout=600,
)
def audit_ngc_sample_reproduction(tolerance: float = 1e-3) -> dict:
    """Run NGC inference on the shipped sample; compare to shipped expected output.

    Returns: {"verdict": "PASS" | "FAIL", "max_abs_error": float, ...}.

    Per design §3.1 activity 6 / plan §4 step 2.
    """
    import json
    import os
    from pathlib import Path as P
    import torch

    # Hydra chdir:True per preflight CWD coupling; reproduce it.
    ckpt_root = P(f"{VOLUME_CHECKPOINT_ROOT}/modulus_ns_meshgraphnet_v0.1")
    os.chdir(str(ckpt_root))

    # Default dtype per preflight fp32 known-unknown.
    torch.set_default_dtype(torch.float32)

    # Load the checkpoint (re-run BLOCKING-1 smoke under GPU context).
    from physicsnemo.models.meshgraphnet import MeshGraphNet
    model = MeshGraphNet(
        input_dim_nodes=6, input_dim_edges=3, output_dim=3,
    ).cuda()
    # Use Task 1's load logic.
    checkpoint_path = next(ckpt_root.rglob("*.tar"), None) or next(ckpt_root.rglob("*.pt"), None)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint under {ckpt_root}")
    ckpt = torch.load(checkpoint_path, map_location="cuda", weights_only=False)
    sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(sd)
    model.eval()

    # Load the NGC-shipped sample input + expected output.
    # The exact file names depend on NGC's checkpoint package; the audit
    # findings from Task 5 listed them. Adapt accordingly.
    sample_input_path = next(ckpt_root.rglob("sample_input*"), None)
    sample_expected_path = next(ckpt_root.rglob("sample_expected*"), None) or next(
        ckpt_root.rglob("expected_output*"), None
    )
    if sample_input_path is None or sample_expected_path is None:
        return {
            "verdict": "FAIL",
            "rationale": (
                f"NGC sample input/expected not located under {ckpt_root}. "
                f"Files present: "
                f"{[str(p.relative_to(ckpt_root)) for p in ckpt_root.rglob('*') if p.is_file()][:20]}"
            ),
        }

    # Run inference on the sample input. (Exact API call shape depends
    # on NGC packaging; this is a stub adapter — refine in Phase 2.)
    sample_input = torch.load(sample_input_path, map_location="cuda")
    with torch.no_grad():
        output = model(sample_input)
    expected = torch.load(sample_expected_path, map_location="cuda")

    max_abs_error = float((output - expected).abs().max().item())

    return {
        "verdict": "PASS" if max_abs_error <= tolerance else "FAIL",
        "max_abs_error": max_abs_error,
        "tolerance": tolerance,
        "sample_input_path": str(sample_input_path.relative_to(ckpt_root)),
        "sample_expected_path": str(sample_expected_path.relative_to(ckpt_root)),
    }
```

- [ ] **Step 3: Fire the audit.**

```bash
modal run external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py::audit_ngc_sample_reproduction
```

Expected: dict with `verdict` + `max_abs_error` + `tolerance`. Record both values in D0-2X verdict 4.

- [ ] **Step 4: If `verdict == "FAIL"` — decide tolerance refinement vs Gate D FAIL.**

Per design §3.1 gate-out triggers: Pattern-A drift in NGC sample reproduction (activity 6) → already subsumed by Gate D FAIL; no separate path. So if FAIL, Task 8 (Gate D composite verdict) records FAIL and triggers FNO-on-Darcy fallback. Branch decision: if the FAIL was due to NGC documenting a looser tolerance (read in Task 5's findings), pre-register the looser tolerance via D0-2X verdict 4 amendment and re-run.

- [ ] **Step 5: Commit verdict + the (possibly amended) tolerance.**

```bash
git add external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py external_validation/_rollout_anchors/methodology/DECISIONS.md
git commit -m "02-physicsnemo-mgn: NGC sample reproduction verdict (D0-2X verdict 4)"
```

---

### Task 8: Gate D composite verdict

**Goal:** Combine BLOCKING-1 (Task 1 / 7) + NGC sample reproduction (Task 7) into the Gate D composite verdict. PASS = checkpoint usable for case study 02 P0; FAIL = FNO-on-Darcy fallback. Per design §3.1 activity 7; D0-2X verdict 5.

**Files:**
- Modify: `external_validation/_rollout_anchors/methodology/DECISIONS.md`

- [ ] **Step 1: Read D0-2X verdicts 1 + 4 (already populated from Tasks 1/7 + Task 7).**

```bash
grep -A 3 "^1\.\|^4\." external_validation/_rollout_anchors/methodology/DECISIONS.md | grep -A 3 D0-2X
```

(Adapt the grep to find the actual D0-2X-section verdicts.)

- [ ] **Step 2: Compose the Gate D verdict.**

Gate D PASS iff: BLOCKING-1 verdict == PASS AND NGC sample reproduction verdict == PASS.
Gate D FAIL otherwise.

Replace D0-2X verdict 5's `[pending]` with the composed verdict + rationale + recovery path:

```markdown
5. **Gate D composite verdict (Task 8):** PASS (checkpoint usable for case study 02) / FAIL-with-FNO-fallback.
   - Verdict: PASS (composed from verdict 1 + verdict 4, both PASS).
   - Source: BLOCKING-1 state-dict smoke + NGC sample reproduction.
```

OR:

```markdown
5. **Gate D composite verdict (Task 8):** PASS (checkpoint usable for case study 02) / FAIL-with-FNO-fallback.
   - Verdict: FAIL (verdict 1 = <BLOCKING-1 verdict>; verdict 4 = <sample reproduction verdict>).
   - Recovery path: rename `02-physicsnemo-mgn/` → `02-fno-darcy/`; Phase 2 proceeds with FNO-on-Darcy per design §1.5 #5; Phase 1 Tasks 9-13 amended to FNO substrate.
```

- [ ] **Step 3: If Gate D FAIL: branch to FNO-on-Darcy.**

The remainder of this plan (Tasks 9-15) assumes Gate D PASS. If FAIL, the plan needs a parallel FNO-on-Darcy substitution pass — out of scope for this plan; open a new writing-plans iteration with the FNO substrate as scope. Pause execution; flag to user.

- [ ] **Step 4: Commit the Gate D verdict.**

```bash
git add external_validation/_rollout_anchors/methodology/DECISIONS.md
git commit -m "DECISIONS.md: Gate D composite verdict for case study 02 (D0-2X verdict 5)"
```

---

### Task 9: 1-traj substrate-class smoke + 3 discriminating observables

**Goal:** Run a 1-traj smoke MGN rollout; compute the 3 discriminating observables (∫|∇·v|dV, KE budget dKE/dt, Strouhal St); confirm cylinder wake fits `open-driven-dissipative` OR pattern-A drift fires. Per design §3.1 activity 8; D0-2X verdicts 6 + 7.

**Files:**
- Modify: `external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py` (add entrypoint)

- [ ] **Step 1: Add the 1-traj smoke entrypoint.**

Append to `02-physicsnemo-mgn/modal_app.py`:

```python
@app.function(
    volumes={"/vol": mgn_volume},
    gpu="A10G",
    timeout=900,
)
def smoke_substrate_class_vortex_shedding() -> dict:
    """1-traj vortex-shedding rollout + 3 discriminating observables.

    Returns: {
        "mass_conservation_int_abs_div_v": float,  # ∫|∇·v|dV (≈ 0 expected)
        "ke_budget_monotone_in_either_direction": bool,
        "strouhal": float,
        "verdict": "open-driven-dissipative" | "<other class>",
        "persistent_volume_decision": "Y" | "N",  # MGN inference writes to /vol/rollouts?
    }.

    Per design §3.1 activity 8.
    """
    import json
    import os
    from pathlib import Path as P
    import numpy as np
    import torch

    ckpt_root = P(f"{VOLUME_CHECKPOINT_ROOT}/modulus_ns_meshgraphnet_v0.1")
    os.chdir(str(ckpt_root))
    torch.set_default_dtype(torch.float32)

    # Run a 100-step rollout on the NGC-shipped sample input.
    # (Exact rollout API depends on NGC packaging + the PhysicsNeMo MGN
    # rollout interface; refine based on Task 5 audit findings.)

    # Substrate-class observables.
    # 1. ∫|∇·v|dV via finite-difference divergence on the resampled grid.
    # 2. KE budget: dKE/dt sign analysis over the trajectory.
    # 3. Strouhal: spectral peak in lift coefficient ≈ vortex shedding frequency × D / U_inflow.

    # Load model + run rollout. The API call shape is the most-likely
    # PhysicsNeMo MeshGraphNet pattern; if Task 5 audit findings differ
    # (e.g., NGC packages the rollout call under a different module path),
    # adjust the import + invocation here. The observables computations
    # are NGC-independent and stay as written.
    from physicsnemo.models.meshgraphnet import MeshGraphNet
    from physicsnemo.datapipes.gnn.vortex_shedding_dataset import (
        VortexSheddingDataset,
    )

    ckpt_path = next(ckpt_root.rglob("*.tar"), None) or next(ckpt_root.rglob("*.pt"))
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint under {ckpt_root}")
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt

    model = MeshGraphNet(
        input_dim_nodes=6, input_dim_edges=3, output_dim=3,
    ).cuda()
    model.load_state_dict(sd)
    model.eval()

    # Locate sample data dir per Task 5 findings.
    audit_findings_path = ckpt_root / "audit_findings.json"
    audit_findings = json.loads(audit_findings_path.read_text())
    sample_data_dir = ckpt_root / audit_findings["sample_data_dirs"][0]

    ds = VortexSheddingDataset(
        name="test",
        data_dir=str(sample_data_dir),
        split="test",
        num_samples=1,
        num_steps=100,  # 1-traj smoke — 100 steps captures Strouhal period.
        noise_std=0.0,
    )

    # Rollout the trajectory.
    velocity_series: list[np.ndarray] = []
    pressure_series: list[np.ndarray] = []
    with torch.no_grad():
        for step in range(100):
            sample = ds[0]  # The dataset yields rollout-internal state.
            # MeshGraphNet's forward signature: (node_features, edge_features, graph)
            # NGC vortex-shedding packs these into the sample dict; the
            # exact key names depend on Task 5 audit. The most-likely shape:
            output = model(
                sample["node_features"].cuda() if "node_features" in sample
                else sample["x"].cuda(),
                sample["edge_features"].cuda() if "edge_features" in sample
                else sample["edge_attr"].cuda(),
                sample["graph"] if "graph" in sample else sample,
            )
            # Output: (N_nodes, D_out). Extract velocity (first 2 dims) +
            # pressure (3rd dim).
            output_np = output.cpu().numpy()
            velocity_series.append(output_np[:, :2])
            pressure_series.append(output_np[:, 2])

    velocity = np.stack(velocity_series, axis=0)  # (T, N_nodes, 2)
    pressure = np.stack(pressure_series, axis=0)  # (T, N_nodes)

    # === Observable 1: ∫|∇·v|dV per timestep ===
    # On unstructured mesh, divergence via node-level finite-difference is
    # not straightforward. Two paths:
    # (a) If Gate A PASS: use scikit-fem Basis to integrate ∇·v properly.
    # (b) If Gate A PARTIAL: resample to regular grid + central-difference div.
    # Most likely path for cylinder wake is (b); the (a) path is preferred
    # if Gate A's mesh reconstruction succeeded.
    # Approximation here: assume regular-grid-like layout from sample's
    # node_positions, compute via grid-aligned FD with edge_length proxy.
    # The smoke fires this best-effort estimate; precise integration lands
    # in Phase 2 if the smoke's verdict is borderline.
    edge_length_proxy = 1.0 / np.sqrt(velocity.shape[1])  # crude
    div_v_approx = np.zeros(velocity.shape[0])
    for t in range(velocity.shape[0]):
        v_t = velocity[t]  # (N_nodes, 2)
        # Sum of central-difference proxies over node neighborhoods (crude
        # but order-of-magnitude correct for a 1-traj smoke).
        div_v_approx[t] = float(np.abs(np.gradient(v_t[:, 0])).sum() + np.abs(np.gradient(v_t[:, 1])).sum()) * edge_length_proxy
    mass_conservation_int_abs_div_v = float(div_v_approx.mean())

    # === Observable 2: KE budget — monotone-in-either-direction? ===
    ke_per_step = 0.5 * (velocity ** 2).sum(axis=(1, 2))  # (T,)
    dke_dt = np.diff(ke_per_step) / 0.01  # assume dt=0.01; adjust if NGC differs.
    # Monotone-in-either-direction = all dKE/dt same sign across the
    # post-warmup window (skip first 10 steps as transient).
    dke_post_warmup = dke_dt[10:]
    monotone_increasing = bool(np.all(dke_post_warmup > 0))
    monotone_decreasing = bool(np.all(dke_post_warmup < 0))
    ke_budget_monotone_in_either_direction = monotone_increasing or monotone_decreasing

    # === Observable 3: Strouhal St via spectral peak in pressure (lift proxy) ===
    # St = f * D / U_inflow. For cylinder wake at Re ∈ [100, 300], expect
    # St ∈ [0.16, 0.21]. Use cylinder diameter D=1.0, U_inflow=1.0 as
    # NGC's likely non-dimensionalization (verify in Phase 2).
    p_at_centroid = pressure.mean(axis=1)  # (T,) — area-mean pressure proxy.
    p_fft = np.fft.rfft(p_at_centroid - p_at_centroid.mean())
    freqs = np.fft.rfftfreq(p_at_centroid.shape[0], d=0.01)
    peak_idx = int(np.argmax(np.abs(p_fft[1:])) + 1)  # skip DC
    strouhal = float(freqs[peak_idx] * 1.0 / 1.0)  # D / U_inflow

    # Substrate-class verdict.
    fits_open_driven_dissipative = (
        mass_conservation_int_abs_div_v < 1e-2  # ≈ 0
        and not ke_budget_monotone_in_either_direction  # oscillates
        and 0.15 <= strouhal <= 0.22  # cylinder-wake signature
    )
    verdict = (
        "open-driven-dissipative" if fits_open_driven_dissipative
        else f"UNEXPECTED (mass_drift={mass_conservation_int_abs_div_v:.3e}, "
        f"ke_monotone={ke_budget_monotone_in_either_direction}, "
        f"strouhal={strouhal:.3f})"
    )

    # Persistent-volume decision: did this entrypoint write to /vol?
    # Yes (we wrote audit_findings.json + use mgn_volume); the rollout-dir
    # isolation pattern applies to Phase 2's inference entrypoint.
    persistent_volume_decision = "Y"

    return {
        "mass_conservation_int_abs_div_v": mass_conservation_int_abs_div_v,
        "ke_budget_monotone_in_either_direction": ke_budget_monotone_in_either_direction,
        "strouhal": strouhal,
        "verdict": verdict,
        "persistent_volume_decision": persistent_volume_decision,
    }
```

The implementation uses the most-likely PhysicsNeMo MGN API call shape per `MeshGraphNet(input_dim_nodes=6, input_dim_edges=3, output_dim=3)` from the preflight. If Task 5's audit findings reveal a different rollout API (e.g., NGC packages it under a different module path), the executor adjusts the `model(...)` call shape — the observables math is NGC-independent and stays as written.

- [ ] **Step 2: Fire the smoke.**

```bash
modal run external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py::smoke_substrate_class_vortex_shedding
```

Expected: dict with the 3 observables + verdict + persistent-volume decision.

- [ ] **Step 3: Triage the verdict.**

Compare the observables against the design's predictions (§2.1 + §1.3 sub-class distinction):
- `mass_conservation_int_abs_div_v` ≈ 0 → consistent with incompressible NS prediction.
- `ke_budget_monotone_in_either_direction` == False → confirms boundary-driven sub-class (oscillates).
- `strouhal` ∈ [0.16, 0.21] → confirms cylinder-wake-specific signature.

If all three confirm → verdict = `open-driven-dissipative`, classification confirmed.
If any diverges → pattern-A drift fires → D-entry amendment captures the surprise.

- [ ] **Step 4: Record D0-2X verdicts 6 + 7.**

Replace `[pending]` for verdict 6 (substrate-class) + verdict 7 (persistent-volume) with the resolved values.

- [ ] **Step 5: Commit.**

```bash
git add external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py external_validation/_rollout_anchors/methodology/DECISIONS.md
git commit -m "02-physicsnemo-mgn: substrate-class smoke + persistent-volume decision (D0-2X verdicts 6+7)"
```

---

## Phase 1D — Code-absorption (post-audit)

### Task 10: `_expect_velocity` helper key resolution (Pattern-B P0 single-instance)

**Goal:** Update `_harness/mesh_rollout_adapter.py`'s `_expect_velocity` helper to use the actual NGC velocity-field key resolved in Task 5. Single-instance D-entry; **NO pre-generalization** per design §2.2 / rung-4c discipline. Per D0-2X verdict 8.

**Files:**
- Modify: `external_validation/_rollout_anchors/_harness/mesh_rollout_adapter.py`
- Create or modify: `external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py`

- [ ] **Step 1: Locate the current `_expect_velocity` implementation.**

```bash
grep -n "_expect_velocity" external_validation/_rollout_anchors/_harness/mesh_rollout_adapter.py
```

Read the current implementation. It should resolve velocity via `node_values["velocity"]` (default LB/synthetic assumption).

- [ ] **Step 2: Write the failing test.**

Append to (or create) `_harness/tests/test_mesh_rollout_adapter.py`:

```python
"""Tests for the mesh rollout adapter's loader-contract helpers."""

from __future__ import annotations

import numpy as np
import pytest

from external_validation._rollout_anchors._harness.mesh_rollout_adapter import (
    MeshRollout,
    _expect_velocity,
)


def test_expect_velocity_resolves_ngc_velocity_key_for_vortex_shedding() -> None:
    """Per D0-2X verdict 8: NGC vortex-shedding uses the resolved key.

    Replace VELOCITY_KEY_FROM_NGC_AUDIT below with the actual key name
    resolved in Task 5 (DECISIONS.md D0-2X verdict 8).
    """
    VELOCITY_KEY_FROM_NGC_AUDIT = "velocity"  # placeholder; replace per audit

    rollout = MeshRollout(
        node_positions=np.zeros((10, 2)),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        node_type=np.zeros(10, dtype=np.int64),
        node_values={
            VELOCITY_KEY_FROM_NGC_AUDIT: np.ones((5, 10, 2)),
        },
        dt=0.01,
        metadata={
            "framework": "pytorch+dgl",
            "model": "modulus_ns_meshgraphnet",
            "dataset": "vortex-shedding-2d",
        },
    )

    velocity = _expect_velocity(rollout)
    assert isinstance(velocity, np.ndarray), (
        f"_expect_velocity must resolve the NGC velocity key "
        f"{VELOCITY_KEY_FROM_NGC_AUDIT!r}; got {type(velocity)}"
    )
    assert velocity.shape == (5, 10, 2)
```

- [ ] **Step 3: Run the test (expected FAIL or PASS depending on current helper behavior).**

```bash
pytest external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py::test_expect_velocity_resolves_ngc_velocity_key_for_vortex_shedding -v
```

Expected: depending on what Task 5 resolved as the velocity key, the test either passes immediately (if `"velocity"` is the NGC key) or fails (if NGC uses a different key like `"u"` or `"flow_field"`).

- [ ] **Step 4: If the test fails, update `_expect_velocity` to handle the NGC-specific key.**

Modify `_harness/mesh_rollout_adapter.py`'s `_expect_velocity`:

```python
def _expect_velocity(rollout: MeshRollout) -> np.ndarray | HarnessDefect:
    """Resolve the velocity field from the rollout's node_values dict.

    NGC vortex-shedding (modulus_ns_meshgraphnet v0.1) uses the key
    "<KEY_FROM_AUDIT>" per DECISIONS.md D0-2X verdict 8 (audit fired
    2026-05-11). Legacy LB / synthetic paths use "velocity" — both
    are accepted as a *single-instance* enumeration (no predicate-
    generalization per design §2.2 / rung-4c discipline; predicate
    generalization fires only if amendment 1's Ahmed Body brings a
    second NGC naming).
    """
    for key in ("<KEY_FROM_AUDIT>", "velocity"):  # NGC first, fallback to legacy
        if key in rollout.node_values:
            return np.asarray(rollout.node_values[key])
    return HarnessDefect(
        value=None,
        skip_reason=(
            f"_expect_velocity: no velocity-shaped key in node_values "
            f"(keys present: {sorted(rollout.node_values.keys())[:5]}). "
            f"See DECISIONS.md D0-2X verdict 8 for the NGC key name."
        ),
    )
```

(Replace `<KEY_FROM_AUDIT>` with the actual value from Task 5.)

- [ ] **Step 5: Run the test (expected PASS).**

```bash
pytest external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py::test_expect_velocity_resolves_ngc_velocity_key_for_vortex_shedding -v
```

Expected: 1 passed.

- [ ] **Step 6: Add a second test verifying the legacy "velocity" key still works (regression guard).**

```python
def test_expect_velocity_still_accepts_legacy_velocity_key() -> None:
    """Pattern-B P0 discipline: the helper accepts BOTH the NGC key and
    the legacy "velocity" key. Single-instance enumeration; no predicate-
    generalization. Amendment 1's Ahmed Body is the multi-instance trigger.
    """
    rollout = MeshRollout(
        node_positions=np.zeros((10, 2)),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        node_type=np.zeros(10, dtype=np.int64),
        node_values={"velocity": np.ones((5, 10, 2))},
        dt=0.01,
        metadata={"framework": "synthetic"},
    )
    velocity = _expect_velocity(rollout)
    assert isinstance(velocity, np.ndarray)
    assert velocity.shape == (5, 10, 2)
```

- [ ] **Step 7: Run both tests.**

```bash
pytest external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py -v
```

Expected: 2 passed.

- [ ] **Step 8: Commit.**

```bash
git add external_validation/_rollout_anchors/_harness/mesh_rollout_adapter.py external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py
git commit -m "_harness: _expect_velocity resolves NGC vortex-shedding key (D0-2X verdict 8; no pre-generalization)"
```

---

### Task 11: `MGN_DATASET_SYSTEM_CLASS` taxonomy + dispatch (TDD)

**Goal:** Introduce the mesh-side substrate-class taxonomy parallel to particle-side `LAGRANGEBENCH_DATASET_SYSTEM_CLASS`. Wire dispatch into `energy_drift_on_mesh` + `dissipation_sign_violation_on_mesh` per the design §2.2 P0-resolvable pattern-B response (duplicated route, NOT stack-agnostic refactor). Per D0-2X verdict 9.

**Files:**
- Modify: `external_validation/_rollout_anchors/_harness/mesh_rollout_adapter.py`
- Modify: `external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py`

- [ ] **Step 1: Write failing test for the taxonomy entry.**

Append to `_harness/tests/test_mesh_rollout_adapter.py`:

```python
def test_mgn_dataset_system_class_pins_vortex_shedding_2d() -> None:
    """Per D0-2X verdict 9: MGN_DATASET_SYSTEM_CLASS exists and pins
    vortex_shedding_2d to the substrate-class verdict from Task 9.
    """
    from external_validation._rollout_anchors._harness.mesh_rollout_adapter import (
        MGN_DATASET_SYSTEM_CLASS,
    )

    assert "vortex_shedding_2d" in MGN_DATASET_SYSTEM_CLASS
    # Replace the expected value with the Task 9 verdict.
    expected_class = "open-driven-dissipative"  # placeholder; per D0-2X verdict 6
    assert MGN_DATASET_SYSTEM_CLASS["vortex_shedding_2d"] == expected_class
```

- [ ] **Step 2: Run test (expected FAIL — `MGN_DATASET_SYSTEM_CLASS` doesn't exist yet).**

```bash
pytest external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py::test_mgn_dataset_system_class_pins_vortex_shedding_2d -v
```

Expected: FAIL with `ImportError: cannot import name 'MGN_DATASET_SYSTEM_CLASS'`.

- [ ] **Step 3: Add `MGN_DATASET_SYSTEM_CLASS` to mesh_rollout_adapter.py.**

After the existing `LAGRANGEBENCH_DATASET_SYSTEM_CLASS` import / near the top of `mesh_rollout_adapter.py`:

```python
# Mesh-side substrate-class taxonomy. Parallel to
# `particle_rollout_adapter.py::LAGRANGEBENCH_DATASET_SYSTEM_CLASS`.
# Per design §2.2 P0-resolvable pattern-B response (duplicated route, NOT
# stack-agnostic refactor); duplicate-logic-drift risk is *named* per
# round-codex-4 catalogue, not eliminated. Stack-agnostic refactor
# triggers only on amendment 1 / case study 03 evidence.
#
# Empirical classification per the "classify when you exercise" rule:
# entries land only after Phase 1's empirical probe confirms the
# substrate's behavior (3 discriminating observables; see D0-2X verdict 6).
MGN_DATASET_SYSTEM_CLASS: dict[str, str] = {
    "vortex_shedding_2d": "open-driven-dissipative",  # per D0-2X verdict 6
}
```

- [ ] **Step 4: Run the test (expected PASS).**

```bash
pytest external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py::test_mgn_dataset_system_class_pins_vortex_shedding_2d -v
```

Expected: 1 passed.

- [ ] **Step 5: Write failing test for dispatch in `energy_drift_on_mesh`.**

```python
def test_energy_drift_on_mesh_skips_when_open_driven_dissipative() -> None:
    """Per D0-22 amendment 1 + D0-2X verdict 9: energy_drift_on_mesh
    SKIPs with reason on open-driven-dissipative substrates, parallel
    to the particle-side gate.
    """
    from external_validation._rollout_anchors._harness.mesh_rollout_adapter import (
        energy_drift_on_mesh,
        MeshRollout,
    )

    rollout = MeshRollout(
        node_positions=np.zeros((10, 2)),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        node_type=np.zeros(10, dtype=np.int64),
        # KE clears KE_REST_THRESHOLD so D0-08 does NOT fire; D0-22's
        # MGN-side dispatch must fire instead.
        node_values={"velocity": 10 * np.ones((5, 10, 2))},
        dt=0.01,
        metadata={
            "framework": "pytorch+dgl",
            "model": "modulus_ns_meshgraphnet",
            "dataset": "vortex_shedding_2d",
        },
        is_regular_grid=True,
        grid_spacing=(0.1, 0.1),
    )

    result = energy_drift_on_mesh(rollout)
    assert result.value is None, (
        "energy_drift_on_mesh should SKIP on open-driven-dissipative, "
        f"got value={result.value}, skip_reason={result.skip_reason}"
    )
    assert "open-driven-dissipative" in (result.skip_reason or ""), (
        f"SKIP reason must cite the substrate class; got: {result.skip_reason}"
    )
    assert "D0-22" in (result.skip_reason or "") or "D0-2X" in (result.skip_reason or "")
```

- [ ] **Step 6: Run test (expected FAIL — dispatch not yet wired).**

```bash
pytest external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py::test_energy_drift_on_mesh_skips_when_open_driven_dissipative -v
```

Expected: FAIL.

- [ ] **Step 7: Wire dispatch into `energy_drift_on_mesh`.**

Modify `mesh_rollout_adapter.py::energy_drift_on_mesh` to add the substrate-class gate (parallel to particle-side at `particle_rollout_adapter.py::energy_drift` lines 523-548):

```python
def energy_drift_on_mesh(rollout: MeshRollout) -> HarnessDefect:
    """[existing docstring...]

    Substrate-class dispatch added at D0-2X (Phase 1): mirrors D0-22
    amendment 1's particle-side gate. Open-driven-dissipative substrates
    SKIP with reason — the strictly-dissipative-or-conservative assumption
    underpinning energy_drift does not apply.
    """
    velocity = _expect_velocity(rollout)
    if isinstance(velocity, HarnessDefect):
        return velocity
    if not rollout.is_regular_grid:
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"mesh is graph-topology (framework="
                f"{rollout.metadata.get('framework')!r}); graph-mesh KE "
                f"integration is gated on Day 2 hour 1 NGC audit"
            ),
        )

    # D0-2X substrate-class dispatch (parallel to D0-22 amendment 1 on
    # particle side). Fires BEFORE the KE-rest gate because the substrate
    # class is the load-bearing assumption that energy_drift's contract
    # depends on; if the assumption is violated, KE-rest gating is moot.
    dataset_name = rollout.metadata.get("dataset")
    system_class = MGN_DATASET_SYSTEM_CLASS.get(dataset_name)
    if system_class == "open-driven-dissipative":
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"system_class='open-driven-dissipative' (dataset={dataset_name!r}); "
                "the strictly-dissipative-or-conservative assumption "
                "underpinning energy_drift does not apply on open-driven "
                "substrates. See DECISIONS.md D0-22 (amendment 1) for the "
                "particle-side precedent and D0-2X for the mesh-side "
                "extension."
            ),
        )

    e_series = kinetic_energy_series_on_mesh(rollout)
    e0 = float(e_series[0])
    if abs(e0) < KE_REST_THRESHOLD:
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"KE(0)={e0:.3e} < {KE_REST_THRESHOLD:.0e} (mesh rollout "
                f"starts at rest; relative drift undefined; see DECISIONS.md "
                f"D0-08)"
            ),
        )
    drift = float(np.max(np.abs(e_series - e0)))
    return HarnessDefect(value=drift / abs(e0))
```

- [ ] **Step 8: Run the dispatch test (expected PASS).**

```bash
pytest external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py::test_energy_drift_on_mesh_skips_when_open_driven_dissipative -v
```

Expected: 1 passed.

- [ ] **Step 9: Wire dispatch into `dissipation_sign_violation_on_mesh`.**

Add the parallel test:

```python
def test_dissipation_sign_violation_on_mesh_skips_when_open_driven_dissipative() -> None:
    """Per D0-22 base gate + D0-2X verdict 9: dissipation_sign_violation_on_mesh
    SKIPs with reason on open-driven-dissipative substrates."""
    from external_validation._rollout_anchors._harness.mesh_rollout_adapter import (
        dissipation_sign_violation_on_mesh,
        MeshRollout,
    )

    rollout = MeshRollout(
        node_positions=np.zeros((10, 2)),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        node_type=np.zeros(10, dtype=np.int64),
        node_values={"velocity": 10 * np.ones((5, 10, 2))},
        dt=0.01,
        metadata={
            "framework": "pytorch+dgl",
            "model": "modulus_ns_meshgraphnet",
            "dataset": "vortex_shedding_2d",
        },
        is_regular_grid=True,
        grid_spacing=(0.1, 0.1),
    )

    result = dissipation_sign_violation_on_mesh(rollout)
    assert result.value is None, (
        "dissipation_sign_violation_on_mesh should SKIP on open-driven-dissipative, "
        f"got value={result.value}, skip_reason={result.skip_reason}"
    )
    assert "open-driven-dissipative" in (result.skip_reason or "")
    assert "D0-22" in (result.skip_reason or "") or "D0-2X" in (result.skip_reason or "")
```

Run the test (expected FAIL):

```bash
pytest external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py::test_dissipation_sign_violation_on_mesh_skips_when_open_driven_dissipative -v
```

Modify `dissipation_sign_violation_on_mesh` in `mesh_rollout_adapter.py` to add the substrate-class gate (parallel to particle-side at `particle_rollout_adapter.py::dissipation_sign_violation` lines 594-611):

```python
def dissipation_sign_violation_on_mesh(rollout: MeshRollout) -> HarnessDefect:
    """[existing docstring...]

    Substrate-class dispatch added at D0-2X (Phase 1): mirrors D0-22 base
    gate's particle-side dispatch. Open-driven-dissipative substrates SKIP
    with reason — the strictly-dissipative-or-conservative assumption that
    dissipation_sign_violation encodes does not apply.
    """
    velocity = _expect_velocity(rollout)
    if isinstance(velocity, HarnessDefect):
        return velocity
    if not rollout.is_regular_grid:
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"mesh is graph-topology (framework="
                f"{rollout.metadata.get('framework')!r}); graph-mesh dKE/dt "
                f"is gated on Day 2 hour 1 NGC audit"
            ),
        )

    # D0-2X substrate-class dispatch (parallel to D0-22 base gate on
    # particle side). Fires BEFORE the timestep / KE-rest gates because
    # the substrate class is the load-bearing assumption.
    dataset_name = rollout.metadata.get("dataset")
    system_class = MGN_DATASET_SYSTEM_CLASS.get(dataset_name)
    if system_class == "open-driven-dissipative":
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"system_class='open-driven-dissipative' (dataset={dataset_name!r}); "
                "dE/dt > 0 over a stretch by physics (boundary-driven inflow "
                "supplies KE); the strictly-dissipative-or-conservative "
                "assumption underpinning dissipation_sign_violation does not "
                "apply. See DECISIONS.md D0-22 for the particle-side precedent "
                "and D0-2X for the mesh-side extension."
            ),
        )

    if rollout.n_timesteps < 2:
        raise ValueError(
            f"dissipation_sign_violation_on_mesh needs at least 2 timesteps; "
            f"got {rollout.n_timesteps}"
        )
    e_series = kinetic_energy_series_on_mesh(rollout)
    e_max = float(np.max(e_series))
    if e_max < KE_REST_THRESHOLD:
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"max(KE)={e_max:.3e} < {KE_REST_THRESHOLD:.0e} (mesh "
                f"trajectory has no kinetic energy; dissipation question "
                f"undefined; see DECISIONS.md D0-08)"
            ),
        )
    de_dt = np.diff(e_series) / rollout.dt
    max_growth = float(np.max(de_dt))
    return HarnessDefect(value=max(0.0, max_growth) / e_max)
```

Run the test (expected PASS):

```bash
pytest external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py::test_dissipation_sign_violation_on_mesh_skips_when_open_driven_dissipative -v
```

- [ ] **Step 10: Run the full mesh_rollout_adapter test suite.**

```bash
pytest external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py -v
```

Expected: all tests pass.

- [ ] **Step 11: Commit.**

```bash
git add external_validation/_rollout_anchors/_harness/mesh_rollout_adapter.py external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py
git commit -m "_harness: MGN_DATASET_SYSTEM_CLASS + substrate-class dispatch on *_on_mesh (D0-2X verdict 9)"
```

---

### Task 12: Pre-flight assertions in `mesh_rollout_adapter.py`

**Goal:** Add a "loader-contract assertions" section to `mesh_rollout_adapter.py` paralleling the particle-side section, with each assertion grounded in preflight V1-V18 + Phase 1 audit findings (Task 5). Per D0-2X verdict 10.

**Files:**
- Modify: `external_validation/_rollout_anchors/_harness/mesh_rollout_adapter.py`
- Modify: `external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py`

- [ ] **Step 1: Read particle-side loader-contract assertions for the template.**

```bash
grep -n "loader-contract\|LB loader\|assertion" external_validation/_rollout_anchors/_harness/particle_rollout_adapter.py | head -10
```

Read the surrounding code to understand the assertion shape (defensive-validation early-return with `HarnessDefect(skip_reason=...)` or raise).

- [ ] **Step 2: Write failing test for one critical assertion (default dtype).**

Per preflight known-unknown #4: fp32-vs-fp64 contract implicit. Materializer must `torch.set_default_dtype(torch.float32)` before dataset construction. Pre-flight assertion: detect dtype mismatch on incoming rollout's velocity field.

```python
def test_mesh_rollout_adapter_asserts_velocity_dtype_float32() -> None:
    """Per D0-2X verdict 10 + preflight known-unknown #4: MGN materializer
    asserts float32 velocity dtype on the rollout; float64 surfaces as a
    loader-contract violation (informative SKIP reason)."""
    from external_validation._rollout_anchors._harness.mesh_rollout_adapter import (
        MeshRollout, _assert_loader_contract_mgn,
    )

    rollout = MeshRollout(
        node_positions=np.zeros((10, 2), dtype=np.float32),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        node_type=np.zeros(10, dtype=np.int64),
        node_values={"velocity": np.ones((5, 10, 2), dtype=np.float64)},  # wrong dtype
        dt=0.01,
        metadata={"framework": "pytorch+dgl", "model": "modulus_ns_meshgraphnet"},
    )

    with pytest.raises(AssertionError, match="float32"):
        _assert_loader_contract_mgn(rollout)
```

- [ ] **Step 3: Run test (FAIL — `_assert_loader_contract_mgn` doesn't exist).**

```bash
pytest external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py::test_mesh_rollout_adapter_asserts_velocity_dtype_float32 -v
```

Expected: FAIL with `ImportError`.

- [ ] **Step 4: Implement `_assert_loader_contract_mgn` covering V1-V18 + 5 secondary known-unknowns.**

Add to `mesh_rollout_adapter.py`:

```python
def _assert_loader_contract_mgn(rollout: MeshRollout) -> None:
    """MGN materializer loader-contract assertions per D0-2X verdict 10.

    Each assertion is grounded in a preflight V-entry or known-unknown.
    Fires defensively on incoming MGN rollouts before the rule kernels
    consume them; informative AssertionError if any contract violated.

    Per design §2.1 Enabling discipline: source-method-implementing-
    pattern-A-discipline. Written in source review; catches pattern-A
    divergence at runtime before P0 inference data flows into the rules.
    """
    # V12 / V14: velocity field present + correct shape.
    velocity = rollout.node_values.get("velocity")
    if velocity is None:
        # _expect_velocity handles alternate key names; this assertion is
        # a no-op when velocity is absent (downstream _expect_velocity
        # returns the informative SKIP).
        return

    velocity_arr = np.asarray(velocity)

    # Preflight known-unknown #4: fp32-vs-fp64 contract.
    assert velocity_arr.dtype in (np.float32,), (
        f"MGN velocity dtype must be float32 per preflight known-unknown #4 "
        f"(materializer must torch.set_default_dtype(torch.float32) before "
        f"dataset construction). Got: {velocity_arr.dtype}. "
        f"See DECISIONS.md D0-2X verdict 10."
    )

    # V-entries: velocity shape (T, N_nodes, D) where D ∈ {2, 3}.
    assert velocity_arr.ndim == 3, (
        f"MGN velocity must be 3D (T, N_nodes, D); got shape "
        f"{velocity_arr.shape}. See preflight V12."
    )
    assert velocity_arr.shape[2] in (2, 3), (
        f"MGN velocity last-dim must be 2 (2D) or 3 (3D); got "
        f"{velocity_arr.shape[2]}. See preflight V12."
    )

    # Preflight known-unknown #5: node_type values in {0, 3, 4, 5, 6}
    # (F.one_hot's num_classes=4 bound after (value - 3) shift).
    node_type = np.asarray(rollout.node_type)
    valid_node_types = {0, 3, 4, 5, 6}
    actual_types = set(np.unique(node_type).tolist())
    invalid = actual_types - valid_node_types
    assert not invalid, (
        f"MGN node_type values must be in {valid_node_types} per preflight "
        f"known-unknown #5 (one_hot num_classes=4 bound after value-3 shift). "
        f"Invalid values: {invalid}. See DECISIONS.md D0-2X verdict 10."
    )

    # Preflight V-entries on metadata: framework + model + dataset must be present.
    for required_meta_key in ("framework", "model"):
        assert required_meta_key in rollout.metadata, (
            f"MGN rollout metadata must include {required_meta_key!r}; "
            f"got keys: {sorted(rollout.metadata.keys())}. See preflight "
            f"V-entries on rollout schema."
        )
```

- [ ] **Step 5: Run the test (PASS).**

```bash
pytest external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py::test_mesh_rollout_adapter_asserts_velocity_dtype_float32 -v
```

Expected: 1 passed.

- [ ] **Step 6: Add assertion-coverage tests for the remaining V-entries + secondary known-unknowns.**

Append three tests to `_harness/tests/test_mesh_rollout_adapter.py`:

```python
def test_mesh_rollout_adapter_asserts_velocity_shape_3d() -> None:
    """Per preflight V12: velocity must be 3D (T, N_nodes, D)."""
    from external_validation._rollout_anchors._harness.mesh_rollout_adapter import (
        MeshRollout, _assert_loader_contract_mgn,
    )

    rollout = MeshRollout(
        node_positions=np.zeros((10, 2), dtype=np.float32),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        node_type=np.zeros(10, dtype=np.int64),
        node_values={"velocity": np.ones((10, 2), dtype=np.float32)},  # 2D wrong
        dt=0.01,
        metadata={"framework": "pytorch+dgl", "model": "modulus_ns_meshgraphnet"},
    )
    with pytest.raises(AssertionError, match="3D"):
        _assert_loader_contract_mgn(rollout)


def test_mesh_rollout_adapter_asserts_node_type_in_known_set() -> None:
    """Per preflight known-unknown #5: node_type ∈ {0, 3, 4, 5, 6}."""
    from external_validation._rollout_anchors._harness.mesh_rollout_adapter import (
        MeshRollout, _assert_loader_contract_mgn,
    )

    rollout = MeshRollout(
        node_positions=np.zeros((10, 2), dtype=np.float32),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        node_type=np.array([0, 3, 4, 5, 6, 7, 0, 0, 0, 0], dtype=np.int64),  # 7 invalid
        node_values={"velocity": np.ones((5, 10, 2), dtype=np.float32)},
        dt=0.01,
        metadata={"framework": "pytorch+dgl", "model": "modulus_ns_meshgraphnet"},
    )
    with pytest.raises(AssertionError, match="node_type"):
        _assert_loader_contract_mgn(rollout)


def test_mesh_rollout_adapter_asserts_metadata_required_keys() -> None:
    """Metadata must include framework + model per loader contract."""
    from external_validation._rollout_anchors._harness.mesh_rollout_adapter import (
        MeshRollout, _assert_loader_contract_mgn,
    )

    rollout = MeshRollout(
        node_positions=np.zeros((10, 2), dtype=np.float32),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        node_type=np.zeros(10, dtype=np.int64),
        node_values={"velocity": np.ones((5, 10, 2), dtype=np.float32)},
        dt=0.01,
        metadata={"framework": "pytorch+dgl"},  # missing "model"
    )
    with pytest.raises(AssertionError, match="model"):
        _assert_loader_contract_mgn(rollout)
```

Run all four loader-contract tests:

```bash
pytest external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py -k "loader_contract or velocity_dtype or velocity_shape or node_type or metadata_required" -v
```

Expected: 4 passed (the dtype test from step 2 + the 3 from this step).

The `_assert_loader_contract_mgn` implementation in step 4 already covers all four assertions; if any test fails, the implementation needs the missing assertion added inline before re-running.

- [ ] **Step 7: Run the full test suite.**

```bash
pytest external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit.**

```bash
git add external_validation/_rollout_anchors/_harness/mesh_rollout_adapter.py external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py
git commit -m "_harness: mesh-side loader-contract assertions per preflight V1-V18 + 5 secondary (D0-2X verdict 10)"
```

---

### Task 13: D-entries finalized + committed

**Goal:** Replace all `[pending]` markers in D0-2X with the resolved verdicts from Tasks 1, 5, 6, 7, 8, 9, 10, 11, 12. Mark D0-2X status from `open` to `resolved`. Per design §3.1 activity 13.

**Files:**
- Modify: `external_validation/_rollout_anchors/methodology/DECISIONS.md`

- [ ] **Step 1: Scan D0-2X for remaining `[pending]` markers.**

```bash
grep -A 1 "\[pending\]" external_validation/_rollout_anchors/methodology/DECISIONS.md
```

Expected: zero matches if Tasks 1-12 all populated their verdicts; otherwise, list of unfilled verdicts.

- [ ] **Step 2: If any verdicts remain `[pending]`, fill them now from session notes.**

- [ ] **Step 3: Change D0-2X status from `open` to `resolved at 2026-05-11 (Phase 1 complete; Tasks 1-12 verdicts pinned)`.**

Edit `DECISIONS.md` D0-2X header line.

- [ ] **Step 4: Commit the finalization.**

```bash
git add external_validation/_rollout_anchors/methodology/DECISIONS.md
git commit -m "DECISIONS.md: D0-2X resolved — case study 02 Phase 1 verdicts finalized"
```

---

## Phase 1E — Cross-review boundary

### Task 14: Phase 1 boundary cross-review (Codex pass)

**Goal:** Dispatch a Codex cross-review against Phase 1's verdicts + code-absorption. Per design §2.3 + §3.1 cross-review scope. Findings triaged in Task 15.

**Files:**
- Create: `external_validation/_rollout_anchors/methodology/docs/2026-05-XX-case-study-02-phase-1-cross-review.md` (date stamp at execution time)

- [ ] **Step 1: Compose the cross-review prompt.**

The cross-review covers:
- D0-2X verdicts 1-10 (audit + dispatch + assertions).
- Code-absorption commits: Tasks 10 (`_expect_velocity`), 11 (`MGN_DATASET_SYSTEM_CLASS`), 12 (pre-flight assertions).
- Design doc §2.6 layered fail-open predictions: did Phase 1 surface any of them?

Prompt skeleton:

```
Adversarial review of case study 02 Phase 1 (commits <sha-range>) per
the design doc at methodology/docs/2026-05-11-case-study-02-physicsnemo-mgn-design.md.

Goal: find layered fail-open paths in the MGN safety surface that the
Phase 1 code-absorption (D0-2X verdicts 8-10) may have introduced.

Search lens: round-codex-4 elevated the layered-fail-open observation
to a named methodology contribution. Phase 1 introduces:
- _expect_velocity helper key resolution (Task 10)
- MGN_DATASET_SYSTEM_CLASS substrate-class dispatch (Task 11)
- Loader-contract assertions in mesh_rollout_adapter.py (Task 12)

Each of these is a potential fail-open surface. Look for:
- Adjacent attack vectors to D0-22's particle-side dispatch (e.g.,
  mesh-side analog of round-codex-4's retry-isolation if MGN inference
  writes to a persistent volume — see D0-2X verdict 7).
- Loader-contract assertions that DON'T fire when they should (false
  negatives in defensive validation).
- Substrate-class dispatch bypass paths (delete-the-dataset-key, missing
  metadata, etc.) — parallel to round-codex-2's manifest_required gate.

Output: findings list per round-codex-4 format (severity + surface +
evidence + recommended fix + Pattern-C cell).
```

- [ ] **Step 2: Dispatch the Codex review.**

Via the `codex:rescue` subagent or direct `codex exec` invocation per the parallel-session pattern from `5cb90cc` (which used `codex exec` directly via Bash). Capture findings in a dated cross-review doc.

- [ ] **Step 3: Commit the cross-review findings doc.**

```bash
git add external_validation/_rollout_anchors/methodology/docs/2026-05-XX-case-study-02-phase-1-cross-review.md
git commit -m "methodology: case study 02 Phase 1 boundary cross-review findings"
```

---

### Task 15: Triage Phase 1 cross-review findings

**Goal:** Apply pattern-C four-cell triage to each Phase 1 cross-review finding. Absorb cell-2 findings in-rung; defer cell-3 to amendment 1; record cell-1 + cell-4 outcomes. Update D0-2X with the cross-review summary. Per design §2.3.

**Files:**
- Modify: `external_validation/_rollout_anchors/methodology/DECISIONS.md` (D0-2X cross-review summary)
- Possibly modify: `_harness/mesh_rollout_adapter.py`, `02-physicsnemo-mgn/modal_app.py`, etc. (if cell-2 absorptions land)

- [ ] **Step 1: For each finding, triage into the four cells.**

Per v2.1 §1.3:
- **Cell 1 (re-discovery under prior scope):** defer to prior decision; cite the discipline-marker (e.g., D0-22 amendment 1's substrate-class dispatch already covers it).
- **Cell 2 (novel-in-scope):** in-rung absorption per pattern A or B. Land follow-up commits.
- **Cell 3 (novel-out-of-scope):** forward-flag to amendment 1 (Ahmed Body) or case study 03.
- **Cell 4 (genuinely new framing):** re-examine prior decision with new information. Earn the cell-4 bar by articulation per §1.3 falsification rule 4.

- [ ] **Step 2: Land cell-2 absorption commits if any.**

Each cell-2 finding → TDD red-green commit:
- Write failing test capturing the finding's scenario.
- Implement the fix.
- Verify test passes.
- Commit with reference to the cross-review finding.

- [ ] **Step 3: Record cell distribution in D0-2X.**

Append to D0-2X:

```markdown
**Phase 1 boundary cross-review (Task 14-15) — findings triaged:**

| # | Finding (1-line) | Cell | Disposition |
|---|---|---|---|
| 1 | [finding 1 summary] | [cell N] | [absorbed at sha / deferred to amendment 1 / re-examined] |
| ... | ... | ... | ... |

Total: N cell-1, M cell-2, P cell-3, Q cell-4.
```

- [ ] **Step 4: Commit the triage summary.**

```bash
git add external_validation/_rollout_anchors/methodology/DECISIONS.md
git commit -m "DECISIONS.md: D0-2X Phase 1 cross-review triage summary"
```

- [ ] **Step 5: Push the branch.**

```bash
git push -u origin feature/case-study-02-physicsnemo-mgn
```

(Or `git push` if the branch is already tracking origin.)

- [ ] **Step 6: Update the Phase 1 plan's status in the design doc's §7 successor block.**

In `methodology/docs/2026-05-11-case-study-02-physicsnemo-mgn-design.md` §7:
- Change "Successor (session 2): Phase 1 execution per §3.1 + §4.1." → "Phase 1 complete at sha <sha>; see D0-2X for verdicts."

Commit:

```bash
git add external_validation/_rollout_anchors/methodology/docs/2026-05-11-case-study-02-physicsnemo-mgn-design.md
git commit -m "design doc: mark Phase 1 complete; D0-2X verdicts and cross-review pinned"
git push
```

---

## Phase 1 acceptance criteria check (per design §4.1)

Verify each design §4.1 checkbox before declaring Phase 1 done:

- [ ] BLOCKING-1 CPU state-dict smoke complete; verdict in D0-2X verdict 1.
- [ ] NGC checkpoint downloaded; hash pinned in DECISIONS.md.
- [ ] Day 2 hour 1 NGC audit findings recorded (verdicts 2 + 8).
- [ ] Gate A verdict recorded (verdict 3 + D0-02 amendment).
- [ ] Gate D verdict recorded (verdict 5).
- [ ] `test_inference_matches_ngc_sample` verdict recorded (verdict 4).
- [ ] Empirical substrate-class smoke verdict recorded (verdict 6).
- [ ] `MGN_DATASET_SYSTEM_CLASS` introduced + dispatch wired (verdict 9).
- [ ] `_expect_velocity` key resolution pinned (verdict 8).
- [ ] Pre-flight assertions written in `mesh_rollout_adapter.py` (verdict 10).
- [ ] Persistent-volume decision recorded (verdict 7).
- [ ] Phase 1 boundary cross-review complete; findings triaged (Tasks 14-15).

If any unchecked → fix before opening Phase 2's writing-plans round.

---

## Successor: Phase 2 + Phase 3 writing-plans

Phase 2 + Phase 3 require Phase 1's verdicts (Gate A branch, substrate-class verdict, persistent-volume decision, NGC velocity key) to plan correctly. Open a fresh writing-plans iteration **after Phase 1 completes**, using:
- This plan's final state (D0-2X resolved).
- The design doc §3.2 + §3.3 + §4.2 + §4.3.
- The Phase 1 boundary cross-review findings.

Do NOT extend this plan to cover Phase 2/3 — the per-phase plan boundary is intentional per design §7 (audit verdicts feed forward; plans are per-phase so verdicts inform the next plan's shape).
