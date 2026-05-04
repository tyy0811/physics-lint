"""Modal entrypoint for LagrangeBench rollout generation (Day 1).

Two functions live here, one per rung in the test ladder:

- ``jax_micro_gate`` (rung 1, plan §7 / D0-10): JAX-CUDA install
  visibility check. Returns ``jax.devices()``. Cheapest possible
  failure-mode-isolated test for the JAX-on-Modal-CUDA install path.
- ``lagrangebench_install_smoke`` (rung 2, plan §3.2 step 1):
  LagrangeBench install + toy-config inference smoke. Three sub-checks
  (import → JAX-still-sees-GPU → ``python main.py mode=infer
  dataset=tgv2d``). Surfaces version-pinning conflicts where
  LagrangeBench's deps would silently downgrade JAX, and surfaces
  Hydra/CLI breakage and missing-data failure modes via captured
  stdout/stderr.

Both functions run on T4 per D0-13 stage-1 ("smoke tests use the
cheapest CUDA-JAX path"). Day 1 §3.2 step 3 production rollouts will
land on A10G (D0-13 stage 2) in a separate function; A100 is reserved
as OOM fallback only — see DECISIONS D0-13 for the full stage-by-stage
matrix and the OOM escalation criterion.

Per D0-10 + D0-13: if ``jax.devices()`` does not return at least one
device with ``platform == "gpu"``, the agent pivots to JAX-CPU
read-only mode (synthetic rollouts, already landed on this branch);
the Modal-image debugging side-quest is OUT OF SCOPE without explicit
user re-authorisation. The 2h plan cap is a ceiling, not an
authorisation.

Subsequent commits will layer in:
- Dataset download (``bash download_data.sh tgv2d|dam2d`` into a
  Modal Volume so the download persists across runs)
- Checkpoint download (gdown URLs from the LagrangeBench README)
- Inference + rollout export to ``particle_rollout.npz`` schema
- ``particle_rollout_adapter`` invocation

Run with:
    modal run external_validation/_rollout_anchors/01-lagrangebench/modal_app.py
    modal run external_validation/_rollout_anchors/01-lagrangebench/modal_app.py::lagrangebench_smoke
"""

from __future__ import annotations

import modal

app = modal.App("rollout-anchors-lagrangebench")

# Hour-0 image: JAX with CUDA 12 only. Heavy installs deferred so a
# micro-gate FAIL does not waste image-build time on this side-quest.
# Mirrors plan §3.2 step 1 verbatim ("pip install -U 'jax[cuda12]' jaxlib").
#
# Python 3.10 (not 3.11): LagrangeBench's pyproject.toml pins
# ``python_requires = ">=3.9, <=3.11"`` which PEP 440 normalises as
# (3, 9, 0) <= python <= (3, 11, 0), excluding all 3.11.x patch releases.
# Modal's debian_slim(python_version="3.11") ships 3.11.x where x>0, which
# fails the pin. 3.10 is the highest version satisfying both the
# LagrangeBench pin and JAX 0.10's range. Upstream-forced — not a
# methodology choice — and therefore not pre-registered in DECISIONS.md;
# this comment is the audit trail. If LagrangeBench relaxes the pin
# upstream (e.g., to ``<3.12``), bumping to 3.11 is a clean change.
jax_image = modal.Image.debian_slim(python_version="3.10").pip_install("jax[cuda12]", "jaxlib")


MICRO_GATE_GPU_CLASS = (
    "T4"  # D0-13 stage-1 default; drift-guarded by tests/test_modal_app_gpu_class.py
)


@app.function(image=jax_image, gpu=MICRO_GATE_GPU_CLASS, timeout=600)
def jax_micro_gate() -> dict:
    """Hour-2 micro-gate per plan §7 / D0-10 (refined by D0-13).

    Returns a dict with the device list, default backend, and a derived
    ``has_gpu`` boolean. Caller (``main``) classifies against the
    D0-10 + D0-13 spirit-reading: any CUDA-compatible GPU device passes;
    CPU-only return triggers the D0-10 pivot. Also returns the resolved
    jax + jaxlib versions so the audit trail captures *which* JAX-CUDA
    stack passed the gate (matters when the Python pin or upstream
    constraints change which versions pip resolves to).
    """
    import jax
    import jaxlib

    devices = jax.devices()
    backend = jax.default_backend()
    has_gpu = any(d.platform == "gpu" for d in devices)
    return {
        "devices": [str(d) for d in devices],
        "default_backend": backend,
        "has_gpu": has_gpu,
        "device_count": len(devices),
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
    }


@app.local_entrypoint()
def main() -> None:
    """Fire the JAX micro-gate; classify against D0-10 + D0-13."""
    result = jax_micro_gate.remote()
    print("=== JAX micro-gate verdict (D0-10 + D0-13) ===")
    print(f"  gpu_class:       {MICRO_GATE_GPU_CLASS}")
    print(f"  jax_version:     {result['jax_version']}")
    print(f"  jaxlib_version:  {result['jaxlib_version']}")
    print(f"  default_backend: {result['default_backend']}")
    print(f"  device_count:    {result['device_count']}")
    print(f"  devices:         {result['devices']}")
    print(f"  has_gpu:         {result['has_gpu']}")
    if result["has_gpu"]:
        print(
            f"  -> verdict: PASS — CUDA GPU ({MICRO_GATE_GPU_CLASS}) visible "
            "to JAX; proceed to step 2."
        )
    else:
        print("  -> verdict: FAIL — no GPU device returned; pivot per D0-10:")
        print("     JAX-CPU read-only path (synthetic rollouts already landed);")
        print("     no Modal-image debugging without explicit user re-auth.")


# Rung-2 image: jax_image + LagrangeBench clone + ``pip install -e ".[dev]"``.
# Layered on top of jax_image so the cached JAX layer (~95s build) is reused.
# `--depth 1` keeps the clone small; the working tree only needs HEAD.
lagrangebench_image = jax_image.apt_install("git").run_commands(
    "git clone --depth 1 https://github.com/tumaer/lagrangebench.git /opt/lagrangebench",
    "cd /opt/lagrangebench && pip install -e '.[dev]'",
)


LAGRANGEBENCH_SMOKE_GPU_CLASS = (
    "T4"  # D0-13 stage-1 (smoke tests); drift-guarded by tests/test_modal_app_gpu_class.py
)


@app.function(image=lagrangebench_image, gpu=LAGRANGEBENCH_SMOKE_GPU_CLASS, timeout=900)
def lagrangebench_install_smoke() -> dict:
    """Rung 2: LagrangeBench install + toy-config inference smoke.

    Three sub-checks, each independently reportable:

    1. ``import lagrangebench`` succeeds — package is on the path and
       importable. Surfaces install-time errors.
    2. JAX still sees a CUDA GPU after the LagrangeBench import — surfaces
       version-pinning conflicts where LagrangeBench's deps would silently
       downgrade JAX to a CPU-only or broken-CUDA variant.
    3. ``python main.py mode=infer dataset=tgv2d`` runs end-to-end on the
       cloned repo — surfaces Hydra/CLI breakage and missing-data
       failure modes (rung 3 work). Captured stdout/stderr (last 2 KB
       each) is returned for diagnosis even when the command fails.
    """
    import os
    import subprocess

    # Sub-check 1: import lagrangebench
    lb_import_error: str | None = None
    lb_version: str | None = None
    lb_file: str | None = None
    try:
        import lagrangebench

        lb_version = getattr(lagrangebench, "__version__", "no_version_attr")
        lb_file = getattr(lagrangebench, "__file__", "no_file_attr")
        lb_import_ok = True
    except Exception as e:
        lb_import_ok = False
        lb_import_error = f"{type(e).__name__}: {e}"

    # Sub-check 2: JAX still sees GPU after the LagrangeBench import
    import jax
    import jaxlib

    devices = jax.devices()
    has_gpu = any(d.platform == "gpu" for d in devices)

    # Sub-check 3: try the toy infer command
    smoke_returncode: int | None = None
    smoke_stdout_tail = ""
    smoke_stderr_tail = ""
    try:
        os.chdir("/opt/lagrangebench")
        result = subprocess.run(
            ["python", "main.py", "mode=infer", "dataset=tgv2d"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        smoke_returncode = result.returncode
        smoke_stdout_tail = result.stdout[-2000:]
        smoke_stderr_tail = result.stderr[-2000:]
    except subprocess.TimeoutExpired as e:
        smoke_returncode = -1
        smoke_stderr_tail = f"TimeoutExpired after 300s: {e}"
    except Exception as e:
        smoke_returncode = -2
        smoke_stderr_tail = f"{type(e).__name__}: {e}"

    return {
        "lagrangebench_import_ok": lb_import_ok,
        "lagrangebench_version": lb_version,
        "lagrangebench_file": lb_file,
        "lagrangebench_import_error": lb_import_error,
        "jax_version_after_lb_import": jax.__version__,
        "jaxlib_version_after_lb_import": jaxlib.__version__,
        "jax_has_gpu_after_lb_import": has_gpu,
        "jax_devices_after_lb_import": [str(d) for d in devices],
        "smoke_returncode": smoke_returncode,
        "smoke_stdout_tail": smoke_stdout_tail,
        "smoke_stderr_tail": smoke_stderr_tail,
    }


@app.local_entrypoint()
def lagrangebench_smoke() -> None:
    """Fire the LagrangeBench install + toy-config smoke (rung 2)."""
    result = lagrangebench_install_smoke.remote()
    print("=== LagrangeBench install smoke (rung 2, D0-13 stage-1) ===")
    print(f"  gpu_class:                     {LAGRANGEBENCH_SMOKE_GPU_CLASS}")
    print(f"  lagrangebench_import_ok:       {result['lagrangebench_import_ok']}")
    print(f"  lagrangebench_version:         {result['lagrangebench_version']}")
    print(f"  lagrangebench_file:            {result['lagrangebench_file']}")
    if result["lagrangebench_import_error"]:
        print(f"  lagrangebench_import_error:    {result['lagrangebench_import_error']}")
    print(f"  jax_version_after_lb_import:    {result['jax_version_after_lb_import']}")
    print(f"  jaxlib_version_after_lb_import: {result['jaxlib_version_after_lb_import']}")
    print(f"  jax_has_gpu_after_lb_import:    {result['jax_has_gpu_after_lb_import']}")
    print(f"  jax_devices_after_lb_import:    {result['jax_devices_after_lb_import']}")
    print(f"  smoke_returncode:              {result['smoke_returncode']}")
    print("  --- smoke stdout (last 2 KB) ---")
    print(result["smoke_stdout_tail"] or "(empty)")
    print("  --- smoke stderr (last 2 KB) ---")
    print(result["smoke_stderr_tail"] or "(empty)")
    print("  --- verdict ---")
    install_ok = result["lagrangebench_import_ok"] and result["jax_has_gpu_after_lb_import"]
    if install_ok and result["smoke_returncode"] == 0:
        print("  -> rung 2 verdict: PASS — install clean + toy infer succeeded.")
    elif install_ok:
        print(
            f"  -> rung 2 verdict: PARTIAL — install + JAX-GPU clean, but "
            f"toy infer returned {result['smoke_returncode']}. Inspect stderr "
            f"above; commonly a missing-dataset or missing-checkpoint failure "
            f"that resolves at rung 3 (data + checkpoint download)."
        )
    else:
        print(
            "  -> rung 2 verdict: FAIL — install or post-install JAX-GPU "
            "regression. Inspect import_error / jax_devices_after_lb_import."
        )
