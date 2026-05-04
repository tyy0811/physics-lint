"""Modal entrypoint for LagrangeBench rollout generation (Day 1).

Hour-0 scope (this commit): **JAX micro-gate only** — verify
``jax.devices()`` returns at least one CUDA-compatible GPU device on
the Modal image. This is the hour-2 micro-gate from plan §7 /
DECISIONS D0-10 (refined by D0-13 from "A100" to "any CUDA GPU"),
fired immediately rather than after a heavy image build.

Per D0-10 + D0-13: if ``jax.devices()`` does not return at least one
device with ``platform == "gpu"``, the agent pivots to JAX-CPU
read-only mode (synthetic rollouts, already landed on this branch);
the Modal-image debugging side-quest is OUT OF SCOPE without explicit
user re-authorisation. The 2h plan cap is a ceiling, not an
authorisation. GPU-class default is T4 for the micro-gate (cheapest
CUDA-JAX path; same epistemic content as A100); A10G is the planned
default for Day 1 §3.2 step 3 inference; A100 is reserved as OOM
fallback only — see DECISIONS D0-13 for the full stage-by-stage
matrix and the OOM escalation criterion.

Subsequent commits will layer in:
- LagrangeBench clone + ``pip install -e ".[dev]"``
- Dataset download (``bash download_data.sh tgv2d|dam2d``)
- Checkpoint download (gdown URLs from the LagrangeBench README)
- Inference + rollout export to ``particle_rollout.npz`` schema
- ``particle_rollout_adapter`` invocation

Run with:
    modal run external_validation/_rollout_anchors/01-lagrangebench/modal_app.py
"""

from __future__ import annotations

import modal

app = modal.App("rollout-anchors-lagrangebench")

# Hour-0 image: JAX with CUDA 12 only. Heavy installs deferred so a
# micro-gate FAIL does not waste image-build time on this side-quest.
# Mirrors plan §3.2 step 1 verbatim ("pip install -U 'jax[cuda12]' jaxlib").
jax_image = modal.Image.debian_slim(python_version="3.11").pip_install("jax[cuda12]", "jaxlib")


MICRO_GATE_GPU_CLASS = (
    "T4"  # D0-13 stage-1 default; drift-guarded by tests/test_modal_app_gpu_class.py
)


@app.function(image=jax_image, gpu=MICRO_GATE_GPU_CLASS, timeout=600)
def jax_micro_gate() -> dict:
    """Hour-2 micro-gate per plan §7 / D0-10 (refined by D0-13).

    Returns a dict with the device list, default backend, and a derived
    ``has_gpu`` boolean. Caller (``main``) classifies against the
    D0-10 + D0-13 spirit-reading: any CUDA-compatible GPU device passes;
    CPU-only return triggers the D0-10 pivot.
    """
    import jax

    devices = jax.devices()
    backend = jax.default_backend()
    has_gpu = any(d.platform == "gpu" for d in devices)
    return {
        "devices": [str(d) for d in devices],
        "default_backend": backend,
        "has_gpu": has_gpu,
        "device_count": len(devices),
        "jax_version": jax.__version__,
    }


@app.local_entrypoint()
def main() -> None:
    """Fire the JAX micro-gate; classify against D0-10 + D0-13."""
    result = jax_micro_gate.remote()
    print("=== JAX micro-gate verdict (D0-10 + D0-13) ===")
    print(f"  gpu_class:       {MICRO_GATE_GPU_CLASS}")
    print(f"  jax_version:     {result['jax_version']}")
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
