"""Modal entrypoint for LagrangeBench rollout generation (Day 1).

Hour-0 scope (this commit): **JAX micro-gate only** — verify
``jax.devices()`` returns at least one A100 device on the Modal A100
image. This is the hour-2 micro-gate from plan §7 / DECISIONS D0-10,
fired immediately rather than after a heavy image build.

Per D0-10: if ``jax.devices()`` does not return at least one A100, the
agent pivots to JAX-CPU read-only mode (synthetic rollouts, already
landed on this branch); the Modal-image debugging side-quest is OUT
OF SCOPE without explicit user re-authorisation. The 2h plan cap is
a ceiling, not an authorisation.

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


@app.function(image=jax_image, gpu="A100", timeout=600)
def jax_micro_gate() -> dict:
    """Hour-2 micro-gate per plan §7 / D0-10.

    Returns a dict with the device list, default backend, and a derived
    ``has_gpu`` boolean. Caller (``main``) classifies against D0-10.
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
    """Fire the JAX micro-gate; classify against D0-10."""
    result = jax_micro_gate.remote()
    print("=== JAX micro-gate verdict (D0-10) ===")
    print(f"  jax_version:     {result['jax_version']}")
    print(f"  default_backend: {result['default_backend']}")
    print(f"  device_count:    {result['device_count']}")
    print(f"  devices:         {result['devices']}")
    print(f"  has_gpu:         {result['has_gpu']}")
    if result["has_gpu"]:
        print("  -> verdict: PASS — A100 visible to JAX; proceed to step 2.")
    else:
        print("  -> verdict: FAIL — no GPU device returned; pivot per D0-10:")
        print("     JAX-CPU read-only path (synthetic rollouts already landed);")
        print("     no Modal-image debugging without explicit user re-auth.")
