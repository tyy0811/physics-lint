"""Modal entrypoint for LagrangeBench rollout generation (Day 1).

Three functions live here, one per rung in the test ladder:

- ``jax_micro_gate`` (rung 1, plan §7 / D0-10): JAX-CUDA install
  visibility check. Returns ``jax.devices()``. Cheapest possible
  failure-mode-isolated test for the JAX-on-Modal-CUDA install path.
  Runs on T4 per D0-13 stage-1.
- ``lagrangebench_install_smoke`` (rung 2, plan §3.2 step 1):
  LagrangeBench install + toy-config inference smoke. Three sub-checks
  (import → JAX-still-sees-GPU → ``python main.py mode=infer``).
  Surfaces version-pinning conflicts where LagrangeBench's deps would
  silently downgrade JAX, and surfaces Hydra/CLI breakage and
  missing-data failure modes via captured stdout/stderr. Runs on T4
  per D0-13 stage-1.
- ``lagrangebench_rollout_p0_segnn_tgv2d`` (rung 3, plan §3.2 step 3,
  D0-15): SEGNN-10-64 checkpoint inference on TGV 2D test split.
  Downloads the checkpoint via gdown into the
  ``rollout-anchors-artifacts`` Modal Volume per the D0-14 layout,
  runs inference, captures any rollout files written. Runs on A10G
  per D0-13 stage-2.

Per D0-10 + D0-13: if ``jax.devices()`` does not return at least one
device with ``platform == "gpu"``, the agent pivots to JAX-CPU
read-only mode (synthetic rollouts, already landed on this branch);
the Modal-image debugging side-quest is OUT OF SCOPE without explicit
user re-authorisation. The 2h plan cap is a ceiling, not an
authorisation.

Subsequent commits will layer in:
- ``.npz`` materialization in particle_rollout.npz schema (rung 3.5,
  if LagrangeBench's native rollout output format is not already
  schema-conformant)
- ``particle_rollout_adapter`` invocation (rung 4)
- P1 (GNS-TGV2D) and P2/P3 rollouts (separate functions, mostly
  reusing rung-3's volume + checkpoint download infrastructure)

Run with:
    modal run external_validation/_rollout_anchors/01-lagrangebench/modal_app.py::main
    modal run external_validation/_rollout_anchors/01-lagrangebench/modal_app.py::lagrangebench_smoke
    modal run external_validation/_rollout_anchors/01-lagrangebench/modal_app.py::rollout_p0_segnn_tgv2d
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("rollout-anchors-lagrangebench")

# Hour-0 image: JAX with CUDA 12 only. Heavy installs deferred so a
# micro-gate FAIL does not waste image-build time on this side-quest.
#
# Python 3.10 (not 3.11): LagrangeBench's pyproject.toml pins
# ``python_requires = ">=3.9, <=3.11"`` which PEP 440 normalises as
# (3, 9, 0) <= python <= (3, 11, 0), excluding all 3.11.x patch releases.
# Modal's debian_slim(python_version="3.11") ships 3.11.x where x>0, which
# fails the pin. 3.10 is the highest version satisfying both the
# LagrangeBench pin and JAX 0.4.29's range. Upstream-forced — not a
# methodology choice — and therefore not pre-registered in DECISIONS.md;
# this comment is the audit trail. If LagrangeBench relaxes the pin
# upstream (e.g., to ``<3.12``), bumping to 3.11 is a clean change.
#
# jax[cuda12]==0.4.29 + jaxlib==0.4.29: matched-stack pin to align
# jax_image's framework with lagrangebench_image's pinned framework.
# Without this pin, jax_image resolves to the latest 3.10-compatible
# JAX (was 0.6.2), and the plugin/pjrt packages installed alongside
# (also 0.6.x) carry a different PJRT API version (0.70) than the
# framework jax 0.4.29 ends up using after LagrangeBench's
# pin-driven downgrade in lagrangebench_image (PJRT API 0.54). The
# resulting framework/plugin mismatch breaks GPU access at runtime:
# "INVALID_ARGUMENT: Mismatched PJRT plugin PJRT API version (0.70)
# and framework PJRT API version 0.54". Pinning jax[cuda12] to the
# LagrangeBench-required version ensures plugin/pjrt match the
# framework from the start in BOTH images. Discovered via rung 2
# sub-check 2 FAIL at 60cacea; this commit applies option (i) from
# the discussion (matched-stack pin); option (ii) (bundled-cuda
# jaxlib==0.4.29+cuda12.cudnn89 from JAX's release storage) is the
# fallback if pip cannot resolve jax[cuda12]==0.4.29 here.
jax_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "jax[cuda12]==0.4.29",
    "jaxlib==0.4.29",
)


MICRO_GATE_GPU_CLASS = (
    "T4"  # D0-13 stage-1 default; drift-guarded by tests/test_modal_app_gpu_class.py
)


def _package_version(name: str) -> str:
    """Return the installed version of ``name`` or ``"<not_installed>"``.

    Used inside Modal functions to capture cross-image dependency stack
    identity for the audit trail. ``importlib.metadata`` is stdlib and
    available on both the local entrypoint side and inside Modal
    containers, so this helper does not require any extra image deps.
    """
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "<not_installed>"


def _capture_pjrt_api_version() -> str:
    """Best-effort capture of the framework PJRT API version observed at runtime.

    PJRT API version is the load-bearing identity for plugin/framework
    compatibility — rung 2 sub-check 2's first FAIL surfaced at this
    axis specifically (framework 0.54 vs plugin 0.70). Package versions
    (jax/jaxlib/plugin/pjrt) are reliable proxies under matched-stack
    installs, but the runtime PJRT API value is what determines
    compatibility, so it's worth capturing directly when possible.

    JAX does not expose the PJRT API version through a stable public
    API; this helper probes several known internal paths and falls back
    to ``"<unknown>"`` if all probes fail. Bounded best-effort — keeps
    the audit trail useful when introspection works without being a
    silent failure when it doesn't.
    """
    import importlib

    candidates = (
        ("jax._src.lib.xla_client", "pjrt_api_version"),
        ("jax._src.lib.xla_client", "_pjrt_api_version"),
        ("jaxlib.xla_client", "pjrt_api_version"),
        ("jaxlib.xla_extension", "pjrt_api_version"),
    )
    for module_path, attr in candidates:
        try:
            module = importlib.import_module(module_path)
            value = getattr(module, attr, None)
            if value is not None:
                return str(value)
        except Exception:
            continue
    return "<unknown>"


@app.function(image=jax_image, gpu=MICRO_GATE_GPU_CLASS, timeout=600)
def jax_micro_gate() -> dict:
    """Hour-2 micro-gate per plan §7 / D0-10 (refined by D0-13).

    Returns a dict with the device list, default backend, and a derived
    ``has_gpu`` boolean. Caller (``main``) classifies against the
    D0-10 + D0-13 spirit-reading: any CUDA-compatible GPU device passes;
    CPU-only return triggers the D0-10 pivot. Also returns the resolved
    jax + jaxlib + cuda12 plugin + cuda12 pjrt versions so the audit
    trail captures *which* JAX-CUDA stack passed the gate (matters when
    the Python pin or upstream constraints change which versions pip
    resolves to, and when downstream images downgrade jax/jaxlib but
    leave the plugin/pjrt versions unchanged).
    """
    import jax
    import jaxlib

    # jax.devices() raises RuntimeError on plugin/framework PJRT API
    # mismatch (rung 2's first FAIL surfaced at this shape). Wrap to
    # let the verdict ladder report cleanly through the structured
    # return rather than via uncaught exception.
    gpu_init_error: str | None = None
    try:
        devices = jax.devices()
        backend = jax.default_backend()
        has_gpu = any(d.platform == "gpu" for d in devices)
        device_strs = [str(d) for d in devices]
        device_count = len(devices)
    except RuntimeError as e:
        gpu_init_error = f"jax_devices_raised: {type(e).__name__}: {e}"
        backend = "<errored>"
        has_gpu = False
        device_strs = []
        device_count = 0
    return {
        "devices": device_strs,
        "default_backend": backend,
        "has_gpu": has_gpu,
        "device_count": device_count,
        "gpu_init_error": gpu_init_error,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "jax_cuda12_plugin_version": _package_version("jax-cuda12-plugin"),
        "jax_cuda12_pjrt_version": _package_version("jax-cuda12-pjrt"),
        "pjrt_api_version": _capture_pjrt_api_version(),
    }


@app.local_entrypoint()
def main() -> None:
    """Fire the JAX micro-gate; classify against D0-10 + D0-13."""
    result = jax_micro_gate.remote()
    print("=== JAX micro-gate verdict (D0-10 + D0-13) ===")
    print(f"  gpu_class:                  {MICRO_GATE_GPU_CLASS}")
    print(f"  jax_version:                {result['jax_version']}")
    print(f"  jaxlib_version:             {result['jaxlib_version']}")
    print(f"  jax_cuda12_plugin_version:  {result['jax_cuda12_plugin_version']}")
    print(f"  jax_cuda12_pjrt_version:    {result['jax_cuda12_pjrt_version']}")
    print(f"  pjrt_api_version:           {result['pjrt_api_version']}")
    print(f"  default_backend:            {result['default_backend']}")
    print(f"  device_count:               {result['device_count']}")
    print(f"  devices:                    {result['devices']}")
    print(f"  has_gpu:                    {result['has_gpu']}")
    if result["gpu_init_error"]:
        print(f"  gpu_init_error:             {result['gpu_init_error']}")
    if result["has_gpu"]:
        print(
            f"  -> verdict: PASS — CUDA GPU ({MICRO_GATE_GPU_CLASS}) visible "
            "to JAX; proceed to step 2."
        )
    else:
        print("  -> verdict: FAIL — no GPU device returned; pivot per D0-10:")
        print("     JAX-CPU read-only path (synthetic rollouts already landed);")
        print("     no Modal-image debugging without explicit user re-auth.")
    # Build-phase observation: under Modal's eager-build behaviour, reaching
    # this print means the lagrangebench_image build also succeeded as part
    # of the unified build phase. This is NOT the rung-1 verdict (which is
    # printed above and concerns only jax_image's runtime path); it is a
    # separate audit-trail record so future-me reading the logs knows that
    # the rung-2 image was buildable at the moment rung-1 PASSed, even
    # though rung-2's smoke function had not yet been invoked.
    print(
        "  build-phase observed: lagrangebench_image built clean (Modal "
        "eager-build corollary; runtime verdict pending rung 2)."
    )


# Rung-2 image: jax_image + LagrangeBench clone + ``pip install -e ".[dev]"``.
# Layered on top of jax_image so the cached JAX layer (~95s build) is reused.
# `--depth 1` keeps the clone small; the working tree only needs HEAD.
#
# --extra-index-url for torch CPU wheels: LagrangeBench pins
# ``torch==2.3.1+cpu`` (a CPU-only build hosted on PyTorch's wheel index,
# not Modal's default PyPI mirror). The ``+cpu`` local-version-segment
# suffix is PyPA's mechanism for "this version comes from a different
# index"; --extra-index-url is the canonical response. We do NOT pass
# --ignore-requires-python here — that would defeat the exact mechanism
# upstream uses to communicate "tested only on these interpreters".
lagrangebench_image = jax_image.apt_install("git").run_commands(
    "git clone --depth 1 https://github.com/tumaer/lagrangebench.git /opt/lagrangebench",
    "cd /opt/lagrangebench && pip install"
    " --extra-index-url https://download.pytorch.org/whl/cpu"
    " -e '.[dev]'",
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

    # Sub-check 2: JAX still sees GPU after the LagrangeBench import.
    # jax.devices() raises RuntimeError on plugin/framework PJRT API
    # mismatch (rung 2's first FAIL at 60cacea surfaced at this shape).
    # Wrap to let the verdict ladder report cleanly through the
    # structured return rather than via uncaught exception that exits
    # the function before the verdict logic runs.
    import jax
    import jaxlib

    gpu_init_error: str | None = None
    try:
        devices = jax.devices()
        has_gpu = any(d.platform == "gpu" for d in devices)
        device_strs = [str(d) for d in devices]
    except RuntimeError as e:
        gpu_init_error = f"jax_devices_raised: {type(e).__name__}: {e}"
        has_gpu = False
        device_strs = []

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
        "jax_cuda12_plugin_version_after_lb_import": _package_version("jax-cuda12-plugin"),
        "jax_cuda12_pjrt_version_after_lb_import": _package_version("jax-cuda12-pjrt"),
        "pjrt_api_version_after_lb_import": _capture_pjrt_api_version(),
        "jax_has_gpu_after_lb_import": has_gpu,
        "jax_devices_after_lb_import": device_strs,
        "gpu_init_error_after_lb_import": gpu_init_error,
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
    print(f"  jax_version_after_lb_import:               {result['jax_version_after_lb_import']}")
    print(
        f"  jaxlib_version_after_lb_import:            {result['jaxlib_version_after_lb_import']}"
    )
    print(
        f"  jax_cuda12_plugin_version_after_lb_import: "
        f"{result['jax_cuda12_plugin_version_after_lb_import']}"
    )
    print(
        f"  jax_cuda12_pjrt_version_after_lb_import:   "
        f"{result['jax_cuda12_pjrt_version_after_lb_import']}"
    )
    print(
        f"  pjrt_api_version_after_lb_import:          {result['pjrt_api_version_after_lb_import']}"
    )
    # Cross-image matched-stack assertion: the load-bearing claim that
    # explains why sub-check 2 PASSes (or, if it FAILs, why) is whether
    # all four jax-cuda12 packages observed inside lagrangebench_image
    # agree on a single version. Capturing this as a derived line in
    # the verdict log preserves the claim in the audit trail rather
    # than leaving it implicit in the four version lines above.
    _stack_versions = [
        result["jax_version_after_lb_import"],
        result["jaxlib_version_after_lb_import"],
        result["jax_cuda12_plugin_version_after_lb_import"],
        result["jax_cuda12_pjrt_version_after_lb_import"],
    ]
    if len(set(_stack_versions)) == 1:
        print(
            f"  cross-image stack:                         MATCHED at "
            f"{_stack_versions[0]} (jax + jaxlib + plugin + pjrt agree)"
        )
    else:
        print(
            f"  cross-image stack:                         MISMATCHED — "
            f"jax={_stack_versions[0]}, jaxlib={_stack_versions[1]}, "
            f"plugin={_stack_versions[2]}, pjrt={_stack_versions[3]}"
        )
    print(f"  jax_has_gpu_after_lb_import:               {result['jax_has_gpu_after_lb_import']}")
    print(f"  jax_devices_after_lb_import:               {result['jax_devices_after_lb_import']}")
    if result["gpu_init_error_after_lb_import"]:
        print(
            f"  gpu_init_error_after_lb_import:            "
            f"{result['gpu_init_error_after_lb_import']}"
        )
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


# Rung-3 image: lagrangebench_image + gdown (Google-Drive checkpoint
# download) + unzip (unpack zipped checkpoint archives) + wget (used by
# upstream's download_data.sh to fetch datasets from Zenodo). Layered
# on top of lagrangebench_image so cached layers (jax + lagrangebench
# install + torch CPU) are reused; this is a thin extension.
#
# wget specifically: discovered missing at rung-3's first fire (c2e3bc3)
# when bash download_data.sh tgv_2d /vol/... returned 1 because wget
# wasn't on the image. download_data.sh has no `set -e`, so wget's
# "command not found" doesn't halt the script — it continues to the
# python3 unzip step which fails on the non-existent zipfile. The
# resulting returncode 1 carried no specific signal about wget being
# the cause; we needed to pull the script from upstream to confirm.
# Adding wget to apt_install + the structured returncode capture in
# the function below means the next class of similar failures surfaces
# cleanly via the manifest without needing to pull upstream sources.
#
# add_local_file shipping the conversion module (rung 3.5, D0-15
# amendment 4): the post-inference conversion from LagrangeBench's
# native pkl shape to SCHEMA.md §1 particle_rollout.npz lives in the
# pure-Python module ``_harness/lagrangebench_pkl_to_npz.py``. Adding
# it via add_local_file (rather than building it into the image)
# means edits to the conversion don't trigger an image rebuild — the
# file is sent over the wire each function invocation. Same
# reuse-across-rungs reasoning as the harness package itself
# (P1 GNS-TGV2D will reuse the conversion verbatim; P2/P3 likewise).
_HARNESS_CONVERSION_MODULE = (
    Path(__file__).resolve().parent.parent / "_harness" / "lagrangebench_pkl_to_npz.py"
)
_REMOTE_HARNESS_DIR = "/opt/physics_lint_harness"
rollout_image = (
    lagrangebench_image.apt_install("unzip", "wget")
    .pip_install("gdown")
    .add_local_file(
        str(_HARNESS_CONVERSION_MODULE),
        f"{_REMOTE_HARNESS_DIR}/lagrangebench_pkl_to_npz.py",
    )
)


# Modal Volume: persistent storage for rollout-anchors artifacts. Layout
# pre-registered in DECISIONS D0-14:
#   /vol/checkpoints/<provider>/<name>/best/   (unzipped checkpoint dirs)
#   /vol/datasets/<provider>/<dataset_name>/   (downloaded datasets)
#   /vol/rollouts/<provider>/<name>_<git_sha>.<ext>  (generated rollouts)
rollout_volume = modal.Volume.from_name("rollout-anchors-artifacts", create_if_missing=True)


# LagrangeBench checkpoint catalogue: gdown file IDs from the README's
# pretrained-models table (https://github.com/tumaer/lagrangebench).
# URLs are code artifacts (they change when upstream reorganises),
# so they live here rather than in DECISIONS.md prose. The methodology
# decision is "we use the URLs LagrangeBench upstream publishes"
# (DECISIONS D0-15); the URLs themselves live with the code that uses
# them.
LAGRANGEBENCH_CHECKPOINT_GDOWN_IDS: dict[str, str] = {
    # 2D
    "segnn_tgv2d": "1llGtakiDmLfarxk6MUAtqj6sLleMQ7RL",
    "gns_tgv2d": "19TO4PaFGcryXOFFKs93IniuPZKEcaJ37",
    "segnn_rpf2d": "108dZVWs2qxAvKiboeEBW-nIcv-aslhYP",
    "gns_rpf2d": "1uYusVlP1ykUNuw58vo7Wss-xyTMmopAn",
    "segnn_ldc2d": "1D_wgs2pD9pTXoJK76yi-R0K2tY_T6lPn",
    "gns_ldc2d": "1JvdsW0H6XrgC2_cwV3pP66cAm9j1-AXc",
    "segnn_dam2d": "1_6rHxK81vzrdIMPtJ7rIkeoUgsTeKmSn",
    "gns_dam2d": "16bJz3VfSMxOG1II8kCg5DlzGhjvdip2p",
    # 3D (P3 stretch only per plan §3.1)
    "segnn_tgv3d": "1ivJnHTgfbQ0IJujc5O0CUoQNiGU4zi_d",
    "gns_tgv3d": "1DEkXxrebS9eyLSMlc_ztHrqlh29NgLXC",
    "segnn_rpf3d": "1Qczh3Z_z0grTuRuPDHyiYLzV1zg7Liz9",
    "gns_rpf3d": "1yo-qgShLd1sgS1u5zkMXdJvhuPBwEQQE",
    "segnn_ldc3d": "1ZIg7FXc1l3C4ekc9WvVvjHEl5KKxOA_U",
    "gns_ldc3d": "1b3IIkxk5VcWiT8Oyqg1wex8-ZfJv2g_v",
}


# Map LagrangeBench dataset short names to the directory name produced by
# ``download_data.sh`` (matches each checkpoint config's ``dataset.src``).
# Discovered via inspection of upstream configs/<dataset>/base.yaml.
LAGRANGEBENCH_DATASET_DIRS: dict[str, str] = {
    "tgv_2d": "2D_TGV_2500_10kevery100",
    # extend as P1+ work scales
}


ROLLOUT_GENERATION_GPU_CLASS = (
    "A10G"  # D0-13 stage-2 default; drift-guarded by tests/test_modal_app_gpu_class.py
)


# Cumulative upstream-compatibility patches required to reach a clean
# rung-3 verdict. Each entry captures (layer, methodology-relevance,
# git ref, summary). The structured-record framing replaces "patch
# count feels like a lot" with "N library-internal-regression
# patches with load-bearing methodology relevance" — actionable,
# auditable, surface-able in the rung-3 verdict log at a glance.
#
# Methodology relevance is defined relative to the physics-lint
# harness (the consumer of rollout artifacts): "load-bearing" means
# the patch affects what the harness reads (positions, velocities,
# KE, particle types, dt, domain box, metadata); "non-load-bearing"
# means the patch is in a layer the harness doesn't touch.
#
# Discipline going forward: when a new patch is needed and falls
# into the "library-internal regression with load-bearing methodology
# relevance" bucket, surface immediately for higher-level review
# rather than apply a workaround. The other buckets (infrastructure,
# config, library-internal-non-load-bearing) are bounded and well-
# understood; the load-bearing-library-internal bucket is the signal
# that the cumulative-patch posture has shifted from "stand up a
# 2026 JAX-CUDA-Modal stack" to "patch around upstream's actual
# correctness regressions in our consumed surface."
UPSTREAM_COMPAT_PATCHES: list[dict[str, str]] = [
    {
        "ref": "physics-lint 9276596",
        "layer": "infrastructure",
        "relevance": "non-load-bearing",
        "summary": (
            "Python pin 3.11.x -> 3.10 (LagrangeBench pyproject "
            "<=3.11 normalises to <=3.11.0 under PEP 440, excluding "
            "Modal debian_slim:3.11's 3.11.x patch level)"
        ),
    },
    {
        "ref": "physics-lint 32b6b79",
        "layer": "infrastructure",
        "relevance": "non-load-bearing",
        "summary": (
            "torch CPU index --extra-index-url for torch==2.3.1+cpu "
            "(LagrangeBench pins +cpu local-version-segment hosted on "
            "PyTorch's wheel index, not Modal's default PyPI mirror)"
        ),
    },
    {
        "ref": "physics-lint 91d3ce7",
        "layer": "infrastructure",
        "relevance": "non-load-bearing",
        "summary": (
            "jax[cuda12]==0.4.29 + jaxlib==0.4.29 matched-stack pin "
            "(without it, jax_image's plugin/pjrt at 0.6.x carry PJRT "
            "API 0.70 vs LagrangeBench's pinned jax 0.4.29 framework "
            "PJRT API 0.54; mismatch breaks GPU access)"
        ),
    },
    {
        "ref": "physics-lint 63e33e9",
        "layer": "config",
        "relevance": "non-load-bearing",
        "summary": (
            "dataset.name=tgv2d CLI arg (older SEGNN-TGV2D checkpoint "
            "config doesn't carry dataset.name; current LagrangeBench "
            "runner.py:148 reads cfg.dataset.name with no auto-inference "
            "fallback)"
        ),
    },
    {
        "ref": "physics-lint 7254193",
        "layer": "library-internal",
        "relevance": "non-load-bearing",
        "summary": (
            "Sinkhorn metric drop via eval.infer.metrics=[mse,e_kin] "
            "(ott-jax 0.4.9 self-inconsistency: sinkhorn_divergence "
            "forwards sinkhorn_kwargs to Geometry.__init__() which "
            "doesn't accept it; Sinkhorn is not consumed by any "
            "in-scope physics-lint rule)"
        ),
    },
    {
        "ref": "physics-lint 9f7df91",
        "layer": "config",
        "relevance": "non-load-bearing",
        "summary": (
            "eval.infer.n_trajs=20 (replacing plan's eval.n_trajs=20): "
            "current upstream rollout.py:351 reads "
            "cfg.eval.infer.n_trajs in infer mode, NOT the plan's "
            "top-level cfg.eval.n_trajs (which is a no-op for our "
            "mode); without this fix the checkpoint's default 200 "
            "trajectories was in effect, causing the 7254193 "
            "subprocess timeout. The pre-registered value (20) is "
            "unchanged; only the CLI key path updates"
        ),
    },
    {
        "ref": "physics-lint 334cafd",
        "layer": "config",
        "relevance": "non-load-bearing",
        "summary": (
            "LagrangeBench's eval.rollout_dir defaults to null "
            "(metrics-only mode); enabling rollout persistence "
            "requires both eval.rollout_dir=<path> (top-level per "
            "rollout.py:319) and eval.infer.out_type=pkl (nested "
            "per rollout.py:396), reflecting LB's split config "
            "layout. One feature-toggle, two CLI args at two "
            "different config-key paths because of LB's layout"
        ),
    },
]


@app.function(
    image=rollout_image,
    gpu=ROLLOUT_GENERATION_GPU_CLASS,
    volumes={"/vol": rollout_volume},
    timeout=3600,
)
def lagrangebench_rollout_p0_segnn_tgv2d(git_sha: str, full_git_sha: str) -> dict:
    """Rung 3 + 3.5 P0: SEGNN-10-64 inference + SCHEMA.md npz materialization (D0-15).

    Steps:
    1. Ensure SEGNN-TGV2D checkpoint is unpacked under
       /vol/checkpoints/lagrangebench/segnn_tgv2d/best/. If missing,
       gdown the zip from the LagrangeBench README's pretrained-models
       table, unzip, commit the volume.
    2. Ensure TGV2D dataset is unpacked under
       /vol/datasets/lagrangebench/2D_TGV_2500_10kevery100/. If missing,
       run ``bash download_data.sh tgv_2d /vol/datasets/lagrangebench/``
       from /opt/lagrangebench, commit the volume. (LagrangeBench's
       ``mode=infer`` also auto-downloads if dataset.src is missing
       per the README; explicit download here makes the volume layout
       deterministic and avoids the auto-download running inside the
       inference subprocess on every invocation.)
    3. Run ``python main.py mode=infer eval.test=True
       load_ckp=<ckpt>/best eval.n_rollout_steps=100
       eval.infer.n_trajs=20 dataset.src=<dataset>
       dataset.name=tgv2d eval.infer.metrics=[mse,e_kin]
       eval.rollout_dir=<rollout_subdir>
       eval.infer.out_type=pkl seed=0`` per D0-15 (incl. amendment 4).
       The two-arg ``eval.rollout_dir`` (top-level) +
       ``eval.infer.out_type=pkl`` (nested) toggle persists native
       per-trajectory rollout pkls into the volume; ``seed=0``
       pre-registers the inference seed so the rung-4 regeneration
       test asserts byte-identical contents-SHA-256.
    4. Capture lagrangebench_sha (``git -C /opt/lagrangebench
       rev-parse HEAD``) for the rollout metadata audit trail.
    5. Convert each ``rollout_*.pkl`` LagrangeBench wrote to a
       SCHEMA.md §1 ``particle_rollout_traj{j:02d}.npz`` via the
       ``lagrangebench_pkl_to_npz`` harness module (added to the
       image via ``add_local_file``; pure-Python; the five
       validation assertions enumerated in DECISIONS.md D0-15
       amendment 4 fire here, not at rung 4 read time).
    6. Walk /opt/lagrangebench (and /vol/rollouts) for any files
       written during inference. Compute SHA-256 of each. Return a
       manifest of (path, size, sha256) tuples plus stdout/stderr
       tails so future-me can reconstruct what got written.
    """
    import hashlib
    import os
    import subprocess
    import time

    manifest: dict = {
        "git_sha": git_sha,
        "full_git_sha": full_git_sha,
        "lagrangebench_sha": None,
        "inference_seed": 0,  # pre-registered per D0-15 amendment 4
        "aborted_at_step": None,
        "checkpoint_download_skipped": False,
        "checkpoint_gdown_returncode": None,
        "checkpoint_gdown_stderr_tail": "",
        "checkpoint_unzip_returncode": None,
        "checkpoint_unzip_stderr_tail": "",
        "dataset_download_skipped": False,
        "dataset_download_returncode": None,
        "dataset_download_stderr_tail": "",
        "inference_returncode": None,
        "inference_wall_seconds": None,
        "inference_stdout_tail": "",
        "inference_stderr_tail": "",
        "rollout_subdir": None,
        # Rung 3.5 conversion-step fields (D0-15 amendment 4):
        "conversion_attempted": False,
        "conversion_returncode": None,  # 0 success, 1 failure
        "conversion_error": None,
        "converted_npz_paths": [],
        "files_written": [],  # list of {path, size, sha256}
    }

    # All subprocess.run calls in this function use check=False (no
    # automatic raise) and capture stdout/stderr explicitly. Same
    # discipline as the rung-2 try/except fix at 91d3ce7: structured
    # failure-reporting through the manifest, not via uncaught
    # exception that bypasses the verdict logic. Each step records its
    # returncode + stderr tail; on non-zero, manifest["aborted_at_step"]
    # is set and the function returns early so downstream steps don't
    # run on a broken predecessor.

    # Step 1: checkpoint download (gdown) + unzip
    ckpt_root = "/vol/checkpoints/lagrangebench/segnn_tgv2d"
    ckpt_dir = f"{ckpt_root}/best"
    if os.path.isdir(ckpt_dir):
        manifest["checkpoint_download_skipped"] = True
    else:
        os.makedirs(ckpt_root, exist_ok=True)
        zip_path = f"{ckpt_root}/segnn_tgv2d.zip"
        gdown_id = LAGRANGEBENCH_CHECKPOINT_GDOWN_IDS["segnn_tgv2d"]

        gdown_proc = subprocess.run(
            ["gdown", gdown_id, "-O", zip_path],
            capture_output=True,
            text=True,
            timeout=600,
        )
        manifest["checkpoint_gdown_returncode"] = gdown_proc.returncode
        manifest["checkpoint_gdown_stderr_tail"] = gdown_proc.stderr[-2000:]
        if gdown_proc.returncode != 0:
            manifest["aborted_at_step"] = "checkpoint_gdown"
            return manifest

        unzip_proc = subprocess.run(
            ["unzip", "-o", zip_path, "-d", ckpt_root],
            capture_output=True,
            text=True,
            timeout=300,
        )
        manifest["checkpoint_unzip_returncode"] = unzip_proc.returncode
        manifest["checkpoint_unzip_stderr_tail"] = unzip_proc.stderr[-2000:]
        if unzip_proc.returncode != 0:
            manifest["aborted_at_step"] = "checkpoint_unzip"
            return manifest

        # Some checkpoint zips unpack into a subdir named like the zip;
        # the README's load_ckp= pattern expects /best directly under
        # ckpt_root. Walk one level if needed.
        if not os.path.isdir(ckpt_dir):
            for entry in os.listdir(ckpt_root):
                candidate = os.path.join(ckpt_root, entry, "best")
                if os.path.isdir(candidate):
                    os.rename(candidate, ckpt_dir)
                    break
        rollout_volume.commit()

    # Step 2: dataset download via upstream's download_data.sh
    dataset_root = "/vol/datasets/lagrangebench"
    dataset_dir = f"{dataset_root}/{LAGRANGEBENCH_DATASET_DIRS['tgv_2d']}"
    if os.path.isdir(dataset_dir):
        manifest["dataset_download_skipped"] = True
    else:
        os.makedirs(dataset_root, exist_ok=True)
        os.chdir("/opt/lagrangebench")
        ds_proc = subprocess.run(
            ["bash", "download_data.sh", "tgv_2d", dataset_root + "/"],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        manifest["dataset_download_returncode"] = ds_proc.returncode
        # download_data.sh writes wget progress to stderr even on success;
        # capture both streams so we can distinguish "wget not found" from
        # "wget ran but Zenodo timed out" without needing a re-fire.
        manifest["dataset_download_stderr_tail"] = (
            (ds_proc.stdout[-1000:] + "\n--- stderr ---\n" + ds_proc.stderr[-1500:])
            if ds_proc.stderr
            else ds_proc.stdout[-2000:]
        )
        if ds_proc.returncode != 0:
            manifest["aborted_at_step"] = "dataset_download"
            return manifest
        rollout_volume.commit()

    # Step 3: inference
    # Per-checkpoint per-git_sha subdirectory under the D0-14 layout.
    # eval.rollout_dir (top-level per rollout.py:319) writes LB's
    # native rollout_*.pkl files here; the rung-3.5 conversion step
    # below reads them and writes particle_rollout_traj{j:02d}.npz
    # files alongside per the D0-15 amendment 4 directory layout.
    rollout_dir_root = "/vol/rollouts/lagrangebench"
    rollout_subdir = f"{rollout_dir_root}/segnn_tgv2d_{git_sha}"
    os.makedirs(rollout_subdir, exist_ok=True)
    manifest["rollout_subdir"] = rollout_subdir

    os.chdir("/opt/lagrangebench")
    inf_args = [
        "python",
        "main.py",
        "mode=infer",
        "eval.test=True",
        f"load_ckp={ckpt_dir}",
        "eval.n_rollout_steps=100",
        # eval.infer.n_trajs (NOT eval.n_trajs): current upstream
        # rollout.py:351 reads cfg.eval.infer.n_trajs in infer mode.
        # The plan's literal eval.n_trajs=20 was a no-op for our
        # mode and left the checkpoint default of 200 in effect,
        # causing the 2400-sec timeout at 7254193. Pre-registered in
        # DECISIONS D0-15 (amendment 3); methodology-relevant value
        # (20) is unchanged from the plan, only the CLI key updates.
        "eval.infer.n_trajs=20",
        f"dataset.src={dataset_dir}",
        # dataset.name=tgv2d (no underscore): required by current
        # upstream runner.py:148 — the publicly distributed SEGNN-
        # TGV2D checkpoint's bundled config doesn't carry
        # dataset.name (older runner inferred from path; current
        # HEAD doesn't). Pre-registered in DECISIONS D0-15
        # (amendment 1). Valid name space per upstream data.py:
        # {tgv2d, tgv3d, rpf2d, rpf3d, ldc2d, ldc3d, dam2d}.
        "dataset.name=tgv2d",
        # eval.infer.metrics=[mse,e_kin]: drops Sinkhorn divergence
        # from the per-rollout evaluation list. ott-jax 0.4.9 is
        # internally self-inconsistent — sinkhorn_divergence
        # forwards sinkhorn_kwargs to Geometry.__init__() which
        # doesn't accept it. Default eval.infer.metrics is
        # [mse, sinkhorn, e_kin]; this drops Sinkhorn. Sinkhorn is
        # NOT consumed by any in-scope physics-lint rule, so the
        # methodology cost of the drop is zero (DECISIONS D0-15
        # amendment 2 carries the full reasoning).
        "eval.infer.metrics=[mse,e_kin]",
        # Rung-3.5 rollout-persistence enablement (DECISIONS D0-15
        # amendment 4). Two args at two different config-key paths
        # because of LagrangeBench's split layout: eval.rollout_dir
        # is top-level (rollout.py:319 reads defaults.eval.rollout_dir)
        # while eval.infer.out_type is nested (rollout.py:396 reads
        # cfg_eval_infer.out_type). LB's default out_type=none
        # discards rollouts after metric computation; we need
        # out_type=pkl to persist them for the conversion step
        # below.
        f"eval.rollout_dir={rollout_subdir}",
        "eval.infer.out_type=pkl",
        # Pre-registered seed=0 per DECISIONS D0-15 amendment 4.
        # The rung-4 regeneration test asserts that re-invoking with
        # the same checkpoint and seed produces particle_rollout_*.npz
        # with bit-identical contents-SHA-256, validating that the
        # full pipeline (LagrangeBench inference -> pkl -> npz
        # conversion) is end-to-end deterministic.
        "seed=0",
    ]
    # subprocess.run wrapped in try/except for TimeoutExpired and
    # general Exception (coverage extension; previously missed during
    # the bf3741d structured-manifest refactor — the third rung-3
    # attempt at 7254193 hit TimeoutExpired which propagated as
    # uncaught traceback rather than landing in the manifest). Same
    # discipline as the rung-2 smoke function and the rung-3
    # checkpoint/dataset steps.
    inf_start = time.monotonic()
    try:
        inf_proc = subprocess.run(
            inf_args,
            capture_output=True,
            text=True,
            timeout=2400,
        )
        manifest["inference_returncode"] = inf_proc.returncode
        manifest["inference_stdout_tail"] = inf_proc.stdout[-3000:]
        manifest["inference_stderr_tail"] = inf_proc.stderr[-3000:]
    except subprocess.TimeoutExpired as e:
        manifest["inference_returncode"] = -1
        manifest["inference_stdout_tail"] = (
            e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        )[-3000:]
        manifest["inference_stderr_tail"] = (
            f"TimeoutExpired after {e.timeout}s. Stderr tail: "
            + ((e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or ""))[-2500:])
        )
    except Exception as e:
        manifest["inference_returncode"] = -2
        manifest["inference_stderr_tail"] = f"Uncaught {type(e).__name__}: {e}"
    manifest["inference_wall_seconds"] = round(time.monotonic() - inf_start, 1)
    if manifest["inference_returncode"] != 0:
        manifest["aborted_at_step"] = "inference"
        # Still walk for files written (partial rollouts can be
        # informative); don't return early here.

    # Step 4: capture lagrangebench_sha for the rollout metadata
    # audit trail. The image's /opt/lagrangebench is a --depth 1
    # clone, so .git is present and `git rev-parse HEAD` is the
    # source-of-truth identity for the upstream code that produced
    # the rollouts in this run.
    lb_sha_proc = subprocess.run(
        ["git", "-C", "/opt/lagrangebench", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if lb_sha_proc.returncode == 0:
        manifest["lagrangebench_sha"] = lb_sha_proc.stdout.strip()
    else:
        manifest["lagrangebench_sha"] = f"<git_rev_parse_failed_rc={lb_sha_proc.returncode}>"

    # Step 5: rung-3.5 conversion — LagrangeBench rollout_*.pkl ->
    # SCHEMA.md §1 particle_rollout_traj{j:02d}.npz. Only attempt if
    # inference returned 0 (partial rollouts may have malformed pkls
    # that surface as conversion failures, which would obscure the
    # real inference-step failure). Conversion failures are recorded
    # in manifest fields rather than raised — the LB native pkls are
    # still on volume and the failure is reproducible from the
    # captured manifest.
    if manifest["inference_returncode"] == 0:
        manifest["conversion_attempted"] = True
        try:
            import sys as _sys

            if _REMOTE_HARNESS_DIR not in _sys.path:
                _sys.path.insert(0, _REMOTE_HARNESS_DIR)
            from lagrangebench_pkl_to_npz import (
                RolloutMetadata,
                convert_rollout_dir,
            )

            converted = convert_rollout_dir(
                rollout_dir=rollout_subdir,
                dataset_metadata_path=os.path.join(dataset_dir, "metadata.json"),
                ckpt_dir=ckpt_dir,
                metadata=RolloutMetadata(
                    git_sha=full_git_sha,
                    lagrangebench_sha=manifest["lagrangebench_sha"],
                    dataset="tgv2d",
                    model="segnn",
                    seed=manifest["inference_seed"],
                    framework="jax+haiku",
                    framework_version=_package_version("jax"),
                ),
            )
            manifest["conversion_returncode"] = 0
            manifest["converted_npz_paths"] = [str(p) for p in converted]
        except Exception as e:
            manifest["conversion_returncode"] = 1
            manifest["conversion_error"] = f"{type(e).__name__}: {e}"
            manifest["aborted_at_step"] = "conversion"

    # Step 6: walk for files written
    candidate_roots = [
        "/opt/lagrangebench/outputs",
        "/opt/lagrangebench/rollouts",
        "/opt/lagrangebench/wandb",
        rollout_subdir,
    ]
    for root in candidate_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                try:
                    size = os.path.getsize(fp)
                    if size > 50 * 1024 * 1024:
                        manifest["files_written"].append(
                            {"path": fp, "size": size, "sha256": "<skipped_large>"}
                        )
                        continue
                    h = hashlib.sha256()
                    with open(fp, "rb") as f:
                        for chunk in iter(lambda: f.read(65536), b""):
                            h.update(chunk)
                    manifest["files_written"].append(
                        {"path": fp, "size": size, "sha256": h.hexdigest()}
                    )
                except OSError as e:
                    manifest["files_written"].append(
                        {"path": fp, "size": -1, "sha256": f"<read_error:{e}>"}
                    )

    rollout_volume.commit()
    return manifest


# P1 GNS-TGV2D: structurally identical to P0 above, with checkpoint
# name + model name + rollout-subdir prefix replaced. Deliberate
# minimal-edit copy rather than parameterized refactor — P0 is
# already well-tested (D0-16 PASS at f75e22d8dd; D0-17/amendment-1
# regen+convert PASS at 8c3d080397/5857144) and refactoring it
# carries drift risk (e.g., a parameterizing edit might miss a
# hardcoded "segnn" somewhere). When P2 lands, refactor both into a
# shared helper. For now, two functions; differences enumerated:
#
#   - ckpt_root         segnn_tgv2d -> gns_tgv2d
#   - zip_path basename segnn_tgv2d.zip -> gns_tgv2d.zip
#   - LAGRANGEBENCH_CHECKPOINT_GDOWN_IDS["segnn_tgv2d"] -> [...]["gns_tgv2d"]
#   - rollout_subdir    segnn_tgv2d_<git_sha> -> gns_tgv2d_<git_sha>
#   - RolloutMetadata.model "segnn" -> "gns"
#
# All other logic (image, GPU class, volume, image-build cache,
# conversion module, all CLI args including dataset.name=tgv2d,
# patches-required field) is identical and shared by Modal's
# function-table mechanics — both functions resolve to the same
# image hash and rollout_image cache.
@app.function(
    image=rollout_image,
    gpu=ROLLOUT_GENERATION_GPU_CLASS,
    volumes={"/vol": rollout_volume},
    timeout=3600,
)
def lagrangebench_rollout_p1_gns_tgv2d(git_sha: str, full_git_sha: str) -> dict:
    """Rung 3 + 3.5 P1: GNS-10-128 inference + SCHEMA.md npz materialization.

    Mirrors lagrangebench_rollout_p0_segnn_tgv2d with checkpoint /
    model substitutions per the comment block above. Same TGV2D
    dataset, same image, same conversion pipeline, same patch
    ledger. The cross-stack comparison (SEGNN vs GNS on identical
    test split) is the load-bearing claim of plan §3.1's P0+P1 work.
    """
    import hashlib
    import os
    import subprocess
    import time

    manifest: dict = {
        "git_sha": git_sha,
        "full_git_sha": full_git_sha,
        "lagrangebench_sha": None,
        "inference_seed": 0,
        "aborted_at_step": None,
        "checkpoint_download_skipped": False,
        "checkpoint_gdown_returncode": None,
        "checkpoint_gdown_stderr_tail": "",
        "checkpoint_unzip_returncode": None,
        "checkpoint_unzip_stderr_tail": "",
        "dataset_download_skipped": False,
        "dataset_download_returncode": None,
        "dataset_download_stderr_tail": "",
        "inference_returncode": None,
        "inference_wall_seconds": None,
        "inference_stdout_tail": "",
        "inference_stderr_tail": "",
        "rollout_subdir": None,
        "conversion_attempted": False,
        "conversion_returncode": None,
        "conversion_error": None,
        "converted_npz_paths": [],
        "files_written": [],
    }

    # Step 1: checkpoint download (gdown) + unzip
    ckpt_root = "/vol/checkpoints/lagrangebench/gns_tgv2d"
    ckpt_dir = f"{ckpt_root}/best"
    if os.path.isdir(ckpt_dir):
        manifest["checkpoint_download_skipped"] = True
    else:
        os.makedirs(ckpt_root, exist_ok=True)
        zip_path = f"{ckpt_root}/gns_tgv2d.zip"
        gdown_id = LAGRANGEBENCH_CHECKPOINT_GDOWN_IDS["gns_tgv2d"]

        gdown_proc = subprocess.run(
            ["gdown", gdown_id, "-O", zip_path],
            capture_output=True,
            text=True,
            timeout=600,
        )
        manifest["checkpoint_gdown_returncode"] = gdown_proc.returncode
        manifest["checkpoint_gdown_stderr_tail"] = gdown_proc.stderr[-2000:]
        if gdown_proc.returncode != 0:
            manifest["aborted_at_step"] = "checkpoint_gdown"
            return manifest

        unzip_proc = subprocess.run(
            ["unzip", "-o", zip_path, "-d", ckpt_root],
            capture_output=True,
            text=True,
            timeout=300,
        )
        manifest["checkpoint_unzip_returncode"] = unzip_proc.returncode
        manifest["checkpoint_unzip_stderr_tail"] = unzip_proc.stderr[-2000:]
        if unzip_proc.returncode != 0:
            manifest["aborted_at_step"] = "checkpoint_unzip"
            return manifest

        if not os.path.isdir(ckpt_dir):
            for entry in os.listdir(ckpt_root):
                candidate = os.path.join(ckpt_root, entry, "best")
                if os.path.isdir(candidate):
                    os.rename(candidate, ckpt_dir)
                    break
        rollout_volume.commit()

    # Step 2: dataset download (cached from P0; skip path likely hit)
    dataset_root = "/vol/datasets/lagrangebench"
    dataset_dir = f"{dataset_root}/{LAGRANGEBENCH_DATASET_DIRS['tgv_2d']}"
    if os.path.isdir(dataset_dir):
        manifest["dataset_download_skipped"] = True
    else:
        os.makedirs(dataset_root, exist_ok=True)
        os.chdir("/opt/lagrangebench")
        ds_proc = subprocess.run(
            ["bash", "download_data.sh", "tgv_2d", dataset_root + "/"],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        manifest["dataset_download_returncode"] = ds_proc.returncode
        manifest["dataset_download_stderr_tail"] = (
            (ds_proc.stdout[-1000:] + "\n--- stderr ---\n" + ds_proc.stderr[-1500:])
            if ds_proc.stderr
            else ds_proc.stdout[-2000:]
        )
        if ds_proc.returncode != 0:
            manifest["aborted_at_step"] = "dataset_download"
            return manifest
        rollout_volume.commit()

    # Step 3: inference — same CLI args as P0 (dataset.name=tgv2d
    # stays; only load_ckp + rollout_subdir differ).
    rollout_dir_root = "/vol/rollouts/lagrangebench"
    rollout_subdir = f"{rollout_dir_root}/gns_tgv2d_{git_sha}"
    os.makedirs(rollout_subdir, exist_ok=True)
    manifest["rollout_subdir"] = rollout_subdir

    os.chdir("/opt/lagrangebench")
    inf_args = [
        "python",
        "main.py",
        "mode=infer",
        "eval.test=True",
        f"load_ckp={ckpt_dir}",
        "eval.n_rollout_steps=100",
        "eval.infer.n_trajs=20",
        f"dataset.src={dataset_dir}",
        "dataset.name=tgv2d",
        "eval.infer.metrics=[mse,e_kin]",
        f"eval.rollout_dir={rollout_subdir}",
        "eval.infer.out_type=pkl",
        "seed=0",
    ]
    inf_start = time.monotonic()
    try:
        inf_proc = subprocess.run(
            inf_args,
            capture_output=True,
            text=True,
            timeout=2400,
        )
        manifest["inference_returncode"] = inf_proc.returncode
        manifest["inference_stdout_tail"] = inf_proc.stdout[-3000:]
        manifest["inference_stderr_tail"] = inf_proc.stderr[-3000:]
    except subprocess.TimeoutExpired as e:
        manifest["inference_returncode"] = -1
        manifest["inference_stdout_tail"] = (
            e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        )[-3000:]
        manifest["inference_stderr_tail"] = (
            f"TimeoutExpired after {e.timeout}s. Stderr tail: "
            + ((e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or ""))[-2500:])
        )
    except Exception as e:
        manifest["inference_returncode"] = -2
        manifest["inference_stderr_tail"] = f"Uncaught {type(e).__name__}: {e}"
    manifest["inference_wall_seconds"] = round(time.monotonic() - inf_start, 1)
    if manifest["inference_returncode"] != 0:
        manifest["aborted_at_step"] = "inference"

    # Step 4: capture lagrangebench_sha
    lb_sha_proc = subprocess.run(
        ["git", "-C", "/opt/lagrangebench", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if lb_sha_proc.returncode == 0:
        manifest["lagrangebench_sha"] = lb_sha_proc.stdout.strip()
    else:
        manifest["lagrangebench_sha"] = f"<git_rev_parse_failed_rc={lb_sha_proc.returncode}>"

    # Step 5: rung-3.5 conversion (model="gns" is the only diff vs P0)
    if manifest["inference_returncode"] == 0:
        manifest["conversion_attempted"] = True
        try:
            import sys as _sys

            if _REMOTE_HARNESS_DIR not in _sys.path:
                _sys.path.insert(0, _REMOTE_HARNESS_DIR)
            from lagrangebench_pkl_to_npz import (
                RolloutMetadata,
                convert_rollout_dir,
            )

            converted = convert_rollout_dir(
                rollout_dir=rollout_subdir,
                dataset_metadata_path=os.path.join(dataset_dir, "metadata.json"),
                ckpt_dir=ckpt_dir,
                metadata=RolloutMetadata(
                    git_sha=full_git_sha,
                    lagrangebench_sha=manifest["lagrangebench_sha"],
                    dataset="tgv2d",
                    model="gns",
                    seed=manifest["inference_seed"],
                    framework="jax+haiku",
                    framework_version=_package_version("jax"),
                ),
            )
            manifest["conversion_returncode"] = 0
            manifest["converted_npz_paths"] = [str(p) for p in converted]
        except Exception as e:
            manifest["conversion_returncode"] = 1
            manifest["conversion_error"] = f"{type(e).__name__}: {e}"
            manifest["aborted_at_step"] = "conversion"

    # Step 6: walk for files written
    candidate_roots = [
        "/opt/lagrangebench/outputs",
        "/opt/lagrangebench/rollouts",
        "/opt/lagrangebench/wandb",
        rollout_subdir,
    ]
    for root in candidate_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                fp = os.path.join(dirpath, fn)
                try:
                    size = os.path.getsize(fp)
                    if size > 50 * 1024 * 1024:
                        manifest["files_written"].append(
                            {"path": fp, "size": size, "sha256": "<skipped_large>"}
                        )
                        continue
                    h = hashlib.sha256()
                    with open(fp, "rb") as f:
                        for chunk in iter(lambda: f.read(65536), b""):
                            h.update(chunk)
                    manifest["files_written"].append(
                        {"path": fp, "size": size, "sha256": h.hexdigest()}
                    )
                except OSError as e:
                    manifest["files_written"].append(
                        {"path": fp, "size": -1, "sha256": f"<read_error:{e}>"}
                    )

    rollout_volume.commit()
    return manifest


@app.local_entrypoint()
def rollout_p0_segnn_tgv2d() -> None:
    """Fire the rung-3 P0 SEGNN-TGV2D rollout (D0-15)."""
    import subprocess

    repo_root = "/Users/zenith/Desktop/physics-lint"
    git_sha_short = subprocess.check_output(
        ["git", "rev-parse", "--short=10", "HEAD"], cwd=repo_root, text=True
    ).strip()
    git_sha_full = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()

    print("=== Rung 3 P0 SEGNN-TGV2D rollout (D0-15) ===")
    print(f"  gpu_class:           {ROLLOUT_GENERATION_GPU_CLASS}")
    print(f"  git_sha (short):     {git_sha_short}")
    print(f"  git_sha (full):      {git_sha_full}")
    print(f"  checkpoint gdown id: {LAGRANGEBENCH_CHECKPOINT_GDOWN_IDS['segnn_tgv2d']}")
    print()
    result = lagrangebench_rollout_p0_segnn_tgv2d.remote(
        git_sha=git_sha_short, full_git_sha=git_sha_full
    )
    print("--- manifest ---")
    print(f"  aborted_at_step:                {result['aborted_at_step'] or '<none>'}")
    print(f"  lagrangebench_sha:              {result['lagrangebench_sha']}")
    print(f"  inference_seed:                 {result['inference_seed']}")
    print(f"  rollout_subdir:                 {result['rollout_subdir']}")
    print(f"  checkpoint_download_skipped:    {result['checkpoint_download_skipped']}")
    print(f"  checkpoint_gdown_returncode:    {result['checkpoint_gdown_returncode']}")
    print(f"  checkpoint_unzip_returncode:    {result['checkpoint_unzip_returncode']}")
    print(f"  dataset_download_skipped:       {result['dataset_download_skipped']}")
    print(f"  dataset_download_returncode:    {result['dataset_download_returncode']}")
    print(f"  inference_returncode:           {result['inference_returncode']}")
    print(f"  inference_wall_seconds:         {result['inference_wall_seconds']}")
    print(f"  conversion_attempted:           {result['conversion_attempted']}")
    print(f"  conversion_returncode:          {result['conversion_returncode']}")
    if result["conversion_error"]:
        print(f"  conversion_error:               {result['conversion_error']}")
    print(f"  converted_npz_count:            {len(result['converted_npz_paths'])}")
    print()
    if result["converted_npz_paths"]:
        print("--- converted npz paths (rung 3.5) ---")
        for p in result["converted_npz_paths"][:25]:
            print(f"  {p}")
        if len(result["converted_npz_paths"]) > 25:
            print(f"  ... ({len(result['converted_npz_paths']) - 25} more)")
        print()
    if result["checkpoint_gdown_stderr_tail"]:
        print("--- checkpoint gdown stderr (last 2 KB) ---")
        print(result["checkpoint_gdown_stderr_tail"])
        print()
    if result["checkpoint_unzip_stderr_tail"]:
        print("--- checkpoint unzip stderr (last 2 KB) ---")
        print(result["checkpoint_unzip_stderr_tail"])
        print()
    if result["dataset_download_stderr_tail"]:
        print("--- dataset download stdout/stderr (last 2 KB) ---")
        print(result["dataset_download_stderr_tail"])
        print()
    print("--- inference stdout (last 3 KB) ---")
    print(result["inference_stdout_tail"] or "(empty)")
    print()
    print("--- inference stderr (last 3 KB) ---")
    print(result["inference_stderr_tail"] or "(empty)")
    print()
    print(f"--- files written ({len(result['files_written'])} entries) ---")
    for f in result["files_written"][:80]:
        print(f"  {f['size']:>12}  {f['sha256'][:16]}...  {f['path']}")
    if len(result["files_written"]) > 80:
        print(f"  ... ({len(result['files_written']) - 80} more)")
    print()
    # Cumulative-patch-shape distribution: structured signal for
    # whether proceed-with-LagrangeBench-HEAD remains the right
    # posture, vs. pause-for-higher-level-review. See
    # UPSTREAM_COMPAT_PATCHES module constant for the full reasoning.
    print(
        f"--- patches required to reach this verdict ({len(UPSTREAM_COMPAT_PATCHES)} entries) ---"
    )
    _layer_counts: dict[str, int] = {}
    _relevance_counts: dict[str, int] = {}
    for p in UPSTREAM_COMPAT_PATCHES:
        _layer_counts[p["layer"]] = _layer_counts.get(p["layer"], 0) + 1
        _relevance_counts[p["relevance"]] = _relevance_counts.get(p["relevance"], 0) + 1
        print(f"  [{p['layer']:>16}] [{p['relevance']:>16}] {p['ref']}\n    {p['summary']}")
    _layer_summary = ", ".join(f"{n} {layer}" for layer, n in _layer_counts.items())
    _relevance_summary = ", ".join(f"{n} {rel}" for rel, n in _relevance_counts.items())
    print(f"  distribution: {_layer_summary} | {_relevance_summary}")
    print()
    print("--- verdict ---")
    inf_ok = result["inference_returncode"] == 0
    conv_ok = result["conversion_returncode"] == 0
    if inf_ok and conv_ok:
        n = len(result["converted_npz_paths"])
        print(
            f"  -> rung 3 + 3.5 P0 verdict: PASS — inference returncode 0; "
            f"conversion produced {n} schema-conformant npz file(s)."
        )
        print("     Next: rung 4 (particle_rollout_adapter invocation against the npzs).")
    elif inf_ok and not conv_ok:
        print(
            f"  -> rung 3 PASS / rung 3.5 FAIL — inference returncode 0 but "
            f"conversion failed: {result['conversion_error']}. Native pkls "
            f"persisted at {result['rollout_subdir']}; reproduce conversion "
            f"locally with the captured manifest."
        )
    else:
        print(
            f"  -> rung 3 P0 verdict: FAIL — inference returncode "
            f"{result['inference_returncode']}. Inspect stderr above; "
            f"conversion not attempted (skipped on inference failure)."
        )


@app.local_entrypoint()
def rollout_p1_gns_tgv2d() -> None:
    """Fire the rung-3 P1 GNS-TGV2D rollout (plan §3.1, post-D0-18 unblock).

    Mirrors rollout_p0_segnn_tgv2d in printer shape; the P0 vs P1
    cross-stack comparison is the load-bearing claim of plan §3.1
    (SEGNN equivariant vs GNS approximate-equivariant on TGV2D).
    """
    import subprocess

    repo_root = "/Users/zenith/Desktop/physics-lint"
    git_sha_short = subprocess.check_output(
        ["git", "rev-parse", "--short=10", "HEAD"], cwd=repo_root, text=True
    ).strip()
    git_sha_full = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()

    print("=== Rung 3 P1 GNS-TGV2D rollout (plan §3.1) ===")
    print(f"  gpu_class:           {ROLLOUT_GENERATION_GPU_CLASS}")
    print(f"  git_sha (short):     {git_sha_short}")
    print(f"  git_sha (full):      {git_sha_full}")
    print(f"  checkpoint gdown id: {LAGRANGEBENCH_CHECKPOINT_GDOWN_IDS['gns_tgv2d']}")
    print()
    result = lagrangebench_rollout_p1_gns_tgv2d.remote(
        git_sha=git_sha_short, full_git_sha=git_sha_full
    )
    print("--- manifest ---")
    print(f"  aborted_at_step:                {result['aborted_at_step'] or '<none>'}")
    print(f"  lagrangebench_sha:              {result['lagrangebench_sha']}")
    print(f"  inference_seed:                 {result['inference_seed']}")
    print(f"  rollout_subdir:                 {result['rollout_subdir']}")
    print(f"  checkpoint_download_skipped:    {result['checkpoint_download_skipped']}")
    print(f"  checkpoint_gdown_returncode:    {result['checkpoint_gdown_returncode']}")
    print(f"  checkpoint_unzip_returncode:    {result['checkpoint_unzip_returncode']}")
    print(f"  dataset_download_skipped:       {result['dataset_download_skipped']}")
    print(f"  dataset_download_returncode:    {result['dataset_download_returncode']}")
    print(f"  inference_returncode:           {result['inference_returncode']}")
    print(f"  inference_wall_seconds:         {result['inference_wall_seconds']}")
    print(f"  conversion_attempted:           {result['conversion_attempted']}")
    print(f"  conversion_returncode:          {result['conversion_returncode']}")
    if result["conversion_error"]:
        print(f"  conversion_error:               {result['conversion_error']}")
    print(f"  converted_npz_count:            {len(result['converted_npz_paths'])}")
    print()
    if result["converted_npz_paths"]:
        print("--- converted npz paths (rung 3.5 P1) ---")
        for p in result["converted_npz_paths"][:25]:
            print(f"  {p}")
        if len(result["converted_npz_paths"]) > 25:
            print(f"  ... ({len(result['converted_npz_paths']) - 25} more)")
        print()
    if result["checkpoint_gdown_stderr_tail"]:
        print("--- checkpoint gdown stderr (last 2 KB) ---")
        print(result["checkpoint_gdown_stderr_tail"])
        print()
    if result["checkpoint_unzip_stderr_tail"]:
        print("--- checkpoint unzip stderr (last 2 KB) ---")
        print(result["checkpoint_unzip_stderr_tail"])
        print()
    print("--- inference stdout (last 3 KB) ---")
    print(result["inference_stdout_tail"] or "(empty)")
    print()
    print("--- inference stderr (last 3 KB) ---")
    print(result["inference_stderr_tail"] or "(empty)")
    print()
    print(f"--- files written ({len(result['files_written'])} entries) ---")
    for f in result["files_written"][:80]:
        print(f"  {f['size']:>12}  {f['sha256'][:16]}...  {f['path']}")
    if len(result["files_written"]) > 80:
        print(f"  ... ({len(result['files_written']) - 80} more)")
    print()
    print("--- verdict ---")
    inf_ok = result["inference_returncode"] == 0
    conv_ok = result["conversion_returncode"] == 0
    if inf_ok and conv_ok:
        n = len(result["converted_npz_paths"])
        print(
            f"  -> rung 3 + 3.5 P1 verdict: PASS — inference returncode 0; "
            f"conversion produced {n} schema-conformant npz file(s)."
        )
        print("     Next: rung 4 (cross-stack SEGNN vs GNS comparison via harness adapter).")
    elif inf_ok and not conv_ok:
        print(
            f"  -> rung 3 PASS / rung 3.5 FAIL — inference returncode 0 but "
            f"conversion failed: {result['conversion_error']}. Native pkls "
            f"persisted at {result['rollout_subdir']}; reproduce conversion "
            f"locally with the captured manifest or via "
            f"convert_pkls_p1_gns_tgv2d (TBD if needed)."
        )
    else:
        print(
            f"  -> rung 3 P1 verdict: FAIL — inference returncode "
            f"{result['inference_returncode']}. Inspect stderr above; "
            f"conversion not attempted (skipped on inference failure)."
        )


# --- Standalone rung-3.5 conversion (D0-17 amendment 1) ---
# CPU-only re-conversion against existing pkls in the Volume. Use when:
# (a) a prior rung-3 inference PASSed but rung-3.5 conversion FAILed
#     (e.g. the post-D0-17 regen at 8c3d080397 hit the strict PBC
#     length validation; D0-17 amendment 1 fixes it; this function
#     re-runs conversion against the persisted pkls without re-firing
#     inference);
# (b) the conversion module changes and existing rollouts need fresh
#     npzs without re-firing GPU inference.
#
# Methodology motivation (per DECISIONS.md D0-17 amendment 1): keeps
# P0 and P1 conversion environments identical (both Modal-side, same
# image, same numpy version) so the eventual SEGNN-vs-GNS cross-stack
# table doesn't mix conversion environments.


@app.function(
    image=rollout_image,
    volumes={"/vol": rollout_volume},
    timeout=600,  # CPU-only, 20 pkls -> 20 npzs is fast
)
def lagrangebench_convert_pkls_in_volume(
    rollout_subdir_name: str,
    dataset_lb_dirname: str,
    dataset_logical_name: str,
    checkpoint_subdir: str,
    model_name: str,
    git_sha_full: str,
    inference_seed: int = 0,
) -> dict:
    """Re-run rung-3.5 conversion against existing pkls in the Volume.

    Pulls pkls from /vol/rollouts/lagrangebench/<rollout_subdir_name>/,
    reads dataset metadata from /vol/datasets/lagrangebench/<dataset_lb_dirname>/,
    hashes checkpoint at /vol/checkpoints/lagrangebench/<checkpoint_subdir>/best/,
    invokes convert_rollout_dir, writes npzs back to the same rollout
    subdir.

    Returns the same manifest shape as the inference function's
    conversion-step subset (conversion_returncode, converted_npz_paths,
    conversion_error, lagrangebench_sha) for verdict-printer reuse.
    """
    import os

    # JAX_PLATFORMS=cpu before any pickle.load: LB rollout pkls contain
    # jnp.ndarray references, so pickle.load transitively imports jax,
    # which on this CPU-only function (no gpu= arg, runs on CPU
    # instance) tries to init the CUDA backend and raises
    # FAILED_PRECONDITION. The standalone-conversion function doesn't
    # need GPU compute at all (pure numpy float64 ops); telling jax to
    # use CPU avoids the spurious init failure. Discovered when
    # convert_pkls_p0_segnn_tgv2d's first fire FAILed at this exact
    # shape on a CPU Modal instance — the inference-side conversion
    # never hit this because that function runs on A10G (jax CUDA init
    # succeeds there).
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import subprocess
    import sys as _sys

    if _REMOTE_HARNESS_DIR not in _sys.path:
        _sys.path.insert(0, _REMOTE_HARNESS_DIR)
    from lagrangebench_pkl_to_npz import RolloutMetadata, convert_rollout_dir

    rollout_subdir = f"/vol/rollouts/lagrangebench/{rollout_subdir_name}"
    dataset_dir = f"/vol/datasets/lagrangebench/{dataset_lb_dirname}"
    ckpt_dir = f"/vol/checkpoints/lagrangebench/{checkpoint_subdir}/best"

    manifest: dict = {
        "rollout_subdir": rollout_subdir,
        "lagrangebench_sha": None,
        "conversion_returncode": None,
        "conversion_error": None,
        "converted_npz_paths": [],
    }

    # Capture lagrangebench_sha for the rollout metadata audit trail
    # (parity with the inference-side function).
    lb_sha_proc = subprocess.run(
        ["git", "-C", "/opt/lagrangebench", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    manifest["lagrangebench_sha"] = (
        lb_sha_proc.stdout.strip()
        if lb_sha_proc.returncode == 0
        else f"<git_rev_parse_failed_rc={lb_sha_proc.returncode}>"
    )

    if not os.path.isdir(rollout_subdir):
        manifest["conversion_returncode"] = 1
        manifest["conversion_error"] = f"rollout_subdir {rollout_subdir} does not exist"
        return manifest
    if not os.path.isdir(dataset_dir):
        manifest["conversion_returncode"] = 1
        manifest["conversion_error"] = f"dataset_dir {dataset_dir} does not exist"
        return manifest
    if not os.path.isdir(ckpt_dir):
        manifest["conversion_returncode"] = 1
        manifest["conversion_error"] = f"ckpt_dir {ckpt_dir} does not exist"
        return manifest

    try:
        converted = convert_rollout_dir(
            rollout_dir=rollout_subdir,
            dataset_metadata_path=os.path.join(dataset_dir, "metadata.json"),
            ckpt_dir=ckpt_dir,
            metadata=RolloutMetadata(
                git_sha=git_sha_full,
                lagrangebench_sha=manifest["lagrangebench_sha"],
                dataset=dataset_logical_name,
                model=model_name,
                seed=inference_seed,
                framework="jax+haiku",
                framework_version=_package_version("jax"),
            ),
        )
        manifest["conversion_returncode"] = 0
        manifest["converted_npz_paths"] = [str(p) for p in converted]
        rollout_volume.commit()
    except Exception as e:
        manifest["conversion_returncode"] = 1
        manifest["conversion_error"] = f"{type(e).__name__}: {e}"

    return manifest


@app.local_entrypoint()
def convert_pkls_p0_segnn_tgv2d(rollout_subdir_name: str) -> None:
    """Standalone rung-3.5 conversion for P0 SEGNN-TGV2D pkls (D0-17 amendment 1).

    Use after a rung-3 inference PASSes but the embedded conversion
    step FAILs (e.g. the f75e22d / 8c3d080397 runs hit pre-D0-17 /
    pre-amendment-1 conversion bugs but the inference output is fine
    and persisted to Volume). Hardcodes the SEGNN-TGV2D dataset/
    checkpoint mappings; sibling functions land for P1 GNS-TGV2D and
    P2/P3 when needed.

    Usage:
        modal run external_validation/.../modal_app.py::convert_pkls_p0_segnn_tgv2d \\
            --rollout-subdir-name segnn_tgv2d_8c3d080397
    """
    import subprocess

    repo_root = "/Users/zenith/Desktop/physics-lint"
    git_sha_full = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()

    print("=== Standalone rung-3.5 conversion: P0 SEGNN-TGV2D (D0-17 amendment 1) ===")
    print(f"  rollout_subdir_name: {rollout_subdir_name}")
    print(f"  git_sha (full):      {git_sha_full}")
    print()

    result = lagrangebench_convert_pkls_in_volume.remote(
        rollout_subdir_name=rollout_subdir_name,
        dataset_lb_dirname="2D_TGV_2500_10kevery100",
        dataset_logical_name="tgv2d",
        checkpoint_subdir="segnn_tgv2d",
        model_name="segnn",
        git_sha_full=git_sha_full,
    )

    print("--- manifest ---")
    print(f"  rollout_subdir:        {result['rollout_subdir']}")
    print(f"  lagrangebench_sha:     {result['lagrangebench_sha']}")
    print(f"  conversion_returncode: {result['conversion_returncode']}")
    if result["conversion_error"]:
        print(f"  conversion_error:      {result['conversion_error']}")
    print(f"  converted_npz_count:   {len(result['converted_npz_paths'])}")
    print()
    if result["converted_npz_paths"]:
        print("--- converted npz paths ---")
        for p in sorted(result["converted_npz_paths"]):
            print(f"  {p}")
        print()
    print("--- verdict ---")
    if result["conversion_returncode"] == 0:
        n = len(result["converted_npz_paths"])
        print(
            f"  -> standalone rung-3.5 conversion: PASS — produced {n} "
            f"schema-conformant npz file(s) at {result['rollout_subdir']}."
        )
    else:
        print(f"  -> standalone rung-3.5 conversion: FAIL — {result['conversion_error']}")
