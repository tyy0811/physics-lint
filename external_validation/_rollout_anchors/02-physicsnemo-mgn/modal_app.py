"""Modal entrypoint for Case Study 02 — PhysicsNeMo MeshGraphNet.

Parallel to 01-lagrangebench/modal_app.py. Builds the MGN inference image
with nvidia-physicsnemo pinned at sha 1ca85d65 (tag v2.0.0, 2026-03-10)
per preflight/mgn_loader_contract.md. torch-geometric + tfrecord for the
VortexSheddingDataset loader (it returns PyG `Data`, not DGL); scikit-fem
for the Gate A mesh-coercion audit (Task 6). torch is pinned to the 2.10
line — physicsnemo v2.0.0 (2026-03-10) predates torch 2.11 (2026-03-23) and
its domain_parallel code references a DTensor internal removed in 2.11.
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
        "scikit-fem",  # Gate A (D0-23 v3) — DGL was the plan's wrong assumption; the
        # v2.0.0 VortexSheddingDataset returns PyG `Data`, so `dgl` is dropped
        # (it was unused, and its torch-2.2-ABI wheel was dead weight).
        # torch pin: physicsnemo v2.0.0 was published 2026-03-10, one month after
        # torch 2.10.0 (2026-02-10) and 13 days before torch 2.11.0 (2026-03-23).
        # Its `domain_parallel/custom_ops/_tensor_ops.py` does
        # `from torch.distributed.tensor._ops.registration import ...`, which
        # exists in torch 2.10 but was moved/removed in 2.11 — so `import
        # physicsnemo.models` (pulled in by `from .dit import DiT` ->
        # `domain_parallel`) breaks on torch >= 2.11. Pin to the 2.10 line.
        "torch>=2.10.0,<2.11.0",
        # physicsnemo v2.0.0 pyproject declares warp-lang>=1.5.0 with no
        # upper bound; warp-lang >=1.13 removed wp.context.Device which
        # physicsnemo's nn.functional.radius_search._warp_impl still
        # references (import-time AttributeError). Pin to the floor.
        "warp-lang==1.5.0",
        # VortexSheddingDataset reads DeepMind .tfrecord via the `tfrecord`
        # package and builds PyG graphs — both are `OptionalImport`s in
        # physicsnemo/datapipes/gnn/vortex_shedding_dataset.py (lines 30-31 @
        # 1ca85d65), so neither is pulled by the physicsnemo install. Needed
        # for compute_cylinder_flow_stats, the V1-V18 loader-contract audit,
        # Gate A, and the Gate D inference reproduction.
        "tfrecord==1.14.6",
        "torch-geometric==2.7.0",
    )
    # MeshGraphNet's forward aggregation calls `torch_scatter.scatter`
    # (physicsnemo/nn/module/gnn_layers/utils.py:84,276 @ 1ca85d65 — without it the
    # forward raises "MeshGraphNet requires PyTorch Geometric and torch_scatter").
    # torch_scatter is a compiled extension; install the prebuilt wheel matching
    # the image's torch 2.10.0+cu128 from the PyG wheel index (a separate call so
    # it resolves against the already-installed torch).
    .pip_install(
        "torch-scatter",
        find_links="https://data.pyg.org/whl/torch-2.10.0+cu128.html",
    )
    # NGC CLI install per https://docs.ngc.nvidia.com/cli/cmd.html
    .run_commands(
        "wget -q https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.41.4/files/ngccli_linux.zip -O /tmp/ngccli.zip",
        "unzip -q /tmp/ngccli.zip -d /opt/ngc",
        "ln -s /opt/ngc/ngc-cli/ngc /usr/local/bin/ngc",
        "rm /tmp/ngccli.zip",
    )
    # Sibling-module: the hyphenated parent dir blocks normal Python packaging,
    # so we add the adapter file explicitly so the container can `import` it.
    .add_local_file(
        "external_validation/_rollout_anchors/02-physicsnemo-mgn/_legacy_checkpoint_name_remap.py",
        "/root/_legacy_checkpoint_name_remap.py",
    )
    # Phase 2 Tasks 5-7 import the mesh-side harness modules (lint_mesh_rollout,
    # mesh_rollout_adapter, sarif_emitter, particle_rollout_adapter) inside the
    # container. add_local_python_source ships the entire `external_validation`
    # package to /root/external_validation/ on container start; the hyphenated
    # `02-physicsnemo-mgn` subdir is skipped on import (not Python-importable
    # anyway). Skips dot-prefixed subdirs + __pycache__/.pyc by default; only
    # .py files are included.
    .add_local_python_source("external_validation")
)

# Modal Volume for NGC checkpoints + rollout outputs.
mgn_volume = modal.Volume.from_name("case-study-02-physicsnemo-artifacts", create_if_missing=True)

app = modal.App(
    "physics-lint-case-study-02-physicsnemo-mgn",
    image=mgn_image,
)


NGC_VORTEX_MODEL = "nvidia/modulus/modulus_ns_meshgraphnet"
# Catalog page shows only the "latest" tag (uploaded 2023-05-26, 24.79 MB);
# the plan's initial `:v0.1` guess was a path-invention that returned
# "could not be found". Pinning to "latest" + relying on sha256 for
# provenance, since NGC does not expose date-stable version tags here.
NGC_VORTEX_VERSION = "latest"
VOLUME_CHECKPOINT_ROOT = "/vol/checkpoints"


@app.function(
    timeout=300,
    secrets=[modal.Secret.from_name("ngc-api-key")],
)
def probe_ngc_direct_urls() -> dict:
    """Direct HTTP probes against NGC storage URLs, per documented patterns.

    Tests:
    - Ahmed Body sibling at the documented public URL (control)
    - modulus_ns_meshgraphnet at multiple version strings
    - Each tried unauth, ApiKey, and Bearer

    Distinguishes scope/auth issues from deprecation issues from genuine ACLs.
    """
    import json as _json
    import os
    import urllib.error
    import urllib.request

    api_key = os.environ["NGC_API_KEY"]

    urls = {
        "ahmed_body_v0.2": "https://api.ngc.nvidia.com/v2/models/nvidia/modulus/modulus_ahmed_body_meshgraphnet/versions/v0.2/files/ahmed_body_mgn.zip",
        "modulus_ns_latest": "https://api.ngc.nvidia.com/v2/models/nvidia/modulus/modulus_ns_meshgraphnet/versions/latest/files/vortex_shedding_mgn.zip",
        "modulus_ns_v0.1": "https://api.ngc.nvidia.com/v2/models/nvidia/modulus/modulus_ns_meshgraphnet/versions/v0.1/files/vortex_shedding_mgn.zip",
        "modulus_ns_v0.2": "https://api.ngc.nvidia.com/v2/models/nvidia/modulus/modulus_ns_meshgraphnet/versions/v0.2/files/vortex_shedding_mgn.zip",
        "modulus_ns_1": "https://api.ngc.nvidia.com/v2/models/nvidia/modulus/modulus_ns_meshgraphnet/versions/1/files/vortex_shedding_mgn.zip",
    }

    auth_variants = {
        "unauth": None,
        "bearer": f"Bearer {api_key}",
        "apikey": f"ApiKey {api_key}",
    }

    def probe(url: str, auth_header: str | None) -> dict:
        # HEAD request to avoid downloading bytes; capture status + final URL after redirects.
        req = urllib.request.Request(url, method="HEAD")
        if auth_header:
            req.add_header("Authorization", auth_header)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return {
                    "status": resp.status,
                    "final_url": resp.url,
                    "content_length": resp.headers.get("Content-Length"),
                    "content_type": resp.headers.get("Content-Type"),
                }
        except urllib.error.HTTPError as e:
            return {
                "status": e.code,
                "reason": e.reason,
                "headers": dict(e.headers) if e.headers else {},
                "body_excerpt": (e.read(500).decode("utf-8", errors="replace") if e.fp else ""),
            }
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    results: dict[str, dict] = {}
    for url_key, url in urls.items():
        results[url_key] = {}
        for auth_key, auth_header in auth_variants.items():
            results[url_key][auth_key] = probe(url, auth_header)

    # Also: list model versions via metadata API (unauth).
    for which in ("modulus_ns_meshgraphnet", "modulus_ahmed_body_meshgraphnet"):
        url = f"https://api.ngc.nvidia.com/v2/models/nvidia/modulus/{which}/versions"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read(2000).decode("utf-8", errors="replace")
                results[f"versions_metadata_{which}"] = {
                    "status": resp.status,
                    "body_excerpt": body,
                }
        except urllib.error.HTTPError as e:
            results[f"versions_metadata_{which}"] = {
                "status": e.code,
                "reason": e.reason,
                "body_excerpt": e.read(1000).decode("utf-8", errors="replace") if e.fp else "",
            }
        except Exception as e:
            results[f"versions_metadata_{which}"] = {"error": f"{type(e).__name__}: {e}"}

    print("=== NGC DIRECT-URL PROBE RESULTS ===")
    print(_json.dumps(results, indent=2, default=str))
    return results


@app.function(
    volumes={"/vol": mgn_volume},
    timeout=120,
)
def dump_full_keysets_for_rename_map() -> dict:
    """Dump expected (v2.0.0 MeshGraphNet) + actual (NGC checkpoint) full
    keysets so the rename map can be constructed precisely. Refinement 2
    prerequisite: the adapter cannot ship without an exhaustive key-by-key
    rename rule (or an explicit, falsifying-on-extra-keys policy).
    """
    import json as _json

    import torch
    from physicsnemo.models.meshgraphnet import MeshGraphNet

    checkpoint_path = Path(
        f"{VOLUME_CHECKPOINT_ROOT}/modulus_ns_meshgraphnet_{NGC_VORTEX_VERSION}/"
        f"vortex_shedding_mgn/model.pt"
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actual = list(ckpt["model_state_dict"].keys())

    model = MeshGraphNet(
        input_dim_nodes=6,
        input_dim_edges=3,
        output_dim=3,
    )
    expected = list(model.state_dict().keys())

    # Group by leading prefix to see structural categories.
    def prefix(k: str) -> str:
        return ".".join(k.split(".")[:2])

    from collections import Counter

    expected_prefixes = Counter(prefix(k) for k in expected)
    actual_prefixes = Counter(prefix(k) for k in actual)

    # Trial: try the trivial .mlp. → .model. rename and report coverage.
    renamed_actual = [k.replace(".mlp.", ".model.") for k in actual]
    matched_after_simple = sum(1 for k in renamed_actual if k in set(expected))

    # Also try the broader rename: .mlp. → .model. AND .edge_mlp. → .edge_mlp.
    # (if v2.0.0 uses something else for processor edges, we'll see it here).

    result = {
        "expected_count": len(expected),
        "actual_count": len(actual),
        "expected_prefix_groups": dict(expected_prefixes),
        "actual_prefix_groups": dict(actual_prefixes),
        "expected_keys_full": expected,
        "actual_keys_full": actual,
        "simple_rename_match_count": matched_after_simple,
    }
    print("=== FULL KEYSET DUMP ===")
    print(_json.dumps(result, indent=2, default=str))
    return result


@app.function(
    volumes={"/vol": mgn_volume},
    timeout=120,
)
def inspect_ngc_checkpoint_extras() -> dict:
    """Refinement 1 — characterize `device_buffer` (and any other non-renamed
    extras) before deciding whether the rename adapter can safely ignore them.

    If `device_buffer` is normalization statistics (1-D feature-dim tensor with
    typical mean/std-like values), ignoring it produces un-normalized
    predictions — silent wrong outputs that look plausible. If it's a
    scalar/0-element device flag, ignoring is safe. The adapter cannot be
    written until this classification is recorded.
    """
    import json as _json

    import torch

    checkpoint_path = Path(
        f"{VOLUME_CHECKPOINT_ROOT}/modulus_ns_meshgraphnet_{NGC_VORTEX_VERSION}/"
        f"vortex_shedding_mgn/model.pt"
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    # Classify each key by whether it matches the .mlp. → .model. rename pattern.
    rename_pattern_keys: list[str] = []
    extras: list[str] = []
    for k in sd:
        if ".mlp." in k:
            rename_pattern_keys.append(k)
        else:
            extras.append(k)

    extras_detail: dict[str, dict] = {}
    for k in extras:
        v = sd[k]
        if torch.is_tensor(v):
            try:
                stats = {
                    "shape": list(v.shape),
                    "dtype": str(v.dtype),
                    "numel": int(v.numel()),
                    "min": float(v.min().item()) if v.numel() > 0 else None,
                    "max": float(v.max().item()) if v.numel() > 0 else None,
                    "mean": float(v.float().mean().item()) if v.numel() > 0 else None,
                    "std": float(v.float().std().item()) if v.numel() > 1 else None,
                    "first_values": v.flatten()[:10].tolist() if v.numel() > 0 else [],
                    "last_values": v.flatten()[-10:].tolist() if v.numel() > 10 else [],
                }
            except Exception as e:
                stats = {"error": f"{type(e).__name__}: {e}"}
        else:
            stats = {"non_tensor_type": type(v).__name__, "repr_excerpt": repr(v)[:200]}
        extras_detail[k] = stats

    result = {
        "total_keys": len(sd),
        "rename_pattern_key_count": len(rename_pattern_keys),
        "extras_keys": extras,
        "extras_detail": extras_detail,
    }
    print("=== NGC CHECKPOINT EXTRAS INSPECTION ===")
    print(_json.dumps(result, indent=2, default=str))
    return result


@app.function(
    volumes={"/vol": mgn_volume},
    timeout=120,
)
def verify_ngc_checkpoint_state_dict_compat() -> dict:
    """BLOCKING-1 verdict: NGC checkpoint ↔ physicsnemo v2.0.0 state-dict compat.

    Pure-CPU state-dict-key smoke (no GPU). Loads the NGC checkpoint from the
    Modal Volume, passes it through the legacy-modulus name-remap adapter
    (per D0-23 verdict 1; rationale: rename-only refactor between modulus
    and physicsnemo v2.0.0), then compares the remapped keys against the
    v2.0.0 MeshGraphNet constructor (per
    examples/cfd/vortex_shedding_mgn/conf/config.yaml: input_dim_nodes=6,
    input_dim_edges=3, output_dim=3).

    PASS here verifies key-set match only. Architecture identity (the
    remap's load-bearing assumption) is empirically verified by Gate D's
    test_inference_matches_ngc_sample (Task 7); see _legacy_checkpoint_name_remap.py.
    """
    import sys

    sys.path.insert(0, "/root")  # adapter is a sibling .py file
    import torch
    from _legacy_checkpoint_name_remap import (
        remap_modulus_to_physicsnemo_state_dict,
    )
    from physicsnemo.models.meshgraphnet import MeshGraphNet

    checkpoint_path = Path(
        f"{VOLUME_CHECKPOINT_ROOT}/modulus_ns_meshgraphnet_{NGC_VORTEX_VERSION}/"
        f"vortex_shedding_mgn/model.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint absent at {checkpoint_path}; "
            f"run download_ngc_vortex_shedding_checkpoint first."
        )

    model = MeshGraphNet(
        input_dim_nodes=6,
        input_dim_edges=3,
        output_dim=3,
    )
    expected_keys = set(model.state_dict().keys())

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        raw_state_dict = ckpt["model_state_dict"]
        wrap_key = "model_state_dict"
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        raw_state_dict = ckpt["state_dict"]
        wrap_key = "state_dict"
    else:
        raw_state_dict = ckpt
        wrap_key = None

    # Apply the legacy-modulus → physicsnemo v2.0.0 name remap.
    ckpt_state_dict = remap_modulus_to_physicsnemo_state_dict(raw_state_dict)
    actual_keys = set(ckpt_state_dict.keys())

    missing_in_ckpt = sorted(expected_keys - actual_keys)
    extra_in_ckpt = sorted(actual_keys - expected_keys)
    common = expected_keys & actual_keys

    verdict = "PASS" if not missing_in_ckpt and not extra_in_ckpt else "FAIL"

    result = {
        "verdict": verdict,
        "wrap_key": wrap_key,
        "expected_key_count": len(expected_keys),
        "actual_key_count": len(actual_keys),
        "common_key_count": len(common),
        "missing_in_ckpt_count": len(missing_in_ckpt),
        "extra_in_ckpt_count": len(extra_in_ckpt),
        "missing_in_ckpt_sample": missing_in_ckpt[:10],
        "extra_in_ckpt_sample": extra_in_ckpt[:10],
        "ckpt_top_level_type": type(ckpt).__name__,
        "ckpt_top_level_keys": (sorted(ckpt.keys())[:20] if isinstance(ckpt, dict) else None),
    }
    import json as _json

    print("=== BLOCKING-1 STATE-DICT VERDICT ===")
    print(_json.dumps(result, indent=2, default=str))
    return result


@app.function(
    volumes={"/vol": mgn_volume},
    timeout=600,
    secrets=[modal.Secret.from_name("ngc-api-key")],  # NGC_API_KEY env var
)
def download_ngc_vortex_shedding_checkpoint() -> dict:
    """Download modulus_ns_meshgraphnet:latest checkpoint via direct HTTP.

    Returns: {"zip_sha256", "zip_size_bytes", "extracted_files", "checkpoint_path",
              "checkpoint_sha256", "checkpoint_size_bytes", ...}.

    NOTE on mechanism: an earlier draft used `ngc registry model download-version`
    via the NGC CLI 3.41.4. That path failed silently with HTTP 403 on the
    storage backend, likely a CLI auth-resolution bug consistent with NVIDIA's
    documented Modulus → PhysicsNeMo deprecation of the old NGC artifact stack.
    Direct HTTP GET against the same backing storage URL works unauthenticated
    (verified by `probe_ngc_direct_urls`), so we use urllib.request to fetch
    the zip directly.
    """
    import hashlib
    import io
    import urllib.request
    import zipfile

    download_url = (
        f"https://api.ngc.nvidia.com/v2/models/nvidia/modulus/"
        f"modulus_ns_meshgraphnet/versions/{NGC_VORTEX_VERSION}/files/vortex_shedding_mgn.zip"
    )
    dest_root = Path(f"{VOLUME_CHECKPOINT_ROOT}/modulus_ns_meshgraphnet_{NGC_VORTEX_VERSION}")
    dest_root.mkdir(parents=True, exist_ok=True)

    # Stream the zip to memory + hash on the way.
    print(f"Downloading {download_url}")
    h_zip = hashlib.sha256()
    buf = io.BytesIO()
    with urllib.request.urlopen(download_url, timeout=300) as resp:
        if resp.status != 200:
            raise RuntimeError(f"NGC direct GET failed: status={resp.status}")
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            h_zip.update(chunk)
            buf.write(chunk)
    zip_size = buf.tell()
    zip_sha256 = h_zip.hexdigest()
    print(f"Downloaded {zip_size} bytes, sha256={zip_sha256}")

    # Persist the raw zip for provenance, then extract.
    zip_path = dest_root / "vortex_shedding_mgn.zip"
    zip_path.write_bytes(buf.getvalue())

    extracted_files: list[str] = []
    with zipfile.ZipFile(io.BytesIO(buf.getvalue())) as zf:
        zf.extractall(dest_root)
        extracted_files = sorted(zf.namelist())
    print(f"Extracted {len(extracted_files)} files")

    # Locate the model checkpoint (.pt / .tar / .pth / .ckpt) inside the extracted tree.
    candidates: list[Path] = []
    for pattern in ("*.pt", "*.tar", "*.pth", "*.ckpt"):
        candidates.extend(dest_root.rglob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No .pt/.tar/.pth/.ckpt checkpoint found under {dest_root} after extraction; "
            f"got files: {extracted_files[:20]}"
        )
    if len(candidates) > 1:
        # Multiple candidates is fine for the bundle (model + optimizer state etc.);
        # we hash the largest one which is conventionally the main weights file.
        candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    checkpoint_path = candidates[0]

    h_ckpt = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h_ckpt.update(chunk)
    checkpoint_sha256 = h_ckpt.hexdigest()

    mgn_volume.commit()

    result = {
        "download_url": download_url,
        "zip_sha256": zip_sha256,
        "zip_size_bytes": zip_size,
        "extracted_files": extracted_files,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "all_checkpoint_candidates": [str(p) for p in candidates],
        "ngc_model": NGC_VORTEX_MODEL,
        "ngc_version": NGC_VORTEX_VERSION,
        "physicsnemo_sha": PHYSICSNEMO_SHA,
    }
    import json as _json

    print("=== DOWNLOAD RESULT ===")
    print(_json.dumps(result, indent=2, default=str))
    return result


# DeepMind MeshGraphNets public dataset (Pfaff et al. 2020). NVIDIA's
# VortexSheddingDataset reads exactly this `meta.json` + `<split>.tfrecord`
# format (the physicsnemo MGN datapipe was ported from DeepMind's reference).
DM_MESHGRAPHNETS_BASE = "https://storage.googleapis.com/dm-meshgraphnets"
DM_CYLINDER_FLOW_DIR = "/vol/datasets/cylinder_flow"
# Provenance pins (mirror DECISIONS.md D0-23 "Phase 1 data provenance"); any
# Phase 1 entrypoint consuming these files asserts on-disk sha == pin before
# proceeding (catches upstream-bucket drift). `train.tfrecord` has no pin —
# it is fetched transiently inside compute_cylinder_flow_stats and deleted.
DM_CYLINDER_FLOW_META_SHA256 = "2a3e39429a55a0cf47355717cc07f4b292629daeb48a89abd518ea0402033e96"
DM_CYLINDER_FLOW_TEST_SHA256 = "8522932a23da6ccdee996c56158e4b908f7091f0ade11e8acea700be321af8c3"


def _stream_download_and_hash(url: str, dest: Path, chunk_size: int = 1024 * 1024) -> dict:
    """Stream a URL to `dest`, computing sha256 on the way. Returns
    {"url", "path", "sha256", "size_bytes"}.
    """
    import hashlib
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(url, timeout=600) as resp, open(dest, "wb") as out:
        if resp.status != 200:
            raise RuntimeError(f"GET {url} → status {resp.status}")
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            out.write(chunk)
            size += len(chunk)
    return {"url": url, "path": str(dest), "sha256": h.hexdigest(), "size_bytes": size}


@app.function(
    volumes={"/vol": mgn_volume},
    timeout=1800,
)
def download_dm_cylinder_flow_dataset() -> dict:
    """Tier-1: download DeepMind's public cylinder_flow dataset to the Volume.

    Pulls `meta.json` (883 B) + `test.tfrecord` (~1.3 GB) — enough for the
    V1-V18 loader-contract audit (Task 5), the NGC sample reproduction
    (Task 7), and the 1-traj substrate-class smoke (Task 9). `train.tfrecord`
    (~13.6 GB / 13,645,805,387 B) is intentionally NOT pulled here — see
    `compute_cylinder_flow_stats` which pulls + fits + deletes it atomically.
    """
    import json as _json

    dest_dir = Path(DM_CYLINDER_FLOW_DIR)
    results: dict[str, dict] = {}

    results["meta.json"] = _stream_download_and_hash(
        f"{DM_MESHGRAPHNETS_BASE}/cylinder_flow/meta.json",
        dest_dir / "meta.json",
    )
    print(
        f"meta.json: {results['meta.json']['size_bytes']} bytes, sha256={results['meta.json']['sha256']}"
    )

    results["test.tfrecord"] = _stream_download_and_hash(
        f"{DM_MESHGRAPHNETS_BASE}/cylinder_flow/test.tfrecord",
        dest_dir / "test.tfrecord",
    )
    print(
        f"test.tfrecord: {results['test.tfrecord']['size_bytes']} bytes, sha256={results['test.tfrecord']['sha256']}"
    )

    mgn_volume.commit()

    print("=== DM CYLINDER_FLOW DOWNLOAD RESULT ===")
    print(_json.dumps(results, indent=2, default=str))
    return results


@app.function(
    volumes={"/vol": mgn_volume},
    # VortexSheddingDataset materializes every trajectory in memory: at the
    # full-train defaults (num_samples=1000, num_steps=600) the node-feature /
    # node-target tensors are ~30 GB total (vortex_shedding_dataset.py:81-159 @
    # 1ca85d65 — two in-memory lists of (num_steps-1, n_nodes, {2,2,1}) float32
    # tensors, no streaming accumulation). 96 GB gives ~3x headroom.
    memory=98304,
    timeout=3600,
)
def compute_cylinder_flow_stats(
    num_samples: int = 1000,
    num_steps: int = 600,
    noise_std: float = 0.02,
) -> dict:
    """Tier-1: fit edge_stats.json / node_stats.json from train.tfrecord, atomically.

    Why this exists: the NGC checkpoint zip ships only `vortex_shedding_mgn/model.pt`
    — no stats — yet `VortexSheddingDataset(split="test")` loads `edge_stats.json`
    and `node_stats.json` from CWD (vortex_shedding_dataset.py:103,141 @ 1ca85d65),
    so the test-split loader (Task 7's NGC sample reproduction, Task 9's substrate
    smoke) cannot run without them. The example workflow produces them as a side
    effect of `train.py` on the train split; we reproduce that fit directly.

    Atomicity (handoff Refinement 2): this single entrypoint
      1. asserts meta.json sha == D0-23 pin (catches upstream-bucket drift),
      2. downloads DeepMind's `train.tfrecord` (~13.6 GB) to *local container
         disk* — never to the Modal Volume,
      3. constructs `VortexSheddingDataset(split="train", ...)` which fits and
         `save_json`s both stats files via `_get_edge_stats` / `_get_node_stats`
         (vortex_shedding_dataset.py:188-265),
      4. copies the two JSONs onto the Volume next to meta.json/test.tfrecord,
      5. deletes the local ~13.6 GB in a `finally` block.
    No persistent ~13.6 GB anywhere; a hard crash before the finally still leaves
    the Volume clean (the big file only ever lived on ephemeral local disk).

    Stat-fit parameters. Defaults `(num_samples=1000, num_steps=600, noise_std=0.02)`
    = `VortexSheddingDataset.__init__` defaults = full DeepMind cylinder_flow
    train split with DeepMind/modulus-standard input-noise injection (noise is
    added to the train-split velocities *before* the stats are computed —
    vortex_shedding_dataset.py:127-133, so `velocity_std` carries a ~noise_std**2
    term, which is what the checkpoint was normalized against). The example
    `conf/config.yaml` uses `(400, 300)` for faster training; the modulus-era
    config the 2023 NGC checkpoint was trained against is not documented, so we
    fit on the full train distribution (the most reproducible choice — no
    arbitrary subsample) and let Task 7's NGC sample reproduction be the
    empirical test of whether these stats match the checkpoint's expectations
    (Gate D; FNO-on-Darcy fallback pre-registered per design §3.1.A if not).
    `torch.manual_seed(0)` is set so the noise-injected fit is reproducible.
    """
    import hashlib
    import json as _json
    import os
    import shutil

    import torch

    fit_dir = Path("/tmp/cf_stats_fit")
    if fit_dir.exists():
        shutil.rmtree(fit_dir)
    fit_dir.mkdir(parents=True)
    stats_work_dir = fit_dir / "stats_out"
    stats_work_dir.mkdir()

    # --- Pre-flight: meta.json provenance assertion ---
    meta_dl = _stream_download_and_hash(
        f"{DM_MESHGRAPHNETS_BASE}/cylinder_flow/meta.json", fit_dir / "meta.json"
    )
    if meta_dl["sha256"] != DM_CYLINDER_FLOW_META_SHA256:
        raise RuntimeError(
            f"cylinder_flow meta.json sha256 mismatch: got {meta_dl['sha256']}, "
            f"pinned {DM_CYLINDER_FLOW_META_SHA256} (DECISIONS D0-23). "
            f"Upstream bucket drift — investigate before recomputing stats."
        )
    print(f"meta.json: {meta_dl['size_bytes']} bytes, sha256 OK ({meta_dl['sha256']})")

    # --- Download train.tfrecord to LOCAL disk only (never persisted) ---
    print(f"Downloading train.tfrecord (~13.6 GB) to {fit_dir}/train.tfrecord ...")
    train_dl = _stream_download_and_hash(
        f"{DM_MESHGRAPHNETS_BASE}/cylinder_flow/train.tfrecord",
        fit_dir / "train.tfrecord",
    )
    print(f"train.tfrecord: {train_dl['size_bytes']} bytes, sha256={train_dl['sha256']}")

    out_dir = Path(DM_CYLINDER_FLOW_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    cwd_before = os.getcwd()

    result: dict[str, object] = {
        "meta_json": meta_dl,
        "train_tfrecord": {k: train_dl[k] for k in ("url", "sha256", "size_bytes")},
        "num_samples": num_samples,
        "num_steps": num_steps,
        "noise_std": noise_std,
        "physicsnemo_sha": PHYSICSNEMO_SHA,
        "seed": 0,
    }
    try:
        torch.set_default_dtype(torch.float32)  # preflight known-unknown 5.6
        torch.manual_seed(0)  # noise_std injection -> seed for a reproducible fit

        # _get_edge_stats / _get_node_stats save_json to CWD.
        os.chdir(stats_work_dir)

        from physicsnemo.datapipes.gnn.vortex_shedding_dataset import (
            VortexSheddingDataset,
        )

        print(
            f"Constructing VortexSheddingDataset(split='train', "
            f"num_samples={num_samples}, num_steps={num_steps}, noise_std={noise_std}) ..."
        )
        ds = VortexSheddingDataset(
            name="cylinder_flow",
            data_dir=str(fit_dir),
            split="train",
            num_samples=num_samples,
            num_steps=num_steps,
            noise_std=noise_std,
        )
        print(f"Dataset constructed: len={len(ds)} ({num_samples} traj x {num_steps - 1} steps)")

        edge_stats_src = stats_work_dir / "edge_stats.json"
        node_stats_src = stats_work_dir / "node_stats.json"
        if not (edge_stats_src.exists() and node_stats_src.exists()):
            raise RuntimeError(
                f"expected edge_stats.json + node_stats.json written by "
                f"_get_edge_stats / _get_node_stats into {stats_work_dir}; "
                f"found {sorted(p.name for p in stats_work_dir.iterdir())}"
            )

        edge_stats = _json.loads(edge_stats_src.read_text())
        node_stats = _json.loads(node_stats_src.read_text())
        # Surface the shapes so a downstream V15 dim-mismatch is visible here.
        print("edge_stats keys/lens: " + ", ".join(f"{k}={len(v)}" for k, v in edge_stats.items()))
        print("node_stats keys/lens: " + ", ".join(f"{k}={len(v)}" for k, v in node_stats.items()))

        def _sha256(p: Path) -> str:
            h = hashlib.sha256()
            h.update(p.read_bytes())
            return h.hexdigest()

        edge_stats_dst = out_dir / "edge_stats.json"
        node_stats_dst = out_dir / "node_stats.json"
        shutil.copy2(edge_stats_src, edge_stats_dst)
        shutil.copy2(node_stats_src, node_stats_dst)
        mgn_volume.commit()

        result["edge_stats"] = {
            "path": str(edge_stats_dst),
            "sha256": _sha256(edge_stats_dst),
            "size_bytes": edge_stats_dst.stat().st_size,
            "value": edge_stats,
        }
        result["node_stats"] = {
            "path": str(node_stats_dst),
            "sha256": _sha256(node_stats_dst),
            "size_bytes": node_stats_dst.stat().st_size,
            "value": node_stats,
        }
        result["status"] = "ok"
    finally:
        os.chdir(cwd_before)  # leave the temp dir before removing it
        shutil.rmtree(fit_dir, ignore_errors=True)
        # The Modal Volume never held train.tfrecord; nothing to delete there.

    print("=== CYLINDER_FLOW STATS FIT RESULT ===")
    print(_json.dumps(result, indent=2, default=str))
    return result


def _sha256_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


@app.function(
    volumes={"/vol": mgn_volume},
    timeout=600,
    # CPU-only: a single decoded record + a 1-trajectory test-split dataset.
)
def audit_cylinder_flow_loader_contract() -> dict:
    """Audit the VortexSheddingDataset loader contract against preflight V1-V18
    + the 5 secondary known-unknowns, on the actual DeepMind cylinder_flow data.

    Per `02-physicsnemo-mgn/preflight/mgn_loader_contract.md` §3.1 + §5; produces
    DECISIONS D0-23 verdicts 2 (loader-contract findings) and 8 (the velocity-field
    key for Task 10's `_expect_velocity` helper). Each V-check records pass/fail/error
    rather than aborting, so the audit always emits a complete report; only the
    provenance-sha asserts (V1/V2 + drift guard) abort. CPU-only — loader-side only,
    no inference.

    Citations are `vortex_shedding_dataset.py:<line>` at sha `1ca85d65`.
    """
    import json as _json
    import os
    import shutil
    import tempfile

    import numpy as np
    import torch

    data_dir = Path(DM_CYLINDER_FLOW_DIR)
    meta_path = data_dir / "meta.json"
    test_tfrecord_path = data_dir / "test.tfrecord"
    for p in (meta_path, test_tfrecord_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run download_dm_cylinder_flow_dataset first.")

    findings: dict[str, object] = {
        "physicsnemo_sha": PHYSICSNEMO_SHA,
        "data_dir": str(data_dir),
        "torch_default_dtype_at_entry": str(torch.get_default_dtype()),
    }

    # --- V1 / V2 + drift guard: provenance shas must match the D0-23 pins ---
    meta_sha = _sha256_of_file(meta_path)
    test_sha = _sha256_of_file(test_tfrecord_path)
    findings["meta_json_sha256"] = meta_sha
    findings["test_tfrecord_sha256"] = test_sha
    if meta_sha != DM_CYLINDER_FLOW_META_SHA256:
        raise RuntimeError(
            f"meta.json sha mismatch: {meta_sha} vs pinned {DM_CYLINDER_FLOW_META_SHA256} (D0-23)"
        )
    if test_sha != DM_CYLINDER_FLOW_TEST_SHA256:
        raise RuntimeError(
            f"test.tfrecord sha mismatch: {test_sha} vs pinned {DM_CYLINDER_FLOW_TEST_SHA256} (D0-23)"
        )
    findings["V1_meta_json_present"] = True  # opened above
    findings["V2_test_tfrecord_present"] = True

    torch.set_default_dtype(torch.float32)  # preflight 5.6 — must precede dataset construction

    # --- meta.json contract: V3 / V4 / V5 / V6 / V7 ---
    meta = _json.loads(meta_path.read_text())
    field_names = meta.get("field_names")
    findings["V3_field_names"] = field_names
    findings["V3_field_names_is_list_of_str"] = isinstance(field_names, list) and all(
        isinstance(k, str) for k in (field_names or [])
    )
    supported_dtypes = {"float32", "float64", "int32", "int64"}
    dtype_report: dict[str, dict] = {}
    for k, v in (meta.get("features") or {}).items():
        dt = v.get("dtype")
        in_supported = dt in supported_dtypes
        getattr_ok = in_supported or (isinstance(dt, str) and hasattr(np, dt))
        dtype_report[k] = {
            "dtype": dt,
            "in_supported_map": in_supported,
            "resolvable": getattr_ok,
            "type": v.get("type"),
            "shape": v.get("shape"),
        }
    findings["V4_V5_per_feature"] = dtype_report
    findings["V4_all_dtypes_resolvable"] = all(r["resolvable"] for r in dtype_report.values())
    findings["V5_all_features_have_shape"] = all(
        r["shape"] is not None for r in dtype_report.values()
    )
    findings["V6_trajectory_length"] = meta.get("trajectory_length")
    dynamic_varlen_fields = [k for k, r in dtype_report.items() if r["type"] == "dynamic_varlen"]
    findings["V7_dynamic_varlen_fields"] = dynamic_varlen_fields
    findings["V7_note"] = (
        "no dynamic_varlen fields in cylinder_flow (cells/mesh_pos/node_type are 'static', "
        "velocity/pressure are 'dynamic') — the length_<k> sibling requirement is vacuous here"
        if not dynamic_varlen_fields
        else "dynamic_varlen present — decoded record must carry 'length_<k>' siblings"
    )

    # --- Decode one test.tfrecord record with the loader's own machinery ---
    # Mirrors vortex_shedding_dataset.py:286-306 (_load_tfrecord_dataset).
    import tfrecord.torch.dataset as tfrecord_torch
    from physicsnemo.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset

    description = {k: "byte" for k in field_names}
    tfr = tfrecord_torch.TFRecordDataset(
        str(test_tfrecord_path),
        None,  # no .tfindex -> single-worker sequential read
        description,
        transform=lambda rec: VortexSheddingDataset._decode_record(rec, meta),
    )
    rec0 = next(iter(tfr))  # first decoded record (dict of numpy arrays)
    rec_shapes = {k: list(np.shape(v)) for k, v in rec0.items()}
    findings["decoded_record_shapes"] = rec_shapes
    required_keys = {"cells", "mesh_pos", "node_type", "velocity", "pressure"}
    findings["V8_required_keys_present"] = sorted(required_keys & set(rec0.keys())) == sorted(
        required_keys
    )
    findings["V8_record_keys"] = sorted(rec0.keys())

    cells0 = np.asarray(rec0["cells"][0])
    findings["V9_cells_frame0_shape"] = list(cells0.shape)
    findings["V9_cells_is_triangle"] = cells0.ndim == 2 and cells0.shape[-1] == 3
    findings["V9_n_cells"] = int(cells0.shape[0]) if cells0.ndim == 2 else None
    # V10: stationary mesh. cells/mesh_pos are 'static' type, so _decode_record tiled
    # them to trajectory_length identical frames — constant across t by construction.
    cells_all = np.asarray(rec0["cells"])
    mesh_pos_all = np.asarray(rec0["mesh_pos"])
    findings["V10_cells_constant_over_t"] = bool(np.all(cells_all == cells_all[0:1]))
    findings["V10_mesh_pos_constant_over_t"] = bool(np.all(mesh_pos_all == mesh_pos_all[0:1]))
    findings["V10_note"] = (
        "True by construction — cells/mesh_pos are 'static' fields tiled by _decode_record"
    )
    velocity_all = np.asarray(rec0["velocity"])
    findings["V11_velocity_trajectory_len"] = int(velocity_all.shape[0])
    findings["V11_velocity_len_ge_2"] = velocity_all.shape[0] >= 2
    first_axes = {k: int(np.shape(v)[0]) for k, v in rec0.items()}
    findings["V12_first_axis_lengths"] = first_axes
    findings["V12_all_equal_trajectory_length"] = len(set(first_axes.values())) == 1 and next(
        iter(first_axes.values())
    ) == meta.get("trajectory_length")
    node_type0 = np.asarray(rec0["node_type"][0]).reshape(-1)
    unique_node_types = sorted(int(x) for x in np.unique(node_type0))
    findings["V16_node_type_unique_frame0"] = unique_node_types
    findings["V16_node_type_subset_of_0_3_4_5_6"] = set(unique_node_types) <= {0, 3, 4, 5, 6}
    findings["secondary_5_7_node_type_one_hot_bound"] = (
        "OK — non-zero node_type values map to {value-3} in [0,3] for one_hot(num_classes=4); "
        f"observed unique values {unique_node_types}"
    )
    findings["secondary_5_5_num_steps_le_trajectory_length"] = (
        "audit constructs split='test' with num_steps=5 <= trajectory_length="
        f"{meta.get('trajectory_length')}"
    )
    findings["secondary_5_6_default_dtype"] = (
        f"set to {torch.get_default_dtype()} before dataset construction (was "
        f"{findings['torch_default_dtype_at_entry']} at entry)"
    )

    # --- V14 / V15 + V13 / V17 / V18 + verdict 8: construct the test-split dataset ---
    # The loader reads edge_stats.json / node_stats.json from CWD (lines 103, 141),
    # so the audit stages them in a temp dir and chdirs there (secondary 5.4 — the CWD
    # coupling is demonstrated by this very dance).
    cwd_before = os.getcwd()
    work_dir = Path(tempfile.mkdtemp(prefix="cf_audit_"))
    edge_stats_vol = data_dir / "edge_stats.json"
    node_stats_vol = data_dir / "node_stats.json"
    findings["V14_edge_stats_on_volume"] = edge_stats_vol.exists()
    findings["V14_node_stats_on_volume"] = node_stats_vol.exists()
    try:
        if edge_stats_vol.exists() and node_stats_vol.exists():
            shutil.copy2(edge_stats_vol, work_dir / "edge_stats.json")
            shutil.copy2(node_stats_vol, work_dir / "node_stats.json")
            edge_stats = _json.loads((work_dir / "edge_stats.json").read_text())
            node_stats = _json.loads((work_dir / "node_stats.json").read_text())
            stat_lens = {k: len(v) for k, v in {**edge_stats, **node_stats}.items()}
            findings["V15_stats_value_lengths"] = stat_lens
            findings["V15_dims_match_feature_widths"] = (
                stat_lens.get("edge_mean") == 3
                and stat_lens.get("edge_std") == 3
                and stat_lens.get("velocity_mean") == 2
                and stat_lens.get("velocity_std") == 2
                and stat_lens.get("velocity_diff_mean") == 2
                and stat_lens.get("velocity_diff_std") == 2
                and stat_lens.get("pressure_mean") == 1
                and stat_lens.get("pressure_std") == 1
            )

            os.chdir(work_dir)
            findings["secondary_5_4_stats_cwd"] = os.getcwd()
            ds = VortexSheddingDataset(
                name="cylinder_flow",
                data_dir=str(data_dir),
                split="test",
                num_samples=1,
                num_steps=5,
                noise_std=0.0,  # ignored for split != "train" (line 127) — secondary 5.3
            )
            findings["V13_len"] = len(ds)
            findings["V13_len_eq_num_samples_times_num_steps_minus_1"] = len(ds) == 1 * (5 - 1)
            item = ds[0]
            findings["dataset_getitem_returns"] = (
                "(graph, cells, rollout_mask)" if isinstance(item, tuple) else type(item).__name__
            )
            graph = item[0] if isinstance(item, tuple) else item
            findings["graph_type"] = type(graph).__name__
            try:
                findings["graph_data_keys"] = sorted(str(k) for k in graph.to_dict())
            except Exception as ke:
                findings["graph_data_keys_error"] = f"{type(ke).__name__}: {ke}"
            findings["V18_node_feature_width_x"] = int(graph.x.shape[-1])
            findings["V18_target_width_y"] = int(graph.y.shape[-1])
            findings["V17_edge_attr_width"] = int(graph.edge_attr.shape[-1])
            findings["V17_V18_widths_match_config"] = (
                int(graph.x.shape[-1]) == 6
                and int(graph.y.shape[-1]) == 3
                and int(graph.edge_attr.shape[-1]) == 3
            )
            if isinstance(item, tuple) and len(item) >= 2:
                cells_t = np.asarray(item[1])
                findings["test_split_cells_shape"] = list(cells_t.shape)
                findings["test_split_cells_is_triangle"] = (
                    cells_t.ndim == 2 and cells_t.shape[-1] == 3
                )
            # V13 boundary
            _ = ds[len(ds) - 1]
            try:
                _ = ds[len(ds)]
                findings["V13_out_of_range_raises"] = False
            except (IndexError, KeyError):
                findings["V13_out_of_range_raises"] = True
            # Verdict 8: the velocity-field key.
            findings["verdict_8_velocity_field_key"] = "velocity"
            findings["verdict_8_note"] = (
                "raw record key 'velocity' (meta['field_names']); the dataset keys it internally "
                "as node_features[g]['velocity'] and __getitem__ emits it as the first 2 columns of "
                "graph.x (= cat(velocity, one_hot(node_type))) — it is NOT a named attr on the "
                "returned PyG Data. For Task 10's _expect_velocity helper: the key is 'velocity'."
            )
            findings["secondary_5_3_noise_split_conditional"] = (
                "confirmed — _add_noise fires only when split=='train' (line 127); this audit "
                "constructed split='test' so no noise was applied to the loaded velocities"
            )
        else:
            findings["V14_status"] = (
                "stats files absent on Volume — run compute_cylinder_flow_stats first"
            )
    except Exception as e:
        # The audit must always emit a complete report — record the failure, don't abort.
        findings["dataset_construction_error"] = f"{type(e).__name__}: {e}"
    finally:
        os.chdir(cwd_before)
        shutil.rmtree(work_dir, ignore_errors=True)

    # Emit findings JSON to the Volume + return.
    findings_path = data_dir / "loader_contract_audit.json"
    findings_path.write_text(_json.dumps(findings, indent=2, default=str))
    mgn_volume.commit()

    print("=== CYLINDER_FLOW LOADER-CONTRACT AUDIT ===")
    print(_json.dumps(findings, indent=2, default=str))
    return findings


@app.function(
    volumes={"/vol": mgn_volume},
    timeout=600,
)
def audit_gate_a_pyg_to_meshfield() -> dict:
    """Gate A (verdict 3 / amends D0-02): can the cylinder_flow PyG mesh be coerced
    to a scikit-fem `Basis` (the precondition for the mesh-side physics-lint harness)?

    Returns `{"verdict": "PASS" | "PARTIAL" | "FAIL", "rationale": str, ...}`:
      PASS    — `skfem.MeshTri(p=mesh_pos.T, t=cells.T)` + `Basis(ElementTriP1())`
                construct cleanly from the loader's `mesh_pos` + `cells`.
      PARTIAL — the data is mesh-shaped (valid triangle connectivity over node
                positions) but the scikit-fem coercion itself fails — recovery is
                GridField resampling (`GridField(values=resampled, h=spacing, periodic=False)`).
      FAIL    — the data is fundamentally graph-shaped / scikit-fem absent — the
                mesh harness SKIPs (cover-letter Appendix A.4 variant fires).

    Per design §3.1 activity 5. The plan's `audit_gate_a_dgl_to_meshfield` skeleton
    assumed DGL `ndata`/`edata`; the part-3 loader-contract audit
    (`preflight/loader_contract_audit.json`) showed v2.0.0's `VortexSheddingDataset`
    returns PyG `Data` with `mesh_pos` set and a separate `cells` tensor in the
    test-split `__getitem__` tuple — so this constructs the dataset directly and
    reads the mesh off `ds[0]`. A loud pre-flight assertion on the PyG-extracted
    array shapes/dtypes fires *before* scikit-fem touches them, so a fresh
    PyG→scikit-fem contract surface (axis order, dtype) is caught as such rather
    than absorbed silently. CPU-only.
    """
    import json as _json
    import os
    import shutil
    import tempfile

    import numpy as np
    import torch

    data_dir = Path(DM_CYLINDER_FLOW_DIR)
    needed = [
        data_dir / "meta.json",
        data_dir / "test.tfrecord",
        data_dir / "edge_stats.json",
        data_dir / "node_stats.json",
    ]
    for p in needed:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing — run download_dm_cylinder_flow_dataset + "
                f"compute_cylinder_flow_stats first."
            )
    if _sha256_of_file(data_dir / "meta.json") != DM_CYLINDER_FLOW_META_SHA256:
        raise RuntimeError("meta.json sha mismatch vs D0-23 pin")
    if _sha256_of_file(data_dir / "test.tfrecord") != DM_CYLINDER_FLOW_TEST_SHA256:
        raise RuntimeError("test.tfrecord sha mismatch vs D0-23 pin")

    torch.set_default_dtype(torch.float32)
    out: dict[str, object] = {"physicsnemo_sha": PHYSICSNEMO_SHA}

    try:
        import skfem

        out["skfem_version"] = getattr(skfem, "__version__", "unknown")
    except ImportError as e:
        result = {
            "verdict": "FAIL",
            "rationale": f"scikit-fem ImportError: {e}",
            "recovery_path": "mesh harness SKIPs; cover-letter Appendix A.4 variant fires",
            **out,
        }
        print("=== GATE A VERDICT ===")
        print(_json.dumps(result, indent=2, default=str))
        return result

    from physicsnemo.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset

    cwd_before = os.getcwd()
    work_dir = Path(tempfile.mkdtemp(prefix="cf_gatea_"))
    shutil.copy2(data_dir / "edge_stats.json", work_dir / "edge_stats.json")
    shutil.copy2(data_dir / "node_stats.json", work_dir / "node_stats.json")
    verdict: str | None = None
    rationale = ""
    try:
        os.chdir(work_dir)  # loader reads {edge,node}_stats.json from CWD
        ds = VortexSheddingDataset(
            name="cylinder_flow",
            data_dir=str(data_dir),
            split="test",
            num_samples=1,
            num_steps=5,
            noise_std=0.0,
        )
        item = ds[0]
        if not (isinstance(item, tuple) and len(item) >= 2):
            verdict, rationale = (
                "FAIL",
                f"test-split __getitem__ returned {type(item).__name__}, not the "
                f"(graph, cells, rollout_mask) tuple — cannot recover mesh connectivity",
            )
        else:
            graph, cells_t = item[0], item[1]
            mesh_pos_t = (
                graph["mesh_pos"] if "mesh_pos" in graph else getattr(graph, "mesh_pos", None)
            )
            if mesh_pos_t is None:
                verdict, rationale = (
                    "FAIL",
                    "graph has no `mesh_pos` attribute — no node coordinates to build a Mesh",
                )
            else:
                mesh_pos_np = (
                    mesh_pos_t.numpy() if hasattr(mesh_pos_t, "numpy") else np.asarray(mesh_pos_t)
                )
                cells_np = cells_t.numpy() if hasattr(cells_t, "numpy") else np.asarray(cells_t)

                # --- Pre-flight on the PyG-extracted arrays, BEFORE scikit-fem touches
                # them. Hard asserts (catch a fresh PyG->scikit-fem contract surface
                # loudly). The exact (1923, 3612) come from the part-3 audit on this
                # sha-pinned test.tfrecord.
                out["mesh_pos_shape"] = list(mesh_pos_np.shape)
                out["mesh_pos_dtype"] = str(mesh_pos_np.dtype)
                out["cells_shape"] = list(cells_np.shape)
                out["cells_dtype"] = str(cells_np.dtype)
                assert tuple(mesh_pos_np.shape) == (1923, 2), (
                    f"mesh_pos shape {mesh_pos_np.shape} != (1923, 2) — test trajectory 0 "
                    f"changed? (test.tfrecord is sha-pinned in D0-23)"
                )
                assert tuple(cells_np.shape) == (3612, 3), (
                    f"cells shape {cells_np.shape} != (3612, 3) — test trajectory 0 changed? "
                    f"(test.tfrecord is sha-pinned in D0-23)"
                )
                assert np.issubdtype(mesh_pos_np.dtype, np.floating), (
                    f"mesh_pos dtype {mesh_pos_np.dtype} not floating — unexpected for node coords"
                )
                assert np.issubdtype(cells_np.dtype, np.integer), (
                    f"cells dtype {cells_np.dtype} not integer — unexpected for triangle connectivity"
                )
                cmin, cmax = int(cells_np.min()), int(cells_np.max())
                out["cells_index_range"] = [cmin, cmax]
                out["cells_index_in_node_range"] = (cmin >= 0) and (cmax < mesh_pos_np.shape[0])
                out["mesh_pos_bbox"] = {
                    "min": [float(x) for x in mesh_pos_np.min(axis=0)],
                    "max": [float(x) for x in mesh_pos_np.max(axis=0)],
                }
                if not out["cells_index_in_node_range"]:
                    verdict, rationale = (
                        "FAIL",
                        f"cell vertex indices {[cmin, cmax]} fall outside "
                        f"[0, {mesh_pos_np.shape[0]}) — connectivity does not reference the node "
                        f"array; not a coercible mesh",
                    )
                else:
                    # scikit-fem: MeshTri(p, t) wants p=(ndim, npoints), t=(nverts, ncells).
                    try:
                        p = np.ascontiguousarray(mesh_pos_np.astype(np.float64).T)  # (2, 1923)
                        t = np.ascontiguousarray(cells_np.astype(np.int64).T)  # (3, 3612)
                        mesh = skfem.MeshTri(p, t)
                        out["skfem_mesh_repr"] = repr(mesh)[:200]
                        out["skfem_n_nodes"] = int(mesh.p.shape[1])
                        out["skfem_n_elements"] = int(mesh.t.shape[1])
                        try:
                            basis = skfem.Basis(mesh, skfem.ElementTriP1())
                            out["skfem_basis_repr"] = repr(basis)[:200]
                            out["skfem_basis_ndofs"] = int(basis.N)
                            verdict = "PASS"
                            rationale = (
                                f"scikit-fem MeshTri + Basis(ElementTriP1) reconstructed cleanly "
                                f"from the loader's mesh_pos ({mesh_pos_np.shape[0]} nodes) + cells "
                                f"({cells_np.shape[0]} triangles); {basis.N} P1 DOFs. The mesh "
                                f"harness's PyG->MeshField path is unblocked (the MeshField wrapper "
                                f"itself is verified separately in Task 12's mesh_rollout_adapter.py)."
                            )
                        except Exception as be:
                            verdict = "PARTIAL"
                            rationale = (
                                f"scikit-fem MeshTri constructed but Basis(ElementTriP1) failed: "
                                f"{type(be).__name__}: {be}. Data is mesh-shaped — recover via "
                                f"GridField resampling."
                            )
                            out["recovery_path"] = (
                                "GridField(values=resampled, h=spacing, periodic=False)"
                            )
                    except Exception as me:
                        verdict = "PARTIAL"
                        rationale = (
                            f"data is mesh-shaped (valid triangle connectivity over "
                            f"{mesh_pos_np.shape[0]} node positions, indices in range) but "
                            f"scikit-fem MeshTri coercion failed: {type(me).__name__}: {me}. "
                            f"Recover via GridField resampling."
                        )
                        out["recovery_path"] = (
                            "GridField(values=resampled, h=spacing, periodic=False)"
                        )
    except AssertionError:
        raise  # pre-flight assertions abort loudly — a fresh PyG->scikit-fem surface
    except Exception as e:
        verdict = "FAIL"
        rationale = f"unexpected error before/around scikit-fem coercion: {type(e).__name__}: {e}"
    finally:
        os.chdir(cwd_before)
        shutil.rmtree(work_dir, ignore_errors=True)

    if verdict is None:  # defensive — no branch set it
        verdict, rationale = "FAIL", "audit produced no verdict (unexpected control flow)"
    result = {"verdict": verdict, "rationale": rationale, **out}
    findings_path = data_dir / "gate_a_audit.json"
    findings_path.write_text(_json.dumps(result, indent=2, default=str))
    mgn_volume.commit()
    print("=== GATE A VERDICT ===")
    print(_json.dumps(result, indent=2, default=str))
    return result


def _gate_d_band(err: float) -> str:
    """Pre-registered D0-23 threshold bands for a Gate-D failure (err = max-abs
    velocity error over rollout-mask nodes; only meaningful when err > tolerance)."""
    if err <= 1e-3:
        return "PASS (err <= 1e-3)"
    if err <= 5e-3:
        return "A (near-miss, (1e-3, ~5e-3]): live suspects = (2) edge_stats impl drift vs modulus-DGL [primary]; (3) small (400,300) effect [secondary]. (1)+(4) excluded. Also weigh: the 1e-3 tol was inherited from the LB shipped-reference analog — here it bounds the model's *intrinsic* one-step learned-prediction accuracy (no shipped reference), so tol re-derivation is a candidate ahead of suspect (2)."
    if err <= 0.1:
        return "B (moderate, (~5e-3, ~0.1]): live suspects = (3) (400,300) re-fit [primary, cheap]; if no improvement -> (1)-subtle [re-audit processor interleave restructure]; fix-or-FNO. (2)+(4) excluded."
    return "C (cliff, > ~0.1): clean (1) architecture FAIL -> FNO-on-Darcy fallback (design §3.1.A); upper diagnostic exit."


# Pre-registered D0-23 verdict-confirmation bands (2026-05-12, user-directed Option 3),
# on the 1-step velocity RMSE over rollout-mask nodes in physical units. Paper baseline:
# Pfaff et al. 2020 (MeshGraphNets, ICLR 2021) Table 1, CylinderFlow RMSE-1-step =
# 2.34 +/- 0.12 e-3. See DECISIONS D0-23 "Verdict-confirmation re-fire".
_GATE_D_RMSE_PASS = 5e-3  # ~2x paper baseline
_GATE_D_RMSE_MARGINAL_HI = 1.5e-2  # ~6x paper baseline


def _gate_d_rmse_band(rmse: float) -> str:
    """Pre-registered D0-23 verdict bands on the 1-step velocity RMSE (physical, rollout-mask nodes)."""
    if rmse <= _GATE_D_RMSE_PASS:
        return (
            f"PASS (RMSE-1 <= {_GATE_D_RMSE_PASS:.0e}, ~2x paper baseline 2.34e-3): the recalibrated "
            f"Gate-D criterion IS 'RMSE-1 <= 5e-3'; the 1e-3-max-abs criterion is retired as a category "
            f"error. Finalize D0-23 v1=CONFIRMED, v4=PASS, v5=PASS; proceed to Tasks 8-9."
        )
    if rmse <= _GATE_D_RMSE_MARGINAL_HI:
        return (
            f"MARGINAL (RMSE-1 in ({_GATE_D_RMSE_PASS:.0e}, {_GATE_D_RMSE_MARGINAL_HI:.1e}], ~2-6x paper "
            f"2.34e-3): architecture identity confirmed; reproduction below paper baseline but consistent "
            f"with checkpoint usability -> CS02 proceeds with the limitation named in the v2.1 methodology "
            f"trail (resolution floor of Phase 2's verdict bands). v4=PASS-with-limitation, v5=PASS-with-limitation."
        )
    return (
        f"FAIL (RMSE-1 > {_GATE_D_RMSE_MARGINAL_HI:.1e}, >~6x paper 2.34e-3): adapter structurally correct "
        f"(edge-MLP concat fix verified) but reproduction quality meaningfully below paper baseline "
        f"-> FNO-on-Darcy fallback (design §3.1.A). v4=FAIL, v5=FAIL."
    )


@app.function(
    volumes={"/vol": mgn_volume},
    gpu="A10G",
    timeout=600,
)
def audit_ngc_sample_reproduction(num_steps: int = 50, tolerance: float = 1e-3) -> dict:
    """Gate D falsification test (D0-23 verdict 4): does the NGC checkpoint —
    loaded via the `_legacy_checkpoint_name_remap` adapter into `MeshGraphNet(6,3,3)`
    and run through the v2.0.0 one-step inference protocol
    (examples/cfd/vortex_shedding_mgn/inference.py @ 1ca85d65: denormalize x/y, build
    `invar`, re-normalize velocity cols, `model(invar, edge_attr, graph)`, denormalize
    the prediction, mask non-rollout nodes, integrate `v_pred[t+1] = v_diff_pred + v[t]`)
    — reproduce the next frame of a real cylinder_flow `test.tfrecord` trajectory?

    There is **no bundled "expected output"** in the NGC zip (it ships only `model.pt`),
    so "reproduction" here is "the checkpoint's one-step prediction matches the test
    record's next frame", not "matches a shipped tensor" — the metric therefore bounds
    the model's *intrinsic* one-step accuracy, not a load self-consistency. This is the
    empirical test of verdict-1's amendment ("architecture identity confirmed by Gate D")
    AND of the Task-5-pt2 stats fit.

    Loops one-step predictions over all `num_steps - 1` consecutive frame pairs of the
    first test trajectory (each step uses the *true* previous frame — not the previous
    prediction — so this isolates one-step accuracy from rollout error accumulation).
    **Verdict basis (D0-23 "Verdict-confirmation re-fire", pre-registered 2026-05-12):**
    the 1-step **velocity RMSE** over rollout-mask nodes in physical (denormalized) units
    — directly comparable to Pfaff et al. 2020's CylinderFlow RMSE-1-step = 2.34e-3 (the
    paper's metric; no max-abs-vs-RMSE conversion). Bands: `<= 5e-3` (~2x paper) → PASS;
    `(5e-3, 1.5e-2]` (~2-6x) → MARGINAL (CS02 proceeds, limitation named); `> 1.5e-2` →
    FAIL → FNO-on-Darcy. Also reports, for completeness/diagnosis: the same RMSE over all
    nodes (deflated — masked-out preds are zeroed) and in velocity-std units (convention
    hedge), per-component (vx/vy) RMSE, the |error| p50/p90/p99 quantiles, the per-frame
    max-abs + RMSE, and the legacy max-abs metric/band (now informational — the 1e-3-max-abs
    pass bar is retired as a category error: it was a bit-exact-tensor-repro tolerance from
    the LagrangeBench analog, inapplicable absent a shipped reference tensor). The `tolerance`
    arg is kept for signature compat (recorded as `legacy_max_abs_tolerance_retired`). GPU (A10G).
    """
    import json as _json
    import os
    import shutil
    import sys
    import tempfile

    import torch

    data_dir = Path(DM_CYLINDER_FLOW_DIR)
    for p in (
        data_dir / "meta.json",
        data_dir / "test.tfrecord",
        data_dir / "edge_stats.json",
        data_dir / "node_stats.json",
    ):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing — run download_dm_cylinder_flow_dataset + "
                f"compute_cylinder_flow_stats first."
            )
    if _sha256_of_file(data_dir / "meta.json") != DM_CYLINDER_FLOW_META_SHA256:
        raise RuntimeError("meta.json sha mismatch vs D0-23 pin")
    if _sha256_of_file(data_dir / "test.tfrecord") != DM_CYLINDER_FLOW_TEST_SHA256:
        raise RuntimeError("test.tfrecord sha mismatch vs D0-23 pin")

    sys.path.insert(0, "/root")  # the name-remap adapter is a sibling .py file
    from _legacy_checkpoint_name_remap import remap_modulus_to_physicsnemo_state_dict
    from physicsnemo.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset
    from physicsnemo.models.meshgraphnet import MeshGraphNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float32)
    out: dict[str, object] = {
        "physicsnemo_sha": PHYSICSNEMO_SHA,
        "device": device,
        "tolerance": tolerance,
        "num_steps": num_steps,
    }

    # --- Load the NGC checkpoint via the name-remap adapter into MeshGraphNet(6,3,3) ---
    ckpt_path = Path(
        f"{VOLUME_CHECKPOINT_ROOT}/modulus_ns_meshgraphnet_{NGC_VORTEX_VERSION}/"
        f"vortex_shedding_mgn/model.pt"
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    remapped = remap_modulus_to_physicsnemo_state_dict(raw_sd)
    model = MeshGraphNet(
        input_dim_nodes=6, input_dim_edges=3, output_dim=3
    )  # defaults: relu, no concat trick
    try:
        model.load_state_dict(remapped, strict=True)
    except RuntimeError as e:
        result = {
            "verdict": "FAIL",
            "rationale": (
                f"model.load_state_dict(strict=True) raised — the name-remap adapter renamed keys "
                f"but the *shapes* mismatch (architecture is genuinely different, not a rename): "
                f"{e}. Per D0-23 ladder this is suspect (1), clean FAIL -> FNO-on-Darcy fallback "
                f"(design §3.1.A)."
            ),
            **out,
        }
        print("=== GATE D (NGC SAMPLE REPRODUCTION) VERDICT ===")
        print(_json.dumps(result, indent=2, default=str))
        return result
    model = model.to(device).eval()
    out["model_loaded"] = "strict=True OK (keys + shapes match after name-remap)"

    # --- Construct the test-split dataset (stats staged in CWD) ---
    cwd_before = os.getcwd()
    work_dir = Path(tempfile.mkdtemp(prefix="cf_gated_"))
    shutil.copy2(data_dir / "edge_stats.json", work_dir / "edge_stats.json")
    shutil.copy2(data_dir / "node_stats.json", work_dir / "node_stats.json")
    verdict: str | None = None
    rationale = ""
    try:
        os.chdir(work_dir)
        ds = VortexSheddingDataset(
            name="cylinder_flow",
            data_dir=str(data_dir),
            split="test",
            num_samples=1,
            num_steps=num_steps,
            noise_std=0.0,
        )
        stats = {k: v.to(device) for k, v in ds.node_stats.items()}
        n_pairs = len(ds)  # = 1 * (num_steps - 1)
        out["n_one_step_pairs"] = n_pairs

        worst_mask = 0.0
        worst_all = 0.0
        per_frame = []
        # Accumulators for the RMSE / per-component / quantile metrics (D0-23
        # verdict-confirmation re-fire; physical units = denormalized velocity).
        sq_sum_mask = 0.0  # sum of squared physical errors over (frame, mask-node, {vx,vy})
        n_elem_mask = 0  # element count for the above
        sq_sum_all = 0.0  # ditto over ALL nodes (note: masked-out preds are zeroed, so this
        n_elem_all = 0  #   conflates "model correctly predicts ~0 at boundary" with the zeroing)
        sq_sum_mask_norm = 0.0  # sum of squared *normalized* errors ((v - v_exact)/velocity_std)
        sq_sum_vx = 0.0  # per-component (mask nodes), physical
        sq_sum_vy = 0.0
        n_nodes_mask = 0  # mask-node count (per-component denominator)
        abs_mask_chunks = []  # for the |error| p50/p90/p99 quantiles (mask nodes, physical)
        vstd = stats["velocity_std"]  # (2,) on device
        for tidx in range(n_pairs):
            graph, _cells, mask = ds[tidx]
            graph = graph.to(device)
            # denormalize: x[:,0:2] -> raw v[t]; y[:,0:2] -> raw v_diff_true; y[:,2] -> raw p_true
            graph.x[:, 0:2] = VortexSheddingDataset.denormalize(
                graph.x[:, 0:2], stats["velocity_mean"], stats["velocity_std"]
            )
            graph.y[:, 0:2] = VortexSheddingDataset.denormalize(
                graph.y[:, 0:2], stats["velocity_diff_mean"], stats["velocity_diff_std"]
            )
            graph.y[:, [2]] = VortexSheddingDataset.denormalize(
                graph.y[:, [2]], stats["pressure_mean"], stats["pressure_std"]
            )
            invar = graph.x.clone()
            # one-step (true previous frame): invar[:,0:2] = raw v[t]; re-normalize for the model
            invar[:, 0:2] = VortexSheddingDataset.normalize_node(
                invar[:, 0:2], stats["velocity_mean"], stats["velocity_std"]
            )
            with torch.no_grad():
                pred_i = model(invar, graph.edge_attr, graph).detach()
            pred_i[:, 0:2] = VortexSheddingDataset.denormalize(
                pred_i[:, 0:2], stats["velocity_diff_mean"], stats["velocity_diff_std"]
            )
            pred_i[:, 2] = VortexSheddingDataset.denormalize(
                pred_i[:, 2], stats["pressure_mean"], stats["pressure_std"]
            )
            invar[:, 0:2] = VortexSheddingDataset.denormalize(
                invar[:, 0:2], stats["velocity_mean"], stats["velocity_std"]
            )  # -> raw v[t] again
            mask2 = torch.cat((mask, mask), dim=-1).to(device)  # (n, 2) bool
            pred_diff_masked = torch.where(mask2, pred_i[:, 0:2], torch.zeros_like(pred_i[:, 0:2]))
            v_pred = pred_diff_masked + invar[:, 0:2]  # v_pred[t+1]
            v_exact = graph.y[:, 0:2] + graph.x[:, 0:2]  # v_diff_true + v[t] = v_exact[t+1]
            signed = v_pred - v_exact  # (n, 2)
            diff = signed.abs()
            mask1 = mask.to(device).reshape(-1).bool()
            err_mask = float(diff[mask1].max().item()) if mask1.any() else float("nan")
            err_all = float(diff.max().item())
            worst_mask = max(worst_mask, err_mask)
            worst_all = max(worst_all, err_all)
            # RMSE / per-component / quantile accumulators
            sm = signed[mask1]  # (n_mask, 2)
            sq_sum_mask += float((sm * sm).sum().item())
            n_elem_mask += int(sm.numel())
            sq_sum_all += float((signed * signed).sum().item())
            n_elem_all += int(signed.numel())
            sm_norm = sm / vstd
            sq_sum_mask_norm += float((sm_norm * sm_norm).sum().item())
            sq_sum_vx += float((sm[:, 0] * sm[:, 0]).sum().item())
            sq_sum_vy += float((sm[:, 1] * sm[:, 1]).sum().item())
            n_nodes_mask += int(sm.shape[0])
            abs_mask_chunks.append(diff[mask1].reshape(-1).detach().cpu())
            rmse_frame_mask = (
                float(((sm * sm).mean()).sqrt().item()) if sm.numel() else float("nan")
            )
            per_frame.append(
                {
                    "tidx": tidx,
                    "err_mask": err_mask,
                    "err_all": err_all,
                    "rmse_mask": rmse_frame_mask,
                }
            )

        rmse_mask = (sq_sum_mask / n_elem_mask) ** 0.5 if n_elem_mask else float("nan")
        rmse_all = (sq_sum_all / n_elem_all) ** 0.5 if n_elem_all else float("nan")
        rmse_mask_norm = (sq_sum_mask_norm / n_elem_mask) ** 0.5 if n_elem_mask else float("nan")
        rmse_vx_mask = (sq_sum_vx / n_nodes_mask) ** 0.5 if n_nodes_mask else float("nan")
        rmse_vy_mask = (sq_sum_vy / n_nodes_mask) ** 0.5 if n_nodes_mask else float("nan")
        abs_all_mask = torch.cat(abs_mask_chunks) if abs_mask_chunks else torch.zeros(1)
        q = torch.quantile(abs_all_mask, torch.tensor([0.50, 0.90, 0.99])).tolist()

        out["max_abs_err_velocity_mask_nodes"] = worst_mask
        out["max_abs_err_velocity_all_nodes"] = worst_all
        out["rmse_1step_velocity_mask_nodes"] = rmse_mask  # <-- D0-23 verdict basis
        out["rmse_1step_velocity_all_nodes"] = rmse_all
        out["rmse_1step_velocity_mask_nodes_normalized"] = rmse_mask_norm  # convention hedge
        out["rmse_1step_velocity_vx_mask"] = rmse_vx_mask
        out["rmse_1step_velocity_vy_mask"] = rmse_vy_mask
        out["abs_err_velocity_mask_quantiles"] = {"p50": q[0], "p90": q[1], "p99": q[2]}
        out["n_mask_nodes_per_frame"] = n_nodes_mask // n_pairs if n_pairs else 0
        out["paper_baseline_rmse_1step_cylinderflow"] = 2.34e-3  # Pfaff et al. 2020 Table 1
        out["legacy_max_abs_tolerance_retired"] = tolerance  # was the (category-error) pass bar
        out["rmse_pass_threshold"] = _GATE_D_RMSE_PASS
        out["rmse_marginal_upper"] = _GATE_D_RMSE_MARGINAL_HI
        worst_frame = max(per_frame, key=lambda d: d["err_mask"])
        out["per_frame_first8"] = per_frame[:8]
        out["per_frame_worst"] = worst_frame
        out["gate_d_band"] = _gate_d_band(worst_mask)  # legacy max-abs band, informational
        out["gate_d_rmse_band"] = _gate_d_rmse_band(rmse_mask)  # the verdict basis
        if rmse_mask <= _GATE_D_RMSE_PASS:
            verdict = "PASS"
        elif rmse_mask <= _GATE_D_RMSE_MARGINAL_HI:
            verdict = "MARGINAL"
        else:
            verdict = "FAIL"
        rationale = (
            f"NGC checkpoint one-step reproduction (post edge-MLP-column-reorder adapter; verdict-"
            f"confirmation re-fire, NO adapter change). 1-step velocity RMSE = {rmse_mask:.3e} over "
            f"rollout-mask nodes (physical units; {rmse_all:.3e} over all nodes [masked-out preds zeroed → "
            f"deflated]; {rmse_mask_norm:.3e} in velocity-std units), per-component vx={rmse_vx_mask:.3e} / "
            f"vy={rmse_vy_mask:.3e}, |err| p50/p90/p99 = {q[0]:.3e}/{q[1]:.3e}/{q[2]:.3e}, max-abs "
            f"{worst_mask:.3e}, across {n_pairs} pairs (true previous frame each). Pfaff et al. 2020 "
            f"CylinderFlow RMSE-1-step baseline = 2.34e-3. D0-23 verdict band: {out['gate_d_rmse_band']} "
            f"The legacy 1e-3-max-abs criterion is retired (a bit-exact-tensor-repro tolerance from the "
            f"LagrangeBench analog, inapplicable here — no shipped reference tensor)."
        )
    except Exception as e:
        verdict = "FAIL"
        rationale = f"unexpected error during the reproduction loop: {type(e).__name__}: {e}"
    finally:
        os.chdir(cwd_before)
        shutil.rmtree(work_dir, ignore_errors=True)

    if verdict is None:
        verdict, rationale = "FAIL", "no verdict produced (unexpected control flow)"
    result = {"verdict": verdict, "rationale": rationale, **out}
    findings_path = data_dir / "gate_d_reproduction.json"
    findings_path.write_text(_json.dumps(result, indent=2, default=str))
    mgn_volume.commit()
    print("=== GATE D (NGC SAMPLE REPRODUCTION) VERDICT ===")
    print(_json.dumps(result, indent=2, default=str))
    return result


@app.function(
    volumes={"/vol": mgn_volume},
    gpu="A10G",
    timeout=900,
)
def smoke_substrate_class_vortex_shedding(num_steps: int = 400) -> dict:
    """Task 9 — 1-traj substrate-class smoke for the DeepMind cylinder_flow / NGC MGN
    (design §3.1 activity 8; D0-23 verdicts 6 + 7).

    Runs (a) the ground-truth cylinder_flow test trajectory's first ``num_steps``
    frames and (b) a true MGN rollout over the same window (the corrected name-remap
    adapter; ``examples/cfd/vortex_shedding_mgn/inference.py`` @ 1ca85d65's ``predict()``
    protocol -- denorm, prev-prediction feedback, mask non-rollout (inflow/wall) nodes,
    integrate ``v[t+1] = v_diff_pred + v[t]``), then computes the 3 discriminating
    observables on each via scikit-fem P1 finite elements on the cylinder_flow mesh
    (the Gate A path -- verified PASS in verdict 3):

      1. **relative incompressibility residual** ``rel_div = ∫|∇·v| dV / ∫ ||∇v||_F dV``
         -- 0 for incompressible NS in the continuum; the GT value is the data's
         discretization floor (the COMSOL solver's), the rollout value should stay within
         ~an order of magnitude of it (the MGN does not enforce ∇·v=0). The PH-CON-001 signal.
      2. **KE(t) = 0.5 ∫ |v|² dV** -> dKE/dt monotone (rising OR falling) in the
         post-warmup window? Prediction (open-driven-dissipative, boundary-driven
         sub-class): NOT monotone -- KE oscillates around a steady mean (inflow/outflow/
         dissipation balance, vortex shedding pumps energy in and out of the wake). The
         PH-CON-002 / PH-CON-003 signal (their strictly-dissipative-or-conservative
         assumption fails iff this oscillates).
      3. **Strouhal St = f_s · D / U** -- f_s = dominant FFT peak of the lift proxy
         ``∫ v_y dV`` (the transverse fluid momentum; detrended, post-warmup, Hann-
         windowed); D = cylinder diameter from the WALL nodes off the channel boundary;
         U from the INFLOW nodes (reported with both the mean and the max/centerline
         convention). Prediction: St ∈ [0.16, 0.21] (cylinder wake, Re ∈ [100, 300]);
         smoke band [0.14, 0.24].

    Verdict 6 = ``"open-driven-dissipative"`` iff the GROUND TRUTH satisfies all three
    (the substrate is the physics, not the surrogate); the rollout's values are an "and
    the MGN reproduces this" cross-check, reported but not gating. A GT observable
    diverging -> ``"UNEXPECTED (...)"`` + ``pattern_a_drift=True`` -> the routing is a
    D-entry amendment, **not** a Gate-D FAIL (design §3.1 disambiguation: substrate-class
    divergence is methodology-refinement, not checkpoint-usability failure). Verdict 7 =
    ``"Y"`` -- this entrypoint writes its findings to the persistent Volume and commits,
    and ``inference.py``'s ``os.chdir(rollout_dir)`` isolation pattern carries over for
    Phase 2's inference entrypoint. GPU (A10G); ``dt = 0.01`` per the cylinder_flow
    ``meta.json``.
    """
    import json as _json
    import os
    import shutil
    import sys
    import tempfile

    import numpy as np
    import torch

    data_dir = Path(DM_CYLINDER_FLOW_DIR)
    for p in (
        data_dir / "meta.json",
        data_dir / "test.tfrecord",
        data_dir / "edge_stats.json",
        data_dir / "node_stats.json",
    ):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing -- run download_dm_cylinder_flow_dataset + "
                f"compute_cylinder_flow_stats first."
            )
    if _sha256_of_file(data_dir / "meta.json") != DM_CYLINDER_FLOW_META_SHA256:
        raise RuntimeError("meta.json sha mismatch vs D0-23 pin")
    if _sha256_of_file(data_dir / "test.tfrecord") != DM_CYLINDER_FLOW_TEST_SHA256:
        raise RuntimeError("test.tfrecord sha mismatch vs D0-23 pin")

    import skfem

    sys.path.insert(0, "/root")  # the name-remap adapter is a sibling .py file
    from _legacy_checkpoint_name_remap import remap_modulus_to_physicsnemo_state_dict
    from physicsnemo.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset
    from physicsnemo.models.meshgraphnet import MeshGraphNet

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_dtype(torch.float32)
    dt = 0.01  # cylinder_flow meta.json: "dt": 0.01
    out: dict[str, object] = {
        "physicsnemo_sha": PHYSICSNEMO_SHA,
        "device": device,
        "num_steps": num_steps,
        "dt": dt,
        "skfem_version": getattr(skfem, "__version__", "unknown"),
    }

    # --- Load the NGC checkpoint via the corrected name-remap adapter ---
    ckpt_path = Path(
        f"{VOLUME_CHECKPOINT_ROOT}/modulus_ns_meshgraphnet_{NGC_VORTEX_VERSION}/"
        f"vortex_shedding_mgn/model.pt"
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model = MeshGraphNet(input_dim_nodes=6, input_dim_edges=3, output_dim=3)
    model.load_state_dict(remap_modulus_to_physicsnemo_state_dict(raw_sd), strict=True)
    model = model.to(device).eval()
    out["model_loaded"] = "strict=True OK (corrected name-remap adapter)"

    cwd_before = os.getcwd()
    work_dir = Path(tempfile.mkdtemp(prefix="cf_substrate_"))
    shutil.copy2(data_dir / "edge_stats.json", work_dir / "edge_stats.json")
    shutil.copy2(data_dir / "node_stats.json", work_dir / "node_stats.json")
    verdict: str | None = None
    rationale = ""
    try:
        os.chdir(work_dir)  # loader reads {edge,node}_stats.json from CWD
        ds = VortexSheddingDataset(
            name="cylinder_flow",
            data_dir=str(data_dir),
            split="test",
            num_samples=1,
            num_steps=num_steps,
            noise_std=0.0,
        )
        stats = {k: v.to(device) for k, v in ds.node_stats.items()}
        n_pairs = len(ds)  # = num_steps - 1
        out["n_frames"] = n_pairs

        # --- mesh + node classes (one-hot encodes node_type: class 0=NORMAL(0),
        #     1=INFLOW(4), 2=OUTFLOW(5), 3=WALL(6) per _one_hot_encode's where(==0,0,nt-3)) ---
        graph0, cells0, _mask0 = ds[0]
        mesh_pos = np.asarray(graph0["mesh_pos"]).astype(np.float64)  # (N, 2)
        cells_np = np.asarray(cells0).astype(np.int64)  # (M, 3)
        node_class = np.asarray(graph0.x[:, 2:6].argmax(dim=1)).astype(int)  # (N,)
        n_nodes = mesh_pos.shape[0]
        assert mesh_pos.shape == (n_nodes, 2) and cells_np.shape[1] == 3, (
            f"unexpected mesh shapes: mesh_pos {mesh_pos.shape}, cells {cells_np.shape}"
        )
        out["n_nodes"] = n_nodes
        out["n_elements"] = int(cells_np.shape[0])
        x_min, y_min = mesh_pos.min(axis=0)
        x_max, y_max = mesh_pos.max(axis=0)
        out["mesh_bbox"] = {
            "min": [float(x_min), float(y_min)],
            "max": [float(x_max), float(y_max)],
        }
        out["node_class_counts"] = {
            "NORMAL": int((node_class == 0).sum()),
            "INFLOW": int((node_class == 1).sum()),
            "OUTFLOW": int((node_class == 2).sum()),
            "WALL": int((node_class == 3).sum()),
        }

        # cylinder = WALL nodes off the channel boundary (not near y_min/y_max/x_min/x_max)
        ytol = 0.02 * (y_max - y_min)
        xtol = 0.02 * (x_max - x_min)
        is_wall = node_class == 3
        on_channel = (
            (np.abs(mesh_pos[:, 1] - y_min) < ytol)
            | (np.abs(mesh_pos[:, 1] - y_max) < ytol)
            | (np.abs(mesh_pos[:, 0] - x_min) < xtol)
            | (np.abs(mesh_pos[:, 0] - x_max) < xtol)
        )
        cyl_mask = is_wall & (~on_channel)
        if cyl_mask.sum() < 4:
            raise RuntimeError(
                f"only {int(cyl_mask.sum())} candidate cylinder-surface nodes found "
                f"(WALL nodes off the channel boundary) -- cannot estimate the cylinder "
                f"diameter; mesh geometry differs from the expected channel-with-cylinder."
            )
        cyl_xy = mesh_pos[cyl_mask]
        diam_x = float(cyl_xy[:, 0].max() - cyl_xy[:, 0].min())
        diam_y = float(cyl_xy[:, 1].max() - cyl_xy[:, 1].min())
        cyl_d = max(diam_x, diam_y)
        cyl_center = [float(cyl_xy[:, 0].mean()), float(cyl_xy[:, 1].mean())]
        out["cylinder"] = {
            "n_surface_nodes": int(cyl_mask.sum()),
            "diameter_x": diam_x,
            "diameter_y": diam_y,
            "diameter": cyl_d,
            "center": cyl_center,
        }
        assert 0.0 < cyl_d < 0.5 * (y_max - y_min), (
            f"cylinder diameter estimate {cyl_d:.4f} implausible vs channel height "
            f"{y_max - y_min:.4f} -- WALL-node clustering picked up the wrong nodes"
        )

        # --- scikit-fem P1 basis on the mesh; verify DOF order == node order ---
        mesh = skfem.MeshTri(np.ascontiguousarray(mesh_pos.T), np.ascontiguousarray(cells_np.T))
        basis = skfem.Basis(mesh, skfem.ElementTriP1())
        assert n_nodes == basis.N, f"skfem basis has {basis.N} DOFs != {n_nodes} nodes"
        assert np.allclose(mesh.p.T, mesh_pos), "skfem reordered the nodes -- DOF<->node map broken"

        def _field_observables(v):
            """v: (n_nodes, 2) physical velocity -> dict of FE-integrated observables."""
            vx_f = basis.interpolate(v[:, 0].astype(np.float64))
            vy_f = basis.interpolate(v[:, 1].astype(np.float64))
            gvx = vx_f.grad  # (2, n_elem, n_qp): [dvx/dx, dvx/dy]
            gvy = vy_f.grad  # [dvy/dx, dvy/dy]
            div = gvx[0] + gvy[1]
            frob = np.sqrt(gvx[0] ** 2 + gvx[1] ** 2 + gvy[0] ** 2 + gvy[1] ** 2)
            int_abs_div = float(np.sum(np.abs(div) * basis.dx))
            int_frob = float(np.sum(frob * basis.dx))
            ke = 0.5 * float(np.sum((vx_f.value**2 + vy_f.value**2) * basis.dx))
            lift = float(np.sum(vy_f.value * basis.dx))  # ∫ v_y dV (transverse momentum)
            return {
                "int_abs_div": int_abs_div,
                "int_frob": int_frob,
                "rel_div": int_abs_div / int_frob if int_frob > 0 else float("nan"),
                "ke": ke,
                "lift": lift,
            }

        # --- baked-in metric sanity (per CLAUDE.md pre-flight check 3): the FE
        #     divergence must give 0 for a div-free linear field and total area for v=(x,0).
        total_area = float(np.sum(basis.dx))
        sanity_divfree = _field_observables(
            np.stack([mesh_pos[:, 1], -mesh_pos[:, 0]], axis=1)
        )  # v=(y,-x): div=0 exactly (P1)
        sanity_div1 = _field_observables(
            np.stack([mesh_pos[:, 0], np.zeros(n_nodes)], axis=1)
        )  # v=(x,0): div=1 exactly -> ∫|div|dV = total_area
        out["fe_divergence_self_test"] = {
            "total_area": total_area,
            "int_abs_div_for_v_eq_(y,-x)": sanity_divfree["int_abs_div"],
            "int_abs_div_for_v_eq_(x,0)": sanity_div1["int_abs_div"],
        }
        if not (
            sanity_divfree["int_abs_div"] < 1e-8 * max(1.0, total_area)
            and abs(sanity_div1["int_abs_div"] - total_area) < 1e-6 * total_area
        ):
            raise RuntimeError(
                "FE divergence self-test FAILED: "
                f"∫|∇·v|dV for v=(y,-x) is {sanity_divfree['int_abs_div']:.3e} (expect ~0), "
                f"for v=(x,0) is {sanity_div1['int_abs_div']:.3e} (expect total_area "
                f"{total_area:.3e}). The scikit-fem P1 divergence wiring is wrong -- do not "
                f"trust the substrate-class observables until this passes."
            )

        # --- GT velocity series (frames 0..n_pairs-1; graph.x[:,0:2] is normalized) ---
        v_gt = np.empty((n_pairs, n_nodes, 2), dtype=np.float64)
        for t in range(n_pairs):
            g_t, _c, _m = ds[t]
            v_gt[t] = (
                VortexSheddingDataset.denormalize(
                    g_t.x[:, 0:2].clone(), stats["velocity_mean"].cpu(), stats["velocity_std"].cpu()
                )
                .numpy()
                .astype(np.float64)
            )

        # --- MGN true rollout (inference.py predict() protocol; prev-prediction feedback) ---
        v_pred = np.empty((n_pairs, n_nodes, 2), dtype=np.float64)
        prev_v = None  # raw (denormalized) velocity carried from the previous prediction
        with torch.no_grad():
            for t in range(n_pairs):
                g_t, _c, mask_t = ds[t]
                g_t = g_t.to(device)
                g_t.x[:, 0:2] = VortexSheddingDataset.denormalize(
                    g_t.x[:, 0:2], stats["velocity_mean"], stats["velocity_std"]
                )  # -> raw v[t]
                invar = g_t.x.clone()
                if t != 0:  # rollout: feed the previous prediction's velocity
                    invar[:, 0:2] = prev_v
                invar[:, 0:2] = VortexSheddingDataset.normalize_node(
                    invar[:, 0:2], stats["velocity_mean"], stats["velocity_std"]
                )
                pred_i = model(invar, g_t.edge_attr, g_t).detach()
                pred_i[:, 0:2] = VortexSheddingDataset.denormalize(
                    pred_i[:, 0:2], stats["velocity_diff_mean"], stats["velocity_diff_std"]
                )
                invar[:, 0:2] = VortexSheddingDataset.denormalize(
                    invar[:, 0:2], stats["velocity_mean"], stats["velocity_std"]
                )  # -> raw v[t] (or raw prev-pred) again
                mask2 = torch.cat((mask_t, mask_t), dim=-1).to(device)  # (N, 2) bool
                v_diff_masked = torch.where(mask2, pred_i[:, 0:2], torch.zeros_like(pred_i[:, 0:2]))
                v_next = v_diff_masked + invar[:, 0:2]  # v[t+1]
                v_pred[t] = v_next.cpu().numpy().astype(np.float64)
                prev_v = v_next.clone()

        # --- observables per frame, both series ---
        warmup = min(40, n_pairs // 5)
        out["warmup_frames_skipped"] = warmup

        def _series_observables(v_series, tag):
            obs = [_field_observables(v_series[t]) for t in range(n_pairs)]
            rel_div = np.array([o["rel_div"] for o in obs])
            ke = np.array([o["ke"] for o in obs])
            lift = np.array([o["lift"] for o in obs])
            dke = np.diff(ke)
            dke_post = dke[warmup:]
            ke_post = ke[warmup:]
            monotone = bool(np.all(dke_post > 0) or np.all(dke_post < 0))
            n_sign_changes = int(np.sum(np.diff(np.sign(dke_post)) != 0))
            ke_mean = float(ke_post.mean())
            ke_cv = float(ke_post.std() / ke_mean) if ke_mean != 0 else float("nan")
            # Strouhal: dominant FFT peak of the lift proxy (∫ v_y dV), post-warmup, Hann-windowed
            lift_post = lift[warmup:]
            lp = (lift_post - lift_post.mean()) * np.hanning(len(lift_post))
            spec = np.abs(np.fft.rfft(lp))
            freqs = np.fft.rfftfreq(len(lift_post), d=dt)  # Hz
            peak_idx = int(np.argmax(spec[1:]) + 1)  # skip DC
            f_s = float(freqs[peak_idx])
            peak_prominence = float(spec[peak_idx] / (np.median(spec[1:]) + 1e-30))
            # KE oscillation frequency (should be ~2 f_s for a von Karman street)
            kp = (ke_post - ke_post.mean()) * np.hanning(len(ke_post))
            ke_spec = np.abs(np.fft.rfft(kp))
            ke_peak_idx = int(np.argmax(ke_spec[1:]) + 1)
            f_ke = float(freqs[ke_peak_idx]) if ke_peak_idx < len(freqs) else float("nan")
            return {
                "rel_div_mean": float(rel_div[warmup:].mean()),
                "rel_div_max": float(rel_div[warmup:].max()),
                "ke_mean": ke_mean,
                "ke_cv": ke_cv,
                "ke_dKEdt_monotone": monotone,
                "ke_dKEdt_sign_changes": n_sign_changes,
                "shedding_freq_hz": f_s,
                "shedding_peak_prominence": peak_prominence,
                "ke_osc_freq_hz": f_ke,
                "strouhal_U_mean": f_s * cyl_d / u_mean if u_mean > 0 else float("nan"),
                "strouhal_U_max": f_s * cyl_d / u_max if u_max > 0 else float("nan"),
                "_tag": tag,
            }

        # inflow velocity U (mean=bulk, max=centerline) from the GT INFLOW nodes,
        # averaged over the post-warmup window (robust to a startup transient)
        inflow = node_class == 1
        if inflow.sum() == 0:
            raise RuntimeError("no INFLOW nodes (one-hot class 1) found -- cannot estimate U")
        speed_gt = np.linalg.norm(v_gt[:, inflow, :], axis=2)  # (n_pairs, n_inflow)
        u_mean = float(speed_gt[warmup:].mean())
        u_max = float(speed_gt[warmup:].max())
        out["inflow_velocity"] = {
            "U_mean": u_mean,
            "U_max": u_max,
            "n_inflow_nodes": int(inflow.sum()),
        }

        gt_obs = _series_observables(v_gt, "ground_truth")
        rollout_obs = _series_observables(v_pred, "mgn_rollout")
        out["ground_truth_observables"] = gt_obs
        out["mgn_rollout_observables"] = rollout_obs
        out["reynolds_band_design_target_strouhal"] = [0.16, 0.21]

        # --- substrate-class verdict, anchored on the GROUND TRUTH ---
        st_gt_in_smoke = (0.14 <= gt_obs["strouhal_U_mean"] <= 0.24) or (
            0.14 <= gt_obs["strouhal_U_max"] <= 0.24
        )
        st_gt_in_tight = (0.16 <= gt_obs["strouhal_U_mean"] <= 0.21) or (
            0.16 <= gt_obs["strouhal_U_max"] <= 0.21
        )
        gt_incompressible = gt_obs["rel_div_mean"] < 0.10
        gt_ke_oscillates = (
            (not gt_obs["ke_dKEdt_monotone"])
            and (gt_obs["ke_cv"] > 1e-3)
            and (gt_obs["shedding_peak_prominence"] > 3.0)
        )
        fits_odd = bool(gt_incompressible and gt_ke_oscillates and st_gt_in_smoke)
        # MGN-reproduces cross-check (reported, not gating)
        rollout_reproduces = bool(
            (rollout_obs["rel_div_mean"] < 5.0 * gt_obs["rel_div_mean"] + 0.05)
            and (not rollout_obs["ke_dKEdt_monotone"])
            and (
                (0.12 <= rollout_obs["strouhal_U_mean"] <= 0.26)
                or (0.12 <= rollout_obs["strouhal_U_max"] <= 0.26)
            )
        )
        out["gt_strouhal_in_design_tight_band"] = st_gt_in_tight
        out["mgn_reproduces_substrate_signature"] = rollout_reproduces
        if fits_odd:
            verdict = "open-driven-dissipative"
            out["pattern_a_drift"] = False
            rationale = (
                f"cylinder_flow test trajectory (GT) fits substrate-class "
                f"'open-driven-dissipative' (boundary-driven sub-class): incompressible "
                f"(∫|∇·v|/∫||∇v||_F = {gt_obs['rel_div_mean']:.3e} << 1), KE oscillates "
                f"around a steady mean (dKE/dt sign-changes={gt_obs['ke_dKEdt_sign_changes']}, "
                f"KE CV={gt_obs['ke_cv']:.2e}, not monotone), Strouhal St~"
                f"[{gt_obs['strouhal_U_mean']:.3f} (U_mean), {gt_obs['strouhal_U_max']:.3f} "
                f"(U_max)] (cylinder-wake signature; design tight band [0.16,0.21] "
                f"{'HIT' if st_gt_in_tight else 'near'}, f_s={gt_obs['shedding_freq_hz']:.3f} Hz, "
                f"D={cyl_d:.4f}, peak prominence {gt_obs['shedding_peak_prominence']:.1f}x). The "
                f"MGN rollout {'reproduces' if rollout_reproduces else 'PARTIALLY reproduces'} "
                f"this signature (rel_div={rollout_obs['rel_div_mean']:.3e}, KE monotone="
                f"{rollout_obs['ke_dKEdt_monotone']}, St~[{rollout_obs['strouhal_U_mean']:.3f},"
                f"{rollout_obs['strouhal_U_max']:.3f}]). D0-23 verdict 6 = "
                f"'open-driven-dissipative' -> MGN_DATASET_SYSTEM_CLASS dispatch lands "
                f"per design §2.2; PH-CON-002/003 SKIP-with-reason on this substrate."
            )
        else:
            verdict = (
                f"UNEXPECTED (rel_div_gt={gt_obs['rel_div_mean']:.3e} [need <0.10], "
                f"ke_monotone_gt={gt_obs['ke_dKEdt_monotone']} ke_cv_gt={gt_obs['ke_cv']:.2e} "
                f"prominence={gt_obs['shedding_peak_prominence']:.1f}, "
                f"St_gt~[{gt_obs['strouhal_U_mean']:.3f},{gt_obs['strouhal_U_max']:.3f}] "
                f"[need a value in [0.14,0.24]])"
            )
            out["pattern_a_drift"] = True
            rationale = (
                f"the cylinder_flow GT trajectory does NOT cleanly fit "
                f"'open-driven-dissipative' on all three observables -- {verdict}. Per design "
                f"§3.1 disambiguation this is a Pattern-A drift on the substrate-class smoke "
                f"-> a D0-23 amendment captures the surprise (NOT a Gate-D FAIL: substrate-"
                f"class divergence is methodology-refinement, not checkpoint-usability). "
                f"Inspect the per-observable values; the most likely explanations are the "
                f"Strouhal U-convention (mean vs centerline) or the rel_div discretization "
                f"floor being looser than 0.10 on this mesh -- both would refine the band "
                f"rather than overturn the class."
            )
    except AssertionError:
        raise  # pre-flight assertions abort loudly
    except Exception as e:
        verdict = "FAIL"
        out["pattern_a_drift"] = None
        rationale = f"unexpected error during the substrate-class smoke: {type(e).__name__}: {e}"
    finally:
        os.chdir(cwd_before)
        shutil.rmtree(work_dir, ignore_errors=True)

    if verdict is None:
        verdict, rationale = "FAIL", "no verdict produced (unexpected control flow)"
    out["persistent_volume_decision"] = "Y"  # this entrypoint writes to + commits the Volume
    result = {"verdict": verdict, "rationale": rationale, **out}
    findings_path = data_dir / "substrate_class_smoke.json"
    findings_path.write_text(_json.dumps(result, indent=2, default=str))
    mgn_volume.commit()
    print("=== SUBSTRATE-CLASS SMOKE (vortex_shedding) VERDICT ===")
    print(_json.dumps(result, indent=2, default=str))
    return result


# ---------------------------------------------------------------------------
# Phase 2 — Task 1: Strouhal pre-check across cylinder_flow test trajectories.
# ---------------------------------------------------------------------------


def _strouhal_for_test_trajectory(rec, dt: float, traj_idx: int) -> dict:
    """Compute per-trajectory Strouhal from one raw test.tfrecord record.

    `rec`: a `VortexSheddingDataset._decode_record(...)` output -- dict of
    arrays where `velocity` is `(T, N, 2)` and the static fields
    (`mesh_pos`, `cells`, `node_type`) are tiled to `(T, ...)` by the
    decoder (V10 / V12 per the loader-contract audit). Raw `node_type`
    values are `{0, 3, 4, 5, 6}` per V16 -- the substrate-class smoke's
    one-hot convention maps `4 -> INFLOW`, `5 -> OUTFLOW`, `6 -> WALL`.

    Returns a dict with `strouhal_U_mean` / `strouhal_U_max` (the standard
    two conventions for cylinder-wake Strouhal; the substrate-class smoke
    reports both), the cylinder geometry, and the inflow-speed window.
    On geometry-detection failures returns `{"traj_idx", "error"}` only;
    the caller decides how to count those.
    """
    import numpy as np
    import skfem

    velocity = np.asarray(rec["velocity"]).astype(np.float64)  # (T, N, 2)
    mesh_pos = np.asarray(rec["mesh_pos"][0]).astype(np.float64)  # (N, 2)
    cells = np.asarray(rec["cells"][0]).astype(np.int64)  # (M, 3)
    node_type = np.asarray(rec["node_type"][0]).reshape(-1).astype(int)  # (N,)

    n_frames, n_nodes = velocity.shape[:2]
    x_min, y_min = mesh_pos.min(axis=0)
    x_max, y_max = mesh_pos.max(axis=0)

    # Cylinder surface = WALL (==6) nodes NOT on the channel rectangle perimeter.
    # Same machinery as smoke_substrate_class_vortex_shedding.
    is_wall = node_type == 6
    ytol = 0.02 * (y_max - y_min)
    xtol = 0.02 * (x_max - x_min)
    on_channel = (
        (np.abs(mesh_pos[:, 1] - y_min) < ytol)
        | (np.abs(mesh_pos[:, 1] - y_max) < ytol)
        | (np.abs(mesh_pos[:, 0] - x_min) < xtol)
        | (np.abs(mesh_pos[:, 0] - x_max) < xtol)
    )
    cyl_mask = is_wall & (~on_channel)
    n_cyl = int(cyl_mask.sum())
    if n_cyl < 4:
        return {
            "traj_idx": traj_idx,
            "error": f"only {n_cyl} cylinder-surface nodes (need >= 4)",
        }
    cyl_xy = mesh_pos[cyl_mask]
    diam_x = float(cyl_xy[:, 0].max() - cyl_xy[:, 0].min())
    diam_y = float(cyl_xy[:, 1].max() - cyl_xy[:, 1].min())
    cyl_d = max(diam_x, diam_y)
    cyl_center = [float(cyl_xy[:, 0].mean()), float(cyl_xy[:, 1].mean())]
    if not (0.0 < cyl_d < 0.5 * (y_max - y_min)):
        return {
            "traj_idx": traj_idx,
            "error": (
                f"implausible cylinder diameter {cyl_d:.4f} vs channel height "
                f"{float(y_max - y_min):.4f} (WALL-node clustering picked up "
                f"wrong nodes)"
            ),
        }

    inflow_mask = node_type == 4
    n_inflow = int(inflow_mask.sum())
    if n_inflow == 0:
        return {
            "traj_idx": traj_idx,
            "error": "no INFLOW (node_type==4) nodes; cannot estimate U",
        }

    mesh = skfem.MeshTri(np.ascontiguousarray(mesh_pos.T), np.ascontiguousarray(cells.T))
    basis = skfem.Basis(mesh, skfem.ElementTriP1())
    # Belt-and-braces invariant: skfem may reorder nodes; if so, the
    # downstream FE-interpolate-by-node-array breaks. The substrate-class
    # smoke asserts the same invariant.
    if not np.allclose(mesh.p.T, mesh_pos):
        return {
            "traj_idx": traj_idx,
            "error": "skfem reordered nodes -- DOF<->node map broken",
        }

    # Lift proxy: integral of v_y over the domain, per frame. Same as the
    # substrate-class smoke's lift series. FFT of the post-warmup window
    # yields the vortex-shedding frequency f_s.
    lift = np.empty(n_frames, dtype=np.float64)
    for t in range(n_frames):
        vy_f = basis.interpolate(velocity[t, :, 1])
        lift[t] = float(np.sum(vy_f.value * basis.dx))

    warmup = min(40, n_frames // 5)
    speed_in = np.linalg.norm(velocity[:, inflow_mask, :], axis=2)  # (T, n_inflow)
    u_mean = float(speed_in[warmup:].mean())
    u_max = float(speed_in[warmup:].max())

    lift_post = lift[warmup:]
    if len(lift_post) < 8:
        return {
            "traj_idx": traj_idx,
            "error": f"post-warmup window too short ({len(lift_post)} frames)",
        }
    lp = (lift_post - lift_post.mean()) * np.hanning(len(lift_post))
    spec = np.abs(np.fft.rfft(lp))
    freqs = np.fft.rfftfreq(len(lift_post), d=dt)
    peak_idx = int(np.argmax(spec[1:]) + 1)  # skip DC
    f_s = float(freqs[peak_idx])
    peak_prom = float(spec[peak_idx] / (np.median(spec[1:]) + 1e-30))

    return {
        "traj_idx": traj_idx,
        "n_nodes": int(n_nodes),
        "n_cells": int(cells.shape[0]),
        "n_cyl_surface_nodes": n_cyl,
        "n_inflow_nodes": n_inflow,
        "cyl_diameter": cyl_d,
        "cyl_diameter_x": diam_x,
        "cyl_diameter_y": diam_y,
        "cyl_center": cyl_center,
        "U_mean": u_mean,
        "U_max": u_max,
        "f_s_Hz": f_s,
        "shedding_peak_prominence": peak_prom,
        "strouhal_U_mean": f_s * cyl_d / u_mean if u_mean > 0 else float("nan"),
        "strouhal_U_max": f_s * cyl_d / u_max if u_max > 0 else float("nan"),
        "warmup_frames": warmup,
    }


@app.function(
    volumes={"/vol": mgn_volume},
    timeout=60 * 30,
)
def audit_strouhal_test_trajectories(
    git_sha: str,
    count_only: bool = False,
) -> dict:
    """Phase 2 Task 1 -- pre-fire Strouhal audit across all cylinder_flow test
    trajectories (refinement 1; design D0-24 v0). CPU-only; iterates raw
    test.tfrecord records via the same machinery as
    audit_cylinder_flow_loader_contract (lines 854-861) and reuses the
    substrate-class smoke's Strouhal pipeline (volume-integrated lift proxy
    + Hann-windowed FFT + scikit-fem P1 basis on the per-trajectory mesh).

    For each test trajectory, computes f_s, U_mean, U_max, cylinder D, and
    St in both conventions (U_mean, U_max). A trajectory is "in design band"
    iff EITHER convention lands in [0.16, 0.21] (substrate-class smoke
    convention; either-or hedges the U_mean-vs-U_max ambiguity in the
    literature). Canonical Phase 2 trajectory = median-Strouhal-in-band
    (sorted by strouhal_U_max for a deterministic order).

    Writes findings + selection to
    /vol/datasets/cylinder_flow/strouhal_test_trajectories.json.

    --count-only mode: counts test.tfrecord records and prints N_test;
    no Strouhal computation. Use for the Plan Step 1 sanity check.

    Verdict:
      OK           - >=1 in-band trajectory; canonical pinned. Proceed to Task 2.
      INVESTIGATE  - 0 in-band; do NOT fire Phase 2 Task 6. Inspect band /
                     geometry detection / U convention.
    """
    import json as _json

    import numpy as np

    data_dir = Path(DM_CYLINDER_FLOW_DIR)
    meta_path = data_dir / "meta.json"
    test_tfrecord_path = data_dir / "test.tfrecord"
    for p in (meta_path, test_tfrecord_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing -- run download_dm_cylinder_flow_dataset first.")
    if _sha256_of_file(meta_path) != DM_CYLINDER_FLOW_META_SHA256:
        raise RuntimeError("meta.json sha mismatch vs D0-23 pin")
    if _sha256_of_file(test_tfrecord_path) != DM_CYLINDER_FLOW_TEST_SHA256:
        raise RuntimeError("test.tfrecord sha mismatch vs D0-23 pin")

    meta = _json.loads(meta_path.read_text())
    field_names = meta["field_names"]
    trajectory_length = int(meta["trajectory_length"])
    dt = 0.01  # cylinder_flow meta.json: "dt": 0.01

    import tfrecord.torch.dataset as tfrecord_torch
    from physicsnemo.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset

    description = {k: "byte" for k in field_names}

    def _records():
        tfr = tfrecord_torch.TFRecordDataset(
            str(test_tfrecord_path),
            None,  # no .tfindex -> single-worker sequential read
            description,
            transform=lambda r: VortexSheddingDataset._decode_record(r, meta),
        )
        return iter(tfr)

    if count_only:
        n = sum(1 for _ in _records())
        print(f"N_test = {n}")
        return {
            "N_test": n,
            "physicsnemo_sha": PHYSICSNEMO_SHA,
            "git_sha": git_sha,
        }

    design_band = (0.16, 0.21)
    per_trajectory: list[dict] = []
    for traj_idx, rec in enumerate(_records()):
        try:
            r = _strouhal_for_test_trajectory(rec, dt, traj_idx)
        except Exception as e:
            r = {"traj_idx": traj_idx, "error": f"{type(e).__name__}: {e}"}
        # Either-or band check; the substrate-class smoke (line 1870-1875)
        # treats both conventions as acceptable evidence.
        in_band = False
        if "error" not in r:
            st_mean = r.get("strouhal_U_mean")
            st_max = r.get("strouhal_U_max")
            in_band_mean = isinstance(st_mean, float) and (
                design_band[0] <= st_mean <= design_band[1]
            )
            in_band_max = isinstance(st_max, float) and (design_band[0] <= st_max <= design_band[1])
            in_band = bool(in_band_mean or in_band_max)
        r["in_design_band"] = in_band
        per_trajectory.append(r)
        if (traj_idx + 1) % 10 == 0:
            print(f"  ... processed {traj_idx + 1} trajectories")

    in_band_results = [r for r in per_trajectory if r.get("in_design_band")]
    out_band_results = [
        r for r in per_trajectory if not r.get("in_design_band") and "error" not in r
    ]
    errors = [r for r in per_trajectory if "error" in r]

    if in_band_results:
        # Deterministic ordering: rank by strouhal_U_max so the median pick
        # is reproducible even when both conventions land in-band.
        sorted_in_band = sorted(in_band_results, key=lambda r: r["strouhal_U_max"])
        canonical = sorted_in_band[len(sorted_in_band) // 2]
        verdict = "OK"
        selection_reason = (
            f"{len(in_band_results)}/{len(per_trajectory)} test trajectories "
            f"land in design band [{design_band[0]}, {design_band[1]}] "
            f"(either-or on strouhal_U_mean / strouhal_U_max); canonical = "
            f"median-Strouhal in-band sorted by strouhal_U_max "
            f"(traj_idx={canonical['traj_idx']}, "
            f"St_U_mean={canonical['strouhal_U_mean']:.3f}, "
            f"St_U_max={canonical['strouhal_U_max']:.3f})."
        )
    else:
        canonical = None
        verdict = "INVESTIGATE"
        st_max_values = [
            r["strouhal_U_max"]
            for r in per_trajectory
            if isinstance(r.get("strouhal_U_max"), float) and not (np.isnan(r["strouhal_U_max"]))
        ]
        if st_max_values:
            range_str = f"strouhal_U_max range [{min(st_max_values):.3f}, {max(st_max_values):.3f}]"
        else:
            range_str = "no valid Strouhal computed"
        selection_reason = (
            f"0/{len(per_trajectory)} test trajectories land in design band "
            f"[{design_band[0]}, {design_band[1]}]; {range_str}. The "
            f"literature-anchored band may be wrong for this cylinder_flow "
            f"distribution; investigate before Phase 2 Task 6 fires (see "
            f"Task 1 outcome decision tree)."
        )

    findings = {
        "verdict": verdict,
        "n_test_trajectories": len(per_trajectory),
        "n_in_design_band": len(in_band_results),
        "n_out_of_design_band": len(out_band_results),
        "n_errors": len(errors),
        "design_band": list(design_band),
        "trajectory_length": trajectory_length,
        "dt": dt,
        "per_trajectory": per_trajectory,
        "canonical_trajectory": canonical,
        "selection_reason": selection_reason,
        "physicsnemo_sha": PHYSICSNEMO_SHA,
        "git_sha": git_sha,
    }

    out_path = data_dir / "strouhal_test_trajectories.json"
    out_path.write_text(_json.dumps(findings, indent=2, default=str))
    mgn_volume.commit()
    print(f"=== STROUHAL AUDIT VERDICT: {verdict} ===")
    print(selection_reason)
    return findings


# ---------------------------------------------------------------------------
# Phase 2 -- Task 5: GT-trajectory lint entrypoint + gt.sarif (CPU control arm).
# ---------------------------------------------------------------------------


def _decode_test_trajectory(traj_idx: int) -> dict:
    """Decode the ``traj_idx``-th test.tfrecord record, no dataset
    instantiation (= no CWD-relative stats reads). Iterates raw records
    via the same machinery as ``audit_cylinder_flow_loader_contract``
    lines 854-861 and ``audit_strouhal_test_trajectories``. CPU-only.
    """
    import json as _json

    import tfrecord.torch.dataset as tfrecord_torch
    from physicsnemo.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset

    data_dir = Path(DM_CYLINDER_FLOW_DIR)
    meta = _json.loads((data_dir / "meta.json").read_text())
    description = {k: "byte" for k in meta["field_names"]}
    tfr = tfrecord_torch.TFRecordDataset(
        str(data_dir / "test.tfrecord"),
        None,  # no .tfindex -> sequential read
        description,
        transform=lambda r: VortexSheddingDataset._decode_record(r, meta),
    )
    last_idx = -1
    for idx, rec in enumerate(tfr):
        last_idx = idx
        if idx == traj_idx:
            return rec
    raise IndexError(
        f"traj_idx={traj_idx} not found in test.tfrecord; count exhausted at {last_idx + 1}"
    )


@app.function(
    volumes={"/vol": mgn_volume},
    timeout=60 * 30,
)
def lint_gt_trajectory(git_sha: str, traj_idx: int) -> dict:
    """Phase 2 Task 5 -- control-arm lint of the canonical GT cylinder_flow
    test trajectory through the mesh harness. CPU-only; no inference.

    Loads the trajectory's raw record from test.tfrecord, materializes a
    MeshRollout (framework='deepmind-cylinder-flow-gt', regular_grid=False,
    cells_2d supplied so Task 4's FE path runs), applies the three
    *_on_mesh rule mirrors via lint_mesh_rollout, and writes a
    SARIF artifact at both the Volume and (via the caller's
    ``modal volume get``) the local committed mirror.

    Expected outputs (D0-24 v1 + v3 + v4 bands):
      - harness:mass_conservation_defect: raw_value (the FE-on-P1 floor
        on this trajectory; D0-24 v1 PASS band is <= 6%).
      - harness:energy_drift: SKIP via the v9 substrate-class dispatch
        (vortex_shedding_2d -> open-driven-dissipative); D0-24 v3 PASS.
      - harness:dissipation_sign_violation: SKIP via the same dispatch;
        D0-24 v4 PASS.

    The ``inference_run_status`` run-level property is fixed to
    ``"n/a_gt_control_arm"`` (design §2.5) -- the GT control arm has no
    inference fire, so the field documents the absence rather than
    omitting (a salvage-detection-by-omission anti-pattern).
    """
    import json as _json
    import sys

    import numpy as np

    data_dir = Path(DM_CYLINDER_FLOW_DIR)
    meta_path = data_dir / "meta.json"
    test_tfrecord_path = data_dir / "test.tfrecord"
    for p in (meta_path, test_tfrecord_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing -- run download_dm_cylinder_flow_dataset first.")
    if _sha256_of_file(meta_path) != DM_CYLINDER_FLOW_META_SHA256:
        raise RuntimeError("meta.json sha mismatch vs D0-23 pin")
    if _sha256_of_file(test_tfrecord_path) != DM_CYLINDER_FLOW_TEST_SHA256:
        raise RuntimeError("test.tfrecord sha mismatch vs D0-23 pin")

    sys.path.insert(0, "/root")  # for adapter / harness imports
    from external_validation._rollout_anchors._harness.lint_mesh_rollout import (
        lint_mesh_rollout,
    )
    from external_validation._rollout_anchors._harness.mesh_rollout_adapter import (
        MeshRollout,
    )
    from external_validation._rollout_anchors._harness.sarif_emitter import emit_sarif

    rec = _decode_test_trajectory(traj_idx)
    velocity = np.asarray(rec["velocity"]).astype(np.float32)  # (T, N, 2)
    pressure = np.asarray(rec["pressure"]).astype(np.float32)  # (T, N, 1)
    mesh_pos = np.asarray(rec["mesh_pos"][0]).astype(np.float32)  # (N, 2)
    cells = np.asarray(rec["cells"][0]).astype(np.int64)  # (M, 3)
    node_type = np.asarray(rec["node_type"][0]).reshape(-1).astype(np.int64)

    rollout = MeshRollout(
        node_positions=mesh_pos,
        node_type=node_type,
        node_values={"velocity": velocity, "pressure": pressure},
        dt=0.01,  # cylinder_flow meta.json
        metadata={
            # framework=NOT "pytorch+dgl" -- this is GT, not MGN inference output.
            # is_regular_grid stays False so the FE graph-mesh path is taken.
            "framework": "deepmind-cylinder-flow-gt",
            "model": "deepmind-meshgraphnets-2020",  # NOT "modulus_*" -> F3 bypasses
            "dataset": "vortex_shedding_2d",
            "regular_grid": False,
            "cells_2d": cells,
            "git_sha": git_sha,
            "trajectory_index": traj_idx,
            "physicsnemo_sha": PHYSICSNEMO_SHA,
        },
        edge_index=None,
    )

    rule_results = lint_mesh_rollout(
        rollout,
        case_study="02-physicsnemo-mgn",
        dataset="vortex_shedding_2d",
        model="deepmind-cylinder-flow-gt",
        ckpt_hash="n/a_gt_control_arm",
        extra_properties={
            "arm": "gt-control",
            "trajectory_index": traj_idx,
            "n_timesteps": int(velocity.shape[0]),
            "n_nodes": int(mesh_pos.shape[0]),
            "n_cells": int(cells.shape[0]),
        },
    )

    run_properties: dict[str, object] = {
        "source": "rollout-anchor-harness",
        "harness_sarif_schema_version": "1.0",
        "case_study": "02-physicsnemo-mgn",
        "arm": "gt-control",
        "inference_run_status": "n/a_gt_control_arm",  # design §2.5
        "trajectory_index": traj_idx,
        "dataset_name": "vortex_shedding_2d",
        "model_name": "deepmind-cylinder-flow-gt",
        "checkpoint_id": "n/a_gt_control_arm",
        "physicsnemo_sha": PHYSICSNEMO_SHA,
        "physics_lint_sha_sarif_emission": git_sha,
        "rollout_subdir": f"/vol/rollouts/physicsnemo/vortex_shedding_{git_sha}/",
    }

    out_dir_vol = Path(f"/vol/rollouts/physicsnemo/vortex_shedding_{git_sha}")
    out_dir_vol.mkdir(parents=True, exist_ok=True)
    out_path_vol = out_dir_vol / "gt.sarif"
    emit_sarif(
        rule_results,
        output_path=out_path_vol,
        run_properties=run_properties,
    )
    mgn_volume.commit()

    rule_summary = {}
    for r in rule_results:
        if r.raw_value is not None:
            rule_summary[r.rule_id] = f"raw_value={r.raw_value:.3e}"
        else:
            reason = str(r.extra_properties.get("skip_reason", "(no reason)"))
            rule_summary[r.rule_id] = f"SKIP: {reason[:80]}"
    print(f"=== GT LINT VERDICT (traj_idx={traj_idx}) ===")
    print(_json.dumps(rule_summary, indent=2))
    return {
        "sarif_path": str(out_path_vol),
        "rule_summary": rule_summary,
        "trajectory_index": traj_idx,
        "n_timesteps": int(velocity.shape[0]),
        "n_nodes": int(mesh_pos.shape[0]),
        "n_cells": int(cells.shape[0]),
        "git_sha": git_sha,
    }


# ---------------------------------------------------------------------------
# Phase 2 -- Task 6: MGN inference on A10G (canonical traj N=1; F5 isolation).
# ---------------------------------------------------------------------------


def _preflight_mgn_inference_p0(rollout_dir: str, git_sha: str, ckpt_path: Path) -> dict:
    """Pre-flight assertions before the MGN inference fire (Task 6).

    Captures provenance (ckpt sha256 + git sha + dtype + cwd) and fails
    loud on contract violations. Recorded in the Task 6 findings JSON.

    Phase-1 known-unknowns covered:
      - KU §5.6 fp32 default dtype (must precede dataset construction)
      - KU §5.4 CWD discipline + stats files staged
      - Persistent-volume rollout output path is writable
      - NGC checkpoint file present + sha recorded (no equality pin yet
        -- D0-23 v1 verifies architectural identity via the reproduction
        gate; a hard sha pin would land as a follow-up if upstream drift
        becomes a concern).
    """
    import hashlib
    import os

    import torch

    findings: dict[str, object] = {}

    assert ckpt_path.exists(), f"NGC checkpoint missing at {ckpt_path}"
    h_ckpt = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h_ckpt.update(chunk)
    findings["ckpt_sha256"] = h_ckpt.hexdigest()
    findings["ckpt_path"] = str(ckpt_path)

    assert torch.get_default_dtype() == torch.float32, (
        f"torch.set_default_dtype(torch.float32) must precede dataset "
        f"construction (KU §5.6); got {torch.get_default_dtype()}"
    )
    findings["torch_default_dtype"] = "float32"

    assert os.getcwd() == rollout_dir, (
        f"KU §5.4 CWD discipline: expected cwd={rollout_dir}, got {os.getcwd()}"
    )
    edge_stats_p = os.path.join(rollout_dir, "edge_stats.json")
    node_stats_p = os.path.join(rollout_dir, "node_stats.json")
    assert os.path.isfile(edge_stats_p), f"edge_stats.json not staged at {edge_stats_p}"
    assert os.path.isfile(node_stats_p), f"node_stats.json not staged at {node_stats_p}"
    findings["rollout_dir"] = rollout_dir

    out_dir = Path(f"/vol/rollouts/physicsnemo/vortex_shedding_{git_sha}")
    out_dir.mkdir(parents=True, exist_ok=True)
    test_path = out_dir / ".preflight_writable_check"
    test_path.write_text("ok")
    test_path.unlink()
    findings["rollout_output_dir"] = str(out_dir)

    findings["physicsnemo_sha"] = PHYSICSNEMO_SHA
    findings["git_sha"] = git_sha
    return findings


@app.function(
    volumes={"/vol": mgn_volume},
    gpu="A10G",
    timeout=60 * 60 * 2,
)
def mgn_rollout_p0_vortex_shedding(
    git_sha: str,
    full_git_sha: str,
    traj_idx: int,
    num_steps: int = 600,
) -> dict:
    """Phase 2 Task 6 -- MGN inference on the canonical cylinder_flow test
    trajectory (selected via Task 1's Strouhal audit; D0-24 v0 result =
    traj_idx 44). Reproduces ``inference.py``'s
    denorm-renorm-model-denorm-mask-integrate protocol verbatim from the
    substrate-class smoke, lifts to ``n_rollout_steps = num_steps - 1``
    (= 599 by default; the full cylinder_flow trajectory horizon).

    Writes:
      - /vol/rollouts/physicsnemo/vortex_shedding_<sha>/mgn_rollout.npz
      - /vol/rollouts/physicsnemo/vortex_shedding_<sha>/mgn_rollout_p0_findings.json

    F5 absorption (Phase-1 cross-review Finding 5): rollout-dir isolation
    pattern -- a fresh ``tempfile.mkdtemp`` per fire so parallel/retry runs
    cannot collide on CWD-relative stats reads
    (vortex_shedding_dataset.py:103,141 @ 1ca85d65). Task 7's smoke test
    exercises the Python-level invariant on local pytest tmp_path; this
    entrypoint demonstrates the pattern in the production path.

    Cap discipline ([[feedback_cap_rationale_not_literal]]): ONE A10G fire
    for Task 6. If it fails, diagnose CPU-only before any re-fire; do NOT
    fix-iterate-on-GPU. A *verdict-confirmation* measurement-only re-fire
    (no adapter change, no protocol change, instrumentation only) is a
    separate cap category per the D0-23 refinement.

    The metadata records ``cells_2d`` from the test trajectory's mesh so
    Task 7's lint path (via Task 4's wired FE branch) can compute the
    incompressibility defect on the rollout without inferring triangulation
    from edge_index.
    """
    import json as _json
    import os
    import shutil
    import sys
    import tempfile

    import numpy as np
    import torch

    data_dir = Path(DM_CYLINDER_FLOW_DIR)
    for p in (
        data_dir / "meta.json",
        data_dir / "test.tfrecord",
        data_dir / "edge_stats.json",
        data_dir / "node_stats.json",
    ):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing -- run download_dm_cylinder_flow_dataset + "
                f"compute_cylinder_flow_stats first."
            )
    if _sha256_of_file(data_dir / "meta.json") != DM_CYLINDER_FLOW_META_SHA256:
        raise RuntimeError("meta.json sha mismatch vs D0-23 pin")
    if _sha256_of_file(data_dir / "test.tfrecord") != DM_CYLINDER_FLOW_TEST_SHA256:
        raise RuntimeError("test.tfrecord sha mismatch vs D0-23 pin")

    sys.path.insert(0, "/root")  # for _legacy_checkpoint_name_remap.py

    # F5: rollout-dir isolation. tempfile.mkdtemp guarantees a unique
    # path even for two concurrent fires with the same prefix; the
    # CWD-relative stats reads are therefore container-local.
    rollout_dir = tempfile.mkdtemp(prefix=f"mgn_rollout_p0_{git_sha}_")
    shutil.copy2(data_dir / "edge_stats.json", os.path.join(rollout_dir, "edge_stats.json"))
    shutil.copy2(data_dir / "node_stats.json", os.path.join(rollout_dir, "node_stats.json"))
    old_cwd = os.getcwd()

    try:
        os.chdir(rollout_dir)
        torch.set_default_dtype(torch.float32)

        ckpt_path = Path(
            f"{VOLUME_CHECKPOINT_ROOT}/modulus_ns_meshgraphnet_{NGC_VORTEX_VERSION}/"
            f"vortex_shedding_mgn/model.pt"
        )
        preflight = _preflight_mgn_inference_p0(rollout_dir, git_sha, ckpt_path)

        from _legacy_checkpoint_name_remap import remap_modulus_to_physicsnemo_state_dict
        from physicsnemo.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset
        from physicsnemo.models.meshgraphnet import MeshGraphNet

        device = "cuda" if torch.cuda.is_available() else "cpu"
        preflight["device"] = device

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw_sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model = MeshGraphNet(input_dim_nodes=6, input_dim_edges=3, output_dim=3)
        model.load_state_dict(remap_modulus_to_physicsnemo_state_dict(raw_sd), strict=True)
        model = model.to(device).eval()
        preflight["model_loaded"] = "strict=True OK (corrected name-remap adapter)"

        # Build dataset spanning traj_idx + 1 trajectories. The dataset
        # iterates test.tfrecord in order, so trajectory T's pairs are at
        # indices [T*(num_steps-1), (T+1)*(num_steps-1) - 1].
        n_samples_needed = traj_idx + 1
        ds = VortexSheddingDataset(
            name="cylinder_flow",
            data_dir=str(data_dir),
            split="test",
            num_samples=n_samples_needed,
            num_steps=num_steps,
            noise_std=0.0,  # ignored for split != "train" per loader-contract V-entries
        )
        stats = {k: v.to(device) for k, v in ds.node_stats.items()}
        n_pairs = num_steps - 1
        first_pair_idx = traj_idx * n_pairs

        # Extract static mesh + node_type from the first pair of traj_idx.
        graph0, cells0, _mask0 = ds[first_pair_idx]
        mesh_pos = np.asarray(graph0["mesh_pos"]).astype(np.float32)  # (N, 2)
        cells_np = np.asarray(cells0).astype(np.int64)  # (M, 3)
        # Decode raw node_type from the one-hot encoding in graph.x[:, 2:6].
        # vortex_shedding_dataset.py:363-368: class 0 -> NORMAL(0), class K>=1
        # -> K+3 (so class 1->4=INFLOW, 2->5=OUTFLOW, 3->6=WALL).
        node_class = np.asarray(graph0.x[:, 2:6].argmax(dim=1)).astype(np.int64)
        node_type_raw = np.where(node_class == 0, 0, node_class + 3).astype(np.int64)
        n_nodes = mesh_pos.shape[0]

        # MGN rollout: inference.py predict() protocol with prev-prediction feedback.
        # Mirrors substrate-class smoke lines 1776-1803 verbatim, lifted to traj_idx.
        v_pred = np.empty((n_pairs, n_nodes, 2), dtype=np.float32)
        p_pred = np.empty((n_pairs, n_nodes, 1), dtype=np.float32)
        prev_v = None
        with torch.no_grad():
            for t in range(n_pairs):
                idx = first_pair_idx + t
                g_t, _c, mask_t = ds[idx]
                g_t = g_t.to(device)
                g_t.x[:, 0:2] = VortexSheddingDataset.denormalize(
                    g_t.x[:, 0:2], stats["velocity_mean"], stats["velocity_std"]
                )
                invar = g_t.x.clone()
                if t != 0:
                    invar[:, 0:2] = prev_v
                invar[:, 0:2] = VortexSheddingDataset.normalize_node(
                    invar[:, 0:2], stats["velocity_mean"], stats["velocity_std"]
                )
                pred_i = model(invar, g_t.edge_attr, g_t).detach()
                pred_i_velo = VortexSheddingDataset.denormalize(
                    pred_i[:, 0:2], stats["velocity_diff_mean"], stats["velocity_diff_std"]
                )
                pred_i_pres = VortexSheddingDataset.denormalize(
                    pred_i[:, 2:3], stats["pressure_mean"], stats["pressure_std"]
                )
                invar[:, 0:2] = VortexSheddingDataset.denormalize(
                    invar[:, 0:2], stats["velocity_mean"], stats["velocity_std"]
                )
                mask2 = torch.cat((mask_t, mask_t), dim=-1).to(device)
                v_diff_masked = torch.where(mask2, pred_i_velo, torch.zeros_like(pred_i_velo))
                v_next = v_diff_masked + invar[:, 0:2]
                v_pred[t] = v_next.cpu().numpy().astype(np.float32)
                p_pred[t] = pred_i_pres.cpu().numpy().astype(np.float32)
                prev_v = v_next.clone()
                if (t + 1) % 100 == 0:
                    print(f"  ... rolled out {t + 1}/{n_pairs} steps")

        from external_validation._rollout_anchors._harness.mesh_rollout_adapter import (
            MeshRollout,
            _assert_loader_contract_mgn,
            save_mesh_rollout_npz,
        )

        rollout = MeshRollout(
            node_positions=mesh_pos,
            node_type=node_type_raw,
            node_values={"velocity": v_pred, "pressure": p_pred},
            dt=0.01,  # cylinder_flow meta.json
            metadata={
                "framework": "pytorch+dgl",  # honored even though v2.0.0 is PyG (KU §5.x)
                "model": "modulus_ns_meshgraphnet",
                "dataset": "vortex_shedding_2d",
                "regular_grid": False,
                "cells_2d": cells_np,  # for Task 4's FE path in Task 7's lint
                "git_sha": git_sha,
                "full_git_sha": full_git_sha,
                "trajectory_index": traj_idx,
                "ngc_version": NGC_VORTEX_VERSION,
                "ckpt_sha256": preflight["ckpt_sha256"],
                "ckpt_hash": preflight["ckpt_sha256"][:16],  # short for SARIF display
                "physicsnemo_sha": PHYSICSNEMO_SHA,
                "n_rollout_steps": n_pairs,
            },
            edge_index=None,  # cells_2d is the load-bearing topology for Task 4's FE
        )
        # Belt-and-braces fail-loud: the load_mesh_rollout_npz path also
        # validates on read (Task 3 F3 wiring), but explicit save-side
        # check surfaces materializer bugs at write time too.
        _assert_loader_contract_mgn(rollout)

        out_dir = Path(f"/vol/rollouts/physicsnemo/vortex_shedding_{git_sha}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_npz = out_dir / "mgn_rollout.npz"
        save_mesh_rollout_npz(rollout, out_npz)

        findings = {
            "rollout_dir": rollout_dir,
            "out_npz": str(out_npz),
            "traj_idx": traj_idx,
            "n_rollout_steps": n_pairs,
            "num_steps": num_steps,
            "n_nodes": int(n_nodes),
            "n_cells": int(cells_np.shape[0]),
            "device": device,
            "preflight": preflight,
        }
        out_findings = out_dir / "mgn_rollout_p0_findings.json"
        out_findings.write_text(_json.dumps(findings, indent=2, default=str))
        mgn_volume.commit()
        print(f"=== MGN ROLLOUT P0 COMPLETE (traj_idx={traj_idx}) ===")
        print(_json.dumps(findings, indent=2, default=str))
        return findings
    finally:
        os.chdir(old_cwd)
        # F5: rollout_dir is NOT removed -- Modal containers are destroyed
        # at function exit anyway, so the tempfile.mkdtemp guarantees
        # provide the actual isolation; the persistence here documents the
        # path used in case a post-mortem inspection is needed.
