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
