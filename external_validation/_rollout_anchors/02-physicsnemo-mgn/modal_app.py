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
        # physicsnemo v2.0.0 pyproject declares warp-lang>=1.5.0 with no
        # upper bound; warp-lang >=1.13 removed wp.context.Device which
        # physicsnemo's nn.functional.radius_search._warp_impl still
        # references (import-time AttributeError). Pin to the floor.
        "warp-lang==1.5.0",
        # VortexSheddingDataset reads DeepMind .tfrecord via the `tfrecord`
        # package and builds PyG graphs — both are `OptionalImport`s in
        # physicsnemo/datapipes/gnn/vortex_shedding_dataset.py (lines 30-31 @
        # 1ca85d65), so neither is pulled by the physicsnemo install. Needed
        # for compute_cylinder_flow_stats and the V1-V18 loader-contract audit.
        "tfrecord==1.14.6",
        "torch-geometric==2.6.1",
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
