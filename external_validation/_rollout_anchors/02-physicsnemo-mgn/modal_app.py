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
    (12.7 GB) is intentionally NOT pulled here — see `compute_cylinder_flow_stats`
    which pulls + fits + deletes it atomically.
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
