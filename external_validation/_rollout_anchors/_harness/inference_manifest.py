"""Shared inference-manifest persistence + classification helpers.

The standalone-conversion gate (``modal_app.py``, Modal-side runtime)
and SARIF emission (``emit_sarif.py``, local runtime) both classify a
rollout subdir by the persisted upstream-inference manifest. Pre-rung-
4c §9 fold-in round 3, the logic was duplicated between the two
modules; both implementations collapsed manifest-corruption into the
generic legacy-absent status, fail-opening the gate on corrupt or
stale manifests (Codex adversarial review round 3 findings 1 + 2).

Round 3 promotes the helpers to this shared module and introduces a
fourth status — ``manifest_invalid`` — distinct from legacy absence
(``from_unknown_inference``). The standalone-conversion gate refuses
``manifest_invalid`` unconditionally (no override flag is appropriate
for corruption); SARIF emission with ``required=True`` (rung-4c dam2d
post-fold-in stacks) raises rather than silently omitting the
provenance field.

The promotion resolves plan v2.1 §2.3's deferred shared-helper
candidacy: pre-round-3, two runtime contexts duplicated the logic;
round-3's test code is the third runtime context that pattern B's
"single-instance-vs-multi-instance triggers generalization" rule
names as the promotion trigger.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path

INFERENCE_MANIFEST_FILENAME = "_inference_manifest.json"
INFERENCE_MANIFEST_GATED_FIELDS: tuple[str, ...] = (
    "git_sha",
    "full_git_sha",
    "lagrangebench_sha",
    "inference_seed",
    "inference_returncode",
    "inference_wall_seconds",
    "aborted_at_step",
    "conversion_attempted",
    "conversion_returncode",
    "rollout_subdir",
)
_REQUIRED_CLASSIFICATION_FIELDS: tuple[str, ...] = (
    "inference_returncode",
    "aborted_at_step",
    "rollout_subdir",
)
_SCHEMA_VERSION = "1"

STATUS_FROM_COMPLETED_INFERENCE = "from_completed_inference"
STATUS_FROM_ABORTED_INFERENCE = "from_aborted_inference"
STATUS_FROM_UNKNOWN_INFERENCE = "from_unknown_inference"
STATUS_MANIFEST_INVALID = "manifest_invalid"


class ManifestInvalidError(Exception):
    """Raised when a manifest file is present but cannot be classified.

    Distinguished from legitimate absence (legacy / pre-rung-4c) which
    is reported via ``STATUS_FROM_UNKNOWN_INFERENCE`` and returns
    ``None`` at the ``read_inference_manifest_status`` API. Callers
    should fail-closed on this error rather than treating it as a
    soft warning — corruption is not a legacy-absence case.
    """


def persist_inference_manifest_to_rollout_subdir(manifest: dict, rollout_subdir: object) -> object:
    """Atomic-write the gate-relevant manifest subset to the rollout subdir.

    Skips silently when ``rollout_subdir`` is None or missing on disk
    (covers the early-abort cases in the rollout orchestrators where
    we return before the subdir is created). The atomic write uses a
    tempfile in the target directory plus ``os.replace`` so a half-
    written manifest can never masquerade as complete if the container
    is killed mid-write.

    Returns the absolute path written, or None on skip.
    """
    if not rollout_subdir or not os.path.isdir(rollout_subdir):
        return None

    payload = {k: manifest.get(k) for k in INFERENCE_MANIFEST_GATED_FIELDS}
    payload["_schema_version"] = _SCHEMA_VERSION
    # Round-codex-3 finding 1: rollout_subdir is now a required
    # classification field. If the caller didn't include it in the
    # manifest dict, auto-fill from the destination path so the
    # persist→classify roundtrip still works. Production callers
    # (rollout orchestrators) pre-populate this field; auto-fill
    # is a safety net for minimal-manifest callers (e.g., tests).
    if payload.get("rollout_subdir") is None:
        payload["rollout_subdir"] = str(rollout_subdir)

    target = os.path.join(rollout_subdir, INFERENCE_MANIFEST_FILENAME)
    fd, tmp_path = tempfile.mkstemp(
        prefix="._inference_manifest.",
        suffix=".json.tmp",
        dir=rollout_subdir,
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
    return target


def classify_inference_run_status(
    rollout_subdir: str | os.PathLike[str],
) -> tuple[str, object]:
    """Classify the upstream inference run status from a persisted manifest.

    Returns ``(status, manifest_or_None)`` where ``status`` is one of
    the four ``STATUS_*`` constants:

    - ``STATUS_FROM_COMPLETED_INFERENCE`` — manifest present, parseable,
      with required classification fields, ``inference_returncode == 0``
      and ``aborted_at_step is None``. Safe to convert by default.
    - ``STATUS_FROM_ABORTED_INFERENCE`` — manifest present, parseable,
      with required fields, but ``inference_returncode != 0`` or
      ``aborted_at_step`` is set (timeout-salvage case). Standalone
      conversion default-refuses; caller opts in via
      ``allow_from_aborted_inference=True``.
    - ``STATUS_FROM_UNKNOWN_INFERENCE`` — no manifest file at the
      expected path (rung-4a/4b artifacts predate the convention).
      Standalone conversion warns but does not refuse.
    - ``STATUS_MANIFEST_INVALID`` — manifest file exists but cannot be
      parsed (JSON decode error, OSError, non-dict root, or missing
      required classification fields). Corruption is not a legacy-
      absence case; the gate refuses unconditionally.

    Pure read; no side effects.
    """
    target = os.path.join(rollout_subdir, INFERENCE_MANIFEST_FILENAME)
    if not os.path.isfile(target):
        return STATUS_FROM_UNKNOWN_INFERENCE, None
    try:
        with open(target) as f:
            persisted = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return STATUS_MANIFEST_INVALID, {"_error": f"{type(e).__name__}: {e}"}

    if not isinstance(persisted, dict):
        return STATUS_MANIFEST_INVALID, {
            "_error": f"manifest root is {type(persisted).__name__}, not dict",
        }

    missing = [k for k in _REQUIRED_CLASSIFICATION_FIELDS if k not in persisted]
    if missing:
        return STATUS_MANIFEST_INVALID, {
            **persisted,
            "_error": f"missing required classification fields: {sorted(missing)}",
        }

    # Round-codex-3 finding 1: bind manifest to rollout directory by
    # basename. Pre-fix, a manifest copied from one rollout subdir to
    # another would still classify as completed/aborted based only on
    # inference_returncode + aborted_at_step. The basename check ensures
    # the manifest is *for* the directory being classified.
    persisted_subdir = persisted.get("rollout_subdir")
    if persisted_subdir is None:
        # Caught by the required-field check above for new manifests,
        # but a defensive belt-and-suspenders for legacy persisted
        # manifests where the field was present-but-null.
        return STATUS_MANIFEST_INVALID, {
            **persisted,
            "_error": "rollout_subdir field is None; cannot validate against classified path",
        }
    persisted_basename = Path(str(persisted_subdir)).name
    classified_basename = Path(str(rollout_subdir)).name
    if persisted_basename != classified_basename:
        return STATUS_MANIFEST_INVALID, {
            **persisted,
            "_error": (
                f"rollout_subdir basename mismatch: persisted manifest's "
                f"rollout_subdir basename is {persisted_basename!r}, but the "
                f"classified path's basename is {classified_basename!r}. "
                "This typically indicates a manifest was copied from a "
                "different rollout subdir; refusing to classify."
            ),
        }

    returncode = persisted.get("inference_returncode")
    aborted_at = persisted.get("aborted_at_step")
    if returncode == 0 and aborted_at is None:
        return STATUS_FROM_COMPLETED_INFERENCE, persisted
    return STATUS_FROM_ABORTED_INFERENCE, persisted


def gate_verdict_for_status(
    status: str,
    *,
    allow_from_aborted_inference: bool,
    manifest_required: bool,
) -> tuple[bool, str | None]:
    """Decide whether the standalone-conversion gate allows conversion to proceed.

    Returns ``(allow, refuse_reason)`` where ``refuse_reason`` is one of
    ``None`` (allow), ``"manifest_invalid"``, ``"aborted_inference"``, or
    ``"missing_required_manifest"``.

    Truth table:

    - ``STATUS_FROM_COMPLETED_INFERENCE`` → allow (clean run).
    - ``STATUS_FROM_ABORTED_INFERENCE`` → allow only if
      ``allow_from_aborted_inference=True``; otherwise refuse with reason
      ``"aborted_inference"``. Caller's salvage opt-in.
    - ``STATUS_FROM_UNKNOWN_INFERENCE`` → behavior depends on
      ``manifest_required``:
        - ``manifest_required=False`` (legacy stacks): allow with warning.
          rung-4a/4b artifacts predate the manifest convention.
        - ``manifest_required=True`` (post-fold-in stacks): refuse with
          reason ``"missing_required_manifest"``. **No override flag** —
          corruption of the gate is not a legacy-absence case. Operator
          must repair the manifest rather than delete it to bypass.
          Added at v2.1 round-codex-2 absorption after Codex review
          surfaced that deleting the manifest was a documented bypass
          for the aborted-inference gate.
    - ``STATUS_MANIFEST_INVALID`` → refuse unconditionally with reason
      ``"manifest_invalid"``. No override flag — corruption is structural,
      not policy. Independent of ``manifest_required`` and
      ``allow_from_aborted_inference``.
    """
    if status == STATUS_MANIFEST_INVALID:
        return False, "manifest_invalid"
    if status == STATUS_FROM_ABORTED_INFERENCE:
        if allow_from_aborted_inference:
            return True, None
        return False, "aborted_inference"
    if status == STATUS_FROM_UNKNOWN_INFERENCE:
        if manifest_required:
            return False, "missing_required_manifest"
        return True, None
    # STATUS_FROM_COMPLETED_INFERENCE (and any future allow-by-default status)
    return True, None


def read_inference_manifest_status(
    mirror_subdir: Path,
    *,
    required: bool = False,
) -> str | None:
    """Local-side variant returning a status string or None.

    For SARIF emission and other local consumers where the structured
    ``(status, manifest)`` tuple is unnecessary.

    Returns:
    - The status string ``STATUS_FROM_COMPLETED_INFERENCE`` or
      ``STATUS_FROM_ABORTED_INFERENCE`` for present, valid manifests.
    - ``None`` when ``required=False`` (default) and the manifest is
      absent (legacy / pre-rung-4c convention).

    Raises:
    - ``FileNotFoundError`` when ``required=True`` and the manifest is
      absent. Post-fold-in stacks (rung-4c dam2d) pass ``required=True``
      so a deleted manifest fails closed rather than emitting a
      provenance-stripped SARIF that looks valid.
    - ``ManifestInvalidError`` when the manifest is present but cannot
      be classified (corrupt JSON, non-dict root, or missing required
      fields). Raised regardless of ``required``; corruption is not a
      legacy-absence case.
    """
    status, persisted = classify_inference_run_status(str(mirror_subdir))
    if status == STATUS_FROM_UNKNOWN_INFERENCE:
        if required:
            raise FileNotFoundError(
                f"Inference manifest required but not found at "
                f"{Path(mirror_subdir) / INFERENCE_MANIFEST_FILENAME}. "
                f"Post-fold-in stacks (rung-4c dam2d onward) must carry the "
                f"manifest; legacy stacks (rung-4a/4b tgv2d) should pass "
                f"required=False."
            )
        return None
    if status == STATUS_MANIFEST_INVALID:
        error_detail = (
            persisted.get("_error", "unknown") if isinstance(persisted, dict) else "unknown"
        )
        raise ManifestInvalidError(
            f"Inference manifest at "
            f"{Path(mirror_subdir) / INFERENCE_MANIFEST_FILENAME} is invalid: "
            f"{error_detail}. Refusing to emit SARIF without classifiable "
            f"provenance; corruption is not a legacy-absence case."
        )
    return status
