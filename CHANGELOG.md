# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.1] - 2026-05-21

### Fixed
- `physics_lint.__version__` was a hand-maintained literal in
  `__init__.py` that silently drifted from `pyproject.toml`: the v1.0.0
  wheel shipped `__version__ == "0.0.0.dev0"`. The stale value also
  propagated into every SARIF report's `tool.driver.version` field and
  the documentation-site title. `__version__` now derives from the
  installed package metadata via `importlib.metadata`, making
  `pyproject.toml` the single source of truth.

## [1.0.0] - 2026-05-18

Initial public release on PyPI.

### Added
- 18 physics rules across residual, boundary-condition, conservation,
  positivity, symmetry, and numerical categories — each with a stable
  `PH-<CATEGORY>-<NNN>` ID, a calibrated analytical floor, and a doc page.
  Of the 18, 16 are active; `PH-SYM-004` and `PH-NUM-001` ship as
  `SKIPPED`-with-reason and are scheduled for v1.1 (see `STABILITY.md`).
- SARIF output with GitHub code-scanning integration.
- Hybrid adapter + dump model loading; `GridField` / `CallableField` /
  `MeshField` field abstractions.
- External-validation anchors for all 18 rules; cross-stack rollout
  validation against LagrangeBench and PhysicsNeMo MeshGraphNet.
- **PH-BC-002**: the Poisson arm — divergence-theorem imbalance
  `∫Δu + ∫f` using the spec-plumbed source array, emitting PASS/WARN/FAIL;
  `SKIPPED`-with-reason when no source array is provided.

### Fixed
- `PH-CON-003`: `np.gradient(edge_order=2)` produced spurious endpoint
  artifacts on strictly-dissipative eigenmodes; replaced with a
  forward-difference primitive (`e691dd3`). Surfaced during external
  validation.

### Documented
- `PH-RES-001`: norm-equivalence is configuration-dependent — the
  Bachmayr-Dahmen-Oster claim holds on periodic + spectral
  configurations; non-periodic + FD configurations emit a different norm.
  Characterized in `external_validation/PH-RES-001/CITATION.md`.
