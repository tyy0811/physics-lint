# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-29

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
