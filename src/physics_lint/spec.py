"""DomainSpec pydantic hierarchy — validated config for rules to consume.

Design doc §4. This is the single source of truth for physics-lint config;
rules read DomainSpec, never raw TOML. The hierarchy is a superset of the
user-writable config schema: computed properties (BCSpec.conserves_mass,
GridDomain.spatial_lengths, etc.) and derived fields (adapter_path /
dump_path on FieldSourceSpec) are populated during the merge path in
physics_lint.config.load_spec().
"""

from __future__ import annotations

import warnings
from typing import Literal

from pydantic import BaseModel, Field, model_validator

BCKind = Literal[
    "periodic",
    "dirichlet_homogeneous",
    "dirichlet",
    "neumann_homogeneous",
    "neumann",
]

PDEKind = Literal["laplace", "poisson", "heat", "wave"]

SymmetryLiteral = Literal[
    "D4",
    "C4",
    "reflection_x",
    "reflection_y",
    "translation_x",
    "translation_y",
    "SO2",
]

FieldType = Literal["grid", "callable", "mesh"]
FieldBackend = Literal["fd", "spectral", "auto"]


class GridDomain(BaseModel):
    """Spatial (and optionally temporal) domain extents."""

    x: tuple[float, float]
    y: tuple[float, float]
    t: tuple[float, float] | None = None

    @property
    def spatial_lengths(self) -> tuple[float, ...]:
        return tuple(hi - lo for lo, hi in (self.x, self.y))

    @property
    def is_time_dependent(self) -> bool:
        return self.t is not None


class BCSpec(BaseModel):
    """Boundary condition with computed properties replacing rule-side BC taxonomy.

    The computed properties are the design doc §4.2 deduplication mechanism:
    per-rule conservation and sign-preservation checks read these booleans
    instead of re-encoding the PER/hN/hD logic in each rule.
    """

    kind: BCKind

    @property
    def preserves_sign(self) -> bool:
        return self.kind in {"dirichlet_homogeneous", "periodic"}

    @property
    def conserves_mass(self) -> bool:
        return self.kind in {"periodic", "neumann_homogeneous"}

    @property
    def conserves_energy(self) -> bool:
        return self.kind in {"periodic", "neumann_homogeneous", "dirichlet_homogeneous"}


class SymmetrySpec(BaseModel):
    """User-declared problem-instance symmetries.

    Not auto-detected. Design doc §9.4 explains why: operator-level admissibility
    is PDE-class-dependent but problem-instance symmetry depends on domain,
    source/IC, and BCs in ways physics-lint cannot mechanically verify.
    """

    declared: list[SymmetryLiteral] = Field(default_factory=list)


class FieldSourceSpec(BaseModel):
    """Field source: adapter module or dump file. Exactly one must be set."""

    type: FieldType
    backend: FieldBackend = "auto"
    adapter_path: str | None = None
    dump_path: str | None = None

    @model_validator(mode="after")
    def exactly_one_source(self) -> FieldSourceSpec:
        if (self.adapter_path is None) == (self.dump_path is None):
            raise ValueError(
                "Exactly one of adapter_path or dump_path must be set; "
                f"got adapter_path={self.adapter_path!r}, dump_path={self.dump_path!r}"
            )
        return self


class SARIFSpec(BaseModel):
    """Optional source-mapping config for SARIF Tier 2 (PR-check surfacing)."""

    source_file: str | None = None
    pde_line: int | None = None
    bc_line: int | None = None
    symmetry_line: int | None = None


class DomainSpec(BaseModel):
    """Top-level validated spec consumed by every rule."""

    pde: PDEKind
    grid_shape: tuple[int, ...] = Field(min_length=2, max_length=3)
    domain: GridDomain
    periodic: bool = False
    boundary_condition: BCSpec
    symmetries: SymmetrySpec = Field(default_factory=SymmetrySpec)
    field: FieldSourceSpec

    diffusivity: float | None = None
    wave_speed: float | None = None
    source_term: str | None = None
    sarif: SARIFSpec | None = None

    @model_validator(mode="after")
    def pde_params_consistent(self) -> DomainSpec:
        if self.pde == "heat" and self.diffusivity is None:
            raise ValueError("PDE 'heat' requires 'diffusivity'")
        if self.pde == "wave" and self.wave_speed is None:
            raise ValueError("PDE 'wave' requires 'wave_speed'")
        if self.pde in {"heat", "wave"} and not self.domain.is_time_dependent:
            raise ValueError(
                f"PDE '{self.pde}' requires a time domain 't'; "
                "add 't = [t_start, t_end]' to [tool.physics-lint.domain]"
            )
        return self

    @model_validator(mode="after")
    def symmetries_compatible_with_domain(self) -> DomainSpec:
        if any(s in self.symmetries.declared for s in ("D4", "C4")):
            lx, ly = self.domain.spatial_lengths[:2]
            if abs(lx - ly) / max(lx, ly) > 1e-6:
                warnings.warn(
                    f"D4/C4 symmetry declared but domain is not square "
                    f"({lx} x {ly}); symmetry rules may produce artifacts",
                    stacklevel=2,
                )
        return self
