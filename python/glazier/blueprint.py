"""The model description every engine reads.

`glazier` reads the same JSON file through its own deserialiser, so a
disagreement between two runs is a disagreement about the model rather than
about the file. Anything an engine needs that the description does not state is
a gap in the format. The CompuCell3D backend for it is `glazier.cc3d_backend`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

MEDIUM = "Medium"


@dataclass(frozen=True)
class SpeciesSpec:
    name: str
    diffusion: float
    decay: float = 0.0
    initial: float = 0.0


@dataclass(frozen=True)
class TypeSpec:
    name: str
    target_volume: float
    lambda_volume: float
    target_surface: float = 0.0
    lambda_surface: float = 0.0
    division_volume: float = 0.0
    death_rate: float = 0.0
    target_length: float = 0.0
    lambda_length: float = 0.0
    connected: bool = False
    max_activity: float = 0.0
    lambda_activity: float = 0.0
    external: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    presents: dict[str, float] = field(default_factory=dict)
    secretion: dict[str, float] = field(default_factory=dict)
    uptake: dict[str, float] = field(default_factory=dict)
    chemotaxis: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Blueprint:
    name: str
    width: int
    height: int
    types: list[TypeSpec]
    contact: list[float]
    temperature: float
    neighbour_order: int
    seed: int
    steps: int
    dump_every: int
    initial: dict
    units: dict
    depth: int = 1
    fields: list[SpeciesSpec] = field(default_factory=list)
    adhesion: dict | None = None
    physicell: dict | None = None

    @property
    def domain_microns(self) -> tuple[float, float]:
        """Physical size of the lattice in the plane."""
        scale = self.units["micron_per_site"]
        return self.width * scale, self.height * scale

    @property
    def dimensions(self) -> int:
        """Two for a plane, three for a volume."""
        return 3 if self.depth > 1 else 2

    @property
    def duration_minutes(self) -> float:
        """Physical duration of the run."""
        return self.steps * self.units["minute_per_step"]

    @property
    def type_names(self) -> list[str]:
        return [MEDIUM] + [t.name for t in self.types]

    def contact_energy(self, a: int, b: int) -> float:
        n = len(self.type_names)
        return self.contact[a * n + b]

    @property
    def species_names(self) -> list[str]:
        return [s.name for s in self.fields]

    def effective_contact(self) -> list[float]:
        """The contact matrix with the adhesion molecules folded in.

        Binding lowers the energy of a bond, so what two types' molecules bind
        comes off the contact energy the description states between them. The
        Rust reader does the same, and the conformance test compares them.
        """
        if self.adhesion is None:
            return list(self.contact)

        molecules = self.adhesion["molecules"]
        binding = self.adhesion["binding"]
        n = len(molecules)
        presented = [[0.0] * n]
        for spec in self.types:
            presented.append([spec.presents.get(name, 0.0) for name in molecules])

        n_types = len(presented)
        contact = list(self.contact)
        for a in range(n_types):
            for b in range(n_types):
                bound = sum(
                    binding[i * n + j] * presented[a][i] * presented[b][j]
                    for i in range(n)
                    for j in range(n)
                )
                contact[a * n_types + b] -= bound
        return contact

    @staticmethod
    def load(path: str | Path) -> "Blueprint":
        raw = json.loads(Path(path).read_text())
        types = [TypeSpec(**t) for t in raw.pop("types")]
        species = [SpeciesSpec(**f) for f in raw.pop("fields", [])]
        return Blueprint(types=types, fields=species, **raw)
