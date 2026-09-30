"""Symmetry-inequivalent adsorption sites (ontop, bridge, hollow).

Sites are found with pymatgen's `AdsorbateSiteFinder` on the small unit
slab, where symmetry analysis is cheap and exact. The atoms defining each
site are also chosen on the unit slab (the geometry pymatgen analyzed) and
mapped to the same atoms of the supercell, in the unit-cell image closest to
the supercell center. The site is then snapped onto those atoms in the
(relaxed) supercell: ontop sits on its surface atom, bridge on the midpoint of
its two atoms, hollow on the centroid of its three (or four) atoms. A site
equidistant to more surface atoms than its pymatgen type implies is
reclassified as a hollow of all those atoms.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.geometry import find_mic

SITE_ATOM_COUNTS = {"ontop": 1, "bridge": 2, "hollow": 3}
# Surface atoms this close (Å) to the farthest defining atom of a site count
# as equidistant; e.g. pymatgen's "bridge" across the diagonal of a square
# lattice has four equidistant atoms and is really a fourfold hollow.
EQUIDISTANT_TOLERANCE = 0.1
# Sites closer than this (Å) in the surface plane are duplicates.
DUPLICATE_DISTANCE = 0.1


@dataclass(frozen=True)
class AdsorptionSite:
    """An adsorption site on the top surface of a slab supercell."""

    site_id: str
    kind: str
    position: tuple[float, float, float]
    surface_atoms: tuple[int, ...]
    elements: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "site_id": self.site_id,
            "kind": self.kind,
            "position": [round(x, 6) for x in self.position],
            "surface_atoms": list(self.surface_atoms),
            "elements": list(self.elements),
        }

    @classmethod
    def from_dict(cls, data: dict) -> AdsorptionSite:
        return cls(
            site_id=data["site_id"],
            kind=data["kind"],
            position=tuple(data["position"]),
            surface_atoms=tuple(data["surface_atoms"]),
            elements=tuple(data["elements"]),
        )


def surface_atom_indices(slab: Atoms, depth: float) -> list[int]:
    """Atoms within `depth` Å below the topmost atom."""

    z = slab.positions[:, 2]
    return [int(i) for i in np.flatnonzero(z >= z.max() - depth)]


def find_sites(
    unit_slab: Atoms,
    supercell: Atoms,
    site_types: Sequence[str] = ("ontop", "bridge", "hollow"),
    surface_depth: float = 0.9,
    symprec: float = 0.01,
) -> list[AdsorptionSite]:
    """Symmetry-inequivalent adsorption sites of `supercell`.

    Args:
        unit_slab: The oriented unit slab the supercell was built from with
            ``unit_slab.repeat((n_a, n_b, 1))`` (possibly relaxed afterwards).
        supercell: The (relaxed) slab supercell sites are placed on.
        site_types: Any of ``ontop``, ``bridge``, ``hollow``.
        surface_depth: Depth (Å) below the topmost atom of the unit slab
            within which atoms count as surface atoms.
        symprec: Symmetry / near-duplicate tolerance passed to pymatgen.

    Because site atoms are chosen on the unit slab, a relaxation that moves
    atoms in or out of the surface depth window does not change which atoms
    a site belongs to.

    Site ids are ``<kind>_<elements>`` with elements ordered by decreasing
    atomic number (``ontop_Ti``, ``bridge_Ti-Si``, ``hollow_Ti-Ti-Si``);
    repeated ids get ``_1``, ``_2``, ... suffixes.
    """

    from pymatgen.analysis.adsorption import AdsorbateSiteFinder
    from pymatgen.io.ase import AseAtomsAdaptor

    unknown = set(site_types) - set(SITE_ATOM_COUNTS)
    if unknown:
        raise ValueError(f"unknown site type(s): {', '.join(sorted(unknown))}")
    repeats = _repeats(unit_slab, supercell)

    structure = AseAtomsAdaptor.get_structure(
        Atoms(unit_slab.get_chemical_symbols(), positions=unit_slab.positions, cell=unit_slab.cell, pbc=True)
    )
    finder = AdsorbateSiteFinder(structure, height=surface_depth)
    found = finder.find_adsorption_sites(
        distance=0.0,
        put_inside=True,
        symm_reduce=symprec,
        near_reduce=symprec,
        positions=tuple(site_types),
        no_obtuse_hollow=True,
    )

    unit_surface = surface_atom_indices(unit_slab, surface_depth)
    translations = unit_slab.cell.array[:2]
    center = 0.5 * (supercell.cell[0] + supercell.cell[1])
    symbols = supercell.get_chemical_symbols()

    raw: list[tuple[str, np.ndarray, tuple[int, ...]]] = []
    for pymatgen_kind in site_types:
        for position in found.get(pymatgen_kind, []):
            position = np.asarray(position, dtype=float)
            members = _site_atoms(unit_slab, unit_surface, position, SITE_ATOM_COUNTS[pymatgen_kind])
            kind = pymatgen_kind if len(members) == SITE_ATOM_COUNTS[pymatgen_kind] else "hollow"
            if kind not in site_types:
                continue
            shift = _closest_translation(position, translations, center)
            xy = position + shift @ translations
            atom_indices = tuple(
                _supercell_index(index, image + shift, repeats, len(unit_slab)) for index, image in members
            )
            expected = np.array(
                [unit_slab.positions[index] + (image + shift) @ translations for index, image in members]
            )
            point = _snap(supercell, atom_indices, expected)
            if any(np.linalg.norm((point - other)[:2]) < DUPLICATE_DISTANCE for _, other, _ in raw):
                continue
            raw.append((kind, point, atom_indices))

    labels = []
    for kind, _, atom_indices in raw:
        elements = sorted((symbols[i] for i in atom_indices), key=lambda s: (-atomic_numbers[s], s))
        labels.append((f"{kind}_{'-'.join(elements)}", tuple(elements)))
    counts = {label: sum(1 for other, _ in labels if other == label) for label, _ in labels}
    seen: dict[str, int] = {}
    sites = []
    for (kind, position, atom_indices), (label, elements) in zip(raw, labels):
        if counts[label] > 1:
            seen[label] = seen.get(label, 0) + 1
            label = f"{label}_{seen[label]}"
        sites.append(
            AdsorptionSite(
                site_id=label,
                kind=kind,
                position=tuple(float(x) for x in position),
                surface_atoms=tuple(int(i) for i in atom_indices),
                elements=elements,
            )
        )
    return sites


def _repeats(unit: Atoms, supercell: Atoms) -> tuple[int, int]:
    n_a = int(round(np.linalg.norm(supercell.cell[0]) / np.linalg.norm(unit.cell[0])))
    n_b = int(round(np.linalg.norm(supercell.cell[1]) / np.linalg.norm(unit.cell[1])))
    if n_a < 1 or n_b < 1 or n_a * n_b * len(unit) != len(supercell):
        raise ValueError("supercell is not unit_slab.repeat((n_a, n_b, 1))")
    return n_a, n_b


def _supercell_index(unit_index: int, image: np.ndarray, repeats: tuple[int, int], n_unit: int) -> int:
    """Index in ``unit.repeat((n_a, n_b, 1))`` of `unit_index` in unit-cell image `image`."""

    m_a = int(image[0]) % repeats[0]
    m_b = int(image[1]) % repeats[1]
    return (m_a * repeats[1] + m_b) * n_unit + unit_index


def _closest_translation(position: np.ndarray, translations: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Integer (i, j) so that ``position + i a + j b`` is closest to `center` in plane."""

    fractions = np.linalg.lstsq(translations[:, :2].T, (center - position)[:2], rcond=None)[0]
    base = np.floor(fractions).astype(int)
    best, best_distance = base, np.inf
    for di, dj in itertools.product(range(-1, 3), repeat=2):
        shift = base + np.array([di, dj])
        distance = np.linalg.norm((position + shift @ translations - center)[:2])
        if distance < best_distance - 1e-9:
            best, best_distance = shift, distance
    return best


def _site_atoms(slab: Atoms, surface: Sequence[int], xy: np.ndarray, count: int) -> list[tuple[int, np.ndarray]]:
    """Surface atoms defining a site at `xy`, with the cell image of each.

    Returns ``(atom index, (i, j))`` pairs, where ``(i, j)`` is the lattice
    translation of the atom's periodic image: the `count` surface-atom images
    nearest `xy` in the surface plane, plus for bridge/hollow sites any
    further image within `EQUIDISTANT_TOLERANCE` of the farthest of them.
    Several images of one atom can define a site in a small unit cell.
    """

    cell = slab.cell.array
    candidates = []
    for index in surface:
        for i, j in itertools.product(range(-2, 3), repeat=2):
            image_xy = slab.positions[index, :2] + i * cell[0, :2] + j * cell[1, :2]
            candidates.append((float(np.linalg.norm(image_xy - xy[:2])), int(index), np.array([i, j])))
    candidates.sort(key=lambda item: (round(item[0], 6), item[1], tuple(item[2])))
    if count == 1:
        selected = candidates[:1]
    else:
        limit = candidates[min(count, len(candidates)) - 1][0] + EQUIDISTANT_TOLERANCE
        selected = [item for item in candidates if item[0] <= limit]
    return [(index, image) for _, index, image in selected]


def _snap(slab: Atoms, atom_indices: Sequence[int], expected: np.ndarray) -> np.ndarray:
    """Site position from the actual (relaxed) positions of its atoms.

    Each site atom is taken at its periodic image nearest its `expected`
    (unrelaxed) position; the site is their in-plane centroid at the height
    of the highest of them.
    """

    vectors = slab.positions[list(atom_indices)] - expected
    vectors[:, 2] = 0.0
    mic_vectors, _ = find_mic(vectors, slab.cell, pbc=[True, True, False])
    actual = expected + mic_vectors
    point = actual.mean(axis=0)
    point[2] = slab.positions[list(atom_indices), 2].max()
    return point
