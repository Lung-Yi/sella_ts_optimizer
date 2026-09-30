"""Surface models: bulk/slab detection, slab cutting, orientation, supercells.

All functions here are pure geometry (no calculator calls); the workflow
combines them with relaxations and single points.

Conventions for every slab returned by this module:

* the cell vectors a and b lie in the xy plane (a along x) and c is along +z;
* the slab is contiguous, its lowest atom at z = `bottom` (1 Å by default);
* the adsorption side is the top (largest z) surface;
* ``c = slab thickness + vacuum_above``, so the vacuum between the top
  surface and the periodic image of the bottom surface is `vacuum_above`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms

from ..structures import is_slab

# Atoms whose heights differ by less than this (Å) belong to one layer.
LAYER_TOLERANCE = 0.3
# Vacuum gap (Å) used by is_slab() to recognize a pre-cut slab.
SLAB_MIN_GAP = 8.0


@dataclass(frozen=True)
class SlabCandidate:
    """One termination produced by `generate_slabs()`."""

    miller: tuple[int, int, int]
    shift: float
    atoms: Atoms
    symmetric: bool
    stoichiometric: bool


# ---------------------------------------------------------------------------
# Solid classification and bulk cell
# ---------------------------------------------------------------------------


def classify_solid(atoms: Atoms, input_type: str = "auto") -> tuple[str, int | None]:
    """Return ``("bulk", None)`` or ``("slab", vacuum_axis)``.

    `input_type` ``"bulk"`` or ``"slab"`` overrides the automatic
    `is_slab()` detection; a forced slab without a detectable vacuum gap uses
    cell axis 2.
    """

    detected, axis = is_slab(atoms, min_gap=SLAB_MIN_GAP)
    if input_type == "bulk":
        return "bulk", None
    if input_type == "slab":
        return "slab", axis if detected else 2
    if input_type != "auto":
        raise ValueError(f"input_type must be auto, bulk or slab, got {input_type!r}")
    return ("slab", axis) if detected else ("bulk", None)


def conventional_bulk(atoms: Atoms, symprec: float = 0.01) -> tuple[Atoms, bool]:
    """Bulk cell in which Miller indices are interpreted.

    If the input cell already holds as many atoms as the conventional
    standard cell (e.g. a CIF in any axis setting), it is kept unchanged so
    Miller indices refer to the axes of the input file. A primitive input
    (e.g. ``ase.build.bulk("Cu")``) is converted to the conventional standard
    cell. Returns ``(atoms, converted)``.
    """

    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    structure = AseAtomsAdaptor.get_structure(_plain_copy(atoms))
    conventional = SpacegroupAnalyzer(structure, symprec=symprec).get_conventional_standard_structure()
    if len(conventional) == len(structure):
        return atoms.copy(), False
    result = AseAtomsAdaptor.get_atoms(conventional)
    result.pbc = True
    return _plain_copy(result), True


def _plain_copy(atoms: Atoms) -> Atoms:
    """Copy without calculator, constraints or per-atom extras (for pymatgen)."""

    return Atoms(
        atoms.get_chemical_symbols(),
        positions=atoms.get_positions(),
        cell=atoms.get_cell(),
        pbc=atoms.get_pbc(),
    )


# ---------------------------------------------------------------------------
# Slab generation and orientation
# ---------------------------------------------------------------------------


def generate_slabs(
    bulk: Atoms,
    miller: Sequence[int],
    min_thickness: float,
    vacuum_above: float,
) -> list[SlabCandidate]:
    """All terminations of `bulk` for one Miller index (pymatgen SlabGenerator).

    `bulk` should come from `conventional_bulk()`. Slabs are centered and
    returned oriented (see module docstring) in pymatgen's order. pymatgen
    measures slab size as number of planes times the interplanar spacing,
    which is larger than the atom-to-atom thickness, so the requested size is
    increased until every termination is at least `min_thickness` Å thick.
    """

    from pymatgen.core.surface import SlabGenerator
    from pymatgen.io.ase import AseAtomsAdaptor

    structure = AseAtomsAdaptor.get_structure(_plain_copy(bulk))
    bulk_formula = structure.composition.reduced_formula
    requested = float(min_thickness)
    for _ in range(20):
        generator = SlabGenerator(
            structure,
            tuple(int(i) for i in miller),
            min_slab_size=requested,
            min_vacuum_size=max(vacuum_above, 10.0),
            lll_reduce=True,
            center_slab=True,
            in_unit_planes=False,
        )
        slabs = generator.get_slabs()
        if not slabs:
            raise ValueError(f"pymatgen generated no slab for Miller index {tuple(miller)}")
        oriented = [
            orient_slab(_plain_copy(AseAtomsAdaptor.get_atoms(slab)), vacuum_axis=2, vacuum_above=vacuum_above)
            for slab in slabs
        ]
        thinnest = min(slab_thickness(atoms) for atoms in oriented)
        if thinnest >= min_thickness - 1e-6:
            break
        requested += max(min_thickness - thinnest, 0.1)
    else:
        raise RuntimeError(f"could not reach a slab thickness of {min_thickness} Å for {tuple(miller)}")

    return [
        SlabCandidate(
            miller=tuple(int(i) for i in miller),
            shift=float(slab.shift),
            atoms=atoms,
            symmetric=bool(slab.is_symmetric()),
            stoichiometric=slab.composition.reduced_formula == bulk_formula,
        )
        for slab, atoms in zip(slabs, oriented)
    ]


def orient_slab(atoms: Atoms, vacuum_axis: int = 2, vacuum_above: float = 15.0, bottom: float = 1.0) -> Atoms:
    """Reorient a slab so that a, b lie in the xy plane and c is along +z.

    The two in-plane cell vectors are the ones other than `vacuum_axis`
    (cyclically permuted so the cell keeps its handedness). Atom heights are
    measured along the in-plane normal, the slab is made contiguous across the
    periodic boundary, its lowest atom is placed at z = `bottom`, and the new
    c length is ``thickness + vacuum_above``. Constraints and calculators are
    dropped; per-atom tags and symbols are kept.
    """

    if vacuum_above <= 0:
        raise ValueError("vacuum_above must be positive")
    order = [(vacuum_axis + 1) % 3, (vacuum_axis + 2) % 3, vacuum_axis]
    cell = atoms.get_cell().array[order]
    a, b, c = cell
    normal = np.cross(a, b)
    area = np.linalg.norm(normal)
    if area < 1e-8:
        raise ValueError("the in-plane cell vectors are parallel or zero")
    normal /= area
    if np.dot(normal, c) < 0:
        # Left-handed after the permutation: flip b to stay right-handed.
        b = -b
        normal = -normal

    positions = atoms.get_positions()
    heights = positions @ normal
    c_perp = abs(float(np.dot(c, normal)))
    if atoms.pbc[vacuum_axis] and c_perp > 1e-8:
        heights = _unwrap_heights(heights, c_perp)

    # In-plane fractional coordinates with respect to (a, b).
    in_plane = positions - np.outer(heights, normal)
    basis = np.vstack([a, b])
    fractions = np.linalg.lstsq(basis.T, in_plane.T, rcond=None)[0].T
    fractions -= np.floor(fractions)

    length_a, length_b = np.linalg.norm(a), np.linalg.norm(b)
    cos_gamma = np.dot(a, b) / (length_a * length_b)
    sin_gamma = math.sqrt(max(0.0, 1.0 - cos_gamma**2))
    thickness = float(heights.max() - heights.min())
    new_cell = np.array(
        [
            [length_a, 0.0, 0.0],
            [length_b * cos_gamma, length_b * sin_gamma, 0.0],
            [0.0, 0.0, thickness + vacuum_above],
        ]
    )
    new_positions = fractions @ new_cell[:2]
    new_positions[:, 2] = heights - heights.min() + bottom

    oriented = Atoms(
        atoms.get_chemical_symbols(),
        positions=new_positions,
        cell=new_cell,
        pbc=True,
        tags=atoms.get_tags(),
    )
    return oriented


def _unwrap_heights(heights: np.ndarray, period: float) -> np.ndarray:
    """Shift heights by multiples of `period` so the slab is contiguous.

    The vacuum is the largest gap between sorted heights (the wrap-around gap
    included); atoms below that gap are moved up by one period.
    """

    wrapped = np.mod(heights, period)
    order = np.sort(wrapped)
    gaps = np.diff(order)
    wrap_gap = order[0] + period - order[-1]
    if len(gaps) == 0 or wrap_gap >= gaps.max():
        return wrapped
    cut = order[int(np.argmax(gaps)) + 1]
    return np.where(wrapped < cut, wrapped + period, wrapped)


def slab_thickness(atoms: Atoms) -> float:
    """Height difference between the highest and lowest atom (Å)."""

    z = atoms.positions[:, 2]
    return float(z.max() - z.min())


# ---------------------------------------------------------------------------
# Supercell, layers and fixed atoms
# ---------------------------------------------------------------------------


def perpendicular_widths(atoms: Atoms) -> tuple[float, float]:
    """In-plane widths of the cell perpendicular to b and to a (Å).

    These, not the vector lengths, set the distance between periodic images
    in a non-rectangular cell.
    """

    a, b = atoms.cell[0], atoms.cell[1]
    area = np.linalg.norm(np.cross(a, b))
    return float(area / np.linalg.norm(b)), float(area / np.linalg.norm(a))


def lateral_repeats(atoms: Atoms, min_lateral: float) -> tuple[int, int]:
    """Smallest (n_a, n_b) so that both perpendicular widths reach `min_lateral`."""

    width_a, width_b = perpendicular_widths(atoms)
    return max(1, math.ceil(min_lateral / width_a - 1e-9)), max(1, math.ceil(min_lateral / width_b - 1e-9))


def atomic_layers(atoms: Atoms, tolerance: float = LAYER_TOLERANCE) -> list[list[int]]:
    """Atom indices grouped into layers by height, bottom to top.

    Sorted heights are split wherever consecutive atoms differ by more than
    `tolerance`.
    """

    z = atoms.positions[:, 2]
    order = np.argsort(z, kind="stable")
    layers: list[list[int]] = []
    previous = None
    for index in order:
        if previous is None or z[index] - previous > tolerance:
            layers.append([])
        layers[-1].append(int(index))
        previous = z[index]
    return layers


def fix_bottom_layers(atoms: Atoms, fraction: float, tolerance: float = LAYER_TOLERANCE) -> list[int]:
    """Fix whole bottom layers holding about `fraction` of the atoms.

    The number of bottom layers is chosen so the fixed atom count is closest
    to ``fraction * len(atoms)`` (ties: fewer layers); the top layer is never
    fixed. Sets a `FixAtoms` constraint on `atoms` (in place) and returns the
    fixed indices.
    """

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    layers = atomic_layers(atoms, tolerance)
    target = fraction * len(atoms)
    best_layers, best_error, count = 0, target, 0
    for n_layers, layer in enumerate(layers[:-1], start=1):
        count += len(layer)
        error = abs(count - target)
        if error < best_error - 1e-9:
            best_layers, best_error = n_layers, error
    fixed = sorted(index for layer in layers[:best_layers] for index in layer)
    atoms.set_constraint(FixAtoms(indices=fixed) if fixed else None)
    return fixed


def top_layer_compositions(atoms: Atoms, n_layers: int = 4, tolerance: float = LAYER_TOLERANCE) -> list[dict]:
    """Composition of the top `n_layers` atomic layers, topmost first."""

    from ase.formula import Formula

    symbols = atoms.get_chemical_symbols()
    result = []
    for layer in reversed(atomic_layers(atoms, tolerance)[-n_layers:]):
        heights = atoms.positions[layer, 2]
        result.append(
            {
                "z": round(float(heights.mean()), 4),
                "n_atoms": len(layer),
                "formula": Formula.from_list([symbols[i] for i in layer]).format("hill"),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Energetics helpers
# ---------------------------------------------------------------------------


def surface_energy(slab: Atoms, slab_energy: float, bulk: Atoms, bulk_energy: float) -> float | None:
    """Average surface energy (eV/Å²) of a stoichiometric slab, else None.

    ``gamma = (E_slab - (N_slab / N_bulk) * E_bulk) / (2 A)``. For an
    asymmetric slab this is the mean of its top and bottom surfaces.
    """

    from ase.formula import Formula

    slab_formula = Formula.from_list(slab.get_chemical_symbols()).reduce()[0]
    bulk_formula = Formula.from_list(bulk.get_chemical_symbols()).reduce()[0]
    if slab_formula != bulk_formula:
        return None
    area = np.linalg.norm(np.cross(slab.cell[0], slab.cell[1]))
    return float((slab_energy - len(slab) / len(bulk) * bulk_energy) / (2.0 * area))


def molecule_size(molecule: Atoms) -> float:
    """Largest interatomic distance of the molecule (Å)."""

    positions = molecule.get_positions()
    if len(positions) < 2:
        return 0.0
    distances = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    return float(distances.max())


def termination_id(miller: Sequence[int] | None, index: int) -> str:
    """``<miller>_t<index>``, e.g. ``001_t0``; ``slab_t0`` for a pre-cut slab."""

    prefix = "slab" if miller is None else "".join(str(int(i)) for i in miller)
    return f"{prefix}_t{index}"
