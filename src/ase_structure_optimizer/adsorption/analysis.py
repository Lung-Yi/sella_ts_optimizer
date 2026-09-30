"""Analysis of adsorption configurations (pure functions).

Adsorption structures are ``slab + molecule``: the first `n_slab` atoms are
the slab, the rest the molecule in the atom order of the gas-phase molecule.
"""

from __future__ import annotations

import math

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.geometry import get_distances

from .molecule import MoleculeAnalysis


def tilt_angle(axis: np.ndarray | None, directional: bool) -> float | None:
    """Angle (degree) between the molecular axis `u` and the surface normal +z.

    0° means `u` points away from the surface. Axes without head/tail
    (`directional=False`) are folded into 0-90°. Returns None when the axis
    is undefined.
    """

    if axis is None:
        return None
    cosine = float(np.clip(axis[2] / np.linalg.norm(axis), -1.0, 1.0))
    angle = math.degrees(math.acos(cosine))
    if not directional:
        angle = min(angle, 180.0 - angle)
    return angle


def molecule_tilt(molecule: Atoms, analysis: MoleculeAnalysis) -> float | None:
    """Tilt angle θ of a molecule (molecule atoms only, gas-phase atom order)."""

    axis = analysis.reference_axis
    return tilt_angle(axis.vector(molecule), axis.directional)


def split_system(atoms: Atoms, n_slab: int) -> tuple[Atoms, Atoms]:
    """``(slab, molecule)`` parts of an adsorption structure."""

    return atoms[:n_slab], atoms[n_slab:]


def contact_atoms(atoms: Atoms, n_slab: int, contact_scale: float) -> list[int]:
    """Molecule atoms (molecule-local indices) touching the slab.

    A molecule atom is in contact when its distance to any slab atom
    (minimum image) is below ``contact_scale * (r_cov_i + r_cov_j)``.
    """

    molecule_positions = atoms.positions[n_slab:]
    slab_positions = atoms.positions[:n_slab]
    _, distances = get_distances(molecule_positions, slab_positions, cell=atoms.cell, pbc=atoms.pbc)
    radii = covalent_radii[atoms.numbers]
    limits = contact_scale * (radii[n_slab:, None] + radii[None, :n_slab])
    return [int(i) for i in np.flatnonzero((distances < limits).any(axis=1))]


def contact_label(atoms: Atoms, n_slab: int, analysis: MoleculeAnalysis, contact_scale: float) -> str:
    """Fragments in contact with the surface, e.g. ``C5H5`` or ``CO_1+H``; ``none`` if none.

    Metal atoms of the molecule contribute their element label (e.g. ``Mo``).
    """

    labels = sorted({analysis.atom_labels[i] for i in contact_atoms(atoms, n_slab, contact_scale)})
    return "+".join(labels) if labels else "none"


def facing_label(atoms: Atoms, n_slab: int, analysis: MoleculeAnalysis, contact_scale: float) -> str:
    """The contact label, or ``~<fragments>`` facing the surface when nothing touches it.

    Short prescreening leaves most configurations above the contact range;
    the fragments of the molecule atoms closest to the surface (relative to
    covalent radii, within 5 % of the closest) still tell the orientations
    apart, e.g. ``~C5H5`` versus ``~CO_1``.
    """

    label = contact_label(atoms, n_slab, analysis, contact_scale)
    if label != "none":
        return label
    _, distances = get_distances(atoms.positions[n_slab:], atoms.positions[:n_slab], cell=atoms.cell, pbc=atoms.pbc)
    radii = covalent_radii[atoms.numbers]
    ratios = (distances / (radii[n_slab:, None] + radii[None, :n_slab])).min(axis=1)
    closest = np.flatnonzero(ratios <= ratios.min() * 1.05)
    return "~" + "+".join(sorted({analysis.atom_labels[i] for i in closest}))


def min_molecule_slab_distance(atoms: Atoms, n_slab: int) -> float:
    """Shortest molecule-slab distance (minimum image)."""

    _, distances = get_distances(atoms.positions[n_slab:], atoms.positions[:n_slab], cell=atoms.cell, pbc=atoms.pbc)
    return float(distances.min())
