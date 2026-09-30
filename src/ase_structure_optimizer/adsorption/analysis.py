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


# ---------------------------------------------------------------------------
# Classification, angles, height (M4)
# ---------------------------------------------------------------------------

CLASSES = ("chemisorbed", "physisorbed", "dissociated", "desorbed")
# A free (non-fixed) slab atom moving more than this (Å) marks the surface as distorted.
SURFACE_DISTORTION_LIMIT = 1.0
BOLTZMANN_EV = 8.617333262e-5


def molecule_part(atoms: Atoms, n_slab: int) -> Atoms:
    """The molecule of an adsorption structure as a whole, non-periodic Atoms.

    Atoms split across the periodic boundary are brought next to the first
    molecule atom (minimum image), so bond graphs and axes are meaningful.
    """

    from ase.geometry import find_mic

    positions = atoms.positions[n_slab:].copy()
    vectors, _ = find_mic(positions - positions[0], atoms.cell, pbc=atoms.pbc)
    return Atoms(atoms.get_chemical_symbols()[n_slab:], positions=positions[0] + vectors)


def intact_check(molecule: Atoms, reference: MoleculeAnalysis, bond_scale: float, overrides=()) -> tuple[bool, str]:
    """Whether the molecule is chemically intact compared with `reference`.

    Without metal atoms the bond sets must be identical. With metal centers
    the comparison is made at the ligand level, so that a change of hapticity
    (e.g. eta5 -> eta3 Cp) is not a dissociation: bonds between non-metal
    atoms must be unchanged, metal-metal bonds unchanged, and every ligand
    must still be bonded to at least one of the metal atoms it was bonded to.
    Returns ``(intact, reason)``.
    """

    from ..graphs import build_bond_graph

    graph = build_bond_graph(molecule, scale=bond_scale, overrides=overrides)
    current = {tuple(sorted(edge)) for edge in graph.edges}
    expected = {(i, j) for i, j, _ in reference.bonds}
    metals = set(reference.metal_indices)
    symbols = reference.symbols

    def name(edge):
        return "-".join(f"{symbols[i]}{i}" for i in edge)

    if not metals:
        if current == expected:
            return True, ""
        return False, _bond_change_reason(expected, current, name)

    def ligand_edges(edges):
        return {edge for edge in edges if edge[0] not in metals and edge[1] not in metals}

    def metal_edges(edges):
        return {edge for edge in edges if edge[0] in metals and edge[1] in metals}

    if ligand_edges(current) != ligand_edges(expected):
        return False, _bond_change_reason(ligand_edges(expected), ligand_edges(current), name)
    if metal_edges(current) != metal_edges(expected):
        return False, _bond_change_reason(metal_edges(expected), metal_edges(current), name)
    detached = []
    for fragment in reference.fragments:
        bound = {m for m in metals for i in fragment.indices if (min(m, i), max(m, i)) in expected}
        if bound and not any((min(m, i), max(m, i)) in current for m in bound for i in fragment.indices):
            metal_names = ", ".join(f"{symbols[m]}{m}" for m in sorted(bound))
            detached.append(f"{fragment.name} detached from {metal_names}")
    if detached:
        return False, "; ".join(detached)
    return True, ""


def _bond_change_reason(expected: set, current: set, name) -> str:
    parts = []
    if expected - current:
        parts.append("broken " + ", ".join(name(edge) for edge in sorted(expected - current)))
    if current - expected:
        parts.append("formed " + ", ".join(name(edge) for edge in sorted(current - expected)))
    return "; ".join(parts)


def classify(
    atoms: Atoms,
    n_slab: int,
    reference: MoleculeAnalysis,
    bond_scale: float,
    contact_scale: float,
    desorbed_distance: float,
    overrides=(),
) -> tuple[str, str, str]:
    """``(class, contact_label, reason)`` of an adsorption structure.

    In order: ``dissociated`` (see `intact_check`), ``desorbed`` (molecule
    farther than `desorbed_distance` from the slab), ``chemisorbed`` (at least
    one contact atom), ``physisorbed``.
    """

    molecule = molecule_part(atoms, n_slab)
    label = contact_label(atoms, n_slab, reference, contact_scale)
    intact, reason = intact_check(molecule, reference, bond_scale, overrides)
    if not intact:
        return "dissociated", label, reason
    if min_molecule_slab_distance(atoms, n_slab) > desorbed_distance:
        return "desorbed", label, ""
    return ("chemisorbed" if label != "none" else "physisorbed"), label, ""


def surface_displacement(atoms: Atoms, n_slab: int, clean_slab: Atoms) -> float:
    """Largest displacement (Å, minimum image) of a free slab atom from the clean slab."""

    from ase.geometry import find_mic

    fixed = set()
    for constraint in clean_slab.constraints:
        if hasattr(constraint, "get_indices"):
            fixed.update(int(i) for i in constraint.get_indices())
    free = [i for i in range(n_slab) if i not in fixed]
    if not free:
        return 0.0
    _, distances = find_mic(atoms.positions[free] - clean_slab.positions[free], clean_slab.cell, pbc=clean_slab.pbc)
    return float(np.max(distances))


def _center(molecule: Atoms, reference: MoleculeAnalysis) -> np.ndarray:
    """Metal atom(s) centroid, otherwise the center of mass."""

    if reference.metal_indices:
        return molecule.positions[list(reference.metal_indices)].mean(axis=0)
    return molecule.get_center_of_mass()


def adsorption_height(atoms: Atoms, n_slab: int, reference: MoleculeAnalysis) -> float:
    """Height of the molecule center (metal, else center of mass) above the topmost slab atom."""

    return float(_center(molecule_part(atoms, n_slab), reference)[2] - atoms.positions[:n_slab, 2].max())


def azimuth_sign_atom(molecule: Atoms, reference: MoleculeAnalysis) -> int | None:
    """Atom fixing the sign of the azimuth axis: farthest heavy atom from the axis u.

    Chosen once on the gas-phase molecule so the azimuth is comparable
    between configurations.
    """

    axis = reference.reference_axis.vector(molecule)
    if axis is None:
        return None
    relative = molecule.positions - molecule.get_center_of_mass()
    perpendicular = np.linalg.norm(relative - np.outer(relative @ axis, axis), axis=1)
    heavy = [i for i, s in enumerate(molecule.get_chemical_symbols()) if s != "H"]
    candidates = heavy if heavy and perpendicular[heavy].max() > 0.3 else list(range(len(molecule)))
    return int(max(candidates, key=lambda i: (round(perpendicular[i], 6), -i)))


def azimuth_angle(molecule: Atoms, reference: MoleculeAnalysis, sign_atom: int | None, cell: np.ndarray) -> float | None:
    """Azimuth φ (degree, 0-360) of the second principal axis relative to cell vector a.

    The second axis is the principal direction perpendicular to u with the
    smallest moment (the molecule's longest extent across u), oriented toward
    `sign_atom`, projected onto the surface plane. None if u is undefined or
    the second axis is nearly vertical.
    """

    axis = reference.reference_axis.vector(molecule)
    if axis is None or sign_atom is None:
        return None
    masses = molecule.get_masses()
    relative = molecule.positions - molecule.get_center_of_mass()
    tensor = np.zeros((3, 3))
    for mass, r in zip(masses, relative):
        tensor += mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    projector = np.eye(3) - np.outer(axis, axis)
    moments, vectors = np.linalg.eigh(projector @ tensor @ projector)
    perpendicular = [k for k in range(3) if abs(np.dot(vectors[:, k], axis)) < 0.5]
    second = vectors[:, min(perpendicular, key=lambda k: moments[k])]
    if np.dot(second, relative[sign_atom]) < 0:
        second = -second
    in_plane = second[:2]
    if np.linalg.norm(in_plane) < 0.1:
        return None
    a = cell[0, :2] / np.linalg.norm(cell[0, :2])
    angle = math.degrees(math.atan2(a[0] * in_plane[1] - a[1] * in_plane[0], np.dot(a, in_plane)))
    return angle % 360.0


# ---------------------------------------------------------------------------
# De-duplication and populations
# ---------------------------------------------------------------------------


def deduplicate(
    records: list[dict],
    positions: dict[str, np.ndarray],
    cell: np.ndarray,
    energy_tol: float,
    rmsd_tol: float,
) -> list[dict]:
    """Group equivalent configurations and keep the lowest-energy one of each.

    Two configurations are the same when they share class and contact label,
    their E_ads differ by less than `energy_tol` and the molecule RMSD (up to
    lateral lattice translations) is below `rmsd_tol`. `records` need keys
    ``config_id``, ``class``, ``contact``, ``eads``; `positions` maps config
    id to molecule positions. Returns the representatives sorted by E_ads,
    each with ``n_duplicates`` and ``duplicates`` (ids) added.
    """

    from .sampling import lateral_rmsd

    representatives: list[dict] = []
    for record in sorted(records, key=lambda r: (r["eads"], r["config_id"])):
        for kept in representatives:
            if (
                kept["class"] == record["class"]
                and kept["contact"] == record["contact"]
                and abs(kept["eads"] - record["eads"]) < energy_tol
                and lateral_rmsd(cell, positions[kept["config_id"]], positions[record["config_id"]]) < rmsd_tol
            ):
                kept["duplicates"].append(record["config_id"])
                kept["n_duplicates"] += 1
                break
        else:
            representatives.append({**record, "n_duplicates": 0, "duplicates": []})
    return representatives


def boltzmann_weights(energies: list[float], temperature: float) -> list[float]:
    """Normalized Boltzmann populations of energies (eV) at `temperature` (K)."""

    if not energies:
        return []
    values = np.asarray(energies, dtype=float)
    factors = np.exp(-(values - values.min()) / (BOLTZMANN_EV * temperature))
    return (factors / factors.sum()).tolist()
