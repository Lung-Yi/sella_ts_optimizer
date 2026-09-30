"""Initial adsorption configurations.

Anchored configurations: for every (site, anchor, spin) the molecule is
rotated so that the anchor direction points to -z (toward the surface),
spun about z by ``spin * 360 / spins_per_anchor`` degrees, and translated so
the anchor sits above the site. Random configurations use uniformly random
rotations (random unit quaternions) above randomly chosen sites.

The height is set so that the closest molecule atom is exactly at the
contact gap from the surface: with ``contact_gap: auto`` each molecule-slab
pair must be at least ``0.9 * (r_vdW,i + r_vdW,j)`` apart, otherwise the
given distance. All molecule atoms (not only the anchor) are used, so
several legs of a molecule (e.g. a CO tripod) can rest on the surface
together without clashing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, vdw_radii
from ase.data.vdw_alvarez import vdw_radii as vdw_radii_alvarez
from ase.geometry import find_mic, get_distances

from .analysis import molecule_tilt
from .config import AUTO, SamplingConfig
from .molecule import MoleculeAnalysis
from .sites import AdsorptionSite

AUTO_GAP_SCALE = 0.9
# RMSD (Å) below which two initial configurations are duplicates.
DUPLICATE_RMSD = 0.2
# Only slab atoms this far (Å) below the topmost slab atom are checked when
# placing the molecule (deeper atoms cannot be closest to it).
PLACEMENT_DEPTH = 6.0


def vdw_radius(symbol: str) -> float:
    """Van der Waals radius (Å): Bondi (``ase.data.vdw_radii``), else Alvarez 2013."""

    number = atomic_numbers[symbol]
    radius = vdw_radii[number]
    if not math.isfinite(radius):
        radius = vdw_radii_alvarez[number]
    if not math.isfinite(radius):
        raise ValueError(f"no van der Waals radius available for {symbol}")
    return float(radius)


@dataclass(frozen=True, eq=False)
class Candidate:
    """An initial adsorption configuration (molecule positions over a fixed slab)."""

    config_id: str
    site_id: str
    site_kind: str
    anchor: str
    spin: int | None
    source: str
    tilt: float | None
    molecule_positions: np.ndarray = field(repr=False)

    def atoms(self, slab: Atoms, molecule: Atoms) -> Atoms:
        """``slab + molecule`` with the molecule at this configuration."""

        placed = Atoms(molecule.get_chemical_symbols(), positions=self.molecule_positions)
        system = slab.copy() + placed
        system.cell = slab.cell
        system.pbc = True
        system.info.update(self.metadata())
        return system

    def metadata(self) -> dict:
        return {
            "config_id": self.config_id,
            "site_id": self.site_id,
            "site_kind": self.site_kind,
            "anchor": self.anchor,
            "spin": self.spin,
            "source": self.source,
            "initial_tilt": None if self.tilt is None else round(self.tilt, 3),
        }


@dataclass(frozen=True)
class SamplingReport:
    """How many configurations were generated and why others were dropped."""

    generated: int
    kept: int
    clash: int
    self_image: int
    duplicate: int


def rotation_to_minus_z(direction: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps unit vector `direction` onto -z."""

    v = np.asarray(direction, dtype=float)
    v = v / np.linalg.norm(v)
    target = np.array([0.0, 0.0, -1.0])
    cross = np.cross(v, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.dot(v, target))
    if sine < 1e-10:
        if cosine > 0:
            return np.eye(3)
        return np.diag([1.0, -1.0, -1.0])  # 180° about x
    axis = cross / sine
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + sine * k + (1 - cosine) * (k @ k)


def rotation_about_z(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly random rotation matrix (from a random unit quaternion)."""

    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def place_on_surface(
    slab: Atoms,
    positions: np.ndarray,
    symbols: Sequence[str],
    contact_gap: float | str = AUTO,
) -> np.ndarray:
    """Shift molecule `positions` along z so the closest atom touches the contact gap.

    For every molecule-slab pair with in-plane (minimum-image) distance
    ``r < gap`` the height must satisfy ``dz >= sqrt(gap^2 - r^2)``; the
    molecule is set at the lowest height meeting all of these. Pairs further
    apart in plane impose no constraint; if none does, the lowest molecule
    atom is placed at the height of the topmost slab atom.
    """

    z_slab = slab.positions[:, 2]
    top = z_slab.max()
    surface = np.flatnonzero(z_slab >= top - PLACEMENT_DEPTH)
    slab_positions = slab.positions[surface]
    slab_symbols = [slab.get_chemical_symbols()[i] for i in surface]

    if contact_gap == AUTO:
        mol_r = np.array([vdw_radius(s) for s in symbols])
        slab_r = np.array([vdw_radius(s) for s in slab_symbols])
        gaps = AUTO_GAP_SCALE * (mol_r[:, None] + slab_r[None, :])
    else:
        gaps = np.full((len(symbols), len(surface)), float(contact_gap))

    vectors = slab_positions[None, :, :] - positions[:, None, :]
    vectors[..., 2] = 0.0
    flat, _ = find_mic(vectors.reshape(-1, 3), slab.cell, pbc=[True, True, False])
    in_plane = np.linalg.norm(flat, axis=1).reshape(len(positions), len(surface))

    required = np.where(
        in_plane < gaps,
        slab_positions[None, :, 2] + np.sqrt(np.clip(gaps**2 - in_plane**2, 0.0, None)) - positions[:, None, 2],
        -np.inf,
    )
    shift = required.max()
    if not math.isfinite(shift):
        shift = top - positions[:, 2].min()
    placed = positions.copy()
    placed[:, 2] += shift
    return placed


def has_clash(slab: Atoms, positions: np.ndarray, symbols: Sequence[str], clash_scale: float) -> bool:
    """Any molecule-slab distance (full periodic images) below clash_scale * vdW sum."""

    _, distances = get_distances(positions, slab.positions, cell=slab.cell, pbc=True)
    mol_r = np.array([vdw_radius(s) for s in symbols])
    slab_r = np.array([vdw_radius(s) for s in slab.get_chemical_symbols()])
    return bool((distances < clash_scale * (mol_r[:, None] + slab_r[None, :])).any())


def has_self_image_clash(cell: np.ndarray, positions: np.ndarray, symbols: Sequence[str], clash_scale: float) -> bool:
    """Molecule too close to its own lateral periodic images."""

    radii = np.array([vdw_radius(s) for s in symbols])
    limits = clash_scale * (radii[:, None] + radii[None, :])
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            shifted = positions + i * cell[0] + j * cell[1]
            distances = np.linalg.norm(positions[:, None, :] - shifted[None, :, :], axis=-1)
            if (distances < limits).any():
                return True
    return False


def lateral_rmsd(cell: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    """RMSD of two molecule placements, up to a lateral lattice translation."""

    difference = first - second
    mean = difference.mean(axis=0)
    in_plane = mean.copy()
    in_plane[2] = 0.0
    wrapped, _ = find_mic(in_plane[None, :], cell, pbc=[True, True, False])
    translation = in_plane - wrapped[0]
    return float(np.sqrt(((difference - translation) ** 2).sum(axis=1).mean()))


def generate_candidates(
    slab: Atoms,
    sites: Sequence[AdsorptionSite],
    molecule: Atoms,
    analysis: MoleculeAnalysis,
    sampling: SamplingConfig,
) -> tuple[list[Candidate], SamplingReport]:
    """Anchored and random initial configurations on `slab`.

    `molecule` is the (gas-phase optimized) molecule and `analysis` its
    `MoleculeAnalysis`; atom order must match. Configurations clashing with
    the slab or with the molecule's own periodic images, and duplicates
    (RMSD < 0.2 Å up to lateral translations), are dropped.
    """

    symbols = molecule.get_chemical_symbols()
    center = np.array(analysis.center)
    relative = molecule.get_positions() - center
    cell = slab.cell.array
    rng = np.random.default_rng(sampling.seed)

    raw: list[tuple[AdsorptionSite, str, int | None, str, np.ndarray]] = []
    for site in sites:
        site_xy = np.array(site.position[:2])
        for anchor in analysis.anchors:
            align = rotation_to_minus_z(np.array(anchor.direction))
            anchor_offset = np.array(anchor.position) - center
            for spin in range(sampling.spins_per_anchor):
                rotation = rotation_about_z(spin * 360.0 / sampling.spins_per_anchor) @ align
                rotated = relative @ rotation.T
                anchor_rotated = anchor_offset @ rotation.T
                shift = np.array([site_xy[0] - anchor_rotated[0], site_xy[1] - anchor_rotated[1], 0.0])
                raw.append((site, anchor.label, spin, "anchor", rotated + shift))
    for _ in range(sampling.n_random):
        rotation = random_rotation(rng)
        site = sites[int(rng.integers(len(sites)))]
        rotated = relative @ rotation.T
        raw.append((site, "random", None, "random", rotated + np.array([*site.position[:2], 0.0])))

    kept: list[Candidate] = []
    clash = self_image = duplicate = 0
    for site, anchor, spin, source, positions in raw:
        placed = place_on_surface(slab, positions, symbols, sampling.contact_gap)
        if has_clash(slab, placed, symbols, sampling.clash_scale):
            clash += 1
            continue
        if has_self_image_clash(cell, placed, symbols, sampling.clash_scale):
            self_image += 1
            continue
        if any(lateral_rmsd(cell, placed, other.molecule_positions) < DUPLICATE_RMSD for other in kept):
            duplicate += 1
            continue
        tilt = molecule_tilt(Atoms(symbols, positions=placed), analysis)
        kept.append(
            Candidate(
                config_id=f"c{len(kept):04d}",
                site_id=site.site_id,
                site_kind=site.kind,
                anchor=anchor,
                spin=spin,
                source=source,
                tilt=tilt,
                molecule_positions=placed,
            )
        )
    report = SamplingReport(generated=len(raw), kept=len(kept), clash=clash, self_image=self_image, duplicate=duplicate)
    return kept, report
