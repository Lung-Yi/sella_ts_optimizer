"""Structure file I/O and simple geometric classification."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from ase import Atoms
from ase.io import read, write

_SUPPORTED_SUFFIXES = {
    ".xyz": "extxyz",
    ".extxyz": "extxyz",
    ".cif": "cif",
    ".traj": "traj",
}


def read_structure(path: str | Path) -> Atoms:
    """Read the last frame of an xyz / extxyz / cif / traj file.

    * CIF: `atoms.info["occupancy"]` is removed and all tags are reset to 0
      (both otherwise make `ase.visualize.plot.plot_atoms` raise `KeyError`).
    * XYZ: a plain XYZ file (no `Lattice=` in the comment line) is returned
      with `pbc=False`. An extended-XYZ file that carries a lattice, such as
      the periodic outputs written by `optimize_geometry_atoms()`, keeps its
      cell and pbc.
    """

    path = Path(path)
    suffix = path.suffix.lower()
    try:
        file_format = _SUPPORTED_SUFFIXES[suffix]
    except KeyError as exc:
        choices = ", ".join(sorted(_SUPPORTED_SUFFIXES))
        raise ValueError(f"Unsupported structure file '{path}'. Supported suffixes: {choices}") from exc

    atoms = read(path, index=-1, format=file_format)

    if file_format == "cif":
        atoms.info.pop("occupancy", None)
        atoms.set_tags(0)
    elif suffix == ".xyz" and not atoms.cell.any():
        atoms.pbc = False

    return atoms


def write_trajectory_pair(images: Atoms | Sequence[Atoms], path_stem: str | Path) -> tuple[Path, Path]:
    """Write images to both `<path_stem>.traj` and `<path_stem>.extxyz`.

    The binary `.traj` keeps full ASE fidelity; the text `.extxyz` copy can be
    opened in editors and viewers. Energies/forces stored on the images'
    calculators (e.g. `SinglePointCalculator`) are written to both files.
    Returns `(traj_path, extxyz_path)`.
    """

    if isinstance(images, Atoms):
        images = [images]
    images = list(images)
    if not images:
        raise ValueError("write_trajectory_pair() needs at least one image")

    stem = Path(path_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    traj_path = stem.parent / f"{stem.name}.traj"
    extxyz_path = stem.parent / f"{stem.name}.extxyz"
    write(traj_path, images, format="traj")
    write(extxyz_path, images, format="extxyz")
    return traj_path, extxyz_path


def is_slab(atoms: Atoms, min_gap: float = 8.0) -> tuple[bool, int | None]:
    """Decide whether `atoms` is a slab and along which cell axis the vacuum lies.

    For every cell axis the largest empty gap between atomic planes is
    measured perpendicular to the other two cell vectors (periodic wrap
    included). A non-periodic axis counts as vacuum. The structure is a slab
    when exactly one axis has a gap of at least `min_gap` Å; bulk crystals
    (no such axis) and molecules/wires in a box (two or three such axes)
    return `(False, None)`.
    """

    if len(atoms) == 0:
        return False, None

    gaps = [_axis_gap(atoms, axis) for axis in range(3)]
    vacuum_axes = [axis for axis, gap in enumerate(gaps) if gap >= min_gap]
    if len(vacuum_axes) == 1:
        return True, vacuum_axes[0]
    return False, None


def _axis_gap(atoms: Atoms, axis: int) -> float:
    """Largest perpendicular gap (Å) between atoms along one cell axis."""

    cell = atoms.cell
    if not atoms.pbc[axis]:
        return float("inf")
    if abs(cell.volume) < 1e-8:
        # Degenerate cell: gaps cannot be measured along a periodic axis.
        return 0.0

    reciprocal = cell.reciprocal()
    spacing = 1.0 / np.linalg.norm(reciprocal[axis])
    fractions = np.sort(np.mod(atoms.get_scaled_positions(wrap=False)[:, axis], 1.0))
    diffs = np.diff(fractions)
    wrap_gap = fractions[0] + 1.0 - fractions[-1]
    largest = max(diffs.max(initial=0.0), wrap_gap)
    return float(largest * spacing)
