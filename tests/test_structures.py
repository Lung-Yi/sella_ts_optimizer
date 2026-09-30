"""Tests for structure I/O and slab detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, fcc111, molecule
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write

from ase_structure_optimizer.structures import is_slab, read_structure, write_trajectory_pair

TISI_CIF = """data_TiSi
_cell_length_a 6.544
_cell_length_b 3.638
_cell_length_c 4.997
_cell_angle_alpha 90.0
_cell_angle_beta 90.0
_cell_angle_gamma 90.0
_symmetry_space_group_name_H-M 'P n m a'
_symmetry_Int_Tables_number 62
loop_
_symmetry_equiv_pos_as_xyz
  'x, y, z'
  '-x+1/2, -y, z+1/2'
  '-x, y+1/2, -z'
  'x+1/2, -y+1/2, -z+1/2'
  '-x, -y, -z'
  'x+1/2, y, -z+1/2'
  'x, -y+1/2, z'
  '-x+1/2, y+1/2, z+1/2'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
  Ti1 Ti 0.1768 0.2500 0.1257 1.0
  Si1 Si 0.0358 0.2500 0.6502 1.0
"""


@pytest.fixture
def tisi_cif(tmp_path: Path) -> Path:
    path = tmp_path / "TiSi.cif"
    path.write_text(TISI_CIF, encoding="utf-8")
    return path


def test_read_cif_clears_occupancy_and_tags(tisi_cif: Path):
    raw = read(tisi_cif)
    assert "occupancy" in raw.info  # the problem read_structure() fixes

    atoms = read_structure(tisi_cif)
    assert len(atoms) == 8 and atoms.get_chemical_formula() == "Si4Ti4"
    assert "occupancy" not in atoms.info
    assert (atoms.get_tags() == 0).all()
    assert atoms.pbc.all()


def test_read_cif_can_be_plotted(tisi_cif: Path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ase.visualize.plot import plot_atoms

    fig, ax = plt.subplots()
    plot_atoms(read_structure(tisi_cif), ax, rotation="-90x")
    plt.close(fig)


def test_read_plain_xyz_is_not_periodic(tmp_path: Path):
    path = tmp_path / "water.xyz"
    write(path, molecule("H2O"), format="xyz")
    atoms = read_structure(path)
    assert not atoms.pbc.any()
    assert atoms.get_chemical_formula() == "H2O"


def test_read_xyz_with_lattice_keeps_cell(tmp_path: Path):
    slab = fcc111("Cu", size=(2, 2, 2), vacuum=5.0)
    slab.pbc = True
    path = tmp_path / "geometry_min_optimized.xyz"
    write(path, slab, format="extxyz")
    atoms = read_structure(path)
    assert atoms.pbc.all()
    np.testing.assert_allclose(atoms.cell.array, slab.cell.array)


def test_read_extxyz_and_traj_last_frame(tmp_path: Path):
    frames = [molecule("CO"), molecule("CO")]
    frames[1].positions[1, 2] += 0.2
    for suffix in (".extxyz", ".traj"):
        path = tmp_path / f"frames{suffix}"
        write(path, frames)
        atoms = read_structure(path)
        np.testing.assert_allclose(atoms.positions, frames[1].positions)


def test_read_unsupported_suffix(tmp_path: Path):
    path = tmp_path / "structure.pdb"
    path.write_text("")
    with pytest.raises(ValueError, match="Unsupported"):
        read_structure(path)


def test_write_trajectory_pair_keeps_energies(tmp_path: Path):
    images = []
    for index, height in enumerate([1.5, 2.0, 2.5]):
        atoms = molecule("CO")
        atoms.positions[:, 2] += height
        atoms.calc = SinglePointCalculator(atoms, energy=-float(index), forces=np.zeros((2, 3)))
        images.append(atoms)

    traj, extxyz = write_trajectory_pair(images, tmp_path / "nested" / "scan.v1")
    assert traj == tmp_path / "nested" / "scan.v1.traj"
    assert extxyz == tmp_path / "nested" / "scan.v1.extxyz"
    extxyz.read_text(encoding="utf-8")  # plain text
    for path in (traj, extxyz):
        energies = [image.get_potential_energy() for image in read(path, ":")]
        assert energies == pytest.approx([0.0, -1.0, -2.0])

    single_traj, _ = write_trajectory_pair(images[0], tmp_path / "single")
    assert len(read(single_traj, ":")) == 1

    with pytest.raises(ValueError):
        write_trajectory_pair([], tmp_path / "empty")


def test_is_slab_detects_bulk_and_slabs(tisi_cif: Path):
    assert is_slab(bulk("Cu", "fcc", a=3.6)) == (False, None)
    assert is_slab(read_structure(tisi_cif)) == (False, None)
    assert is_slab(bulk("Cu", "fcc", a=3.6).repeat((3, 3, 3))) == (False, None)

    slab = fcc111("Cu", size=(2, 2, 4), vacuum=6.0)
    slab.pbc = True
    assert is_slab(slab) == (True, 2)
    assert is_slab(slab, min_gap=15.0) == (False, None)


def test_is_slab_vacuum_along_other_axis():
    slab = fcc111("Cu", size=(2, 2, 4), vacuum=6.0)
    slab.pbc = True
    # Rotate axes: vacuum direction becomes cell axis 0.
    permuted = Atoms(
        slab.get_chemical_symbols(),
        positions=slab.positions[:, [2, 0, 1]],
        cell=slab.cell.array[[2, 0, 1]][:, [2, 0, 1]],
        pbc=True,
    )
    assert is_slab(permuted) == (True, 0)


def test_is_slab_non_orthogonal_cell_uses_perpendicular_gap():
    slab = fcc111("Cu", size=(2, 2, 4), vacuum=4.5)  # ~9 Å vacuum in total
    slab.pbc = True
    assert is_slab(slab, min_gap=8.0) == (True, 2)
    sheared = slab.copy()
    cell = sheared.cell.array.copy()
    cell[2] += 0.5 * cell[0]  # tilt c; perpendicular vacuum is unchanged
    sheared.set_cell(cell, scale_atoms=False)
    assert is_slab(sheared, min_gap=8.0) == (True, 2)


def test_is_slab_nonperiodic_axis_counts_as_vacuum():
    slab = fcc111("Cu", size=(2, 2, 3), vacuum=1.0)
    assert slab.pbc.tolist() == [True, True, False]
    assert is_slab(slab) == (True, 2)


def test_molecule_in_box_is_not_a_slab():
    box = molecule("H2O")
    box.center(vacuum=8.0)
    box.pbc = True
    assert is_slab(box) == (False, None)
    assert is_slab(Atoms()) == (False, None)
