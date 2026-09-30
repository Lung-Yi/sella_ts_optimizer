"""Tests for the analysis of adsorption configurations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, molecule
from ase.constraints import FixAtoms
from ase.io import read

pytest.importorskip("networkx")

from ase_structure_optimizer.adsorption.analysis import (  # noqa: E402
    adsorption_height,
    azimuth_angle,
    azimuth_sign_atom,
    boltzmann_weights,
    classify,
    deduplicate,
    intact_check,
    molecule_part,
    molecule_tilt,
    surface_displacement,
    tilt_angle,
)
from ase_structure_optimizer.adsorption.molecule import analyze_molecule  # noqa: E402

CPMO_XYZ = Path(__file__).resolve().parents[1] / "data" / "CpMo_CO3H.xyz"
CP_RING = [8, 9, 10, 11, 12]


@pytest.fixture
def cpmo():
    return read(CPMO_XYZ)


@pytest.fixture
def slab():
    atoms = fcc111("Cu", (4, 4, 3), vacuum=8.0)
    atoms.pbc = True
    atoms.set_constraint(FixAtoms(indices=[a.index for a in atoms if a.tag == 3]))
    return atoms


def _on_slab(slab, molecule_atoms, clearance):
    placed = molecule_atoms.copy()
    top = slab.positions[:, 2].max()
    center = slab.cell[0] / 2 + slab.cell[1] / 2
    placed.translate([center[0], center[1], top + clearance] - np.array([*placed.positions[:, :2].mean(0), placed.positions[:, 2].min()]))
    system = slab + placed
    system.cell = slab.cell
    return system


# -- intactness ---------------------------------------------------------------


def test_hapticity_change_is_not_dissociation(cpmo):
    reference = analyze_molecule(cpmo)
    slipped = cpmo.copy()
    # Tilt the Cp ring about the C8-C9 edge: Mo-C10/C11/C12 bonds break (eta5 -> eta2).
    axis = slipped.positions[9] - slipped.positions[8]
    ring = slipped[CP_RING + [13, 14, 15, 16, 17]]
    ring.rotate(-25, axis, center=slipped.positions[8])
    slipped.positions[CP_RING + [13, 14, 15, 16, 17]] = ring.positions
    bonds = {(i, j) for i, j, _ in analyze_molecule(slipped).bonds}
    assert {(0, 8), (0, 9)} <= bonds and not {(0, 10), (0, 11), (0, 12)} & bonds  # really eta2 now
    intact, reason = intact_check(slipped, reference, 1.2)
    assert intact, reason


def test_ligand_loss_and_bond_breaking_are_dissociation(cpmo):
    reference = analyze_molecule(cpmo)
    no_hydride = cpmo.copy()
    no_hydride.positions[7] += (no_hydride.positions[7] - no_hydride.positions[0]) * 2
    intact, reason = intact_check(no_hydride, reference, 1.2)
    assert not intact and reason == "H detached from Mo0"

    broken_co = cpmo.copy()
    broken_co.positions[2] += (broken_co.positions[2] - broken_co.positions[1]) * 1.5
    intact, reason = intact_check(broken_co, reference, 1.2)
    assert not intact and "broken C1-O2" in reason

    water = molecule("H2O")
    reference = analyze_molecule(water)
    assert intact_check(water, reference, 1.2) == (True, "")
    stretched = water.copy()
    stretched.positions[1] += [0, 2.0, 0]
    intact, reason = intact_check(stretched, reference, 1.2)
    assert not intact and "broken O0-H1" in reason


# -- classification -------------------------------------------------------------


def test_four_classes(slab):
    co = molecule("CO")  # O0 above C1 after flipping
    co.rotate(180, "x")
    reference = analyze_molecule(co)
    n = len(slab)
    settings = dict(bond_scale=1.2, contact_scale=1.25, desorbed_distance=4.5)

    assert classify(_on_slab(slab, co, 1.9), n, reference, **settings)[:2] == ("chemisorbed", "CO")
    assert classify(_on_slab(slab, co, 3.5), n, reference, **settings)[:2] == ("physisorbed", "none")
    assert classify(_on_slab(slab, co, 6.0), n, reference, **settings)[0] == "desorbed"
    split = co.copy()
    split.positions[0] += [2.5, 0, -1.0]
    category, _, reason = classify(_on_slab(slab, split, 1.9), n, reference, **settings)
    assert category == "dissociated" and "broken" in reason


def test_surface_displacement_ignores_fixed_atoms(slab):
    n = len(slab)
    system = _on_slab(slab, molecule("CO"), 2.0)
    assert surface_displacement(system, n, slab) == pytest.approx(0.0)
    fixed = slab.constraints[0].get_indices()[0]
    system.positions[fixed] += [0, 0, 3.0]
    assert surface_displacement(system, n, slab) == pytest.approx(0.0)
    free = [i for i in range(n) if i not in set(slab.constraints[0].get_indices())][0]
    system.positions[free] += [1.2, 0, 0]
    assert surface_displacement(system, n, slab) == pytest.approx(1.2)
    # A lattice translation is not a displacement.
    shifted = _on_slab(slab, molecule("CO"), 2.0)
    shifted.positions[free] += slab.cell[0]
    assert surface_displacement(shifted, n, slab) == pytest.approx(0.0, abs=1e-9)


# -- angles and height ---------------------------------------------------------------


@pytest.mark.parametrize("angle", [0.0, 30.0, 90.0, 135.0, 180.0])
def test_tilt_of_cpmo(cpmo, angle):
    reference = analyze_molecule(cpmo)
    oriented = cpmo.copy()
    u = reference.reference_axis.vector(oriented)
    oriented.rotate(u, [0, 0, 1], center=oriented.positions[0])  # u along +z: theta 0
    oriented.rotate(angle, "x", center=oriented.positions[0])
    assert molecule_tilt(oriented, reference) == pytest.approx(angle, abs=1.0)


def test_tilt_of_undirected_axis_is_folded():
    co = molecule("CO")
    reference = analyze_molecule(co)
    tilted = co.copy()
    tilted.rotate(150, "x")
    assert molecule_tilt(tilted, reference) == pytest.approx(30.0, abs=1.0)
    assert tilt_angle(None, True) is None


def test_azimuth_follows_rotation_about_z(cpmo):
    reference = analyze_molecule(cpmo)
    sign = azimuth_sign_atom(cpmo, reference)
    cell = np.diag([20.0, 20.0, 30.0])
    oriented = cpmo.copy()
    oriented.rotate(reference.reference_axis.vector(oriented), [0, 0, 1], center=oriented.positions[0])
    oriented.rotate(20, "x", center=oriented.positions[0])
    base = azimuth_angle(oriented, reference, sign, cell)
    assert base is not None
    for extra in (45.0, 90.0, 200.0):
        turned = oriented.copy()
        turned.rotate(extra, "z", center=oriented.positions[0])
        assert azimuth_angle(turned, reference, sign, cell) == pytest.approx((base + extra) % 360, abs=1e-6)
    # Relative to cell vector a: rotating the cell's a axis shifts the azimuth the other way.
    rotated_cell = np.array([[0, 20.0, 0], [-20.0, 0, 0], [0, 0, 30.0]])
    assert azimuth_angle(oriented, reference, sign, rotated_cell) == pytest.approx((base - 90) % 360, abs=1e-6)


def test_height_and_unwrapping(slab, cpmo):
    reference = analyze_molecule(cpmo)
    system = _on_slab(slab, cpmo, 2.0)
    n = len(slab)
    top = slab.positions[:, 2].max()
    assert adsorption_height(system, n, reference) == pytest.approx(system.positions[n, 2] - top)

    wrapped = system.copy()
    wrapped.positions[n + 8] += wrapped.cell[0]  # one Cp carbon in the neighboring cell
    whole = molecule_part(wrapped, n)
    np.testing.assert_allclose(whole.positions, system.positions[n:], atol=1e-9)


# -- de-duplication and populations --------------------------------------------------


def test_deduplicate_lattice_translation(slab):
    co = molecule("CO")
    first = _on_slab(slab, co, 2.0).positions[len(slab):]
    positions = {
        "a": first,
        "b": first + slab.cell[0] - 2 * slab.cell[1],  # same configuration, other cell
        "c": first + [0.8, 0, 0],  # shifted: different
        "d": first,  # same geometry but a different class
    }
    records = [
        {"config_id": "a", "class": "chemisorbed", "contact": "CO", "eads": -1.00},
        {"config_id": "b", "class": "chemisorbed", "contact": "CO", "eads": -1.01},
        {"config_id": "c", "class": "chemisorbed", "contact": "CO", "eads": -1.00},
        {"config_id": "d", "class": "physisorbed", "contact": "none", "eads": -1.00},
    ]
    unique = deduplicate(records, positions, slab.cell.array, energy_tol=0.02, rmsd_tol=0.3)
    ids = [row["config_id"] for row in unique]
    assert ids[0] == "b" and set(ids) == {"b", "c", "d"}
    assert unique[0]["duplicates"] == ["a"] and unique[0]["n_duplicates"] == 1

    far_energy = [dict(records[0]), {**records[1], "eads": -1.5}]
    assert len(deduplicate(far_energy, positions, slab.cell.array, 0.02, 0.3)) == 2


def test_boltzmann_weights():
    weights = boltzmann_weights([-1.0, -0.9, -1.0], 298.15)
    assert sum(weights) == pytest.approx(1.0)
    assert weights[0] == pytest.approx(weights[2])
    assert weights[1] / weights[0] == pytest.approx(np.exp(-0.1 / (8.617333262e-5 * 298.15)))
    assert boltzmann_weights([], 300) == []
