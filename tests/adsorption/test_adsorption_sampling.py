"""Tests for initial adsorption configurations."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule
from ase.geometry import get_distances

pytest.importorskip("pymatgen")
pytest.importorskip("networkx")

from ase_structure_optimizer.adsorption.config import SamplingConfig  # noqa: E402
from ase_structure_optimizer.adsorption.molecule import analyze_molecule  # noqa: E402
from ase_structure_optimizer.adsorption.sampling import (  # noqa: E402
    generate_candidates,
    has_clash,
    has_self_image_clash,
    lateral_rmsd,
    place_on_surface,
    random_rotation,
    rotation_to_minus_z,
    vdw_radius,
)
from ase_structure_optimizer.adsorption.sites import find_sites  # noqa: E402
from ase_structure_optimizer.adsorption.surface import (  # noqa: E402
    conventional_bulk,
    generate_slabs,
    lateral_repeats,
)


@pytest.fixture(scope="module")
def cu111():
    unit = generate_slabs(conventional_bulk(bulk("Cu", "fcc", a=3.6))[0], (1, 1, 1), 6.0, 15.0)[0].atoms
    slab = unit.repeat((*lateral_repeats(unit, 10.0), 1))
    return slab, find_sites(unit, slab)


@pytest.mark.parametrize("direction", [[1, 0, 0], [0, 0, 1], [0, 0, -1], [0.3, -0.4, 0.2], [1e-12, 0, 1]])
def test_rotation_to_minus_z(direction):
    rotation = rotation_to_minus_z(np.array(direction, dtype=float))
    v = np.array(direction, dtype=float) / np.linalg.norm(direction)
    np.testing.assert_allclose(rotation @ v, [0, 0, -1], atol=1e-9)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_random_rotations_are_proper_and_isotropic():
    rng = np.random.default_rng(0)
    rotations = [random_rotation(rng) for _ in range(2000)]
    for rotation in rotations[:20]:
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
    mean_z = np.mean([rotation @ [0, 0, 1] for rotation in rotations], axis=0)
    assert np.linalg.norm(mean_z) < 0.1


def test_vdw_radius_falls_back_to_alvarez():
    assert vdw_radius("C") == pytest.approx(1.70)  # Bondi
    assert vdw_radius("Ti") == pytest.approx(2.46)  # not in Bondi


def test_place_on_surface_fixed_and_auto_gap(cu111):
    slab, _ = cu111
    co = molecule("CO")
    top_atom = slab.positions[slab.positions[:, 2].argmax()]
    positions = co.positions - co.positions[1] + [top_atom[0], top_atom[1], -5.0]  # C over a top atom
    placed = place_on_surface(slab, positions, co.get_chemical_symbols(), contact_gap=2.0)
    _, distances = get_distances(placed, slab.positions, cell=slab.cell, pbc=True)
    assert distances.min() == pytest.approx(2.0)
    assert placed[1, 2] - top_atom[2] == pytest.approx(2.0)  # directly above: gap is vertical

    auto = place_on_surface(slab, positions, co.get_chemical_symbols())
    _, distances = get_distances(auto, slab.positions, cell=slab.cell, pbc=True)
    radii_mol = np.array([vdw_radius(s) for s in co.get_chemical_symbols()])
    radii_slab = np.array([vdw_radius(s) for s in slab.get_chemical_symbols()])
    margin = distances - 0.9 * (radii_mol[:, None] + radii_slab[None, :])
    assert margin.min() == pytest.approx(0.0, abs=1e-9)


def test_clash_checks(cu111):
    slab, _ = cu111
    top = slab.positions[:, 2].max()
    assert has_clash(slab, np.array([[slab.positions[-1, 0], slab.positions[-1, 1], top + 1.0]]), ["O"], 0.7)
    assert not has_clash(slab, np.array([[0.0, 0.0, top + 4.0]]), ["O"], 0.7)
    small = np.diag([3.0, 3.0, 20.0])
    line = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    assert has_self_image_clash(small, line, ["C", "C"], 0.7)
    assert not has_self_image_clash(np.diag([30.0, 30.0, 20.0]), line, ["C", "C"], 0.7)


def test_lateral_rmsd_ignores_lattice_translations(cu111):
    slab, _ = cu111
    positions = molecule("CO").positions + [2.0, 2.0, 12.0]
    shifted = positions + slab.cell[0] - slab.cell[1]
    assert lateral_rmsd(slab.cell.array, positions, shifted) == pytest.approx(0.0, abs=1e-9)
    moved = positions + [0.5, 0.0, 0.0]
    assert lateral_rmsd(slab.cell.array, positions, moved) == pytest.approx(0.5)


def test_generate_candidates(cu111):
    slab, sites = cu111
    co = molecule("CO")
    analysis = analyze_molecule(co)
    sampling = SamplingConfig(n_random=8, spins_per_anchor=2, seed=3)
    candidates, report = generate_candidates(slab, sites, co, analysis, sampling)

    assert report.generated == len(sites) * len(analysis.anchors) * 2 + 8
    assert report.kept == len(candidates) == report.generated - report.clash - report.self_image - report.duplicate
    assert report.duplicate > 0  # CO is linear: spinning it about z changes nothing
    assert [c.config_id for c in candidates] == [f"c{i:04d}" for i in range(len(candidates))]
    assert {c.source for c in candidates} == {"anchor", "random"}
    assert {c.site_kind for c in candidates} == {"ontop", "bridge", "hollow"}

    radii_slab = np.array([vdw_radius(s) for s in slab.get_chemical_symbols()])
    radii_mol = np.array([vdw_radius(s) for s in co.get_chemical_symbols()])
    for candidate in candidates:
        _, distances = get_distances(candidate.molecule_positions, slab.positions, cell=slab.cell, pbc=True)
        margin = distances - 0.9 * (radii_mol[:, None] + radii_slab[None, :])
        assert margin.min() == pytest.approx(0.0, abs=1e-9)
        np.testing.assert_allclose(
            np.linalg.norm(candidate.molecule_positions[0] - candidate.molecule_positions[1]),
            co.get_distance(0, 1),
        )  # rigid molecule
        system = candidate.atoms(slab, co)
        assert len(system) == len(slab) + 2 and system.constraints == slab.constraints
        assert system.info["config_id"] == candidate.config_id

    # Both "C down" and "O down" anchored configurations exist, above their site.
    c_down = [c for c in candidates if c.anchor == "C1"]
    o_down = [c for c in candidates if c.anchor == "O0"]
    assert c_down and o_down
    for candidate in c_down:
        assert candidate.molecule_positions[1, 2] < candidate.molecule_positions[0, 2]
        site = next(s for s in sites if s.site_id == candidate.site_id)
        np.testing.assert_allclose(candidate.molecule_positions[1, :2], site.position[:2], atol=1e-9)
        assert candidate.tilt == pytest.approx(0.0, abs=1e-6)

    # No duplicates remain; the same seed gives the same configurations.
    for i, first in enumerate(candidates):
        for second in candidates[i + 1 :]:
            assert lateral_rmsd(slab.cell.array, first.molecule_positions, second.molecule_positions) >= 0.2
    again, _ = generate_candidates(slab, sites, co, analysis, sampling)
    for first, second in zip(candidates, again):
        np.testing.assert_allclose(first.molecule_positions, second.molecule_positions)


def test_fixed_contact_gap_and_clash_scale(cu111):
    slab, sites = cu111
    co = molecule("CO")
    analysis = analyze_molecule(co)
    candidates, report = generate_candidates(
        slab, sites, co, analysis, SamplingConfig(n_random=0, spins_per_anchor=1, contact_gap=1.0, clash_scale=0.7)
    )
    assert report.clash == report.generated and not candidates  # 1 Å is inside 0.7 x vdW sum
