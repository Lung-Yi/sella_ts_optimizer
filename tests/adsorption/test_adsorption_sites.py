"""Tests for adsorption site finding."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import bulk

pytest.importorskip("pymatgen")

from ase_structure_optimizer.adsorption.sites import AdsorptionSite, find_sites  # noqa: E402
from ase_structure_optimizer.adsorption.surface import (  # noqa: E402
    conventional_bulk,
    generate_slabs,
    lateral_repeats,
)


def _cu_slab(miller, min_lateral=10.0):
    conventional, _ = conventional_bulk(bulk("Cu", "fcc", a=3.6))
    unit = generate_slabs(conventional, miller, 8.0, 12.0)[0].atoms
    return unit, unit.repeat((*lateral_repeats(unit, min_lateral), 1))


@pytest.mark.parametrize(
    ("miller", "expected"),
    [
        ((1, 1, 1), {("ontop", 1): 1, ("bridge", 2): 1, ("hollow", 3): 2}),  # fcc + hcp hollow
        ((1, 0, 0), {("ontop", 1): 1, ("bridge", 2): 1, ("hollow", 4): 1}),
        ((1, 1, 0), {("ontop", 1): 1, ("bridge", 2): 2, ("hollow", 4): 1}),  # short + long bridge
    ],
)
def test_textbook_fcc_sites(miller, expected):
    unit, supercell = _cu_slab(miller)
    sites = find_sites(unit, supercell)
    counts: dict = {}
    for site in sites:
        key = (site.kind, len(site.surface_atoms))
        counts[key] = counts.get(key, 0) + 1
    assert counts == expected


def test_site_positions_labels_and_centering():
    unit, supercell = _cu_slab((1, 1, 1))
    sites = find_sites(unit, supercell)
    assert [site.site_id for site in sites] == ["ontop_Cu", "bridge_Cu-Cu", "hollow_Cu-Cu-Cu_1", "hollow_Cu-Cu-Cu_2"]
    center = 0.5 * (supercell.cell[0] + supercell.cell[1])
    top = supercell.positions[:, 2].max()
    unit_width = np.linalg.norm(unit.cell[0])
    for site in sites:
        position = np.array(site.position)
        assert np.linalg.norm((position - center)[:2]) < 1.5 * unit_width
        assert position[2] == pytest.approx(top)
        atoms_xy = supercell.positions[list(site.surface_atoms), :2]
        np.testing.assert_allclose(position[:2], atoms_xy.mean(axis=0), atol=1e-6)
        assert set(site.elements) == {"Cu"}

    ontop = sites[0]
    np.testing.assert_allclose(ontop.position, supercell.positions[ontop.surface_atoms[0]])
    restored = AdsorptionSite.from_dict(ontop.to_dict())
    assert restored.site_id == ontop.site_id and restored.surface_atoms == ontop.surface_atoms
    assert restored.position == pytest.approx(ontop.position, abs=1e-6)


def test_site_types_filter_and_snapping_to_relaxed_slab():
    unit, supercell = _cu_slab((1, 1, 1))
    relaxed = supercell.copy()
    relaxed.positions[:, :2] += 0.05  # small in-plane shift of the relaxed slab
    sites = find_sites(unit, relaxed, site_types=("ontop",))
    assert [site.kind for site in sites] == ["ontop"]
    np.testing.assert_allclose(sites[0].position, relaxed.positions[sites[0].surface_atoms[0]])
    with pytest.raises(ValueError):
        find_sites(unit, supercell, site_types=("fourfold",))


def test_site_atoms_are_chosen_on_the_unit_slab():
    """A relaxed atom sinking below the surface depth must not change site definitions."""

    unit, supercell = _cu_slab((1, 1, 1))
    reference = find_sites(unit, supercell)
    relaxed = supercell.copy()
    for site in reference:
        if site.kind == "hollow":
            relaxed.positions[site.surface_atoms[0], 2] -= 1.2  # below the 0.9 Å window
            break
    moved = find_sites(unit, relaxed)
    assert [(s.site_id, s.surface_atoms) for s in moved] == [(s.site_id, s.surface_atoms) for s in reference]
    for before, after in zip(reference, moved):
        assert after.position[:2] == pytest.approx(before.position[:2], abs=1e-6)


def test_one_by_one_supercell_uses_periodic_images():
    unit, _ = _cu_slab((1, 1, 1))
    sites = find_sites(unit, unit)
    hollow = [site for site in sites if site.kind == "hollow"][0]
    assert hollow.surface_atoms == (0, 0, 0)  # three images of the single top atom
    top = unit.positions[unit.positions[:, 2].argmax()]
    distance = np.linalg.norm(np.array(hollow.position[:2]) - top[:2])
    assert distance == pytest.approx(np.linalg.norm(unit.cell[0]) / np.sqrt(3), abs=1e-6)
    with pytest.raises(ValueError, match="repeat"):
        find_sites(unit, unit.repeat((2, 1, 1))[:-1])


def test_element_order_in_labels():
    from ase import Atoms

    # Checkerboard A-B square surface: bridges are Ti-Si with the heavier element first.
    unit = Atoms(
        "TiSiSiTi",
        positions=[[0, 0, 1.0], [1.5, 1.5, 1.0], [1.5, 0, 2.5], [0, 1.5, 2.5]],
        cell=[3.0, 3.0, 20.0],
        pbc=True,
    )
    sites = find_sites(unit, unit.repeat((4, 4, 1)))
    for site in sites:
        if site.kind == "bridge":
            assert site.site_id.startswith("bridge_Ti-Si")
