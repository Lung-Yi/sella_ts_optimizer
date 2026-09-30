"""Tests for slab geometry: classification, cutting, orientation, supercells."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, fcc111, fcc211
from ase.constraints import FixAtoms

pytest.importorskip("pymatgen")

from ase_structure_optimizer.adsorption.surface import (  # noqa: E402
    atomic_layers,
    classify_solid,
    conventional_bulk,
    fix_bottom_layers,
    generate_slabs,
    lateral_repeats,
    molecule_size,
    orient_slab,
    perpendicular_widths,
    slab_thickness,
    surface_energy,
    termination_id,
    top_layer_compositions,
)


def _orthorhombic_ab() -> Atoms:
    """Small orthorhombic A2B2 cell with a != b != c (non-standard axis order)."""

    return Atoms(
        "Ti2Si2",
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, 0.0, 0.25], [0.0, 0.5, 0.75]],
        cell=[6.5, 3.6, 5.0],
        pbc=True,
    )


def _check_oriented(slab: Atoms, vacuum: float) -> None:
    cell = slab.cell.array
    assert cell[0, 1] == pytest.approx(0.0) and cell[0, 2] == pytest.approx(0.0)
    assert cell[1, 2] == pytest.approx(0.0)
    assert cell[2, :2] == pytest.approx([0.0, 0.0])
    assert slab.positions[:, 2].min() == pytest.approx(1.0)
    assert cell[2, 2] == pytest.approx(slab_thickness(slab) + vacuum)
    assert np.linalg.det(cell) > 0


def test_classify_solid():
    assert classify_solid(bulk("Cu", "fcc", a=3.6)) == ("bulk", None)
    slab = fcc111("Cu", (2, 2, 3), vacuum=6.0)
    slab.pbc = True
    assert classify_solid(slab) == ("slab", 2)
    assert classify_solid(slab, "bulk") == ("bulk", None)
    assert classify_solid(bulk("Cu", "fcc", a=3.6), "slab") == ("slab", 2)
    with pytest.raises(ValueError):
        classify_solid(slab, "surface")


def test_conventional_bulk_keeps_input_axes_or_converts_primitive():
    primitive = bulk("Cu", "fcc", a=3.6)
    conventional, converted = conventional_bulk(primitive)
    assert converted and len(conventional) == 4
    assert conventional.cell.cellpar()[:3] == pytest.approx([3.6] * 3)

    ortho = _orthorhombic_ab()
    kept, converted = conventional_bulk(ortho)
    assert not converted
    np.testing.assert_allclose(kept.cell.array, ortho.cell.array)  # (001) stays the input's (001)


@pytest.mark.parametrize("miller", [(1, 1, 1), (1, 0, 0), (1, 1, 0), (2, 1, 1)])
def test_generated_slabs_are_oriented_and_thick_enough(miller):
    conventional, _ = conventional_bulk(bulk("Cu", "fcc", a=3.6))
    slabs = generate_slabs(conventional, miller, min_thickness=8.0, vacuum_above=12.0)
    assert slabs
    for slab in slabs:
        assert slab.miller == miller and slab.stoichiometric and slab.symmetric
        assert slab_thickness(slab.atoms) >= 8.0 - 1e-6
        _check_oriented(slab.atoms, 12.0)


def test_orthorhombic_terminations_follow_input_axes():
    slabs = generate_slabs(_orthorhombic_ab(), (0, 0, 1), min_thickness=8.0, vacuum_above=15.0)
    assert len(slabs) >= 2
    for slab in slabs:
        # (001) of the input: the in-plane lattice is 6.5 x 3.6 in some order.
        assert sorted(slab.atoms.cell.lengths()[:2]) == pytest.approx([3.6, 6.5])
        _check_oriented(slab.atoms, 15.0)


def test_orient_slab_handles_wrapping_and_other_axes():
    reference = fcc111("Cu", (2, 2, 4), vacuum=6.0)
    reference.pbc = True
    thickness = slab_thickness(reference)

    wrapped = reference.copy()
    wrapped.positions[:, 2] -= 8.0  # split the slab across the periodic boundary
    wrapped.wrap()
    oriented = orient_slab(wrapped, 2, 15.0)
    assert slab_thickness(oriented) == pytest.approx(thickness)
    _check_oriented(oriented, 15.0)

    # Vacuum along cell axis 0.
    permuted = Atoms(
        reference.get_chemical_symbols(),
        positions=reference.positions[:, [2, 0, 1]],
        cell=reference.cell.array[[2, 0, 1]][:, [2, 0, 1]],
        pbc=True,
    )
    oriented = orient_slab(permuted, 0, 10.0)
    assert slab_thickness(oriented) == pytest.approx(thickness)
    _check_oriented(oriented, 10.0)
    assert sorted(oriented.cell.lengths()[:2]) == pytest.approx(sorted(reference.cell.lengths()[:2]))

    # Interatomic distances are preserved (a rigid rotation plus wrapping).
    np.testing.assert_allclose(
        np.sort(oriented.get_all_distances(mic=True).ravel()),
        np.sort(reference.get_all_distances(mic=True).ravel()),
        atol=1e-8,
    )


def test_orient_slab_tilted_c_axis():
    slab = fcc211("Cu", (3, 2, 3), vacuum=5.0)
    slab.pbc = True
    cell = slab.cell.array.copy()
    cell[2] += 0.3 * cell[0]
    slab.set_cell(cell, scale_atoms=False)
    oriented = orient_slab(slab, 2, 12.0)
    _check_oriented(oriented, 12.0)


def test_lateral_repeats_use_perpendicular_widths():
    slab = fcc111("Cu", (1, 1, 3), vacuum=5.0)  # 60° cell: widths < vector lengths
    width_a, width_b = perpendicular_widths(slab)
    assert width_a == pytest.approx(slab.cell.lengths()[0] * np.sin(np.radians(60)))
    n_a, n_b = lateral_repeats(slab, 10.0)
    assert n_a * width_a >= 10.0 and (n_a - 1) * width_a < 10.0
    assert n_b * width_b >= 10.0
    assert lateral_repeats(slab, 0.1) == (1, 1)


def test_layers_and_fixed_atoms():
    slab = fcc111("Cu", (2, 2, 4), vacuum=5.0)
    layers = atomic_layers(slab)
    assert [len(layer) for layer in layers] == [4, 4, 4, 4]
    assert slab.positions[layers[0], 2].max() < slab.positions[layers[1], 2].min()

    fixed = fix_bottom_layers(slab, 0.5)
    assert sorted(fixed) == sorted(layers[0] + layers[1])
    assert isinstance(slab.constraints[0], FixAtoms)

    five = fcc111("Cu", (2, 2, 5), vacuum=5.0)
    assert len(fix_bottom_layers(five, 0.5)) == 8  # 8 and 12 tie with the target 10: fewer
    assert len(fix_bottom_layers(five, 0.0)) == 0 and not five.constraints
    assert len(fix_bottom_layers(five, 1.0)) == 16  # the top layer is never fixed
    with pytest.raises(ValueError):
        fix_bottom_layers(five, 1.5)

    compositions = top_layer_compositions(slab, n_layers=2)
    assert [row["formula"] for row in compositions] == ["Cu4", "Cu4"]
    assert compositions[0]["z"] > compositions[1]["z"]


def test_surface_energy_and_helpers():
    unit = bulk("Cu", "fcc", a=3.6)
    slab = fcc111("Cu", (1, 1, 4), vacuum=5.0)
    area = np.linalg.norm(np.cross(slab.cell[0], slab.cell[1]))
    gamma = surface_energy(slab, slab_energy=-10.0, bulk=unit, bulk_energy=-3.0)
    assert gamma == pytest.approx((-10.0 + 12.0) / (2 * area))
    nonstoich = slab + Atoms("O", positions=[[0, 0, 20]])
    assert surface_energy(nonstoich, -10.0, unit, -3.0) is None

    assert molecule_size(Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.13]])) == pytest.approx(1.13)
    assert molecule_size(Atoms("H")) == 0.0
    assert termination_id((0, 0, 1), 0) == "001_t0"
    assert termination_id((1, -1, 0), 2) == "1-10_t2"
    assert termination_id(None, 0) == "slab_t0"
