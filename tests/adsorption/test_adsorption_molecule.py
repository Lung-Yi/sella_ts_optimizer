"""Tests for adsorbate analysis: fragments, anchors and reference axis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import read

pytest.importorskip("networkx")

from ase_structure_optimizer.adsorption.config import MoleculeProps  # noqa: E402
from ase_structure_optimizer.adsorption.molecule import (  # noqa: E402
    analyze_molecule,
    format_analysis,
    generate_conformers,
)

CPMO_XYZ = Path(__file__).resolve().parents[1] / "data" / "CpMo_CO3H.xyz"
CP_RING = (8, 9, 10, 11, 12)


@pytest.fixture
def cpmo() -> Atoms:
    return read(CPMO_XYZ)


def _anchor(analysis, label):
    for anchor in analysis.anchors:
        if anchor.label == label or label in anchor.merged:
            return anchor
    raise AssertionError(f"no anchor {label}: {[a.label for a in analysis.anchors]}")


def test_cpmo_fragments_and_labels(cpmo):
    analysis = analyze_molecule(cpmo)
    assert [f.name for f in analysis.fragments] == ["CO_1", "CO_2", "CO_3", "H", "C5H5"]
    assert analysis.metal_indices == (0,)
    assert analysis.rings == (CP_RING,)
    assert analysis.atom_labels[0] == "Mo"
    assert analysis.atom_labels[2] == "CO_1" and analysis.atom_labels[7] == "H"
    assert analysis.warnings == ()
    np.testing.assert_allclose(analysis.center, cpmo.positions[0])


def test_cpmo_anchors(cpmo):
    analysis = analyze_molecule(cpmo)
    positions = cpmo.get_positions()

    cp = _anchor(analysis, "ring5@C5H5")  # Cp ring centroid (merged with the C5H5 fragment centroid)
    np.testing.assert_allclose(cp.position, positions[list(CP_RING)].mean(axis=0), atol=0.3)
    for label in ("CO_1", "CO_2", "CO_3"):
        assert _anchor(analysis, label).kind == "fragment"
    hydride = _anchor(analysis, "H7")
    np.testing.assert_allclose(hydride.position, positions[7])
    metal = _anchor(analysis, "Mo0")
    assert metal.kind == "metal"
    for oxygen in ("O2", "O4", "O6"):
        _anchor(analysis, oxygen)

    # Only one terminal Cp hydrogen is kept.
    terminal_h = [a for a in analysis.anchors if a.kind == "terminal" and cpmo[a.atom_indices[0]].symbol == "H"]
    assert len(terminal_h) == 1 and terminal_h[0].atom_indices[0] in range(13, 18)

    # Directions are unit vectors from the metal to the anchor.
    for anchor in analysis.anchors:
        assert np.linalg.norm(anchor.direction) == pytest.approx(1.0)
    cp_direction = positions[list(CP_RING)].mean(axis=0) - positions[0]
    assert np.dot(cp.direction, cp_direction / np.linalg.norm(cp_direction)) > 0.99
    # The metal anchor points to its open side, away from the Cp ring.
    assert np.dot(metal.direction, cp.direction) < -0.8

    # No two anchors closer than the merge distance.
    anchor_positions = np.array([a.position for a in analysis.anchors])
    distances = np.linalg.norm(anchor_positions[:, None] - anchor_positions[None], axis=-1)
    assert distances[np.triu_indices(len(anchor_positions), 1)].min() >= 0.3


def test_cpmo_reference_axis(cpmo):
    axis = analyze_molecule(cpmo).reference_axis
    assert axis.method == "metal_ring" and axis.directional
    assert axis.start == (0,) and axis.end == CP_RING
    expected = cpmo.positions[list(CP_RING)].mean(axis=0) - cpmo.positions[0]
    np.testing.assert_allclose(axis.vector(cpmo), expected / np.linalg.norm(expected))

    # Recomputed in a rotated geometry, the axis rotates with the molecule.
    rotated = cpmo.copy()
    rotated.rotate(90, "x", center=(0, 0, 0))
    expected_rotated = rotated.positions[list(CP_RING)].mean(axis=0) - rotated.positions[0]
    np.testing.assert_allclose(axis.vector(rotated), expected_rotated / np.linalg.norm(expected_rotated))


def test_metal_without_ring_uses_heaviest_ligand():
    # Linear H-Ni-CO with a heavier CO ligand.
    atoms = Atoms("HNiCO", positions=[[0, 0, -1.5], [0, 0, 0], [0, 0, 1.8], [0, 0, 2.95]])
    analysis = analyze_molecule(atoms)
    assert [f.name for f in analysis.fragments] == ["H", "CO"]
    axis = analysis.reference_axis
    assert axis.method == "metal_fragment" and axis.start == (1,) and axis.end == (2, 3)
    np.testing.assert_allclose(axis.vector(atoms), [0, 0, 1])


def test_user_reference_axis(cpmo):
    props = MoleculeProps(reference_axis=((1,), (2,)))
    axis = analyze_molecule(cpmo, props).reference_axis
    assert axis.method == "user" and axis.start == (1,) and axis.end == (2,)
    with pytest.raises(ValueError, match="reference_axis"):
        analyze_molecule(cpmo, MoleculeProps(reference_axis=((0,), (99,))))


def test_organic_axes():
    co = molecule("CO")
    axis = analyze_molecule(co).reference_axis
    assert axis.method == "principal" and not axis.directional
    assert abs(np.dot(axis.vector(co), [0, 0, 1])) == pytest.approx(1.0)

    benzene = molecule("C6H6")
    axis = analyze_molecule(benzene).reference_axis
    assert axis.method == "plane_normal"
    assert abs(axis.vector(benzene)[2]) == pytest.approx(1.0)  # benzene lies in the xy plane

    methane = analyze_molecule(molecule("CH4"))
    assert methane.reference_axis.method == "undefined"
    assert methane.reference_axis.vector(molecule("CH4")) is None
    assert any("reference axis undefined" in warning for warning in methane.warnings)


def test_metal_free_anchors():
    water = molecule("H2O")
    analysis = analyze_molecule(water)
    assert [(a.label, a.kind) for a in analysis.anchors] == [
        ("H2O", "fragment"),
        ("O0", "heteroatom"),
        ("H1", "terminal"),
    ]
    # The whole-molecule anchor coincides with the center: it uses the plane
    # normal, i.e. places water lying flat.
    assert abs(analysis.anchors[0].direction[0]) == pytest.approx(1.0)  # water lies in the yz plane

    co = analyze_molecule(molecule("CO"))
    assert {a.label for a in co.anchors} == {"CO", "O0", "C1"}  # both ends can point down


def test_bond_settings_are_used(cpmo):
    tight = analyze_molecule(cpmo, MoleculeProps(bond_scale=1.0))
    assert len(tight.fragments) > 5
    assert any("disconnected" in warning for warning in tight.warnings)

    overridden = analyze_molecule(cpmo, MoleculeProps(bond_overrides=((0, 7, False),)))
    assert overridden.atom_labels[7] == "H"
    assert (0, 7) not in {(i, j) for i, j, _ in overridden.bonds}
    assert all(anchor.kind != "hydride" for anchor in overridden.anchors)

    with pytest.raises(ValueError, match="bond_overrides"):
        analyze_molecule(cpmo, MoleculeProps(bond_overrides=((0, 40, True),)))


def test_graph_hash_ignores_atom_order(cpmo):
    shuffled = cpmo[np.random.default_rng(1).permutation(len(cpmo))]
    assert analyze_molecule(cpmo).graph_hash == analyze_molecule(shuffled).graph_hash


def test_report_and_json(cpmo):
    analysis = analyze_molecule(cpmo)
    text = format_analysis(analysis, cpmo.get_positions())
    for fragment in ("C5H5", "CO_1", "CO_2", "CO_3"):
        assert fragment in text
    assert "Mo0   - C8" in text and "metal_ring" in text
    data = json.loads(json.dumps(analysis.to_dict()))
    assert data["reference_axis"]["end"] == list(CP_RING)
    assert len(data["bonds"]) == len(analysis.bonds)


def test_conformers_interface(cpmo):
    conformers = generate_conformers(cpmo)
    assert len(conformers) == 1 and conformers[0] is not cpmo
    np.testing.assert_allclose(conformers[0].positions, cpmo.positions)
    with pytest.raises(ValueError):
        generate_conformers(cpmo, max_conformers=0)
