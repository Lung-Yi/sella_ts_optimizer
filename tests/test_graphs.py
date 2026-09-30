"""Tests for bond graphs and ligand fragments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.io import read

pytest.importorskip("networkx")

from ase_structure_optimizer.graphs import (  # noqa: E402
    Fragment,
    build_bond_graph,
    graph_hash,
    same_connectivity,
    split_fragments,
)

CPMO_XYZ = Path(__file__).parent / "data" / "CpMo_CO3H.xyz"


@pytest.fixture
def cpmo() -> Atoms:
    return read(CPMO_XYZ)


def test_cpmo_bonds(cpmo: Atoms):
    graph = build_bond_graph(cpmo, scale=1.2)
    symbols = cpmo.get_chemical_symbols()
    mo = symbols.index("Mo")
    cp_carbons = set(range(8, 13))
    neighbors = set(graph.neighbors(mo))
    assert cp_carbons <= neighbors  # eta5-Cp: five Mo-C bonds
    assert neighbors == cp_carbons | {1, 3, 5, 7}  # plus 3 CO carbons and the hydride
    assert graph.nodes[mo]["symbol"] == "Mo"
    assert graph.edges[mo, 7]["distance"] == pytest.approx(cpmo.get_distance(mo, 7))


def test_cpmo_fragments(cpmo: Atoms):
    fragments = split_fragments(build_bond_graph(cpmo), cpmo.get_chemical_symbols())
    assert [fragment.name for fragment in fragments] == ["CO_1", "CO_2", "CO_3", "H", "C5H5"]
    assert fragments[0] == Fragment(name="CO_1", formula="CO", indices=(1, 2))
    assert fragments[3].indices == (7,)
    assert fragments[4].formula == "C5H5"
    assert fragments[4].indices == tuple(range(8, 18))
    assert all(0 not in fragment.indices for fragment in fragments)  # the metal is in none


def test_metals_can_be_overridden(cpmo: Atoms):
    graph = build_bond_graph(cpmo)
    fragments = split_fragments(graph, cpmo.get_chemical_symbols(), metals=())
    # With no metal set, the whole complex is a single fragment.
    assert [fragment.name for fragment in fragments] == ["C8H6MoO3"]


def test_molecule_without_metal_is_one_fragment():
    water = molecule("H2O")
    fragments = split_fragments(build_bond_graph(water), water.get_chemical_symbols())
    assert fragments == [Fragment(name="H2O", formula="H2O", indices=(0, 1, 2))]

    methanol = molecule("CH3OH")
    fragments = split_fragments(build_bond_graph(methanol), methanol.get_chemical_symbols())
    assert [fragment.name for fragment in fragments] == ["CH4O"]


def test_disconnected_molecules_are_separate_fragments():
    pair = molecule("CO") + molecule("CO")
    pair.positions[2:] += [0.0, 5.0, 0.0]
    fragments = split_fragments(build_bond_graph(pair), pair.get_chemical_symbols())
    assert [fragment.name for fragment in fragments] == ["CO_1", "CO_2"]


def test_scale_and_overrides(cpmo: Atoms):
    tight = build_bond_graph(cpmo, scale=1.0)
    assert not any(tight.has_edge(0, index) for index in range(8, 13))  # Mo-C(Cp) ~2.3-2.4 Å

    forced = build_bond_graph(cpmo, scale=1.0, overrides=[(0, 8, True), (0, 7, False), (1, 2, False)])
    assert forced.has_edge(0, 8)
    assert forced.edges[0, 8]["distance"] == pytest.approx(cpmo.get_distance(0, 8))
    assert not forced.has_edge(0, 7)
    assert not forced.has_edge(1, 2)

    with pytest.raises(ValueError):
        build_bond_graph(cpmo, overrides=[(0, 0, True)])
    with pytest.raises(ValueError):
        build_bond_graph(cpmo, indices=[1, 2], overrides=[(0, 1, True)])
    with pytest.raises(ValueError):
        build_bond_graph(cpmo, scale=0.0)


def test_indices_subset(cpmo: Atoms):
    graph = build_bond_graph(cpmo, indices=list(range(8, 18)))
    assert set(graph.nodes) == set(range(8, 18))
    assert graph.number_of_edges() == 10  # 5 ring C-C + 5 C-H
    with pytest.raises(IndexError):
        build_bond_graph(cpmo, indices=[99])


def test_periodic_minimum_image_bonds():
    atoms = Atoms("H2", positions=[[0.2, 5.0, 5.0], [9.6, 5.0, 5.0]], cell=[10.0, 10.0, 10.0])
    assert build_bond_graph(atoms).number_of_edges() == 0  # pbc=False
    atoms.pbc = True
    graph = build_bond_graph(atoms)
    assert graph.has_edge(0, 1)
    assert graph.edges[0, 1]["distance"] == pytest.approx(0.6)
    atoms.pbc = [False, True, True]
    assert build_bond_graph(atoms).number_of_edges() == 0


def test_same_connectivity_detects_bond_breaking(cpmo: Atoms):
    reference = build_bond_graph(cpmo)
    assert same_connectivity(reference, build_bond_graph(cpmo.copy()))

    broken = cpmo.copy()
    broken.positions[7] += (broken.positions[7] - broken.positions[0]) * 2.0  # pull hydride away
    assert not same_connectivity(reference, build_bond_graph(broken))
    assert not same_connectivity(reference, build_bond_graph(cpmo, indices=range(1, 18)))


def test_graph_hash_is_permutation_invariant(cpmo: Atoms):
    order = np.random.default_rng(0).permutation(len(cpmo))
    shuffled = cpmo[order]
    assert graph_hash(build_bond_graph(cpmo), cpmo.get_chemical_symbols()) == graph_hash(
        build_bond_graph(shuffled), shuffled.get_chemical_symbols()
    )
    water = molecule("H2O")
    assert graph_hash(build_bond_graph(water), water.get_chemical_symbols()) != graph_hash(
        build_bond_graph(cpmo), cpmo.get_chemical_symbols()
    )
