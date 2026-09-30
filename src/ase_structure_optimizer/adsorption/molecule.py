"""Adsorbate analysis: bonds, ligand fragments, anchor points and reference axis.

Everything here is derived from the molecule's bond graph; no molecule- or
surface-specific rules are hard-coded. Conventions:

* The molecule center is the metal atom (the centroid of all metal atoms if
  there are several), otherwise the geometric centroid. The geometric (not
  mass-weighted) centroid is used so that the anchor of a fragment spanning
  the whole molecule coincides with the center.
* Fragment and ring anchors sit at the geometric centroid of their atoms.
* Anchor directions point from the molecule center to the anchor; placing a
  configuration means rotating that direction to -z (toward the surface).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
from ase import Atoms

from ..graphs import (
    TRANSITION_METALS,
    Fragment,
    build_bond_graph,
    find_rings,
    graph_hash,
    split_fragments,
)
from .config import AUTO, MoleculeProps

# Anchors closer than this (Å) are merged; also the distance below which an
# anchor counts as sitting at the molecule center.
ANCHOR_MERGE_DISTANCE = 0.3
RING_SIZES = (3, 8)
# Principal moments all within this relative spread -> no meaningful long axis.
SPHERICAL_TOLERANCE = 0.10
# Minimum length of the summed metal->neighbor unit vectors for an "open side".
OPEN_SIDE_MIN_NORM = 0.1

ANCHOR_KINDS = ("fragment", "ring", "metal", "hydride", "heteroatom", "terminal")


@dataclass(frozen=True)
class Anchor:
    """A point of the molecule that can be pointed at the surface."""

    label: str
    kind: str
    atom_indices: tuple[int, ...]
    position: tuple[float, float, float]
    direction: tuple[float, float, float]
    merged: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReferenceAxis:
    """Molecular reference axis `u`, stored as atom indices.

    `method` is one of ``user``, ``metal_ring``, ``metal_fragment``,
    ``principal``, ``plane_normal`` or ``undefined``. For ``user`` and the
    metal methods the axis runs from the centroid of `start` to the centroid
    of `end`. For ``principal`` it is the lowest-moment (long) principal axis
    of the `start` atoms, oriented toward atom `end[0]`; ``plane_normal``
    (planar symmetric molecules such as benzene, whose long axis is
    degenerate) uses the highest-moment axis instead. Both have no physical
    head/tail (`directional=False`), so tilt angles are folded into 0-90°.
    """

    method: str
    start: tuple[int, ...]
    end: tuple[int, ...]
    directional: bool
    description: str = ""

    def vector(self, molecule: Atoms) -> np.ndarray | None:
        """Unit axis vector in the geometry `molecule` (molecule atoms only)."""

        if self.method == "undefined":
            return None
        positions = molecule.get_positions()
        if self.method in ("principal", "plane_normal"):
            axes, _ = _principal_axes(molecule[list(self.start)])
            if self.method == "plane_normal":
                return axes[:, 2]
            axis = axes[:, 0]
            com = molecule[list(self.start)].get_center_of_mass()
            if np.dot(axis, positions[self.end[0]] - com) < 0:
                axis = -axis
            return axis
        vector = positions[list(self.end)].mean(axis=0) - positions[list(self.start)].mean(axis=0)
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            return None
        return vector / norm


@dataclass(frozen=True)
class MoleculeAnalysis:
    """Bond graph, fragments, anchors and reference axis of a gas-phase molecule."""

    symbols: tuple[str, ...]
    bonds: tuple[tuple[int, int, float], ...]
    fragments: tuple[Fragment, ...]
    metal_indices: tuple[int, ...]
    rings: tuple[tuple[int, ...], ...]
    center: tuple[float, float, float]
    anchors: tuple[Anchor, ...]
    reference_axis: ReferenceAxis
    atom_labels: tuple[str, ...]
    graph_hash: str
    warnings: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form (``molecule_analysis.json``)."""

        return {
            "symbols": list(self.symbols),
            "bonds": [[i, j, round(d, 4)] for i, j, d in self.bonds],
            "fragments": [
                {"name": f.name, "formula": f.formula, "indices": list(f.indices)} for f in self.fragments
            ],
            "metal_indices": list(self.metal_indices),
            "rings": [list(ring) for ring in self.rings],
            "center": [round(x, 6) for x in self.center],
            "anchors": [
                {
                    "label": a.label,
                    "kind": a.kind,
                    "atom_indices": list(a.atom_indices),
                    "position": [round(x, 6) for x in a.position],
                    "direction": [round(x, 6) for x in a.direction],
                    "merged": list(a.merged),
                }
                for a in self.anchors
            ],
            "reference_axis": {
                "method": self.reference_axis.method,
                "start": list(self.reference_axis.start),
                "end": list(self.reference_axis.end),
                "directional": self.reference_axis.directional,
                "description": self.reference_axis.description,
            },
            "atom_labels": list(self.atom_labels),
            "graph_hash": self.graph_hash,
            "warnings": list(self.warnings),
        }


def analyze_molecule(
    atoms: Atoms,
    props: MoleculeProps | None = None,
    metals: Iterable[str] | None = None,
) -> MoleculeAnalysis:
    """Analyze a gas-phase molecule for adsorption sampling.

    Args:
        atoms: The isolated molecule.
        props: Bond perception and reference-axis settings (defaults if None).
        metals: Elements treated as metal centers (default: transition metals).
    """

    props = props or MoleculeProps()
    metal_set = TRANSITION_METALS if metals is None else frozenset(metals)
    n_atoms = len(atoms)
    if n_atoms == 0:
        raise ValueError("the molecule has no atoms")
    for i, j, _ in props.bond_overrides:
        if max(i, j) >= n_atoms:
            raise ValueError(f"bond_overrides refers to atom {max(i, j)}, but the molecule has {n_atoms} atoms")

    atoms = atoms.copy()
    atoms.pbc = False
    symbols = tuple(atoms.get_chemical_symbols())
    positions = atoms.get_positions()
    graph = build_bond_graph(atoms, scale=props.bond_scale, overrides=props.bond_overrides)
    fragments = tuple(split_fragments(graph, symbols, metals=metal_set))
    metal_indices = tuple(i for i, symbol in enumerate(symbols) if symbol in metal_set)
    ligand_nodes = [i for i in range(n_atoms) if i not in metal_indices]
    rings = tuple(find_rings(graph, nodes=ligand_nodes, min_size=RING_SIZES[0], max_size=RING_SIZES[1]))

    if metal_indices:
        center = positions[list(metal_indices)].mean(axis=0)
    else:
        center = positions.mean(axis=0)

    warnings: list[str] = []
    import networkx as nx

    n_components = nx.number_connected_components(graph)
    if n_components > 1:
        warnings.append(
            f"the bond graph has {n_components} disconnected parts; check bond_scale "
            "or add bond_overrides if the molecule should be connected"
        )
    for index in metal_indices:
        if graph.degree(index) == 0:
            warnings.append(f"metal atom {symbols[index]}{index} has no bonds")

    axis = _reference_axis(atoms, graph, fragments, metal_indices, rings, props)
    if axis.method == "undefined":
        warnings.append(f"reference axis undefined ({axis.description}); tilt angles will not be reported")

    anchors = _anchors(atoms, graph, fragments, metal_indices, rings, center, axis)

    return MoleculeAnalysis(
        symbols=symbols,
        bonds=tuple(
            (int(i), int(j), float(data["distance"])) for i, j, data in sorted(graph.edges(data=True))
        ),
        fragments=fragments,
        metal_indices=metal_indices,
        rings=rings,
        center=tuple(float(x) for x in center),
        anchors=anchors,
        reference_axis=axis,
        atom_labels=_atom_labels(symbols, fragments, metal_indices),
        graph_hash=graph_hash(graph, symbols),
        warnings=tuple(warnings),
    )


def molecule_in_box(molecule: Atoms, padding: float) -> Atoms:
    """The molecule centered in a periodic rectangular box.

    Box edges are the molecule's extent along x, y, z plus `padding`, plus
    0, 0.5 and 1.0 Å respectively so the three edges differ (no artificial
    cubic symmetry), matching the VASP gas-phase reference.
    """

    positions = molecule.get_positions()
    extent = positions.max(axis=0) - positions.min(axis=0)
    lengths = extent + padding + np.array([0.0, 0.5, 1.0])
    boxed = Atoms(molecule.get_chemical_symbols(), positions=positions, cell=np.diag(lengths), pbc=True)
    boxed.center()
    return boxed


def generate_conformers(atoms: Atoms, max_conformers: int = 1) -> list[Atoms]:
    """Conformers of the adsorbate to sample.

    The first version treats the molecule as rigid and returns only a copy of
    the input geometry; the signature is the extension point for a future
    conformer search.
    """

    if max_conformers < 1:
        raise ValueError("max_conformers must be >= 1")
    return [atoms.copy()]


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def _atom_labels(
    symbols: Sequence[str], fragments: Sequence[Fragment], metal_indices: Sequence[int]
) -> tuple[str, ...]:
    """Per-atom label: its fragment name, or the element for metal atoms."""

    labels = [""] * len(symbols)
    for fragment in fragments:
        for index in fragment.indices:
            labels[index] = fragment.name
    metal_symbols = [symbols[i] for i in metal_indices]
    seen: dict[str, int] = {}
    for index in metal_indices:
        symbol = symbols[index]
        if metal_symbols.count(symbol) > 1:
            seen[symbol] = seen.get(symbol, 0) + 1
            labels[index] = f"{symbol}_{seen[symbol]}"
        else:
            labels[index] = symbol
    return tuple(labels)


def _atom_label(symbols: Sequence[str], index: int) -> str:
    return f"{symbols[index]}{index}"


# ---------------------------------------------------------------------------
# Reference axis
# ---------------------------------------------------------------------------


def _principal_axes(atoms: Atoms) -> tuple[np.ndarray, np.ndarray]:
    """Principal axes (columns, ascending moments) and moments about the COM."""

    masses = atoms.get_masses()
    relative = atoms.get_positions() - atoms.get_center_of_mass()
    tensor = np.zeros((3, 3))
    for mass, (x, y, z) in zip(masses, relative):
        tensor += mass * np.array(
            [[y * y + z * z, -x * y, -x * z], [-x * y, x * x + z * z, -y * z], [-x * z, -y * z, x * x + y * y]]
        )
    moments, axes = np.linalg.eigh(tensor)
    return axes, moments


def _reference_axis(
    atoms: Atoms,
    graph,
    fragments: Sequence[Fragment],
    metal_indices: Sequence[int],
    rings: Sequence[tuple[int, ...]],
    props: MoleculeProps,
) -> ReferenceAxis:
    n_atoms = len(atoms)
    symbols = atoms.get_chemical_symbols()

    # 1. User-defined axis.
    if props.reference_axis != AUTO:
        start, end = props.reference_axis
        for index in (*start, *end):
            if index >= n_atoms:
                raise ValueError(f"reference_axis refers to atom {index}, but the molecule has {n_atoms} atoms")
        return ReferenceAxis(
            "user", tuple(start), tuple(end), True, "user-defined: centroid(A) -> centroid(B)"
        )

    if metal_indices:
        # 2. Metal -> centroid of the largest ring whose atoms all bind one metal.
        bound_rings = []
        for ring in rings:
            for metal in metal_indices:
                if all(graph.has_edge(metal, index) for index in ring):
                    bound_rings.append((ring, metal))
                    break
        if bound_rings:
            ring, metal = max(bound_rings, key=lambda item: (len(item[0]), -item[0][0]))
            label = _fragment_of(fragments, ring[0])
            return ReferenceAxis(
                "metal_ring",
                (metal,),
                tuple(ring),
                True,
                f"{symbols[metal]}{metal} -> centroid of {len(ring)}-membered ring of {label}",
            )

        # 3. Metal -> centroid of the heaviest ligand fragment.
        if fragments:
            masses = atoms.get_masses()
            heaviest = max(fragments, key=lambda f: (round(masses[list(f.indices)].sum(), 6), -f.indices[0]))
            bonded = [m for m in metal_indices if any(graph.has_edge(m, i) for i in heaviest.indices)]
            metal = bonded[0] if bonded else metal_indices[0]
            return ReferenceAxis(
                "metal_fragment",
                (metal,),
                heaviest.indices,
                True,
                f"{symbols[metal]}{metal} -> centroid of the heaviest ligand {heaviest.name}",
            )
        return ReferenceAxis("undefined", (), (), False, "bare metal atom(s) without ligands")

    # 4./5. Lowest-moment principal axis, unless the molecule is nearly spherical.
    if n_atoms < 2:
        return ReferenceAxis("undefined", (), (), False, "single atom")
    axes, moments = _principal_axes(atoms)
    if moments[-1] <= 0 or (moments[-1] - moments[0]) / moments[-1] < SPHERICAL_TOLERANCE:
        return ReferenceAxis(
            "undefined", (), (), False, "principal moments differ by less than 10 % (nearly spherical)"
        )
    if (moments[1] - moments[0]) / moments[-1] < SPHERICAL_TOLERANCE:
        # Oblate top: the long axis lies anywhere in the molecular plane.
        return ReferenceAxis(
            "plane_normal",
            tuple(range(n_atoms)),
            (),
            False,
            "normal of the molecular plane (highest-moment axis; the long axis is degenerate)",
        )
    heavy = [i for i, s in enumerate(symbols) if s != "H"] or list(range(n_atoms))
    com = atoms.get_center_of_mass()
    distances = np.linalg.norm(atoms.positions[heavy] - com, axis=1)
    sign_atom = heavy[int(np.argmax(np.round(distances, 6)))]
    return ReferenceAxis(
        "principal",
        tuple(range(n_atoms)),
        (sign_atom,),
        False,
        f"long (lowest-moment) principal axis, oriented toward {symbols[sign_atom]}{sign_atom}",
    )


def _fragment_of(fragments: Sequence[Fragment], index: int) -> str:
    for fragment in fragments:
        if index in fragment.indices:
            return fragment.name
    return "?"


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------


def _anchors(
    atoms: Atoms,
    graph,
    fragments: Sequence[Fragment],
    metal_indices: Sequence[int],
    rings: Sequence[tuple[int, ...]],
    center: np.ndarray,
    axis: ReferenceAxis,
) -> tuple[Anchor, ...]:
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    metal_set = set(metal_indices)

    candidates: list[tuple[str, str, tuple[int, ...]]] = []

    for fragment in fragments:
        candidates.append((fragment.name, "fragment", fragment.indices))

    ring_names: dict[str, int] = {}
    ring_labels = []
    for ring in rings:
        base = f"ring{len(ring)}@{_fragment_of(fragments, ring[0])}"
        ring_names[base] = ring_names.get(base, 0) + 1
        ring_labels.append(base)
    ring_seen: dict[str, int] = {}
    for ring, base in zip(rings, ring_labels):
        if ring_names[base] > 1:
            ring_seen[base] = ring_seen.get(base, 0) + 1
            label = f"{base}_{ring_seen[base]}"
        else:
            label = base
        candidates.append((label, "ring", tuple(ring)))

    for index in metal_indices:
        candidates.append((_atom_label(symbols, index), "metal", (index,)))

    for index, symbol in enumerate(symbols):
        if symbol == "H" and any(neighbor in metal_set for neighbor in graph.neighbors(index)):
            candidates.append((_atom_label(symbols, index), "hydride", (index,)))

    for index, symbol in enumerate(symbols):
        if symbol not in ("C", "H") and index not in metal_set:
            candidates.append((_atom_label(symbols, index), "heteroatom", (index,)))

    # Terminal atoms: every non-H atom with one bond; for terminal H only the
    # one farthest from the center in each fragment (Cp or CH3 hydrogens would
    # otherwise give many nearly equivalent anchors).
    terminal_h: dict[str, int] = {}
    for index, symbol in enumerate(symbols):
        if graph.degree(index) != 1 or index in metal_set:
            continue
        if symbol != "H":
            candidates.append((_atom_label(symbols, index), "terminal", (index,)))
            continue
        owner = _fragment_of(fragments, index)
        best = terminal_h.get(owner)
        distance = np.linalg.norm(positions[index] - center)
        if best is None or distance > np.linalg.norm(positions[best] - center) + 1e-6:
            terminal_h[owner] = index
    for index in sorted(terminal_h.values()):
        candidates.append((_atom_label(symbols, index), "terminal", (index,)))

    # Merge coincident anchors, keeping the first in ANCHOR_KINDS priority order.
    candidates.sort(key=lambda item: ANCHOR_KINDS.index(item[1]))
    kept: list[dict[str, Any]] = []
    for label, kind, indices in candidates:
        position = positions[list(indices)].mean(axis=0)
        for existing in kept:
            if np.linalg.norm(existing["position"] - position) < ANCHOR_MERGE_DISTANCE:
                if label != existing["label"] and label not in existing["merged"]:
                    existing["merged"].append(label)
                break
        else:
            kept.append({"label": label, "kind": kind, "indices": indices, "position": position, "merged": []})

    fallback = _center_direction(atoms, graph, metal_indices, center, axis)
    anchors = []
    for item in kept:
        vector = item["position"] - center
        norm = np.linalg.norm(vector)
        direction = vector / norm if norm >= ANCHOR_MERGE_DISTANCE else fallback
        anchors.append(
            Anchor(
                label=item["label"],
                kind=item["kind"],
                atom_indices=tuple(int(i) for i in item["indices"]),
                position=tuple(float(x) for x in item["position"]),
                direction=tuple(float(x) for x in direction),
                merged=tuple(item["merged"]),
            )
        )
    return tuple(anchors)


def _center_direction(
    atoms: Atoms,
    graph,
    metal_indices: Sequence[int],
    center: np.ndarray,
    axis: ReferenceAxis,
) -> np.ndarray:
    """Direction for anchors that coincide with the molecule center.

    For a metal center this is its least-coordinated ("open") side, the
    negative sum of unit vectors to its bonded neighbors, so the metal faces
    the surface where it is not shielded by ligands. Otherwise (or if that
    sum vanishes) it is the highest-moment principal axis, the normal of a
    planar molecule, so the molecule is placed lying flat.
    """

    positions = atoms.get_positions()
    if metal_indices:
        total = np.zeros(3)
        for metal in metal_indices:
            for neighbor in graph.neighbors(metal):
                if neighbor in metal_indices:
                    continue
                vector = positions[neighbor] - positions[metal]
                total += vector / np.linalg.norm(vector)
        if np.linalg.norm(total) >= OPEN_SIDE_MIN_NORM:
            return -total / np.linalg.norm(total)
    if len(atoms) >= 2:
        axes, _ = _principal_axes(atoms)
        return axes[:, 2]
    return np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Text report (ase-adsorb check-molecule)
# ---------------------------------------------------------------------------


def format_analysis(analysis: MoleculeAnalysis, positions: np.ndarray | None = None) -> str:
    """Human-readable report of a `MoleculeAnalysis`."""

    symbols = analysis.symbols
    lines = [f"Atoms ({len(symbols)}):"]
    for index, symbol in enumerate(symbols):
        coords = "" if positions is None else "  " + " ".join(f"{x:10.4f}" for x in positions[index])
        lines.append(f"  {index:3d}  {symbol:<2s}{coords}  [{analysis.atom_labels[index]}]")

    lines.append(f"\nBonds ({len(analysis.bonds)}):")
    for i, j, distance in analysis.bonds:
        lines.append(f"  {symbols[i]}{i:<3d} - {symbols[j]}{j:<3d}  {distance:6.3f} Å")

    lines.append(f"\nFragments ({len(analysis.fragments)}):")
    if analysis.metal_indices:
        metals = ", ".join(f"{symbols[i]}{i}" for i in analysis.metal_indices)
        lines.append(f"  metal center(s), not part of any fragment: {metals}")
    for fragment in analysis.fragments:
        lines.append(f"  {fragment.name:<10s} atoms {list(fragment.indices)}")
    if analysis.rings:
        lines.append("  rings: " + "; ".join(str(list(ring)) for ring in analysis.rings))

    lines.append(f"\nAnchors ({len(analysis.anchors)}):  direction = anchor - molecule center")
    for anchor in analysis.anchors:
        direction = " ".join(f"{x:6.3f}" for x in anchor.direction)
        merged = f"  (also: {', '.join(anchor.merged)})" if anchor.merged else ""
        lines.append(
            f"  {anchor.label:<12s} {anchor.kind:<10s} atoms {list(anchor.atom_indices)}  dir [{direction}]{merged}"
        )

    axis = analysis.reference_axis
    lines.append("\nReference axis u:")
    lines.append(f"  method: {axis.method} ({axis.description})")
    if axis.method != "undefined":
        lines.append(f"  from atoms {list(axis.start)} to atoms {list(axis.end)}")
        lines.append(
            "  tilt angle range: 0-180° (0° = u points away from the surface)"
            if axis.directional
            else "  tilt angle range: 0-90° (axis without head/tail)"
        )

    if analysis.warnings:
        lines.append("\nWarnings:")
        lines.extend(f"  - {warning}" for warning in analysis.warnings)
    return "\n".join(lines)
