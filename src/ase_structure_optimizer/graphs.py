"""Bond graphs and ligand fragments for molecules and periodic structures.

`networkx` is an optional dependency (``pip install -e ".[adsorption]"``) and
is imported only when a function in this module is called.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

from ase import Atoms
from ase.data import covalent_radii

if TYPE_CHECKING:
    import networkx as nx


# d-block elements plus lanthanides and actinides. Fragments are split at
# these atoms; pass `metals=` to `split_fragments()` to use another set.
TRANSITION_METALS = frozenset(
    {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
        "Ho", "Er", "Tm", "Yb", "Lu",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
        "Es", "Fm", "Md", "No", "Lr",
        "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    }
)  # fmt: skip


@dataclass(frozen=True)
class Fragment:
    """A ligand (or whole-molecule) fragment of a bond graph."""

    name: str
    formula: str
    indices: tuple[int, ...]


def _require_networkx():
    try:
        import networkx
    except ImportError as exc:
        raise RuntimeError(
            "Bond-graph tools require networkx. Install with "
            '`pip install -e ".[adsorption]"` or `pip install networkx`.'
        ) from exc
    return networkx


def build_bond_graph(
    atoms: Atoms,
    indices: Sequence[int] | None = None,
    scale: float = 1.2,
    overrides: Iterable[tuple[int, int, bool]] = (),
) -> nx.Graph:
    """Build a bond graph from covalent radii.

    Atoms `i` and `j` are bonded when `d_ij < scale * (r_cov_i + r_cov_j)`,
    with distances taken under the minimum-image convention along periodic
    directions. Nodes are the original atom indices (restricted to `indices`
    when given) and carry a `symbol` attribute; edges carry the shortest
    bonded `distance`.

    `overrides` is a sequence of `(i, j, bonded)` triples that force a bond to
    exist (`True`) or not (`False`), applied after the distance criterion.
    """

    nx = _require_networkx()
    from ase.neighborlist import neighbor_list

    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")

    symbols = atoms.get_chemical_symbols()
    nodes = list(range(len(atoms))) if indices is None else [int(i) for i in indices]
    node_set = set(nodes)
    if len(node_set) != len(nodes):
        raise ValueError("indices contains duplicates")
    for index in nodes:
        if not 0 <= index < len(atoms):
            raise IndexError(f"atom index {index} out of range for {len(atoms)} atoms")

    graph = nx.Graph()
    for index in nodes:
        graph.add_node(index, symbol=symbols[index])

    if len(atoms):
        cutoffs = scale * covalent_radii[atoms.numbers]
        first, second, distances = neighbor_list("ijd", atoms, cutoffs.tolist())
        for i, j, distance in zip(first, second, distances):
            i, j = int(i), int(j)
            if i >= j or i not in node_set or j not in node_set:
                continue
            if graph.has_edge(i, j):
                distance = min(distance, graph.edges[i, j]["distance"])
            graph.add_edge(i, j, distance=float(distance))

    for i, j, bonded in overrides:
        i, j = int(i), int(j)
        if i == j:
            raise ValueError(f"bond override ({i}, {j}) connects an atom to itself")
        for index in (i, j):
            if index not in node_set:
                raise ValueError(f"bond override refers to atom {index}, which is not in the graph")
        if bonded:
            distance = atoms.get_distance(i, j, mic=bool(atoms.pbc.any()))
            graph.add_edge(i, j, distance=float(distance))
        elif graph.has_edge(i, j):
            graph.remove_edge(i, j)

    return graph


def split_fragments(
    graph: nx.Graph,
    symbols: Sequence[str],
    metals: Iterable[str] | None = None,
) -> list[Fragment]:
    """Split a bond graph into ligand fragments.

    If the graph contains metal atoms (`TRANSITION_METALS` by default), the
    metal nodes are removed and each remaining connected component is one
    ligand fragment; the metal atoms themselves belong to no fragment.
    Without metals each connected component (normally the whole molecule) is
    one fragment.

    Fragments are named by their Hill formula (`C5H5`, `CO`, `H`); formulas
    that occur more than once get a 1-based suffix in order of their lowest
    atom index (`CO_1`, `CO_2`, ...). `symbols` is indexed by atom index.
    Fragments are returned sorted by their lowest atom index.
    """

    nx = _require_networkx()

    metal_set = TRANSITION_METALS if metals is None else frozenset(metals)
    ligand_nodes = [node for node in graph.nodes if symbols[node] not in metal_set]
    components = [
        tuple(sorted(component))
        for component in nx.connected_components(graph.subgraph(ligand_nodes))
    ]
    components.sort(key=lambda component: component[0])

    formulas = [_hill_formula([symbols[i] for i in component]) for component in components]
    totals = {formula: formulas.count(formula) for formula in formulas}
    seen: dict[str, int] = {}
    fragments = []
    for component, formula in zip(components, formulas):
        if totals[formula] > 1:
            seen[formula] = seen.get(formula, 0) + 1
            name = f"{formula}_{seen[formula]}"
        else:
            name = formula
        fragments.append(Fragment(name=name, formula=formula, indices=component))
    return fragments


def same_connectivity(g1: nx.Graph, g2: nx.Graph) -> bool:
    """Return True when both graphs have the same atoms and the same bonds.

    The comparison is index-based (atom `i` in one graph is atom `i` in the
    other), which is what is needed to detect bond breaking or formation
    between two geometries of the same system. Use `graph_hash()` for an
    index-independent comparison.
    """

    if set(g1.nodes) != set(g2.nodes):
        return False
    for node in g1.nodes:
        if g1.nodes[node].get("symbol") != g2.nodes[node].get("symbol"):
            return False
    edges1 = {frozenset(edge) for edge in g1.edges}
    edges2 = {frozenset(edge) for edge in g2.edges}
    return edges1 == edges2


def graph_hash(graph: nx.Graph, symbols: Sequence[str]) -> str:
    """Weisfeiler-Lehman hash of a bond graph with elements as node labels.

    Graphs of the same molecule with a different atom ordering give the same
    hash. `symbols` is indexed by atom index.
    """

    nx = _require_networkx()

    labeled = nx.Graph()
    for node in graph.nodes:
        labeled.add_node(node, element=symbols[node])
    labeled.add_edges_from(graph.edges)
    return nx.weisfeiler_lehman_graph_hash(labeled, node_attr="element")


def _hill_formula(symbols: Sequence[str]) -> str:
    counts: dict[str, int] = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    if "C" in counts:
        order = ["C"] + (["H"] if "H" in counts else [])
        order += sorted(symbol for symbol in counts if symbol not in {"C", "H"})
    else:
        order = sorted(counts)
    return "".join(f"{symbol}{counts[symbol] if counts[symbol] > 1 else ''}" for symbol in order)

