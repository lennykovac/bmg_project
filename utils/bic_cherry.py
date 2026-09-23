from itertools import combinations, product

import networkx as nx
from utils.lrt import is_bmg


###BIC-cherry
def _color_groups(graph: nx.DiGraph) -> dict[str, list]:
    """color -> list of vertices of that color."""
    groups: dict = {}
    for node, color in graph.nodes(data="color"):
        groups.setdefault(color, []).append(node)
    return groups


def bicolored_pairs(graph: nx.DiGraph) -> list[tuple]:
    """All unordered pairs ``{x, y}`` with ``sigma(x) != sigma(y)``, as tuples."""
    groups = _color_groups(graph)
    pairs: list[tuple] = []
    for c1, c2 in combinations(groups.keys(), 2):
        pairs.extend(product(groups[c1], groups[c2]))
    return pairs


def bic_cherry(bmg: nx.DiGraph):
    """
    Construct BIC-cherry network (before any extensions)

    Parameters:
    bmg: valid bmg graph, no self loops, sicor-in-hub property

    Returns:
    network: resulting bic-cherry network
    pairs: all node pairs of different color (needed for extensions)
    """

    if not is_bmg(bmg):
        raise ValueError("input into bic-cherry is not bmg")

    network = nx.DiGraph()
    # careful: bmg.nodes(data="color") returns tuples, not a dict!
    network.add_nodes_from((n, {"color": color}) for n, color in bmg.nodes(data="color"))

    pairs = bicolored_pairs(bmg)

    # |L| == 2: the single cherry vertex takes on the role of the root
    if bmg.number_of_nodes() == 2 and pairs:
        x, y = pairs[0]
        network.add_node(f"p:{x}|{y}", color=None)
        network.add_edge(f"p:{x}|{y}", x)
        network.add_edge(f"p:{x}|{y}", y)
        return network, pairs

    # build root and basic parent nodes and basic edges
    network.add_node("R", color=None)  # new root
    for x, y in pairs:
        for a, b in ((x, y), (y, x)):
            network.add_node(f"p:{a}|{b}", color=None)
            network.add_edge("R", f"p:{a}|{b}")
            network.add_edge(f"p:{a}|{b}", x)
            network.add_edge(f"p:{a}|{b}", y)
    return network, pairs

### EXPANSION
def _non_arcs(bmg: nx.DiGraph, pairs: list[tuple]) -> list[tuple]:
    """Ordered bicolored pairs that are *not* arcs of the bmg."""
    arcs = set(bmg.edges())
    return [e for (u, v) in pairs for e in ((u, v), (v, u)) if e not in arcs]


def _choose_z(bmg: nx.DiGraph, x, y, restricted: bool, arcs: set):
    """Pick the partner ``z`` of the extension ``[xy : xz]``.

    ``z`` has the color of ``y`` and differs from ``y``; in the restricted
    variant ``(x, z)`` additionally has to be an arc of ``graph``.
    """
    target = bmg.nodes[y]["color"]
    for n, color in bmg.nodes(data="color"):
        if n == y or color != target:
            continue
        if restricted and (x, n) not in arcs:
            continue
        return n
    if restricted:
        raise ValueError(
            f"no accepting partner for the extension [{x}{y} : {x}z]: "
            f"{x} has no out-neighbour of color {target!r} besides {y} -- "
            "the graph is not color-sink-free"
        )
    raise ValueError(
        f"no partner for the extension [{x}{y} : {x}z]: {y} is the only vertex "
        f"of color {target!r} but ({x}, {y}) is not an arc -- "
        "the graph violates the sicor-in-hub property"
    )


def expansion(
    bmg: nx.DiGraph,
    restricted: bool = False,
) -> nx.DiGraph:
    """BIC-cherry network plus the expansion step.

    Parameters:
    graph: a vertex-colored digraph with the sicor-in-hub property (a BMG or a
        WBMG of some network)
    restricted: choose the expansion partner ``z`` only among the accepting
        arcs ``(x, z) in E(graph)`` (task 1b)

    Returns:
    network: an explaining network for the input graph
    """
    network, pairs = bic_cherry(bmg)
    arcs = set(bmg.edges())

    for x, y in _non_arcs(bmg, pairs):
        z = _choose_z(bmg, x, y, restricted, arcs)

        network.add_node(f"q:{x}|{z}", color=None)
        network.add_edge(f"p:{x}|{y}", f"q:{x}|{z}")
        network.add_edge(f"q:{x}|{z}", x)
        network.add_edge(f"q:{x}|{z}", z)

    return network


def bic_cherry_extension(bmg: nx.DiGraph) -> nx.DiGraph:
    return expansion(bmg, restricted=False)


def restricted_bic_cherry_extension(bmg: nx.DiGraph) -> nx.DiGraph:
    return expansion(bmg, restricted=True)
