from typing import Any, Hashable, Optional
from utils.graph_utils import bmg_from_network, wbmg_from_network

import networkx as nx


def _is_leaf(network: nx.DiGraph, v: Hashable) -> bool:
    return network.out_degree(v) == 0


def _reattach(network: nx.DiGraph, u: Hashable, v: Hashable, target: Hashable) -> None:
    """Remove edge (u, v) and add (target, v), IN PLACE """

    if target == v:
        raise ValueError(f"target ({target!r}) cannot be v itself")

    candidate = network.copy()
    candidate.remove_edge(u, v)
    candidate.add_edge(target, v)
    if not nx.is_directed_acyclic_graph(candidate):
        raise ValueError(
            f"Reattaching ({u!r}, {v!r}) as ({target!r}, {v!r}) would close a cycle"
        )

    network.remove_edge(u, v)
    network.add_edge(target, v)


def pull_up(network: nx.DiGraph, u: Hashable, v: Hashable, target: Optional[Hashable] = None) -> None:
    """
    Reattach edge (u, v) to a source `target` that is an ANCESTOR of u.
    `target=None` uses u's sole parent 
    IN PLACE.
    """

    if not network.has_edge(u, v):
        raise ValueError(f"No edge ({u!r}, {v!r}) in the network")

    if target is None:
        parents = list(network.predecessors(u))
        if len(parents) != 1:
            raise ValueError(
                f"target=None requires {u!r} to have exactly 1 parent "
                f"(has {len(parents)}); pass target explicitly"
            )
        target = parents[0]

    if target == u or not nx.has_path(network, target, u):
        raise ValueError(f"{target!r} is not an ancestor of {u!r}")

    _reattach(network, u, v, target)


def pull_down(network: nx.DiGraph, u: Hashable, v: Hashable, target: Hashable) -> None:
    """
    The mirror of pull_up: reattach edge (u, v) to a source `target`
    that is a DESCENDANT of u 
    IN PLACE.
    """

    if not network.has_edge(u, v):
        raise ValueError(f"No edge ({u!r}, {v!r}) in the network")

    if target == u or not nx.has_path(network, u, target):
        raise ValueError(f"{target!r} is not a descendant of {u!r}")

    _reattach(network, u, v, target)


def pull_up_to_common_ancestor(network: nx.DiGraph, v: Hashable, ancestor: Hashable) -> None:
    """
    Collapses ALL current parents of v onto a single `ancestor`, by
    applying pull_up pairwise -- the concrete move that reduces the
    in-degree of a multi-parent vertex (what makes N a non-tree) down to
    1. `ancestor` must be a common ancestor of every current parent of v
    (or already be one of them). No-op if v already has exactly that
    single parent.
    """
    parents = list(network.predecessors(v))
    for p in parents:
        if p == ancestor:
            continue
        pull_up(network, p, v, target=ancestor)


def find_twin_vertices(network: nx.DiGraph) -> list[list[Hashable]]:
    """
    Detects groups of internal vertices (never leaves) that share
    EXACTLY the same parent set and the same child set.

    Each group is returned as a sorted list.
    """

    groups: dict[tuple[frozenset, frozenset], list[Hashable]] = {}
    for n in network.nodes:
        if _is_leaf(network, n):
            continue
        key = (
            frozenset(network.predecessors(n)),
            frozenset(network.successors(n)),
        )
        groups.setdefault(key, []).append(n)
    return [sorted(g, key=str) for g in groups.values() if len(g) > 1]


def remove_redundant_vertex(network: nx.DiGraph, u: Hashable) -> None:
    """Removes the internal vertex `u` if (and only if) a "twin" exists"""

    if _is_leaf(network, u):
        raise ValueError(f"{u!r} is a leaf; cannot be removed as redundant")

    parents = frozenset(network.predecessors(u))
    children = frozenset(network.successors(u))

    twin = next(
        (
            n
            for n in network.nodes
            if n != u
            and not _is_leaf(network, n)
            and frozenset(network.predecessors(n)) == parents
            and frozenset(network.successors(n)) == children
        ),
        None,
    )
    if twin is None:
        raise ValueError(f"No twin vertex found for {u!r}")

    network.remove_node(u)


def remove_useless_vertex(network: nx.DiGraph, v: Hashable) -> None:
    """Removes a vertex `v` that has exactly 1 parent and 1 child """

    children = list(network.successors(v))
    if len(children) != 1:
        raise ValueError(
            f"{v!r} must have exactly 1 child to be suppressed "
            f"(has {len(children)})"
        )
    parents = list(network.predecessors(v))
    if len(parents) != 1:
        raise ValueError(
            f"{v!r} must have exactly 1 parent to be suppressed "
            f"(has {len(parents)})"
        )

    parent, child = parents[0], children[0]
    network.remove_node(v)
    network.add_edge(parent, child)


def contract_edge(network: nx.DiGraph, u: Hashable, v: Hashable) -> None:
    """
    Contracts the INTERNAL edge (u, v): v disappears, and u directly inherits all of v's children.

    Used to build the LRT starting from ANY tree that explains a BMG
    
    Only meaningful on TREES (v with exactly 1 parent) -- unlike
    pull_up/pull_down, which reattach an edge while preserving both
    vertices, contract_edge merges the two vertices into ONE. 

    Raises ValueError if:
      - (u, v) is not an edge of the network;
      - v is a leaf (contracting an EXTERNAL edge would remove a leaf,
        changing the leaf set);
      - v has more than 1 parent (contraction is only well-defined when
        v has exactly 1 parent, which always holds in trees).
    """
    if not network.has_edge(u, v):
        raise ValueError(f"No edge ({u!r}, {v!r}) in the network")
    if _is_leaf(network, v):
        raise ValueError(
            f"{v!r} is a leaf; only INTERNAL edges can be contracted"
        )
    if network.in_degree(v) != 1:
        raise ValueError(
            f"contract_edge only applies when {v!r} has exactly 1 parent "
            f"(has {network.in_degree(v)}) -- use pull_up/pull_down on networks"
        )

    children = list(network.successors(v))
    network.remove_node(v)
    for child in children:
        network.add_edge(u, child)

def bmg_is_same(network1, network2, mode='wbmg') -> bool:
    compute_fn = bmg_from_network if mode == 'bmg' else wbmg_from_network
    return set(compute_fn(network1).edges()) == set(compute_fn(network2).edges())

def try_edit(network: nx.DiGraph, edit_fn, *args: Any, still_valid=None, **kwargs: Any) -> tuple[nx.DiGraph, bool]:
    """
    Safe editing driver (the integration point between task 2.2c and
    2.2d): applies `edit_fn(copy, *args, **kwargs)` on a COPY of
    `network` (never mutates the original); the edit is rejected if it
    raises ValueError (broken acyclicity, unmet precondition, etc.) OR if
    `still_valid(network, copy)` is provided and returns False.

    Returns (result, applied):
      - applied=True  -> result is the edited copy
      - applied=False -> result is the original `network`, untouched
    """
    candidate = network.copy()
    try:
        edit_fn(candidate, *args, **kwargs)
    except ValueError:
        return network, False

    if still_valid is not None and not still_valid(network, candidate):
        return network, False

    return candidate, True
