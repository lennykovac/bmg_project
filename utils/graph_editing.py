from collections import defaultdict
from typing import Any, Hashable, Optional

import networkx as nx

from utils.graph_utils import bmg_from_network, wbmg_from_network


def _is_leaf(network: nx.DiGraph, v: Hashable) -> bool:
    return network.out_degree(v) == 0

def _is_root(network: nx.DiGraph, v) -> bool:
    return network.in_degree(v) == 0


# ---------------------------------------------------------------------------
# (B) structural edits
# ---------------------------------------------------------------------------

def _reattach(network: nx.DiGraph, u: Hashable, v: Hashable, target: Hashable) -> None:
    """Remove (u, v), add (target, v), in place.

    If (target, v) already exists this is simply the deletion of (u, v),
    i.e. the in-degree of v drops by one."""
    if target == v:
        raise ValueError(f"target ({target!r}) cannot be v itself")
    if _is_leaf(network, target):
        raise ValueError(f"target {target!r} is a leaf")
    if nx.has_path(network, v, target):
        raise ValueError(f"reattaching ({u!r}, {v!r}) to {target!r} would close a cycle")
    network.remove_edge(u, v)
    network.add_edge(target, v)


def pull_up(network: nx.DiGraph, u: Hashable, v: Hashable, target: Optional[Hashable] = None) -> None:
    """Reattach (u, v) to an ANCESTOR ``target`` of u (default: u's only parent)."""
    if not network.has_edge(u, v):
        raise ValueError(f"No edge ({u!r}, {v!r}) in the network")
    if target is None:
        parents = list(network.predecessors(u))
        if len(parents) != 1:
            raise ValueError(f"target=None requires {u!r} to have exactly 1 parent (has {len(parents)})")
        target = parents[0]
    if target == u or not nx.has_path(network, target, u):
        raise ValueError(f"{target!r} is not an ancestor of {u!r}")
    _reattach(network, u, v, target)


def pull_down(network: nx.DiGraph, u: Hashable, v: Hashable, target: Hashable) -> None:
    """Reattach (u, v) to a (non-leaf) DESCENDANT ``target`` of u."""
    if not network.has_edge(u, v):
        raise ValueError(f"No edge ({u!r}, {v!r}) in the network")
    if target == u or not nx.has_path(network, u, target):
        raise ValueError(f"{target!r} is not a descendant of {u!r}")
    _reattach(network, u, v, target)


def delete_parent_edge(network: nx.DiGraph, u: Hashable, v: Hashable) -> None:
    """Delete (u, v) if v keeps at least one other parent (hybrid resolution)."""
    if not network.has_edge(u, v):
        raise ValueError(f"No edge ({u!r}, {v!r}) in the network")
    if network.in_degree(v) < 2:
        raise ValueError(f"{v!r} would lose its only parent")
    network.remove_edge(u, v)


def pull_up_to_common_ancestor(network: nx.DiGraph, v: Hashable, ancestor: Hashable) -> None:
    """Collapse ALL parents of v onto ``ancestor`` (a common ancestor of them)."""
    for p in list(network.predecessors(v)):
        if p != ancestor:
            pull_up(network, p, v, target=ancestor)


def contract_edge(network: nx.DiGraph, u: Hashable, v: Hashable) -> None:
    """Contract inner edge (u, v): v disappears, u inherits v's children.
    Requires in-degree(v) == 1."""
    if not network.has_edge(u, v):
        raise ValueError(f"No edge ({u!r}, {v!r}) in the network")
    if _is_leaf(network, v):
        raise ValueError(f"{v!r} is a leaf; only INTERNAL edges can be contracted")
    if network.in_degree(v) != 1:
        raise ValueError(f"contract_edge requires {v!r} to have exactly 1 parent")
    children = list(network.successors(v))
    network.remove_node(v)
    network.add_edges_from((u, c) for c in children)


def transfer_edge(network: nx.DiGraph, u: Hashable, v: Hashable, via: Hashable, target: Hashable) -> None:
    """2-move combination: pull_up (u, v) to a parent ``via`` of u, then
    pull_down (via, v) to another child ``target`` of via  (v changes from
    child of u to child of its "uncle"/sibling-vertex target)."""
    if not network.has_edge(via, u):
        raise ValueError(f"{via!r} is not a parent of {u!r}")
    if not network.has_edge(via, target) or target == u:
        raise ValueError(f"{target!r} is not another child of {via!r}")
    pull_up(network, u, v, target=via)
    pull_down(network, via, v, target=target)


def merge_siblings(network: nx.DiGraph, u: Hashable, w: Hashable) -> None:
    """Combination of transfer_edge moves: all children of w are moved below
    its sibling u (common parent required); w is left without children and
    removed. For |child(w)| = k this is a combination of 2k pull moves
    (+ removing the dead vertex)."""
    if u == w or _is_leaf(network, u) or _is_leaf(network, w):
        raise ValueError("merge_siblings needs two distinct inner vertices")
    common = set(network.predecessors(u)) & set(network.predecessors(w))
    if not common:
        raise ValueError(f"{u!r} and {w!r} are not siblings")
    via = min(common, key=str)
    for c in list(network.successors(w)):
        if network.has_edge(u, c):
            network.remove_edge(w, c)      # u already has c
        else:
            transfer_edge(network, w, c, via, u)
    network.remove_node(w)                 # dead vertex


def group_children(network: nx.DiGraph, u: Hashable, c1: Hashable, c2: Hashable, name: Hashable = None) -> None:
    """Inverse of contraction: insert a new vertex w below u and pull_down
    (u, c1), (u, c2) to w. Requires u to keep >= 2 children (w and another),
    so the result stays phylogenetic."""
    if not (network.has_edge(u, c1) and network.has_edge(u, c2)) or c1 == c2:
        raise ValueError("c1, c2 must be distinct children of u")
    if network.out_degree(u) < 3:
        raise ValueError("grouping all children of u would create a single-child vertex")
    if name is None:
        i = 0
        while f"g{i}" in network:
            i += 1
        name = f"g{i}"
    network.add_node(name, color=None)
    network.add_edge(u, name)
    for c in (c1, c2):
        network.remove_edge(u, c)
        network.add_edge(name, c)


def contract_into_parents(network: nx.DiGraph, v: Hashable) -> None:
    """Generalised contraction for networks: v disappears and EVERY parent of
    v inherits all children of v (for in-degree 1 identical to contract_edge).
    As a pull-sequence: pull_up every child edge (v, c) to every parent."""
    if _is_leaf(network, v):
        raise ValueError(f"{v!r} is a leaf")
    parents, children = list(network.predecessors(v)), list(network.successors(v))
    if not parents:
        raise ValueError(f"{v!r} is the root")
    network.remove_node(v)
    network.add_edges_from((p, c) for p in parents for c in children)


# ---------------------------------------------------------------------------
# (A) BMG-invariant edits
# ---------------------------------------------------------------------------

def find_twin_vertices(network: nx.DiGraph) -> list[list[Hashable]]:
    groups: dict = {}
    for n in network.nodes:
        if _is_leaf(network, n):
            continue
        key = (frozenset(network.predecessors(n)), frozenset(network.successors(n)))
        groups.setdefault(key, []).append(n)
    return [sorted(g, key=str) for g in groups.values() if len(g) > 1]


def remove_twin_vertex(network: nx.DiGraph, u: Hashable) -> None:
    """Remove inner vertex u if a twin (same parents, same children) exists.
    Invariant: the twin has identical ancestors/descendants, so every LCA set
    and every ≺-relation between LCA elements is kept."""
    if _is_leaf(network, u):
        raise ValueError(f"{u!r} is a leaf; cannot be removed as redundant")
    parents, children = frozenset(network.predecessors(u)), frozenset(network.successors(u))
    if not any(
        n != u and not _is_leaf(network, n)
        and frozenset(network.predecessors(n)) == parents
        and frozenset(network.successors(n)) == children
        for n in network.nodes
    ):
        raise ValueError(f"No twin vertex found for {u!r}")
    network.remove_node(u)


def remove_single_child_vertex(network: nx.DiGraph, v: Hashable) -> None:
    """Suppress v with exactly one child (parents are linked to the child)."""
    children = list(network.successors(v))
    if len(children) != 1:
        raise ValueError(f"{v!r} must have exactly 1 child (has {len(children)})")
    parents = list(network.predecessors(v))
    network.remove_node(v)
    network.add_edges_from((p, children[0]) for p in parents)


def remove_one_to_one_vertex(network: nx.DiGraph, v: Hashable) -> None:
    if network.in_degree(v) != 1:
        raise ValueError(f"{v!r} must have exactly 1 parent")
    remove_single_child_vertex(network, v)


def remove_dead_vertex(network: nx.DiGraph, v: Hashable, leaves: set) -> None:
    if v in leaves or not _is_leaf(network, v):
        raise ValueError(f"{v!r} is not a dead inner vertex")
    network.remove_node(v)


def remove_shortcut_edge(network: nx.DiGraph, u: Hashable, v: Hashable) -> None:
    """Remove (u, v) if v is still reachable from u without it."""
    network.remove_edge(u, v)
    if not nx.has_path(network, u, v):
        network.add_edge(u, v)
        raise ValueError(f"({u!r}, {v!r}) is not a shortcut")


def normalize(network: nx.DiGraph, leaves: set) -> dict:
    """Apply all BMG-invariant edits (A) in place until nothing changes.

    Afterwards: no dead vertices, no single-child vertices (incl. the root),
    no twins, no shortcut edges. Returns counts per edit type."""
    stats = {"dead": 0, "single_child": 0, "twins": 0, "shortcuts": 0}
    changed = True
    while changed:
        changed = False
        for v in [v for v in network if v not in leaves and network.out_degree(v) == 0]:
            network.remove_node(v)
            stats["dead"] += 1
            changed = True
        for v in [v for v in network if network.out_degree(v) == 1]:
            if v in network and network.out_degree(v) == 1:
                remove_single_child_vertex(network, v)
                stats["single_child"] += 1
                changed = True
        for group in find_twin_vertices(network):
            for u in group[1:]:
                network.remove_node(u)
                stats["twins"] += 1
                changed = True
        # shortcut (u, v): v is a proper descendant of another child w of u
        desc: dict = {}
        for x in reversed(list(nx.topological_sort(network))):
            d = set()
            for c in network.successors(x):
                d.add(c)
                d |= desc[c]
            desc[x] = d
        shortcuts = [
            (u, v)
            for u in network.nodes
            for v in network.successors(u)
            if any(v in desc[w] for w in network.successors(u) if w != v)
        ]
        if shortcuts:
            network.remove_edges_from(shortcuts)  # removing all at once keeps reachability
            stats["shortcuts"] += len(shortcuts)
            changed = True
    return stats


# ---------------------------------------------------------------------------
# task 2.2(d): guarded editing
# ---------------------------------------------------------------------------

def make_guard(reference: nx.DiGraph, mode: str = "bmg") -> Callable[[nx.DiGraph], bool]:
    """Returns ``guard(candidate) -> bool``: does candidate explain the same
    (weak) BMG as ``reference``? Vertex sets (= leaves) and edge sets are compared."""
    if mode not in ("bmg", "wbmg"):
        raise ValueError(f"unknown mode {mode!r} (use 'bmg' or 'wbmg')")

    compute = bmg_from_network if mode == "bmg" else wbmg_from_network
    G = compute(reference)
    nodes, edges = set(G.nodes), set(G.edges)

    def guard(candidate: nx.DiGraph) -> bool:
        H = compute(candidate)
        return set(H.nodes) == nodes and set(H.edges) == edges

    return guard


def bmg_is_same(network1, network2, mode="bmg") -> bool:
    return make_guard(network1, mode)(network2)


def try_edit(network: nx.DiGraph, edit_fn, *args: Any, still_valid=None, **kwargs: Any) -> tuple[nx.DiGraph, bool]:
    """Apply ``edit_fn`` to a COPY; reject on ValueError or if
    ``still_valid(network, copy)`` is False. Returns (result, applied).
    """
    candidate = network.copy()
    try:
        edit_fn(candidate, *args, **kwargs)
    except ValueError:
        return network, False
    if still_valid is not None and not still_valid(network, candidate):
        return network, False
    return candidate, True
