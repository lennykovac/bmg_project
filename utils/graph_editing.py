"""

Normalisation
-------------
After every move :func:`normalize` restores the defining properties of a
phylogenetic network: it removes inner vertices without children, suppresses
inner vertices with a single child, and drops whatever became unreachable from
the root.  
"""

from __future__ import annotations

import hashlib
import itertools
import random as _random
from typing import Hashable, Iterator, NamedTuple, Optional, Callable, Any

import networkx as nx

from utils.graph_utils import bmg_from_network, wbmg_from_network, leaves_from_network

__all__ = [
    "Move",
    "normalize",
    "pull_up",
    "pull_down",
    "merge_twins",
    "delete_parent_edge",
    "pull_up_to_common_ancestor",
    "contract_edge",
    "transfer_edge",
    "group_children",
    "contract_into_parents",

    "apply_move",
    "enumerate_moves",
    "legal_successors",
    "NoLegalMove",
    "single_moves",
    "combination",
    "twin_pairs",
    "reduce_twins",
    "cluster_multiset",
    "target_distance",
    "reticulation_number",
    "tree_likeness",
    "canonical_key",
    "is_network",
    "DEFAULT_MOVES",
    "EXTENDED_MOVES",
]

def _is_leaf(network: nx.DiGraph, v: Hashable) -> bool:
    return network.out_degree(v) == 0

def _is_root(network: nx.DiGraph, v) -> bool:
    return network.in_degree(v) == 0

class Move(NamedTuple):
    """An edit operation. ``args`` is the tuple handed to the move function."""

    kind: str
    args: tuple

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.kind}{self.args}"


DEFAULT_MOVES = ("pull_up", "pull_down", "merge_twins")
EXTENDED_MOVES = ("pull_up", "pull_down", "merge_twins", "delete_parent_edge")
ARTHUR_MOVES = ("pull_up", "pull_down", "merge_twins", "delete_parent_edge", 
                "pull_up_to_common_ancestor", "contract_edge", "transfer_edge",
                "group_children", "contract_into_parents")

# ---------------------------------------------------------------------------
# the moves
# ---------------------------------------------------------------------------

def pull_up(N: nx.DiGraph, u, v, target) -> nx.DiGraph:
    """Replace the arc ``(u, v)`` by ``(p, v)`` for a parent ``p`` of ``u``."""
    if not N.has_edge(u, v):
        raise ValueError(f"{(u, v)} is not an arc of N")
    if not N.has_edge(target, u):
        raise ValueError(f"{target} is not a parent of {u}")
    if target == v:
        raise ValueError("cannot pull an arc up into its own head")
    M = N.copy()
    M.remove_edge(u, v)
    M.add_edge(target, v)
    return M


def pull_down(N: nx.DiGraph, u, v, target) -> nx.DiGraph:
    """Replace the arc ``(u, v)`` by ``(w, v)`` for a sibling ``w`` of ``v``."""
    if not N.has_edge(u, v):
        raise ValueError(f"{(u, v)} is not an arc of N")
    if not N.has_edge(u, target):
        raise ValueError(f"{target} is not a child of {u}")
    if target == v:
        raise ValueError("cannot pull an arc down into itself")
    if N.out_degree(target) == 0:
        raise ValueError(f"{target} is a leaf and must stay one")
    if target in nx.descendants(N, v) or target == v:
        raise ValueError(f"pulling {(u, v)} down to {target} would close a cycle")
    M = N.copy()
    M.remove_edge(u, v)
    M.add_edge(target, v)
    return M


def merge_twins(N: nx.DiGraph, a, b) -> nx.DiGraph:
    """Delete the twin ``b`` of ``a`` (same parents, same children)."""
    if a == b:
        raise ValueError("a and b must be different vertices")
    if N.out_degree(a) == 0 or N.out_degree(b) == 0:
        raise ValueError("only inner vertices can be merged")
    if set(N.predecessors(a)) != set(N.predecessors(b)):
        raise ValueError(f"{a} and {b} do not have the same parents")
    if set(N.successors(a)) != set(N.successors(b)):
        raise ValueError(f"{a} and {b} do not have the same children")
    M = N.copy()
    M.remove_node(b)
    return M


def delete_parent_edge(N: nx.DiGraph, u, v) -> nx.DiGraph:
    """Delete the arc ``(u, v)``; ``v`` has to keep a parent."""
    if not N.has_edge(u, v):
        raise ValueError(f"{(u, v)} is not an arc of N")
    if N.in_degree(v) < 2:
        raise ValueError(f"{v} would lose its last parent")
    M = N.copy()
    M.remove_edge(u, v)
    return M


def pull_up_to_common_ancestor(network: nx.DiGraph, v: Hashable, ancestor: Hashable) -> None:
    """Collapse ALL parents of v onto ``ancestor`` (a common ancestor of them)."""
    for p in list(network.predecessors(v)):
        if p != ancestor:
            pull_up(network, p, v, target==ancestor)


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


_DISPATCH = {
    "pull_up": pull_up,
    "pull_down": pull_down,
    "merge_twins": merge_twins,
    "delete_parent_edge": delete_parent_edge,
    "pull_up_to_common_ancestor": pull_up_to_common_ancestor,
    "contract_edge": contract_edge,
    "transfer_edge": transfer_edge,
    "group_children": group_children,
    "contract_into_parents": contract_into_parents
}


def apply_move(N: nx.DiGraph, move: Move, leaves: set | None = None) -> nx.DiGraph:
    """Apply ``move`` to a copy of ``N`` and normalise the result."""
    if leaves is None:
        leaves = {v for v in N if N.out_degree(v) == 0}
    M = _DISPATCH[move.kind](N, *move.args)
    return normalize(M, leaves)



# ---------------------------------------------------------------------------
# invariants
# ---------------------------------------------------------------------------

def is_network(N: nx.DiGraph, leaves: set | None = None) -> bool:
    """DAG, unique root, no inner vertex with a single child, leaves intact."""
    if N.number_of_nodes() == 0 or not nx.is_directed_acyclic_graph(N):
        return False
    roots = [v for v in N if N.in_degree(v) == 0]
    if len(roots) != 1:
        return False
    if set(nx.descendants(N, roots[0])) | {roots[0]} != set(N.nodes):
        return False
    if any(N.out_degree(v) == 1 for v in N):
        return False
    if leaves is not None and set(leaves_from_network(N)) != set(leaves):
        return False
    return True


def reticulation_number(N: nx.DiGraph) -> int:
    """``sum_v max(0, indeg(v) - 1)``: 0 exactly for trees."""
    return sum(max(0, N.in_degree(v) - 1) for v in N)


def tree_likeness(N: nx.DiGraph) -> tuple[int, int, int]:
    """Cost vector, lexicographically minimised by the search.

    ``(reticulation number, #vertices, #arcs)`` -- the first entry is what
    separates networks from trees, the other two break ties towards the least
    resolved tree, which is the *smallest* tree explaining the graph.
    """
    return (reticulation_number(N), N.number_of_nodes(), N.number_of_edges())


def canonical_key(N: nx.DiGraph) -> tuple:
    """Leaf-labelled canonical signature, used to deduplicate search states.

    Every vertex gets a signature built bottom-up from the sorted signatures of
    its children (leaves use their own name and color). The key is the
    *multiset* of vertex signatures together with the multiset of arc
    signatures; the vertex multiset is what tells a shared vertex apart from
    two duplicated copies of it.
    """
    order = list(nx.topological_sort(N))
    sig: dict[Hashable, str] = {}
    for v in reversed(order):
        children = sorted(sig[c] for c in N.successors(v))
        if not children:
            sig[v] = f"L({N.nodes[v].get('color')!r}:{v!r})"
        else:
            # hashed so that the signatures stay short in deep networks
            raw = "(" + ",".join(children) + ")"
            sig[v] = hashlib.blake2b(raw.encode(), digest_size=12).hexdigest()

    vertices = tuple(sorted(_counter(sig.values()).items()))
    arcs = tuple(sorted(_counter((sig[u], sig[v]) for u, v in N.edges()).items()))
    return (vertices, arcs)


def _counter(iterable) -> dict:
    out: dict = {}
    for item in iterable:
        out[item] = out.get(item, 0) + 1
    return out


# ---------------------------------------------------------------------------
# normalisation
# ---------------------------------------------------------------------------

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

def normalize(N: nx.DiGraph, leaves: set | None = None) -> nx.DiGraph:
    """Turn the result of a move back into a phylogenetic network (copy).

    Raises ``ValueError`` if a leaf was lost, which would mean the move was
    illegal in the first place.
    """

    M = N.copy()
    if leaves is None:
        leaves = {v for v in M if M.out_degree(v) == 0}
    leaves = set(leaves)
 
    # Repeatedly scan and remove reducible vertices until none remain
    changed = True
    while changed:
        changed = False
        for v in list(M.nodes):
            if v not in M or v in leaves:
                continue
            out = M.out_degree(v)
            if out == 0:
                remove_dead_vertex(M, v, leaves)
                changed = True
            elif out == 1:
                remove_single_child_vertex(M, v)
                changed = True
 
    missing = leaves - set(M.nodes)
    if missing:
        raise ValueError(f"normalisation lost the leaves {sorted(map(str, missing))}")

    orphans = [v for v in leaves if M.in_degree(v) == 0 and M.number_of_nodes() > 1]
    if orphans:
        raise ValueError(f"leaves {sorted(map(str, orphans))} lost all parents")
    return M

# ---------------------------------------------------------------------------
# task 2.2(d): guarded editing
# ---------------------------------------------------------------------------

def make_guard(reference: nx.DiGraph, mode: str = "bmg") -> Callable[[nx.DiGraph], bool]:
    """Returns ``guard(candidate) -> bool``: does candidate explain the same
    (weak) BMG as ``reference``? Vertex sets (= leaves) and edge sets are compared."""
    if mode not in ("bmg", "wbmg"):
        raise ValueError(f"unknown mode {mode!r} (use 'bmg' or 'wbmg')")

    compute = bmg_from_network if mode == "bmg" else wbmg_from_network
    G = bmg_from_network(reference, weak=True)
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

# ---------------------------------------------------------------------------
# enumeration
# ---------------------------------------------------------------------------

def twin_pairs(N: nx.DiGraph) -> list[tuple]:
    """Unordered pairs of inner vertices with the same parents and children."""
    buckets: dict = {}
    for v in N:
        if N.out_degree(v) == 0:
            continue
        key = (frozenset(N.predecessors(v)), frozenset(N.successors(v)))
        buckets.setdefault(key, []).append(v)
    out = []
    for group in buckets.values():
        out.extend(itertools.combinations(group, 2))
    return out


def reduce_twins(N: nx.DiGraph) -> nx.DiGraph:
    """Exhaustively merge twin vertices (a provably BMG/WBMG-preserving reduction).

    Because twins are order-theoretically indistinguishable, no best match
    test can tell them apart, so this needs no verification against the
    target graph and is applied before any search starts. On the BIC-cherry
    expansion it removes the two copies ``p_xy``/``p_yx`` of every bicolored
    pair that the construction introduced.
    """
    leaves = {v for v in N if N.out_degree(v) == 0}
    M = N.copy()
    while True:
        pairs = twin_pairs(M)
        if not pairs:
            break
        removed = set()
        for a, b in pairs:
            if a in removed or b in removed:
                continue
            M.remove_node(b)
            removed.add(b)
    return normalize(M, leaves)


def cluster_multiset(N: nx.DiGraph) -> dict:
    """``frozenset of leaf descendants -> number of inner vertices with it``."""
    from utils.graph_utils import clusters

    cl = clusters(N)
    out: dict = {}
    for v in N:
        if N.out_degree(v) == 0:
            continue
        out[cl[v]] = out.get(cl[v], 0) + 1
    return out


def target_distance(N: nx.DiGraph, target: nx.DiGraph) -> tuple:
    """Guided cost: how far the cluster system of ``N`` is from ``target``.

    A phylogenetic tree is determined by its cluster system, so "``N`` is
    ``T*``" is the same as "``N`` is a tree and the multiset of clusters of
    its inner vertices is the cluster system of ``T*``". The cost adds up the
    four ways in which ``N`` can still differ from that:

    ``shape``   ``sum_v min_c |C(v) symdiff c|`` over the inner vertices ``v``
                of ``N`` and the clusters ``c`` of ``T*``. This is the term
                that makes the landscape *smooth*: deleting one arc out of a
                cluster that is one leaf too large lowers it by one, even
                though the cluster is still wrong. A pure "is the cluster
                right" counter is flat almost everywhere and the descent
                stalls immediately.
    ``missing`` clusters of ``T*`` that no vertex of ``N`` realises,
    ``excess``  inner vertices beyond ``|V(T*)|`` (duplicated clusters),
    ``retic``   the reticulation number.

    The sum is ``0`` exactly at ``T*``; ``|V|`` and ``|E|`` only break ties.
    """
    from utils.graph_utils import clusters

    cl_N = clusters(N)
    cl_T = clusters(target)
    want = {cl_T[v] for v in target if target.out_degree(v) > 0}

    inner = [v for v in N if N.out_degree(v) > 0]
    shape = 0
    have = set()
    for v in inner:
        c = cl_N[v]
        have.add(c)
        shape += min(len(c ^ w) for w in want)

    missing = len(want - have)
    excess = max(0, len(inner) - len(want))
    retic = reticulation_number(N)
    return (
        shape + missing + excess + retic,
        retic,
        N.number_of_nodes(),
        N.number_of_edges(),
    )


def enumerate_moves(
    N: nx.DiGraph,
    kinds: tuple = DEFAULT_MOVES,
) -> list[Move]:
    """All legal moves of the requested kinds."""
    leaves = {v for v in N if N.out_degree(v) == 0}
    moves: list[Move] = []
    desc = {v: nx.descendants(N, v) for v in N}

    if "pull_up" in kinds:
        for u, v in N.edges():
            for p in N.predecessors(u):
                if p == v:
                    continue
                moves.append(Move("pull_up", (u, v, p)))

    if "pull_down" in kinds:
        for u, v in N.edges():
            for w in N.successors(u):
                if w == v or w in leaves:
                    continue
                if w in desc[v]:
                    continue
                moves.append(Move("pull_down", (u, v, w)))

    if "merge_twins" in kinds:
        for a, b in twin_pairs(N):
            moves.append(Move("merge_twins", (a, b)))

    if "delete_parent_edge" in kinds:
        for u, v in N.edges():
            if N.in_degree(v) >= 2 and N.out_degree(u) >= 2:
                moves.append(Move("delete_parent_edge", (u, v)))

    moves.sort(key=lambda m: (m.kind, tuple(map(str, m.args))))
    return moves


def legal_successors(
    N: nx.DiGraph,
    kinds: tuple = DEFAULT_MOVES,
    leaves: set | None = None,
) -> Iterator[tuple[Move, nx.DiGraph]]:
    """Yield ``(move, normalised network)`` for every applicable move."""
    if leaves is None:
        leaves = {v for v in N if N.out_degree(v) == 0}
    for move in enumerate_moves(N, kinds):
        try:
            yield move, apply_move(N, move, leaves)
        except (ValueError, nx.NetworkXError):
            continue


# ---------------------------------------------------------------------------
# random editing (task 2d)
# ---------------------------------------------------------------------------

class NoLegalMove(RuntimeError):
    """No move satisfies the constraints.

    Raised by :func:`single_moves` when the network admits no legal move at
    all, or -- with ``preserve_wbmg=True`` -- none that leaves the weak best
    match graph unchanged. The second case is not an edge case: for a complete
    bicolored BMG the BIC-cherry network has *no* preserving move whatsoever
    (see ``TASK2.md``, task 2f), and silently returning the network unchanged
    would hide exactly the finding that matters.
    """


def _rng(rng) -> _random.Random:
    if isinstance(rng, _random.Random):
        return rng
    if rng is None:
        return _random.Random()
    return _random.Random(rng)


def _arcs(G: nx.DiGraph) -> set:
    return set(G.edges())


def single_moves(
    network: nx.DiGraph,
    preserve_wbmg: bool = False,
    kinds: tuple = DEFAULT_MOVES,
    record: list | None = None,
    rng=None,
) -> nx.DiGraph:
    """Apply **one** random edit move and return the resulting network.

    Parameters
    ----------
    network:
        the network to edit; it is not modified (a new graph is returned).
    preserve_wbmg:
        when True, draw only among the moves that leave the weak best match
        graph of ``network`` unchanged. Default False -- any legal move.
    kinds:
        which move kinds to draw from (:data:`DEFAULT_MOVES` is pull up, pull
        down and twin merging; :data:`EXTENDED_MOVES` adds arc deletion).
    record:
        optional list; the move that was applied is appended to it, so the
        caller can report the edit path without changing the return type.
    rng:
        a ``random.Random``, an int seed, or None.

    Raises
    ------
    NoLegalMove
        if no move of the requested kinds applies, or none of them preserves
        the weak best match graph when ``preserve_wbmg`` is set.
    """
    rnd = _rng(rng)
    leaves = {v for v in network if network.out_degree(v) == 0}
    reference = _arcs(bmg_from_network(network, weak=True)) if preserve_wbmg else None

    moves = enumerate_moves(network, kinds)
    rnd.shuffle(moves)

    tried = 0
    for move in moves:
        try:
            candidate = apply_move(network, move, leaves)
        except (ValueError, nx.NetworkXError):
            continue
        tried += 1
        if preserve_wbmg and _arcs(bmg_from_network(candidate, weak=True)) != reference:
            continue
        if record is not None:
            record.append(move)
        return candidate

    if tried == 0:
        raise NoLegalMove(
            f"no applicable move of kinds {kinds} on a network with "
            f"{network.number_of_nodes()} vertices and {network.number_of_edges()} arcs"
        )
    raise NoLegalMove(
        f"none of the {tried} legal moves of kinds {kinds} preserves the WBMG "
        f"(network: {network.number_of_nodes()} vertices, "
        f"{network.number_of_edges()} arcs)"
    )


def combination(
    network: nx.DiGraph,
    k: int = 3,
    preserve_wbmg: bool = False,
    kinds: tuple = DEFAULT_MOVES,
    record: list | None = None,
    rng=None,
) -> nx.DiGraph:
    """Apply ``k`` random edit moves one after another.

    With ``preserve_wbmg=True`` the weak best match graph is checked **after
    every step**, not only at the end: each move is drawn among those that
    preserve the WBMG of the network it is applied to, so every intermediate
    network explains the same weak graph as the input. That is the strict
    reading of task 2d, and it is strictly stronger than "the endpoints agree"
    -- there are pairs of moves that individually destroy the graph and
    jointly restore it, which this rule rules out by construction.

    Parameters as in :func:`single_moves`, plus ``k``, the number of moves.

    Raises
    ------
    NoLegalMove
        propagated from the first step that has no admissible move. Any moves
        already appended to ``record`` did happen; the returned network is
        lost, so catch this if a partial path is acceptable.
    """
    if k < 1:
        raise ValueError(f"k has to be at least 1, got {k}")

    rnd = _rng(rng)
    current = network
    for _ in range(k):
        current = single_moves(
            current,
            preserve_wbmg=preserve_wbmg,
            kinds=kinds,
            record=record,
            rng=rnd,
        )
    return current
