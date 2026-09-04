from collections import defaultdict
from typing import Any, Hashable, Optional

import networkx as nx

def _is_leaf(network: nx.DiGraph, v) -> bool:
    return network.out_degree(v) == 0

def _is_root(network: nx.DiGraph, v) -> bool:
    return network.in_degree(v) == 0


def _reattach(network: nx.DiGraph, old_source, new_source, v) -> nx.DiGraph:
    """Move the edge (old_source, v) so it becomes (new_source, v),
    refusing the move if it would create a self-loop or a cycle.

    Both pull_up and pull_down are this same graph operation; what tells
    them apart is only the *direction* they require new_source to lie in
    relative to old_source (checked by the callers below).
    """
    if not network.has_node(new_source):
        raise ValueError(f"new_source {new_source!r} does not exist in the network")

    if not network.has_edge(old_source, v):
        raise ValueError(f"no edge ({old_source!r}, {v!r}) to reattach")

    if new_source == v:
        raise ValueError("cannot reattach an edge to point from a vertex to itself")

    if nx.has_path(network, v, new_source):
        raise ValueError(
            f"reattaching ({old_source!r}, {v!r}) to source {new_source!r} "
            "would create a cycle"
        )


    '''
    candidate = network.copy()
    candidate.remove_edge(old_source, v)
    candidate.add_edge(new_source, v)
    if not nx.is_directed_acyclic_graph(candidate):
        raise ValueError(
            f"reattaching ({old_source!r}, {v!r}) to source {new_source!r} "
            "would create a cycle"
        )
    '''

    network.remove_edge(old_source, v)
    network.add_edge(new_source, v)

    return network


def pull_up(network: nx.DiGraph, u, v, target=None) -> nx.DiGraph:
    """"Pull up" the edge (u, v): reattach it so v's parent becomes
    `target` -- an ancestor of u, i.e. strictly closer to the root --
    instead of u.

    If `target` is omitted, defaults to u's own parent (the smallest
    possible pull, sliding the attachment exactly one level up); this
    requires u to have exactly one parent, otherwise it's ambiguous and
    must be given explicitly.
    """

    #prüfe ob v leaf ist, falls ja würde u zu einem leaf werden, was wir nicht wollen oder?
    if _is_leaf(network, v):
        raise ValueError(f"refusing to pull up edge to leaf {v!r}")

    if target is None:
        candidates = list(network.predecessors(u))
        if len(candidates) != 1:
            raise ValueError(
                f"{u!r} has {len(candidates)} parent(s); pass `target` explicitly"
            )
        target = candidates[0]

    '''''
    if target != u and not nx.has_path(network, target, u):
        raise ValueError(f"{target!r} is not an ancestor of {u!r} -- not a pull *up*")
    '''

    # richtig für target == u
    if target == u or not nx.has_path(network, target, u):
        raise ValueError(f"{target!r} is not an ancestor of {u!r} -- not a pull *up*")


    return _reattach(network, u, target, v)


def pull_down(network: nx.DiGraph, u, v, target) -> nx.DiGraph:
    """"Pull down" the edge (u, v): reattach it so v's parent becomes
    `target` -- a descendant of u, i.e. strictly closer to the leaves --
    instead of u.

    Unlike pull_up there's no sensible default `target` (u may have many
    children reachable below it), so it must always be given explicitly.
    """
    # falls target ein Blatt ist würde es durch die pull_down operation die Eigenschaft verlieren
    if _is_leaf(network, target):
        raise ValueError(f"refusing to pull down to leaf target {target!r}")

    if target == u or not nx.has_path(network, u, target):
        raise ValueError(f"{target!r} is not a descendant of {u!r} -- not a pull *down*")

    return _reattach(network, u, target, v)


def pull_up_to_common_ancestor(network: nx.DiGraph, v, ancestor) -> nx.DiGraph:
    """Collapse ALL of v's current parents into a single edge from
    `ancestor`. This is the concrete move that turns a many-parents vertex
    (the non-tree-like part of a BIC-cherry+extension network) into a
    single-parent one.

    `ancestor` must be an ancestor of every one of v's current parents
    (each individual reattachment is delegated to `pull_up`, which
    enforces this and raises otherwise).
    """
    for u in list(network.predecessors(v)):
        if u != ancestor:
            pull_up(network, u, v, target=ancestor)
    return network

'''
def _is_leaf(network: nx.DiGraph, v) -> bool:
    return network.nodes[v].get("kind") == "leaf"
'''


def find_twin_vertices(network: nx.DiGraph) -> list:
    """Group internal (non-leaf) vertices that share both the same parent
    set and the same child set -- vertices that play the exact same
    structural role twice. Returns a list of groups (each a list of >= 2
    node names); leaves are never included, since they're data, not
    editable structure.
    """
    groups = defaultdict(list)

    for v in network.nodes():
        if _is_leaf(network, v):
            continue

        key = (frozenset(network.predecessors(v)), frozenset(network.successors(v)))
        groups[key].append(v)
    return [sorted(vs, key=str) for vs in groups.values() if len(vs) > 1]


def remove_redundant_vertex(network: nx.DiGraph, v) -> nx.DiGraph:
    """Remove an internal vertex `v` that has a twin (some other vertex
    with the exact same parents and the exact same children). Nothing
    needs to be rewired: the twin already provides every path `v` did.

    Raises ValueError if `v` is a leaf, a root, or has no twin -- use
    `remove_useless_vertex` instead for a (1 parent, 1 child) vertex that
    has no such duplicate to fall back on.
    """
    if _is_leaf(network, v):
        raise ValueError(
            f"refusing to remove leaf {v!r} -- leaves are data, not structure"
        )

    if _is_root(network, v):
        raise ValueError(
            f"refusing to remove root {v!r} -- roots are data, not structure"
        )

    v_parents = set(network.predecessors(v))
    v_children = set(network.successors(v))

    # Suche nur über Geschwister -> haben garantiert einen gemeinsamen Elter
    one_parent = next(iter(v_parents))
    candidates = network.successors(one_parent)

    has_twin = any(
        u != v
        and not _is_leaf(network, u)
        and not _is_root(network, u)
        and set(network.predecessors(u)) == v_parents
        and set(network.successors(u)) == v_children
        for u in candidates
    )

    if not has_twin:
        raise ValueError(
            f"{v!r} has no twin (same parents and children) to fall back on"
        )

    network.remove_node(v)
    return network


def remove_useless_vertex(network: nx.DiGraph, v) -> nx.DiGraph:
    """Remove a vertex `v` with exactly one parent and one child -- it
    represents no branching, so it's suppressed by connecting its parent
    directly to its child instead.
    """
    if _is_leaf(network, v):
        raise ValueError(f"refusing to suppress leaf {v!r}")

    parents = list(network.predecessors(v))
    children = list(network.successors(v))
    if len(parents) != 1 or len(children) != 1:
        raise ValueError(
            f"{v!r} has {len(parents)} parent(s) and {len(children)} child(ren); "
            "can only suppress a vertex with exactly one of each"
        )

    parent, child = parents[0], children[0]
    network.remove_node(v)
    network.add_edge(parent, child)
    return network


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


def try_edit(network: nx.DiGraph, edit_fn, *args, still_valid=None, **kwargs):
    """Apply `edit_fn(candidate, *args, **kwargs)` to a *copy* of
    `network`, keeping the result only if it's still a DAG and -- when
    `still_valid` is given -- `still_valid(network, candidate)` holds
    (e.g. "network and candidate explain the same weak best match graph",
    task 2.2(d)'s checker, once implemented).

    `network` itself is never mutated. Returns (result, applied): on
    success `result` is the edited copy and `applied` is True; on
    rejection `result` is the original `network` and `applied` is False.
    Any ValueError raised by `edit_fn` itself (an invalid pull, an
    un-suppressible vertex, ...) is treated the same as a rejection rather
    than propagated, so callers can try edits speculatively in a loop.
    """
    candidate = network.copy()
    try:
        result = edit_fn(candidate, *args, **kwargs)
        # Falls edit_fn einen neuen Graphen zurückgibt
        if isinstance(result, nx.DiGraph):
            candidate = result
    except ValueError:
        return network, False

    if not nx.is_directed_acyclic_graph(candidate):
        return network, False
    if still_valid is not None and not still_valid(network, candidate):
        return network, False

    return candidate, True
