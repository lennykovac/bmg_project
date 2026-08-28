"""
Task 2.2(d): checker "does N' still explain the same (weak) best match
graph as N?"

Task 2.2(e)/(f): search heuristic that applies pull_up_to_common_ancestor
(to collapse multi-parent vertices) and the cleanup find_twin_vertices/
remove_redundant_vertex + remove_useless_vertex (task 2.2c), each step
guarded by `try_edit(..., still_valid=...)`, trying to bring the
BIC-cherry+expansion network (N, sigma) as close to "tree-like" as
possible without ceasing to explain the same (weak) BMG.
"""

from dataclasses import dataclass, field
from typing import Callable, Hashable

import networkx as nx

from utils.graph_editing import (
    find_twin_vertices,
    pull_up,
    pull_up_to_common_ancestor,
    remove_redundant_vertex,
    remove_useless_vertex,
    try_edit,
)
from utils.graph_utils import bmg_from_network, root_from_network, wbmg_from_network

_BMG_COMPUTE = {
    "bmg": bmg_from_network,      # strict (Def. 2.1, Ebert & Hellmuth)
    "wbmg": wbmg_from_network,    # weak (projektdescription.pdf)
}


def make_still_valid(mode: str = "wbmg") -> Callable[[nx.DiGraph, nx.DiGraph], bool]:
    """Task 2.2(d). mode='wbmg' (default) or 'bmg'.

    Directly compares the edge sets of the BMG/WBMG induced by `before`
    and by `after` -- no isomorphism check needed, since an edit
    (pull_up/pull_down/remove_*) never renames leaves, it only
    restructures internal vertices.

    Returns a function to be used whereelse
    """

    if mode not in _BMG_COMPUTE:
        raise ValueError(f"unknown mode: {mode!r} (use 'bmg' or 'wbmg')")
    compute = _BMG_COMPUTE[mode]

    def still_valid(before: nx.DiGraph, after: nx.DiGraph) -> bool:
        return set(compute(before).edges()) == set(compute(after).edges())

    return still_valid


@dataclass
class SearchReport:
    mode: str = "wbmg"
    rounds_run: int = 0
    resolved_hybrids: list = field(default_factory=list)   # [(vertex, ancestor_used), ...]
    stuck_hybrids: list = field(default_factory=list)      # vertices where NO pull validated
    twins_removed: int = 0
    useless_removed: int = 0
    is_tree: bool = False


def _hybrid_vertices_by_depth(network: nx.DiGraph, root: Hashable) -> list:
    """
    Vertices with in-degree > 1 (what makes N a non-tree), ordered
    from the root towards the leaves -- resolving the higher ones first
    tends to simplify the lower ones for free.
    """
    depth = nx.single_source_shortest_path_length(network, root)
    hybrids = [v for v in network.nodes if network.in_degree(v) > 1]
    hybrids.sort(key=lambda v: depth.get(v, 0))
    return hybrids


def _candidate_ancestors(network: nx.DiGraph, v: Hashable, root: Hashable) -> list:
    """
    Candidates for `ancestor` in pull_up_to_common_ancestor(v, .).

    A candidate `c` is valid iff, for EVERY current parent `p` of v:
    c == p (already one of the parents -- pull_up_to_common_ancestor
    skips this case) OR c is a proper ancestor of p. Important: a
    candidate may be one of v's CURRENT parents itself (e.g.: v has
    parents {R, q}; R is an ancestor of q, so ancestor=R is valid even
    though R has no ancestors of its own

    Sorted from closest to v (minimal pull, prioritizes reusing an
    already-existing parent) to farthest (the root).
    """
    parents = list(network.predecessors(v))
    if not parents:
        return []

    pool = set(parents)
    for p in parents:
        pool |= nx.ancestors(network, p)

    def is_valid(c):
        return all(c == p or nx.has_path(network, c, p) for p in parents)

    valid = [c for c in pool if is_valid(c)]
    depth = nx.single_source_shortest_path_length(network, root)
    return sorted(valid, key=lambda a: -depth.get(a, 0))


def _comparable_parent_pairs(network: nx.DiGraph, v: Hashable):
    """
    All pairs (descendant, ancestor) among v's CURRENT parents such
    that one is a proper ancestor of the other (⪯-comparable).
    """
    parents = list(network.predecessors(v))
    pairs = []
    for p1 in parents:
        for p2 in parents:
            if p1 != p2 and nx.has_path(network, p2, p1):
                pairs.append((p1, p2))  # p1 descends from p2 -> can be pulled up to p2
    return pairs


def _try_resolve_hybrid(network, v, root, still_valid):
    """
    Task 2.2(d)/(f) applied ONE SINGLE MOVE at a time, in two phases:

    Phase 1 (preferred, found empirically): if two of v's CURRENT parents
    are already comparable to each other (one an ancestor of the other),
    perform only THAT targeted pull_up -- reduces v's in-degree by 1
    without touching the rest of the structure. Repeats until no
    comparable pairs remain.

    Phase 2 (fallback, more aggressive): only when NO parent pair is
    comparable, try pull_up_to_common_ancestor(v, ancestor) for a common
    ancestor of ALL parents (in these graphs that is usually the root,
    which typically changes the LCA structure too much and fails
    still_valid -- see the 2.2(f) note in this module).
    """

    progressed = True
    last_result, last_target = None, None
    while progressed:
        progressed = False
        for p_desc, p_anc in _comparable_parent_pairs(network, v):
            result, applied = try_edit(
                network, pull_up, p_desc, v, target=p_anc, still_valid=still_valid
            )
            if applied:
                network = result
                last_result, last_target = result, p_anc
                progressed = True
                break  # v's parents changed -> recompute pairs from scratch

    if network.in_degree(v) <= 1:
        return last_result if last_result is not None else network, last_target

    for candidate_ancestor in _candidate_ancestors(network, v, root):
        result, applied = try_edit(
            network, pull_up_to_common_ancestor, v, candidate_ancestor,
            still_valid=still_valid,
        )
        if applied:
            return result, candidate_ancestor

    return (last_result, last_target) if last_result is not None else (None, None)


def _cleanup_pass(network, still_valid):
    """
    One round of find_twin_vertices/remove_redundant_vertex and
    remove_useless_vertex, each application guarded by try_edit.
    """

    n_twins = 0
    changed = True
    while changed:
        changed = False
        for group in find_twin_vertices(network):
            # keep the first one in the group, try to remove the rest
            for u in group[1:]:
                result, applied = try_edit(
                    network, remove_redundant_vertex, u, still_valid=still_valid
                )
                if applied:
                    network = result
                    n_twins += 1
                    changed = True
                    break
            if changed:
                break

    n_useless = 0
    changed = True
    while changed:
        changed = False
        candidates = [
            v for v in network.nodes
            if network.in_degree(v) == 1 and network.out_degree(v) == 1
        ]
        for v in candidates:
            result, applied = try_edit(
                network, remove_useless_vertex, v, still_valid=still_valid
            )
            if applied:
                network = result
                n_useless += 1
                changed = True
                break

    return network, n_twins, n_useless


def reduce_to_tree(
    network: nx.DiGraph, mode: str = "wbmg", max_rounds: int = 50
) -> tuple[nx.DiGraph, SearchReport]:
    """
    Main heuristic for tasks 2.2(e)/(f).

    mode: 'wbmg' (default) or 'bmg' -- which (weak) BMG the network must
          keep explaining at every step (task 2.2d, via make_still_valid).

    Returns (reduced_network, SearchReport).
    """
    still_valid = make_still_valid(mode)
    root = root_from_network(network)
    report = SearchReport(mode=mode)
    stuck: set = set()

    for round_i in range(max_rounds):
        report.rounds_run = round_i + 1
        any_change = False

        for v in _hybrid_vertices_by_depth(network, root):
            if v in stuck or network.in_degree(v) <= 1:
                continue
            result, used_ancestor = _try_resolve_hybrid(network, v, root, still_valid)
            if result is not None:
                network = result
                report.resolved_hybrids.append((v, used_ancestor))
                any_change = True
            else:
                stuck.add(v)

        network, n_twins, n_useless = _cleanup_pass(network, still_valid)
        report.twins_removed += n_twins
        report.useless_removed += n_useless
        any_change = any_change or n_twins > 0 or n_useless > 0

        if not any_change:
            break

    report.stuck_hybrids = sorted(stuck, key=str)
    report.is_tree = all(network.in_degree(v) <= 1 for v in network.nodes)
    return network, report
