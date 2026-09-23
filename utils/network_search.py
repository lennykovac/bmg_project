"""
Task 2.2(d)-(f): search for edit paths  BIC-cherry+expansion (N, sigma)  ->  T*.

State space
-----------
States are *normalized* networks (``graph_editing.normalize``: no dead,
single-child or twin vertices, no shortcut edges -- all BMG-invariant).

One search step = one structural edit (pull_up / pull_down / delete_parent_edge)
followed by normalization, accepted only if the guard (same (weak) BMG as the
start network, task 2.2(d)) holds.

Target test
-----------
N equals T*  <=>  N is a phylogenetic tree with the same cluster set as T*
(``graph_utils.same_phylogeny``). Because normalized networks have no twins
and no single-child vertices, cluster distance 0 plus in-degree <= 1 is
exactly that.

Search
------
``edit_search``: beam search (beam width 1 = steepest descent) with a
visited-set on canonical states, a step budget, and full edit-path logging
so failures can be inspected (task 2.2(f)).

"""

from dataclasses import dataclass, field
from typing import Callable, Hashable

import networkx as nx

from utils.graph_editing import (
    contract_into_parents,
    delete_parent_edge,
    group_children,
    make_guard,
    merge_siblings,
    normalize,
    pull_down,
    pull_up,
    pull_up_to_common_ancestor,
)
from utils.graph_utils import clusters, same_phylogeny


# ---------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------

def hybrid_excess(N: nx.DiGraph) -> int:
    return sum(max(0, d - 1) for _, d in N.in_degree())


def inner_clusters(N: nx.DiGraph) -> set:
    return {c for v, c in clusters(N).items() if N.out_degree(v) > 0}


def make_score_guided(lrt: nx.DiGraph) -> Callable[[nx.DiGraph], int]:
    target = inner_clusters(lrt)

    # the lower the better
    def score(N):
        return len(inner_clusters(N) ^ target) + hybrid_excess(N)

    return score


def score_agnostic(N: nx.DiGraph) -> int:
    return N.number_of_edges() + hybrid_excess(N)


# ---------------------------------------------------------------------------
# moves
# ---------------------------------------------------------------------------

def candidate_moves(N: nx.DiGraph, compound: bool = True):
    """Yield (name, fn, args) for structural edits of N.

    single moves: delete_parent_edge, pull_up / pull_down by one level.
    compound moves (small fixed combinations of single moves, task 2.2(d)):
      pull_up_to_common_ancestor(v, c) for hybrids v and c ∈ LCA(parents(v)),
      contract_into_parents(v), merge_siblings(u, w) for siblings whose
      clusters overlap (only those can be merged without creating a new
      cluster that is disjoint-union-like; heuristic restriction)."""
    if compound:
        cl = clusters(N)
        # collapse all parents of a hybrid onto a minimal common ancestor of them
        for v in N.nodes:
            parents = list(N.predecessors(v))
            if len(parents) < 2:
                continue
            common = set.intersection(*(nx.ancestors(N, p) | {p} for p in parents))
            for c in common:
                if not any(ch in common for ch in N.successors(c)):
                    yield ("collapse_parents", pull_up_to_common_ancestor, (v, c))
        for v in N.nodes:
            if N.out_degree(v) > 0 and N.in_degree(v) > 0:
                yield ("contract", contract_into_parents, (v,))
        for p in N.nodes:
            kids = [c for c in N.successors(p) if N.out_degree(c) > 0]
            for i, u in enumerate(kids):
                for w in kids[i + 1:]:
                    if cl[u] & cl[w]:
                        yield ("merge", merge_siblings, (u, w))
        for u in N.nodes:
            kids = sorted(N.successors(u), key=str)
            if len(kids) >= 3:
                for i, c1 in enumerate(kids):
                    for c2 in kids[i + 1:]:
                        yield ("group", group_children, (u, c1, c2))
    for u, v in N.edges:
        if N.in_degree(v) >= 2:
            yield ("delete_parent_edge", delete_parent_edge, (u, v))
        for t in N.predecessors(u):                       # one level up
            if not N.has_edge(t, v):
                yield ("pull_up", pull_up, (u, v, t))
        for t in N.successors(u):                         # one level down
            if t != v and N.out_degree(t) > 0 and not N.has_edge(t, v):
                yield ("pull_down", pull_down, (u, v, t))


def canonical(N: nx.DiGraph) -> frozenset:
    """Isomorphism-invariant key of a normalized network: its edges written as
    (cluster(parent), cluster(child)) -- inner vertex names are irrelevant."""
    cl = clusters(N)
    return frozenset((cl[u], cl[v]) for u, v in N.edges)


def neighbours(N: nx.DiGraph, leaves: set, compound: bool = True):
    """All distinct normalized networks reachable by one (compound) move.
    NOT guarded -- the search evaluates the guard lazily in score order."""
    seen = set()
    for name, fn, args in candidate_moves(N, compound):
        M = N.copy()
        try:
            fn(M, *args)
        except ValueError:
            continue
        normalize(M, leaves)
        key = canonical(M)
        if key in seen:
            continue
        seen.add(key)
        yield (name, args), M, key


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@dataclass
class SearchReport:
    mode: str
    score_name: str
    success: bool = False
    steps: int = 0
    evaluated: int = 0          # number of guard (BMG) evaluations
    start_size: tuple = (0, 0)
    final_size: tuple = (0, 0)
    final_score: int = 0
    final_is_tree: bool = False
    path: list = field(default_factory=list)


def edit_search(
    network: nx.DiGraph,
    lrt: nx.DiGraph,
    mode: str = "bmg",
    guided: bool = True,
    beam_width: int = 1,
    max_steps: int = 500,
    compound: bool = True,
) -> tuple[nx.DiGraph, SearchReport]:
    """Search an edit path from ``network`` to ``lrt``.

    The LRT is used for the success test in every case, and additionally as
    the objective if ``guided`` is True."""
    leaves = {v for v in network if network.out_degree(v) == 0}
    guard = make_guard(network, mode)
    score = make_score_guided(lrt) if guided else score_agnostic
    rep = SearchReport(mode=mode, score_name="guided" if guided else "agnostic")

    N = network.copy()
    normalize(N, leaves)
    rep.start_size = (N.number_of_nodes(), N.number_of_edges())
    if not guard(N):  # cannot happen for invariant edits; kept as a safety net
        raise AssertionError("normalize changed the BMG")

    beam = [(score(N), N, [])]
    visited = {canonical(N)}
    guard_cache: dict = {}
    best = beam[0]

    for step in range(max_steps):
        if same_phylogeny(best[1], lrt):
            break
        pool = []
        for _, B, path in beam:
            for move, M, key in neighbours(B, leaves, compound):
                if key in visited:
                    continue
                pool.append((score(M), M.number_of_edges(), key, M, path + [move]))
        pool.sort(key=lambda t: (t[0], t[1]))
        # lazy guard: only as many BMG checks as needed to fill the beam;
        # results are cached, a state is only marked visited once it was
        # actually selected (unchecked states stay available for later steps)
        new_beam, chosen = [], set()
        for sc, _, key, M, path in pool:
            if key in chosen:
                continue
            if key not in guard_cache:
                rep.evaluated += 1
                guard_cache[key] = guard(M)
            if guard_cache[key]:
                new_beam.append((sc, M, path))
                chosen.add(key)
                if len(new_beam) == beam_width:
                    break
        visited |= chosen
        if not new_beam:
            break
        beam = new_beam
        rep.steps = step + 1
        if (beam[0][0], beam[0][1].number_of_edges()) < (best[0], best[1].number_of_edges()):
            best = beam[0]
        elif beam_width == 1 and beam[0][0] > best[0] + 3:
            break  # wandered too far uphill

    s, N, path = best
    rep.success = same_phylogeny(N, lrt)
    rep.final_score = s
    rep.final_size = (N.number_of_nodes(), N.number_of_edges())
    rep.final_is_tree = all(d <= 1 for _, d in N.in_degree())
    rep.path = path
    return N, rep


# ---------------------------------------------------------------------------
# simple T*-agnostic greedy
# ---------------------------------------------------------------------------

def reduce_to_tree(network: nx.DiGraph, mode: str = "bmg", max_rounds: int = 1000):
    """Greedy hybrid resolution: repeatedly delete an in-edge of a hybrid
    vertex (highest first) or pull it one level up, whenever the guard
    accepts; normalize after each accepted edit. Stuck vertices are retried
    after every change. Returns (network, number_of_accepted_edits)."""
    leaves = {v for v in network if network.out_degree(v) == 0}
    guard = make_guard(network, mode)
    N = network.copy()
    normalize(N, leaves)
    accepted = 0
    for _ in range(max_rounds):
        root = next(v for v in N if N.in_degree(v) == 0)
        depth = nx.single_source_shortest_path_length(N, root)
        hybrids = sorted((v for v in N if N.in_degree(v) > 1), key=lambda v: depth.get(v, 0))
        progressed = False
        for v in hybrids:
            for u in sorted(N.predecessors(v), key=lambda p: -depth.get(p, 0)):
                options = [(delete_parent_edge, (u, v))] + [(pull_up, (u, v, t)) for t in N.predecessors(u)]
                for fn, args in options:
                    M = N.copy()
                    try:
                        fn(M, *args)
                    except ValueError:
                        continue
                    normalize(M, leaves)
                    if guard(M):
                        N, progressed = M, True
                        accepted += 1
                        break
                if progressed:
                    break
            if progressed:
                break
        if not progressed:
            break
    return N, accepted
