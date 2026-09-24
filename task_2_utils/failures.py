"""Task 2f: proven failures and minimal counterexamples.

A failure of a *heuristic* says little -- the budget may just have run out. So
the predicate used here is a **proof**: the search with ``prune=False`` explores
every network reachable from the twin-reduced BIC-cherry expansion under the
invariant; if the frontier runs empty without meeting ``T*``
(``PathResult.exhausted``), no edit path with that move set and invariant
exists.

``minimize_bmg`` (from :mod:`task_2_utils.experiments`) then deletes leaves one at a
time as long as the proof of failure survives. The restriction of a tree-BMG
to a colour-sink-free subset is again a tree-BMG, so the shrunken graph is a
legitimate instance.
"""
from __future__ import annotations

import networkx as nx

from utils.bic_cherry import bic_cherry_expansion
from task_2_utils.edit_search import ATOMIC, WITH_MACROS, find_path
from task_2_utils.experiments import minimize_bmg
from utils.lrt import lrt_from_bmg

__all__ = ["proven_failure", "minimal_counterexample", "signature"]


def proven_failure(G: nx.DiGraph, kinds: tuple = ATOMIC, restricted: bool = True,
                   budget: int = 200_000) -> bool | None:
    """True: provably no strict path; False: a path exists; None: undecided."""
    T = lrt_from_bmg(G)
    if T is None:
        return None
    r = find_path(bic_cherry_expansion(G, restricted=restricted), T, kinds=kinds,
                  mode="strict", prune=False, fallback=False, budget=budget,
                  verify=False)
    if r.reached:
        return False
    return True if r.exhausted else None


def minimal_counterexample(G: nx.DiGraph, kinds: tuple = ATOMIC,
                           restricted: bool = True) -> nx.DiGraph:
    return minimize_bmg(G, lambda H: proven_failure(H, kinds, restricted) is True)


def signature(G: nx.DiGraph) -> str:
    """Isomorphism class of a small coloured digraph (colours renamed a, b, c)."""
    import itertools
    cols = sorted({c for _, c in G.nodes(data="color")}, key=str)
    best = None
    nodes = list(G)
    for perm in itertools.permutations(range(len(cols))):
        cmap = {c: "abcdefg"[perm[i]] for i, c in enumerate(cols)}
        for order in itertools.permutations(nodes):
            idx = {v: i for i, v in enumerate(order)}
            key = (tuple(cmap[G.nodes[v]["color"]] for v in order),
                   tuple(sorted((idx[u], idx[v]) for u, v in G.edges())))
            if best is None or key < best:
                best = key
        if len(nodes) > 6:
            break
    return f"{''.join(best[0])}:{best[1]}"
