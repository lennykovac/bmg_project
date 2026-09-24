"""Bitset implementation of the (weak) best match graphs.

Semantically identical to :func:`utils.graph_utils.bmg_from_network` and
:func:`utils.graph_utils.wbmg_from_network` -- ``tests/test_fast_bmg.py``
asserts arc-by-arc equality on thousands of random networks. The reference
implementations stay in ``graph_utils`` because they are written the way the
definitions read; this module is what the search in task 2e actually calls,
because it evaluates one candidate network per edit move.

What makes it fast
------------------
* ancestor and descendant sets are Python ``int`` bitmasks, so "is ``u``
  strictly below ``v``" is one ``&`` and the minimal-element filter is a loop
  over set bits instead of a quadratic scan over pairs;
* the lca sets of all bicolored leaf pairs are computed once and shared by the
  strict and the weak graph (:func:`best_match_graphs` returns both).

Definitions implemented
-----------------------
``LCA(x, y)``  the ``<=``-minimal common ancestors of ``x`` and ``y``.

strict   ``y`` is a best match of ``x`` iff ``sigma(x) != sigma(y)`` and there
         is no ``u in LCA(x, y')`` with ``sigma(y') = sigma(y)`` that lies
         strictly below some ``v in LCA(x, y)``.

weak     with ``Q(x, tau) := min_<= U {LCA(x, y') : sigma(y') = tau}``, ``y``
         is a weak best match of ``x`` iff ``LCA(x, y) n Q(x, sigma(y)) != {}``.
"""

from __future__ import annotations

from typing import Hashable

import networkx as nx

__all__ = [
    "Closure",
    "closure",
    "lca_table",
    "bmg_from_network_fast",
    "wbmg_from_network_fast",
    "best_match_graphs",
]


class Closure:
    """Index, ancestor masks and descendant masks of a DAG."""

    __slots__ = ("nodes", "index", "anc", "desc")

    def __init__(self, N: nx.DiGraph):
        order = list(nx.topological_sort(N))
        self.nodes: list[Hashable] = order
        self.index: dict[Hashable, int] = {v: i for i, v in enumerate(order)}

        anc = [0] * len(order)
        for v in order:  # topological order: parents come first
            m = 0
            for p in N.predecessors(v):
                m |= anc[self.index[p]] | (1 << self.index[p])
            anc[self.index[v]] = m
        self.anc = anc

        desc = [0] * len(order)
        for v in reversed(order):
            m = 0
            for c in N.successors(v):
                m |= desc[self.index[c]] | (1 << self.index[c])
            desc[self.index[v]] = m
        self.desc = desc

    def minimal(self, mask: int) -> int:
        """Keep only the ``<=``-minimal vertices of ``mask``."""
        out = 0
        rest = mask
        while rest:
            low = rest & -rest
            i = low.bit_length() - 1
            if not (self.desc[i] & mask):
                out |= low
            rest ^= low
        return out


def closure(N: nx.DiGraph) -> Closure:
    return Closure(N)


def _bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def lca_table(N: nx.DiGraph, cl: Closure, leaves: list) -> dict:
    """``(x, y) -> bitmask of LCA(x, y)`` for all bicolored ordered pairs."""
    color = {v: N.nodes[v].get("color") for v in leaves}
    anc = cl.anc
    idx = cl.index
    table: dict[tuple, int] = {}
    for i, x in enumerate(leaves):
        ax = anc[idx[x]]
        for y in leaves[i + 1:]:
            if color[x] == color[y]:
                continue
            common = ax & anc[idx[y]]
            lca = cl.minimal(common)
            table[(x, y)] = lca
            table[(y, x)] = lca
    return table


def _skeleton(N: nx.DiGraph, leaves: list) -> nx.DiGraph:
    G = nx.DiGraph()
    for v in leaves:
        G.add_node(v, color=N.nodes[v].get("color"))
    return G


def fast_bmg(N: nx.DiGraph) -> tuple[nx.DiGraph, nx.DiGraph]:
    """Return ``(BMG, WBMG)`` of the leaf-colored network ``N``."""
    leaves = [v for v in N.nodes if N.out_degree(v) == 0]
    cl = Closure(N)
    table = lca_table(N, cl, leaves)

    color = {v: N.nodes[v].get("color") for v in leaves}
    by_color: dict = {}
    for v in leaves:
        by_color.setdefault(color[v], []).append(v)

    bmg = _skeleton(N, leaves)
    wbmg = _skeleton(N, leaves)

    for x in leaves:
        for tau, same in by_color.items():
            if tau == color[x]:
                continue
            # union of all lca(x, y') with sigma(y') = tau
            union = 0
            for y in same:
                union |= table[(x, y)]
            q = cl.minimal(union)

            # everything strictly below one of the union's vertices
            below = 0
            for i in _bits(union):
                below |= 1 << i

            for y in same:
                lca = table[(x, y)]
                # strict: no vertex of `union` lies strictly below a vertex of lca
                strict = True
                for i in _bits(lca):
                    if cl.desc[i] & below:
                        strict = False
                        break
                if strict:
                    bmg.add_edge(x, y)
                # weak: lca meets Q(x, tau)
                if lca & q:
                    wbmg.add_edge(x, y)

    return bmg, wbmg
