"""Light-weight editing engine for tasks 2c-2f.

The ``networkx`` implementation in :mod:`task_2_utils.graph_editing` is the
*reference*: it is written the way the definitions read. Profiling the search
showed that ~75 % of the time went into ``nx.DiGraph.copy``, so the search runs
on :class:`Net`, a pair of ``dict[vertex, set]`` (children / parents), with

* moves applied **in place** on one copy per candidate,
* normalisation done **locally** (only the vertices a move touched),
* best match graphs computed on **bitsets** (as in :mod:`task_2_utils.fast_bmg`),
* clusters as integer bitmasks, so ``target_distance`` is a few popcounts.

``tests/test_fast_edit.py`` checks move-by-move that ``Net`` and the reference
implementation produce the same network, the same BMG/WBMG and the same cost.

Macro moves
-----------
Besides the four atomic moves the engine knows two *macros*. They are not new
operations: each one is a fixed sequence of the named moves, and the edit path
that is reported is always the expanded, atomic one.

``merge(p, u, w)``   ``pull_down(p, w, u)`` followed by ``pull_up(w, c, u)`` for
                     every child ``c`` of ``w`` -- the two siblings ``u, w`` become
                     one vertex whose cluster is the union.
``contract(u, v)``   ``pull_up(v, c, u)`` for every child ``c`` of ``v`` -- the arc
                     ``(u, v)`` is contracted.
``merge_group(p, u, (w1, ..., wk))``  ``merge(p, u, w1)``, ..., ``merge(p, u, wk)``
                     -- a whole group of siblings becomes one vertex. The search
                     proposes it for the groups of siblings that are fragments
                     of the same vertex of ``T*`` (see ``edit_search``); the
                     candidates themselves are generated there, because the
                     grouping needs the target.

Why they are needed is task 2f: the intermediate networks inside a macro in
general do *not* explain ``G`` (the 'rescued' pairs of 2d), but the macro as a
whole does. Checking invariance per macro instead of per atomic move is what
makes the strict search succeed.
"""

from __future__ import annotations

import hashlib
import itertools
from typing import Hashable, Iterable, NamedTuple

import networkx as nx

__all__ = [
    "Net", "Move", "MOVE_KINDS", "MACRO_KINDS",
    "apply", "enumerate_candidates", "reduce_twins_net",
    "best_match_arcs", "target_cost", "violation", "is_tree", "cluster_set",
]

MOVE_KINDS = ("pull_up", "pull_down", "merge_twins", "remove_arc")
MACRO_KINDS = ("merge", "contract", "merge_group")


class Move(NamedTuple):
    kind: str
    args: tuple

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.kind}{self.args}"


class Net:
    """Mutable DAG: ``ch[v]`` children, ``pa[v]`` parents, ``color`` of leaves."""

    __slots__ = ("ch", "pa", "color", "bit")

    def __init__(self, ch, pa, color, bit):
        self.ch = ch
        self.pa = pa
        self.color = color      # leaf -> colour (shared, never mutated)
        self.bit = bit          # leaf -> 1 << index (shared, never mutated)

    # -- conversion ----------------------------------------------------------
    @classmethod
    def from_nx(cls, N: nx.DiGraph) -> "Net":
        ch = {v: set(N.successors(v)) for v in N}
        pa = {v: set(N.predecessors(v)) for v in N}
        leaves = sorted((v for v in N if N.out_degree(v) == 0), key=str)
        color = {v: N.nodes[v].get("color") for v in leaves}
        bit = {v: 1 << i for i, v in enumerate(leaves)}
        return cls(ch, pa, color, bit)

    def to_nx(self) -> nx.DiGraph:
        D = nx.DiGraph()
        for v in self.ch:
            D.add_node(v, color=self.color.get(v))
        for u, cs in self.ch.items():
            for c in cs:
                D.add_edge(u, c)
        return D

    def copy(self) -> "Net":
        return Net({v: set(s) for v, s in self.ch.items()},
                   {v: set(s) for v, s in self.pa.items()},
                   self.color, self.bit)

    # -- basics ----------------------------------------------------------------
    def is_leaf(self, v) -> bool:
        return v in self.color

    def inner(self) -> list:
        return [v for v in self.ch if v not in self.color]

    def root(self):
        return next(v for v, p in self.pa.items() if not p)

    def n_vertices(self) -> int:
        return len(self.ch)

    def n_arcs(self) -> int:
        return sum(len(s) for s in self.ch.values())

    def reticulations(self) -> int:
        return sum(max(0, len(p) - 1) for p in self.pa.values())

    def reaches(self, a, b) -> bool:
        """Is ``b`` a descendant of ``a`` (or ``a`` itself)?"""
        if a == b:
            return True
        stack, seen = [a], {a}
        while stack:
            for c in self.ch[stack.pop()]:
                if c == b:
                    return True
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return False

    def topo(self) -> list:
        indeg = {v: len(p) for v, p in self.pa.items()}
        order = [v for v, d in indeg.items() if d == 0]
        i = 0
        while i < len(order):
            for c in self.ch[order[i]]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    order.append(c)
            i += 1
        return order

    # -- primitive edits -----------------------------------------------------
    def _add(self, u, v):
        self.ch[u].add(v)
        self.pa[v].add(u)

    def _rem(self, u, v):
        self.ch[u].discard(v)
        self.pa[v].discard(u)

    def _delete(self, v):
        for p in self.pa[v]:
            self.ch[p].discard(v)
        for c in self.ch[v]:
            self.pa[c].discard(v)
        del self.ch[v], self.pa[v]

    def normalize(self, touched: Iterable) -> None:
        """Remove childless inner vertices and suppress single-child ones.

        Local version of :func:`task_2_utils.graph_editing.normalize`: only vertices
        whose out-degree can have changed are inspected.
        """
        work = [v for v in touched]
        while work:
            v = work.pop()
            if v not in self.ch or v in self.color:
                continue
            k = len(self.ch[v])
            if k == 0:
                parents = list(self.pa[v])
                self._delete(v)
                work.extend(parents)
            elif k == 1:
                c = next(iter(self.ch[v]))
                parents = list(self.pa[v])
                self._delete(v)
                for p in parents:
                    self._add(p, c)
                work.extend(parents)


# ---------------------------------------------------------------------------
# atomic moves (in place, validated)
# ---------------------------------------------------------------------------

def _pull_up(N: Net, u, v, p):
    if v not in N.ch[u] or u not in N.ch.get(p, ()) or p == v:
        raise ValueError("illegal pull_up")
    N._rem(u, v)
    N._add(p, v)
    N.normalize((u, p))


def _pull_down(N: Net, u, v, w):
    if v not in N.ch[u] or w not in N.ch[u] or w == v or N.is_leaf(w):
        raise ValueError("illegal pull_down")
    if N.reaches(v, w):
        raise ValueError("pull_down would close a cycle")
    N._rem(u, v)
    N._add(w, v)
    N.normalize((u,))


def _merge_twins(N: Net, a, b):
    if a == b or N.is_leaf(a) or N.is_leaf(b) or N.pa[a] != N.pa[b] or N.ch[a] != N.ch[b]:
        raise ValueError("not twins")
    parents = list(N.pa[b])
    N._delete(b)
    N.normalize(parents)


def _remove_arc(N: Net, u, v):
    if v not in N.ch[u] or len(N.pa[v]) < 2:
        raise ValueError("illegal remove_arc")
    N._rem(u, v)
    N.normalize((u,))


_ATOMIC = {"pull_up": _pull_up, "pull_down": _pull_down,
           "merge_twins": _merge_twins, "remove_arc": _remove_arc}


def _drain(N: Net, v, into, out: list):
    """``pull_up(v, c, into)`` for every child ``c`` until ``v`` is gone."""
    while v in N.ch:
        c = min(N.ch[v], key=str)
        _pull_up(N, v, c, into)
        out.append(Move("pull_up", (v, c, into)))


def apply(N: Net, move: Move, reduce: bool = True) -> tuple[Net, list[Move]]:
    """Apply an atomic move or a macro to a copy; return it and the atomic path."""
    M = N.copy()
    path: list[Move] = []
    if move.kind == "merge":
        p, u, w = move.args
        _pull_down(M, p, w, u)
        path.append(Move("pull_down", (p, w, u)))
        _drain(M, w, u, path)
    elif move.kind == "merge_group":
        p, u, ws = move.args
        for w in ws:
            if w not in M.ch or u not in M.ch or p not in M.ch:
                raise ValueError("merge_group: vertex vanished")
            _pull_down(M, p, w, u)
            path.append(Move("pull_down", (p, w, u)))
            _drain(M, w, u, path)
    elif move.kind == "contract":
        u, v = move.args
        if v not in N.ch[u] or N.is_leaf(v):
            raise ValueError("illegal contract")
        _drain(M, v, u, path)
    else:
        _ATOMIC[move.kind](M, *move.args)
        path.append(move)
    if reduce:
        path.extend(reduce_twins_net(M))
    return M, path


def twin_groups(N: Net) -> list[list]:
    buckets: dict = {}
    for v in sorted(N.inner(), key=str):
        buckets.setdefault((frozenset(N.pa[v]), frozenset(N.ch[v])), []).append(v)
    return sorted((g for g in buckets.values() if len(g) > 1), key=lambda g: str(g[0]))


def reduce_twins_net(N: Net) -> list[Move]:
    """Merge twins in place until none are left; return the merges done."""
    done = []
    while True:
        groups = twin_groups(N)
        if not groups:
            return done
        for g in groups:
            a = g[0]
            for b in g[1:]:
                if b in N.ch and a in N.ch:
                    _merge_twins(N, a, b)
                    done.append(Move("merge_twins", (a, b)))


def enumerate_candidates(N: Net, kinds: tuple = MOVE_KINDS + MACRO_KINDS) -> list[Move]:
    """All candidate moves of the given kinds, in a *deterministic* order.

    Vertex names are mixed ints/strings and Python's string hashing is salted
    per process, so iterating the ``set`` s directly would make the search
    depend on ``PYTHONHASHSEED``. Everything is sorted by ``str``.
    """
    out: list[Move] = []
    S = lambda it: sorted(it, key=str)
    inner = S(N.inner())
    if "pull_up" in kinds:
        for u in inner:
            for v in S(N.ch[u]):
                for p in S(N.pa[u]):
                    if p != v:
                        out.append(Move("pull_up", (u, v, p)))
    if "pull_down" in kinds:
        for u in inner:
            kids = S(N.ch[u])
            for v in kids:
                for w in kids:
                    if w != v and not N.is_leaf(w):
                        out.append(Move("pull_down", (u, v, w)))
    if "remove_arc" in kinds:
        for u in inner:
            if len(N.ch[u]) >= 2:
                for v in S(N.ch[u]):
                    if len(N.pa[v]) >= 2:
                        out.append(Move("remove_arc", (u, v)))
    if "merge_twins" in kinds:
        for g in twin_groups(N):
            for a, b in itertools.combinations(g, 2):
                out.append(Move("merge_twins", (a, b)))
    if "merge" in kinds:
        for p in inner:
            kids = S(c for c in N.ch[p] if not N.is_leaf(c))
            for u, w in itertools.permutations(kids, 2):
                out.append(Move("merge", (p, u, w)))
    if "contract" in kinds:
        for u in inner:
            for v in S(N.ch[u]):
                if not N.is_leaf(v):
                    out.append(Move("contract", (u, v)))
    return out


# ---------------------------------------------------------------------------
# best match graphs on bitsets
# ---------------------------------------------------------------------------

def _closure(N: Net):
    order = N.topo()
    idx = {v: i for i, v in enumerate(order)}
    anc = [0] * len(order)
    for v in order:
        m = 0
        for p in N.pa[v]:
            j = idx[p]
            m |= anc[j] | (1 << j)
        anc[idx[v]] = m
    desc = [0] * len(order)
    for v in reversed(order):
        m = 0
        for c in N.ch[v]:
            j = idx[c]
            m |= desc[j] | (1 << j)
        desc[idx[v]] = m
    return order, idx, anc, desc


def _minimal(mask: int, desc: list) -> int:
    out, rest = 0, mask
    while rest:
        low = rest & -rest
        if not (desc[low.bit_length() - 1] & mask):
            out |= low
        rest ^= low
    return out


def _bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def best_match_arcs(N: Net, weak: bool = False) -> frozenset:
    """Arc set of the (weak) best match graph of ``N``."""
    _order, idx, anc, desc = _closure(N)
    leaves = list(N.color)
    by_color: dict = {}
    for x in leaves:
        by_color.setdefault(N.color[x], []).append(x)
    lca: dict = {}
    for x, y in itertools.combinations(leaves, 2):
        if N.color[x] != N.color[y]:
            m = _minimal(anc[idx[x]] & anc[idx[y]], desc)
            lca[(x, y)] = lca[(y, x)] = m
    arcs = []
    for x in leaves:
        for tau, same in by_color.items():
            if tau == N.color[x]:
                continue
            union = 0
            for y in same:
                union |= lca[(x, y)]
            if weak:
                q = _minimal(union, desc)
                arcs.extend((x, y) for y in same if lca[(x, y)] & q)
            else:
                for y in same:
                    if not any(desc[i] & union for i in _bits(lca[(x, y)])):
                        arcs.append((x, y))
    return frozenset(arcs)


# ---------------------------------------------------------------------------
# costs
# ---------------------------------------------------------------------------

def clusters(N: Net) -> dict:
    """vertex -> leaf bitmask of its cluster."""
    out = {}
    for v in reversed(N.topo()):
        if v in N.bit:
            out[v] = N.bit[v]
        else:
            m = 0
            for c in N.ch[v]:
                m |= out[c]
            out[v] = m
    return out


def cluster_set(N: Net) -> frozenset:
    return frozenset(clusters(N)[v] for v in N.inner())


def target_cost(N: Net, want: frozenset, n_target_inner: int | None = None) -> tuple:
    """Bitset version of :func:`task_2_utils.graph_editing.target_distance`."""
    cl = clusters(N)
    inner = N.inner()
    n_target_inner = len(want) if n_target_inner is None else n_target_inner
    shape = 0
    have = set()
    for v in inner:
        c = cl[v]
        have.add(c)
        shape += min((c ^ w).bit_count() for w in want)
    missing = len(want - have)
    excess = max(0, len(inner) - n_target_inner)
    retic = N.reticulations()
    return (shape + missing + excess + retic, retic, N.n_vertices(), N.n_arcs())


def violation(arcs: frozenset, reference: frozenset) -> int:
    """Number of arcs in which two best match graphs differ."""
    return len(arcs ^ reference)


def is_tree(N: Net) -> bool:
    return (all(len(p) <= 1 for p in N.pa.values())
            and sum(1 for p in N.pa.values() if not p) == 1
            and all(len(c) != 1 for c in N.ch.values()))


def canonical_key(N: Net) -> int:
    """Leaf-labelled isomorphism invariant (vertex and arc multisets)."""
    sig: dict = {}
    for v in reversed(N.topo()):
        if v in N.color:
            sig[v] = hash(("L", str(v)))
        else:
            sig[v] = hash(tuple(sorted(sig[c] for c in N.ch[v])))
    verts = tuple(sorted(sig.values()))
    arcs = tuple(sorted((sig[u], sig[c]) for u in N.ch for c in N.ch[u]))
    return hash((verts, arcs))
