"""Edit-path search from the BIC-cherry explanation to the LRT (tasks 2e/2f).

``find_path(N, T*)`` searches for a sequence of the named edit moves that
turns the BIC-cherry+expansion network ``N`` into the least resolved tree
``T*``. It runs on :class:`task_2_utils.fast_edit.Net` (see there for why).

State space
    networks reachable from ``reduce_twins(N)``; duplicates are removed with a
    leaf-labelled canonical key.
Moves (``kinds``)
    the atomic moves of 2c and the two macros ``merge`` / ``contract``, which
    are fixed sequences of pull downs / pull ups. The reported path is always
    the expanded, atomic one.
Invariant (``mode``)
    ``"strict"``   every accepted search step explains ``G``. With macros this
                   is checked *per macro*; the atomic intermediates inside a
                   macro may not explain ``G`` and are counted
                   (``broken_atomic``).
    ``"relaxed"``  non-explaining steps are allowed but penalised by
                   ``penalty * |E(G(N)) symdiff E(G)|``.
Guidance
    ``h = target_cost(...)[0]`` is 0 exactly at ``T*``. Candidates are ranked
    by the *fragment* rule of :func:`_rank`.
Strategy
    ``"first"`` (default) first-improvement depth-first descent with lazy
    backtracking; ``"best"`` classical best-first search (evaluates every
    candidate of every expanded state -- slower, occasionally shorter paths).

After the search the atomic path is **replayed with the networkx reference
implementation** (:func:`verify_path`), so a reported success does not rest on
the fast engine alone.
"""

from __future__ import annotations

import heapq
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import NamedTuple

import networkx as nx

from task_2_utils.fast_edit import (MACRO_KINDS, MOVE_KINDS, Move, Net, apply,
                             best_match_arcs, canonical_key, cluster_set,
                             enumerate_candidates, is_tree, reduce_twins_net,
                             target_cost)

__all__ = ["PathResult", "find_path", "verify_path", "solve_many", "CONFIGS"]

ATOMIC = MOVE_KINDS
WITH_MACROS = MOVE_KINDS + MACRO_KINDS

#: label -> keyword arguments of find_path; the product of the two knobs
CONFIGS = {
    "atomic/strict": dict(kinds=ATOMIC, mode="strict"),
    "atomic/relaxed": dict(kinds=ATOMIC, mode="relaxed"),
    "macro/strict": dict(kinds=WITH_MACROS, mode="strict"),
    "macro/relaxed": dict(kinds=WITH_MACROS, mode="relaxed"),
}


class PathResult(NamedTuple):
    reached: bool            # final network is T*
    path: list               # atomic moves, start = BIC-cherry expansion itself
    macro_steps: int         # number of search steps (a macro counts once)
    broken_steps: int        # search steps whose network does not explain G
    broken_atomic: int       # atomic intermediates that do not explain G
    final: nx.DiGraph
    final_cost: tuple
    start_cost: tuple
    expanded: int
    seconds: float
    verified: bool | None    # replay with the networkx reference agrees
    exhausted: bool = False  # the whole reachable state space was searched
                             # (only meaningful with prune=False): a failure
                             # is then a proof that no such path exists

    # aliases, so that code written for task_2_utils.search.SearchResult
    # (e.g. task_2_utils.animation.trace_search) accepts a PathResult as well
    @property
    def network(self) -> nx.DiGraph:
        return self.final

    @property
    def reached_target(self) -> bool:
        return self.reached

    @property
    def cost(self) -> tuple:
        return self.final_cost

    strategy = "edit_search"

    def row(self) -> tuple:
        return (self.reached, len(self.path), self.macro_steps, self.broken_steps,
                self.broken_atomic, self.expanded, f"{self.seconds:.1f}s")


def find_path(
    N: nx.DiGraph,
    target: nx.DiGraph,
    kinds: tuple = WITH_MACROS,
    mode: str = "strict",
    weak: bool = False,
    weight: float | None = None,
    penalty: int = 2,
    budget: int = 50_000,
    prune: bool = True,
    tiebreak: str = "size",
    strategy: str = "first",
    width: int = 8,
    per_state: int | None = 30,
    beam_weight: float | None = None,
    fallback: bool = True,
    time_limit: float | None = None,
    verify: bool = True,
) -> PathResult:
    t0 = time.time()
    start = Net.from_nx(N)
    pre = reduce_twins_net(start)                  # provably graph preserving
    reference = best_match_arcs(Net.from_nx(N), weak)
    want = cluster_set(Net.from_nx(target))
    n_inner = sum(1 for v in target if target.out_degree(v) > 0)

    def h(net):
        return target_cost(net, want, n_inner)

    def prio(cost, g, viol):
        base = cost[0] + penalty * viol
        if weight is not None:
            return (weight * base + g, g)
        # greedy: ties broken towards smaller networks, then shorter paths
        return (base, cost[2], cost[3], g) if tiebreak == "size" else (base, g)

    start_cost = h(start)
    if strategy == "beam":
        reached, best, expanded, exhausted = _beam(
            start, pre, start_cost, h, reference, kinds, mode, weak, penalty,
            budget, time_limit, t0, prune, want, width, per_state, beam_weight)
        return _result(N, reached, best, start_cost, expanded, t0, verify, weak,
                       exhausted and not prune and per_state is None)
    if strategy == "first":
        reached, best, expanded, exhausted = _first_improvement(
            start, pre, start_cost, h, reference, kinds, mode, weak, penalty,
            budget, time_limit, t0, prune, want)
        exhausted = exhausted and not prune
        if reached is None and fallback and prune:
            # second attempt with the pruned merges back in (ranked last)
            reached, best2, more, exhausted = _first_improvement(
                start, pre, start_cost, h, reference, kinds, mode, weak, penalty,
                budget, time_limit, t0, False, want)
            expanded += more
            if best2[0] < best[0]:
                best = best2
        return _result(N, reached, best, start_cost, expanded, t0, verify, weak,
                       exhausted)
    tie = itertools.count()
    # heap item: (priority, tie, cost, net, atomic path, macro steps, broken steps)
    heap = [(prio(start_cost, 0, 0), next(tie), start_cost, start, pre, 0, 0)]
    seen = {canonical_key(start)}
    best = (start_cost, start, pre, 0, 0)          # best *explaining* state
    expanded = 0
    reached = None

    while heap and expanded < budget:
        if time_limit and time.time() - t0 > time_limit:
            break
        _p, _t, cost, state, path, steps, broken = heapq.heappop(heap)
        if cost[0] == 0 and is_tree(state):
            reached = (cost, state, path, steps, broken)
            break
        candidates = _ordered(state, enumerate_candidates(state, kinds), want, prune,
                              groups="merge_group" in kinds)
        children = []
        for move in candidates:
            try:
                M, atoms = apply(state, move)
            except ValueError:
                continue
            expanded += 1
            key = canonical_key(M)
            if key in seen:
                continue
            seen.add(key)
            ok = best_match_arcs(M, weak) == reference
            if not ok and mode == "strict":
                continue
            viol = 0 if ok else len(best_match_arcs(M, weak) ^ reference)
            c = h(M)
            g = len(path) + len(atoms)
            item = (prio(c, g, viol), next(tie), c, M, path + atoms, steps + 1,
                    broken + (not ok))
            children.append(item)
            if ok and c < best[0]:
                best = (c, M, path + atoms, steps + 1, broken + (not ok))
        for item in children:
            heapq.heappush(heap, item)

    return _result(N, reached, best, start_cost, expanded, t0, verify, weak,
                   not heap and not prune)


def _result(N, reached, best, start_cost, expanded, t0, verify, weak, exhausted=False):
    cost, net, path, steps, broken = reached if reached else best
    final = net.to_nx()
    broken_atomic, verified = (verify_path(N, path, final, weak) if verify else (None, None))
    return PathResult(
        reached=reached is not None, path=path, macro_steps=steps,
        broken_steps=broken, broken_atomic=broken_atomic, final=final,
        final_cost=cost, start_cost=start_cost, expanded=expanded,
        seconds=time.time() - t0, verified=verified,
        exhausted=reached is None and exhausted,
    )


def _first_improvement(start, pre, start_cost, h, reference, kinds, mode, weak,
                       penalty, budget, time_limit, t0, prune, want):
    """Depth-first descent with first improvement and lazy backtracking.

    A state's candidates are generated lazily, cheapest kinds first
    (twin merges, sibling merges, contractions ...). The first candidate that
    lowers ``h`` and explains ``G`` is taken at once; the parent stays on the
    stack with its half-consumed iterator, so a dead end is left by resuming
    the parent instead of restarting. Only if a state has *no* improving
    child are all of its children ranked and pushed (plateau / uphill moves).
    Compared to full best-first expansion this evaluates a small fraction of
    the candidates, because along most of the path an improving move exists.
    """
    def score(c, viol):
        return c[0] + penalty * viol

    def candidates(state):
        return iter(_ordered(state, enumerate_candidates(state, kinds), want, prune,
                             groups="merge_group" in kinds))

    seen = {canonical_key(start)}
    # entry: [score, cost, state, path, steps, broken, iterator|None, pending]
    stack = [[score(start_cost, 0), start_cost, start, pre, 0, 0, None, []]]
    best = (start_cost, start, pre, 0, 0)
    expanded = 0
    while stack and expanded < budget:
        if time_limit and time.time() - t0 > time_limit:
            break
        entry = stack[-1]
        sc, cost, state, path, steps, broken, it, pending = entry
        if cost[0] == 0 and is_tree(state):
            return (cost, state, path, steps, broken), best, expanded, False
        if it is None:
            it = entry[6] = candidates(state)
        improved = None
        for move in it:
            try:
                M, atoms = apply(state, move)
            except ValueError:
                continue
            expanded += 1
            key = canonical_key(M)
            if key in seen:
                continue
            seen.add(key)
            c = h(M)
            arcs = best_match_arcs(M, weak)
            ok = arcs == reference
            if not ok and mode == "strict":
                continue
            viol = 0 if ok else len(arcs ^ reference)
            child = [score(c, viol), c, M, path + atoms, steps + 1,
                     broken + (not ok), None, []]
            if ok and c < best[0]:
                best = (c, M, path + atoms, steps + 1, broken)
            if child[0] < sc:
                improved = child
                break
            pending.append(child)
            if expanded >= budget:
                break
        if improved is not None:
            stack.append(improved)
            continue
        # iterator exhausted without improvement: rank the rest, best on top
        stack.pop()
        pending.sort(key=lambda e: (e[0], e[1][2], len(e[3])), reverse=True)
        stack.extend(pending)
    return None, best, expanded, not stack


def _beam(start, pre, start_cost, h, reference, kinds, mode, weak, penalty,
          budget, time_limit, t0, prune, want, width, per_state, beam_weight=None):
    """Beam search: keep the ``width`` best networks of every level.

    Every network of the beam is expanded; its candidates are evaluated in the
    fragment order of :func:`_rank` until ``per_state`` admissible children
    (unseen, and explaining ``G`` in strict mode) have been produced
    (``None``: all candidates). The children of all beam members are ranked by
    ``(h + penalty * violation, |V|, |E|, path length)`` and the best
    ``width`` form the next level. Unlike the first-improvement descent the
    beam never commits to a single move, so it can prefer a move that is only
    second best now but leads to a shorter path; the price is that every level
    evaluates up to ``width * per_state`` networks.

    ``beam_weight=w`` ranks by ``g + w * h`` instead (``g`` = atomic moves so
    far), i.e. a beam-limited weighted A*, which aims at *short* paths.
    """
    def score(c, viol, g):
        d = c[0] + penalty * viol
        if beam_weight is not None:        # A*-like: path so far + weighted rest
            return (g + beam_weight * d, d, c[2], c[3])
        return (d, c[2], c[3], g)

    seen = {canonical_key(start)}
    # entry: (score, cost, state, path, steps, broken)
    beam = [(score(start_cost, 0, 0), start_cost, start, pre, 0, 0)]
    best = (start_cost, start, pre, 0, 0)
    expanded = 0
    if start_cost[0] == 0 and is_tree(start):
        return (start_cost, start, pre, 0, 0), best, 0, False
    while beam and expanded < budget:
        if time_limit and time.time() - t0 > time_limit:
            break
        children = []
        for _sc, cost, state, path, steps, broken in beam:
            accepted = 0
            for move in _ordered(state, enumerate_candidates(state, kinds), want, prune,
                                 groups="merge_group" in kinds):
                if per_state is not None and accepted >= per_state:
                    break
                if expanded >= budget:
                    break
                try:
                    M, atoms = apply(state, move)
                except ValueError:
                    continue
                expanded += 1
                key = canonical_key(M)
                if key in seen:
                    continue
                seen.add(key)
                arcs = best_match_arcs(M, weak)
                ok = arcs == reference
                if not ok and mode == "strict":
                    continue
                accepted += 1
                c = h(M)
                viol = 0 if ok else len(arcs ^ reference)
                g = len(path) + len(atoms)
                child = (score(c, viol, g), c, M, path + atoms, steps + 1, broken + (not ok))
                if ok and c < best[0]:
                    best = (c, M, path + atoms, steps + 1, broken)
                if ok and c[0] == 0 and is_tree(M):
                    return child[1:], best, expanded, False
                children.append(child)
        children.sort(key=lambda e: e[0])
        beam = children[:width]
    return None, best, expanded, not beam


def _tlca(mask: int, want: frozenset) -> int:
    """Smallest target cluster containing ``mask`` (the target lca)."""
    best = None
    for c in want:
        if mask & c == mask and (best is None or c.bit_count() < best.bit_count()):
            best = c
    return best


def _phi(state: Net, want: frozenset):
    """``v -> tlca(C(v))``: the vertex of ``T*`` that ``v`` is a fragment of."""
    from task_2_utils.fast_edit import clusters
    cl = clusters(state)
    memo: dict = {}

    def phi(v):
        if v not in memo:
            memo[v] = _tlca(cl[v], want)
        return memo[v]
    return phi


def _rank(state: Net, move: Move, phi) -> int | None:
    """Target-guided ordering of the candidates (heuristic, task 2e).

    Every inner vertex ``v`` of the current network is a *fragment* of the
    target vertex ``phi(v)`` (the lca in ``T*`` of its cluster). ``T*`` is
    reached when every target vertex is represented by exactly one fragment,
    so the useful moves are

    0  twin merges (always graph preserving);
    1  ``contract(u, v)`` with ``phi(v) == phi(u)``: ``v`` is a fragment of the
       same target vertex as its parent -- it has to disappear *into* it;
    2  ``merge(p, u, w)`` with ``phi(u) == phi(w) != phi(p)``: two fragments of
       the same target vertex below a vertex of a different one;
    3  ``merge(p, u, w)`` with ``phi(u) == phi(w) == phi(p)`` (needed: e.g. for
       the complete BMG all cherries are root fragments and must be merged
       *simultaneously*; contracting one of them alone breaks the graph);
    4  arc deletions and contractions across target vertices;
    5  everything else (pull downs, single pull ups).

    ``None`` drops the candidate (with ``prune=True``): merging fragments of
    *different* target vertices creates a cluster that ``T*`` does not have.
    Without this rule the greedy search built vertices whose cluster straddles
    two subtrees of ``T*`` and then could not take them apart again -- the
    failure mode found in 2f.
    """
    k = move.kind
    if k == "merge_twins":
        return 0
    if k == "contract":
        u, v = move.args
        return 1 if phi(u) == phi(v) else 4
    if k == "merge":
        p, u, w = move.args
        if phi(u) != phi(w):
            return None
        return 2 if phi(u) != phi(p) else 3
    if k == "remove_arc":
        return 4
    return 5


def _group_merges(state: Net, phi) -> list:
    """``merge_group`` for every set of >= 3 siblings with the same ``phi``."""
    out = []
    for p in sorted(state.inner(), key=str):
        groups: dict = {}
        for c in sorted(state.ch[p], key=str):
            if not state.is_leaf(c):
                groups.setdefault(phi(c), []).append(c)
        for g in groups.values():
            if len(g) >= 3:
                g = sorted(g, key=str)
                out.append(Move("merge_group", (p, g[0], tuple(g[1:]))))
    return out


def _ordered(state: Net, candidates: list, want: frozenset, prune: bool,
             groups: bool = True) -> list:
    phi = _phi(state, want)
    ranked = []
    if groups:
        for i, m in enumerate(_group_merges(state, phi)):
            # all fragments of one target vertex at once: before pairwise merges
            ranked.append((1.5 if phi(m.args[1]) != phi(m.args[0]) else 2.5, -i, m))
    for i, m in enumerate(candidates):
        r = _rank(state, m, phi)
        if r is None:
            if prune:
                continue
            r = 6
        ranked.append((r, i, m))
    ranked.sort(key=lambda t: (t[0], t[1]))
    return [m for _r, _i, m in ranked]


def verify_path(N: nx.DiGraph, path: list, final: nx.DiGraph, weak: bool = False):
    """Replay ``path`` with :mod:`task_2_utils.graph_editing` (the reference).

    Returns ``(number of intermediate networks not explaining G, agrees)``,
    where ``agrees`` means the replay ends in exactly ``final``.
    """
    from task_2_utils.graph_editing import Move as RefMove, apply_move
    from utils.graph_utils import bmg_from_network

    leaves = {v for v in N if N.out_degree(v) == 0}
    ref = set(bmg_from_network(N, weak=weak).edges())
    cur, broken = N, 0
    for i, m in enumerate(path):
        cur = apply_move(cur, RefMove(m.kind, m.args), leaves)
        if i < len(path) - 1 and set(bmg_from_network(cur, weak=weak).edges()) != ref:
            broken += 1
    agrees = set(cur.edges()) == set(final.edges()) and \
        set(bmg_from_network(cur, weak=weak).edges()) == ref
    return broken, agrees


# ---------------------------------------------------------------------------
# parallel driver
# ---------------------------------------------------------------------------

def _job(args):
    label, N, target, kwargs = args
    return label, find_path(N, target, **kwargs)


def _pool_context():
    """Start method for the worker processes.

    Python >= 3.14 uses ``forkserver`` by default on Linux (``spawn`` on macOS
    and Windows). Both re-import the main script in every worker; a script
    without an ``if __name__ == "__main__":`` guard (like ``task_2.py``) then
    tries to start a pool from inside each worker and crashes. ``fork`` does
    not re-import anything, so it is used wherever it is safe (Linux). On other
    platforms the default is kept, which needs the guard in the main script;
    ``solve_many`` falls back to serial execution if the pool cannot start.
    """
    import multiprocessing as mp
    import sys
    if sys.platform.startswith("linux") and "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return mp.get_context()


def solve_many(jobs: list, workers: int | None = None) -> list:
    """Run ``(label, N, target, kwargs)`` jobs, in parallel processes.

    Parallelism is over *independent searches* (instances x configurations):
    that scales linearly with the number of cores. Parallelising inside one
    search does not pay off -- every candidate is ~1 ms of work and shipping
    the network to another process costs about as much (the GIL rules out
    threads for this CPU-bound code).

    Runs serially if ``workers == 1``, if there is only one job, if called
    from inside a worker process, or if the process pool cannot be started.
    """
    import multiprocessing as mp
    import warnings
    from concurrent.futures.process import BrokenProcessPool

    workers = min(workers or os.cpu_count() or 1, len(jobs))
    if workers <= 1 or mp.parent_process() is not None:
        return [_job(j) for j in jobs]
    try:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=_pool_context()) as pool:
            return list(pool.map(_job, jobs, chunksize=1))
    except (RuntimeError, BrokenProcessPool, OSError) as exc:
        warnings.warn(f"process pool failed ({exc.__class__.__name__}: {exc}); "
                      "running serially", RuntimeWarning)
        return [_job(j) for j in jobs]
