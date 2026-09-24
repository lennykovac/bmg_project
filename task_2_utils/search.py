"""Heuristics that simplify an explaining network towards the LRT (task 2e).

Setting: a colored graph ``(G, sigma)`` and a network ``(N, sigma)`` that
explains it (the BIC-cherry expansion). We look for a sequence of edit moves

    N = N_0 -> N_1 -> ... -> N_k

in which **every** intermediate network still explains ``(G, sigma)``, and in
which ``N_k`` is as tree-like as possible. For a tree-BMG the target is the
least resolved tree ``T*``.

Two cost functions steer the descent.

``cost="tree"``    target-free: ``(reticulations, |V|, |E|)``. Since ``T*`` is
                   the unique *smallest* tree explaining a tree-BMG, a state
                   with cost ``(0, |V(T*)|, |E(T*)|)`` that still explains
                   ``G`` is ``T*``. This is the honest "we do not know where
                   we are going" setting.
``cost="target"``  target-guided: :func:`task_2_utils.graph_editing.target_distance`
                   counts how many clusters of ``N`` are wrong, duplicated or
                   missing with respect to ``T*``. Task 2e explicitly asks for
                   the conversion *into* ``T*``, so the target may be used as
                   a guide; the cost is 0 exactly at ``T*``.

Three strategies.

``greedy``   steepest descent (or first improvement). Plateau moves are
             allowed for a bounded number of consecutive steps -- otherwise
             the search would stop at the very first move that leaves the
             cost unchanged.
``beam``     keeps the ``width`` cheapest states per level, deduplicated by
             :func:`task_2_utils.graph_editing.canonical_key`.
``bfs``      complete breadth-first exploration of the graph-preserving part
             of the move graph, bounded by ``max_states``. Only for tiny
             instances, but it separates "the heuristic is bad" from "the
             move set cannot reach the target" -- which is what task 2f asks.

Before the search starts, :func:`task_2_utils.graph_editing.reduce_twins` is applied.
Twin merging provably preserves both best match graphs, so it costs nothing
and removes the duplicated cherry vertices ``p_xy``/``p_yx`` that the
construction introduces.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Callable, NamedTuple

import networkx as nx

from task_2_utils.fast_bmg import fast_bmg
from utils.graph_utils import is_phylogenetic_tree, same_phylogeny
from utils.bic_cherry import bic_cherry_expansion
from task_2_utils.graph_editing import (
    DEFAULT_MOVES,
    EXTENDED_MOVES,
    Move,
    apply_move,
    canonical_key,
    enumerate_moves,
    reduce_twins,
    target_distance,
    tree_likeness,
)

__all__ = [
    "SearchResult",
    "greedy_simplify",
    "beam_simplify",
    "bfs_simplify",
    "simplify",
    "simplify_bmg",
    "RECOMMENDED",
    "STRATEGIES",
]


class SearchResult(NamedTuple):
    network: nx.DiGraph
    path: list
    cost: tuple
    start_cost: tuple
    is_tree: bool
    reached_target: bool
    explains: bool
    states_expanded: int
    strategy: str

    def summary(self) -> dict:
        return {
            "strategy": self.strategy,
            "start_cost": list(self.start_cost),
            "final_cost": list(self.cost),
            "steps": len(self.path),
            "is_tree": self.is_tree,
            "reached_target": self.reached_target,
            "explains": self.explains,
            "states_expanded": self.states_expanded,
        }


def _graph_index(weak: bool) -> int:
    return 1 if weak else 0


def _arcset(G: nx.DiGraph) -> set:
    return set(G.edges())


def _cost_fn(mode: str, target: nx.DiGraph | None) -> Callable[[nx.DiGraph], tuple]:
    if mode == "target":
        if target is None:
            raise ValueError("cost='target' needs a target tree")
        return lambda N: target_distance(N, target)
    if mode == "tree":
        return tree_likeness
    raise ValueError(f"unknown cost {mode!r}")


def _prepare(N, target, weak, cost, reduce_first):
    idx = _graph_index(weak)
    start = reduce_twins(N) if reduce_first else N
    reference = _arcset(fast_bmg(start)[idx])
    mode = cost if cost is not None else ("target" if target is not None else "tree")
    return start, reference, idx, _cost_fn(mode, target)


def _successors(
    state,
    reference,
    idx,
    kinds,
    leaves,
    limit,
    rng,
    reduce_each=True,
    shuffle=False,
    strict=True,
):
    """Yield ``(move, successor, explains)``.

    With ``strict=True`` only successors that still explain the reference
    graph are produced -- the invariant the work program asks for. With
    ``strict=False`` every successor is produced and the caller is responsible
    for only *accepting* explaining states; that relaxation matters because
    some instances have no explaining neighbour at all (see ``TASK2.md``).

    Twin merging is folded into every step: it is free (provably graph
    preserving) and keeps the states small, which matters because a single
    BIC-cherry expansion of 7 leaves already offers several hundred moves.
    """
    moves = enumerate_moves(state, kinds)
    if limit is not None and len(moves) > limit:
        moves = rng.sample(moves, limit)
    elif shuffle:
        moves = list(moves)
        rng.shuffle(moves)
    for move in moves:
        try:
            M = apply_move(state, move, leaves)
        except (ValueError, nx.NetworkXError):
            continue
        if reduce_each:
            M = reduce_twins(M)
        ok = _arcset(fast_bmg(M)[idx]) == reference
        if ok or not strict:
            yield move, M, ok


# ---------------------------------------------------------------------------
# greedy
# ---------------------------------------------------------------------------

def greedy_simplify(
    N: nx.DiGraph,
    target: nx.DiGraph | None = None,
    weak: bool = False,
    kinds: tuple = DEFAULT_MOVES,
    cost: str | None = None,
    max_steps: int = 300,
    plateau_budget: int = 15,
    first_improvement: bool = True,
    candidate_limit: int | None = 400,
    reduce_first: bool = True,
    restarts: int = 1,
    strict: bool = True,
    seed: int = 0,
) -> SearchResult:
    """Record-to-record descent over the graph-preserving moves.

    At every step the cheapest not-yet-visited successor is taken, *even if it
    is not an improvement*: strict steepest descent stops on the first plateau,
    and the plateaus of this landscape are large (many moves only rearrange
    the duplicated cherry vertices). A run ends after ``plateau_budget``
    consecutive non-improving steps and the best state ever seen is returned.
    ``restarts > 1`` repeats the run with a reshuffled move order and keeps
    the best outcome.
    """
    start, reference, idx, cost_of = _prepare(N, target, weak, cost, reduce_first)
    leaves = {v for v in start if start.out_degree(v) == 0}
    start_cost = cost_of(start)

    overall = None
    expanded = 0

    for run in range(max(1, restarts)):
        rng = random.Random(seed + run)
        shuffle = run > 0
        current, value = start, start_cost
        path: list[Move] = []
        record = (start_cost, [], start)
        seen = {canonical_key(start)}
        stale = 0

        for _ in range(max_steps):
            best_ok = None   # cheapest successor that still explains the graph
            best_any = None  # cheapest successor, explaining or not
            for move, M, ok in _successors(
                current, reference, idx, kinds, leaves, candidate_limit, rng,
                shuffle=shuffle, strict=(strict is True),
            ):
                expanded += 1
                new = cost_of(M)
                if best_any is not None and new >= best_any[0] and (
                    not ok or (best_ok is not None and new >= best_ok[0])
                ):
                    continue
                key = canonical_key(M)
                if key in seen:
                    continue
                cand = (new, key, move, M, ok)
                if best_any is None or new < best_any[0]:
                    best_any = cand
                if ok and (best_ok is None or new < best_ok[0]):
                    best_ok = cand
                    if first_improvement and new < value:
                        break

            # strict  -- never leave the explaining part of the search space
            # False   -- ignore the invariant, only the recorded state matters
            # "prefer" -- take an explaining step whenever one improves, and
            #            detour through non-explaining networks otherwise
            if strict is True:
                best = best_ok
            elif strict is False:
                best = best_any
            elif best_ok is not None and best_ok[0] < value:
                best = best_ok
            else:
                best = best_any if best_any is not None else best_ok
            if best is None:
                break

            new_cost, key, move, M, ok = best
            seen.add(key)
            current, value = M, new_cost
            path = path + [move]
            if ok and new_cost < record[0]:
                record = (new_cost, path, M)
                stale = 0
            else:
                stale += 1
                if stale > plateau_budget:
                    break
            if ok and target is not None and new_cost[0] == 0 and same_phylogeny(current, target):
                record = (new_cost, path, M)
                break

        if overall is None or record[0] < overall[0]:
            overall = record
        if overall[0][0] == 0:
            break

    value, path, network = overall
    return _finish(network, path, value, start_cost, target, reference, idx, expanded, "greedy")


# ---------------------------------------------------------------------------
# beam
# ---------------------------------------------------------------------------

def beam_simplify(
    N: nx.DiGraph,
    target: nx.DiGraph | None = None,
    weak: bool = False,
    kinds: tuple = DEFAULT_MOVES,
    cost: str | None = None,
    width: int = 8,
    max_steps: int = 80,
    candidate_limit: int | None = 250,
    reduce_first: bool = True,
    strict: bool = True,
    seed: int = 0,
) -> SearchResult:
    """Beam search: keep the ``width`` cheapest states of every level."""
    rng = random.Random(seed)
    start, reference, idx, cost_of = _prepare(N, target, weak, cost, reduce_first)

    leaves = {v for v in start if start.out_degree(v) == 0}
    start_cost = cost_of(start)
    beam = [(start_cost, [], start)]
    seen = {canonical_key(start)}
    best = (start_cost, [], start)
    expanded = 0

    for _ in range(max_steps):
        children = []
        for _value, path, state in beam:
            for move, M, ok in _successors(
                state, reference, idx, kinds, leaves, candidate_limit, rng, strict=strict
            ):
                expanded += 1
                key = canonical_key(M)
                if key in seen:
                    continue
                seen.add(key)
                value = cost_of(M)
                children.append((value, path + [move], M))
                if ok and value < best[0]:
                    best = (value, path + [move], M)
        if not children:
            break
        children.sort(key=lambda item: (item[0], len(item[1])))
        beam = children[:width]
        if target is not None and best[0][0] == 0 and same_phylogeny(best[2], target):
            break

    value, path, network = best
    return _finish(network, path, value, start_cost, target, reference, idx, expanded, "beam")


# ---------------------------------------------------------------------------
# complete search
# ---------------------------------------------------------------------------

def bfs_simplify(
    N: nx.DiGraph,
    target: nx.DiGraph | None = None,
    weak: bool = False,
    kinds: tuple = DEFAULT_MOVES,
    cost: str | None = None,
    max_states: int = 3000,
    max_depth: int = 25,
    reduce_first: bool = True,
    strict: bool = True,
    seed: int = 0,
) -> SearchResult:
    """Exhaustive BFS over the graph-preserving move graph (small instances)."""
    rng = random.Random(seed)
    start, reference, idx, cost_of = _prepare(N, target, weak, cost, reduce_first)

    leaves = {v for v in start if start.out_degree(v) == 0}
    start_cost = cost_of(start)
    queue = deque([(start, [], 0)])
    seen = {canonical_key(start)}
    best = (start_cost, [], start)
    expanded = 0

    while queue and expanded < max_states:
        state, path, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for move, M, _ok in _successors(
            state, reference, idx, kinds, leaves, None, rng, strict=strict
        ):
            expanded += 1
            key = canonical_key(M)
            if key in seen:
                continue
            seen.add(key)
            value = cost_of(M)
            if value < best[0]:
                best = (value, path + [move], M)
            if target is not None and same_phylogeny(M, target):
                return _finish(
                    M, path + [move], value, start_cost, target, reference, idx, expanded, "bfs"
                )
            queue.append((M, path + [move], depth + 1))
            if expanded >= max_states:
                break

    value, path, network = best
    return _finish(network, path, value, start_cost, target, reference, idx, expanded, "bfs")


def _finish(network, path, value, start_cost, target, reference, idx, expanded, strategy):
    return SearchResult(
        network=network,
        path=path,
        cost=value,
        start_cost=start_cost,
        is_tree=is_phylogenetic_tree(network),
        reached_target=bool(target is not None and same_phylogeny(network, target)),
        explains=_arcset(fast_bmg(network)[idx]) == reference,
        states_expanded=expanded,
        strategy=strategy,
    )


STRATEGIES = {
    "greedy": greedy_simplify,
    "beam": beam_simplify,
    "bfs": bfs_simplify,
}


def simplify(N: nx.DiGraph, strategy: str = "greedy", **kwargs) -> SearchResult:
    """Dispatch by strategy name."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}, pick one of {sorted(STRATEGIES)}")
    return STRATEGIES[strategy](N, **kwargs)


# ---------------------------------------------------------------------------
# the recommended pipeline
# ---------------------------------------------------------------------------

RECOMMENDED = {
    "strategy": "greedy",
    "cost": "target",
    "kinds": EXTENDED_MOVES,
    "strict": "prefer",
    "restarts": 3,
}


def simplify_bmg(
    bmg: nx.DiGraph,
    target: nx.DiGraph | None = None,
    restricted_bmg: bool = True,
    **overrides,
) -> SearchResult:
    """Graph in, simplified explaining network out -- the whole of task 2 in one call.

    Builds the explaining network of ``bmg`` with the default expansion (the
    *restricted* one, see :data:`task_2_utils.expansions.DEFAULT_EXPANSION`) and runs
    the :data:`RECOMMENDED` search on it. ``target`` defaults to the least
    resolved tree of ``bmg``; pass ``target=None`` together with
    ``cost="tree"`` to search without knowing where to go.

    Raises ``ValueError`` if ``bmg`` is not a BMG and no target is supplied,
    because then there is no least resolved tree to aim at.
    """
    from task_2_utils.expansions import explaining_network
    from utils.lrt import lrt_from_bmg

    settings = {**RECOMMENDED, **overrides}
    strategy = settings.pop("strategy")

    if target is None and settings.get("cost") == "target":
        target = lrt_from_bmg(bmg)
        if target is None:
            raise ValueError(
                "the input graph has no least resolved tree (it is not a BMG); "
                "pass an explicit target or use cost='tree'"
            )

    return STRATEGIES[strategy](bic_cherry_expansion(bmg, restricted=restricted_bmg), target=target, **settings)
