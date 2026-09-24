"""The task 2 pipeline: generate, explain, edit, search, analyse.

One *instance* is one AsymmeTree gene tree. For it we

2a  build the tree-BMG ``G`` (and the WBMG, which coincides with it on trees)
    and the least resolved tree ``T*`` -- the target;
2b  build the BIC-cherry expansions ``N`` (restricted / plain)
    and verify that they explain ``G``;
2c  enumerate the edit moves on ``N``;
2d  test single moves and short combinations of moves for invariance of the
    (weak) best match graph;
2e  run the simplification heuristics from ``task_2_utils.search``;
2f  keep whatever failed, and shrink it to a minimal counterexample.
"""

from __future__ import annotations

import random
from typing import Callable, Iterable

import networkx as nx

from utils.bic_cherry import bic_cherry_expansion
from task_2_utils.expansions import DEFAULT_EXPANSION, EXPANSIONS
from utils.graph_utils import (
    bmg_from_network,
    check_color_sink_free,
    check_sicorinhub,
    is_phylogenetic_tree,
)
from utils.lrt import lrt_from_bmg
from task_2_utils.graph_editing import (
    DEFAULT_MOVES,
    EXTENDED_MOVES,
    apply_move,
    enumerate_moves,
    legal_successors,
    reticulation_number,
)
from task_2_utils.search import STRATEGIES
from utils.tree_utils import create_gene_tree_n_leaves

__all__ = [
    "arcs",
    "explains",
    "single_move_scan",
    "pair_scan",
    "combination_scan",
    "compare_searches",
    "complete_bmg",
    "is_complete_bmg",
    "strict_search_fails",
    "analyse_instance",
    "minimize_bmg",
    "generate_instances",
    "SEARCH_CONFIGS",
    "SEARCH_CONFIGS_2E",
]


def arcs(G: nx.DiGraph) -> set:
    return set(G.edges())


def _both(N: nx.DiGraph) -> tuple[nx.DiGraph, nx.DiGraph]:
    """``(BMG, WBMG)`` of ``N`` -- the two calls the scans need side by side."""
    return bmg_from_network(N), bmg_from_network(N, weak=True)


def explains(N: nx.DiGraph, G: nx.DiGraph, weak: bool = False) -> bool:
    """Does the network ``N`` explain the colored graph ``G``?"""
    bmg, wbmg = _both(N)
    H = wbmg if weak else bmg
    return set(H.nodes) == set(G.nodes) and arcs(H) == arcs(G)


# ---------------------------------------------------------------------------
# 2d -- invariance of single moves and of short combinations
# ---------------------------------------------------------------------------

def single_move_scan(N: nx.DiGraph, kinds: tuple = DEFAULT_MOVES) -> dict:
    """For every legal move: does it keep the BMG / the WBMG?

    Runs on the fast engine (:mod:`task_2_utils.fast_edit`, cross-checked against
    :mod:`task_2_utils.graph_editing` in ``tests/test_fast_edit.py``). Returns
    per-kind counters and ``"ALL"``.
    """
    from task_2_utils.fast_edit import Net, apply, best_match_arcs, enumerate_candidates

    net = Net.from_nx(N)
    a0, w0 = best_match_arcs(net), best_match_arcs(net, weak=True)
    stats: dict = {}
    for move in enumerate_candidates(net, kinds):
        try:
            M, _ = apply(net, move, reduce=False)
        except ValueError:
            continue
        row = stats.setdefault(move.kind, {"total": 0, "bmg": 0, "wbmg": 0, "both": 0})
        keeps_b = best_match_arcs(M) == a0
        keeps_w = best_match_arcs(M, weak=True) == w0
        row["total"] += 1
        row["bmg"] += keeps_b
        row["wbmg"] += keeps_w
        row["both"] += keeps_b and keeps_w

    total = {"total": 0, "bmg": 0, "wbmg": 0, "both": 0}
    for row in stats.values():
        for k in total:
            total[k] += row[k]
    stats["ALL"] = total
    return stats


def pair_scan(N: nx.DiGraph, kinds: tuple = EXTENDED_MOVES, weak: bool = False,
              max_pairs: int = 50_000, seed: int = 0) -> dict:
    """Ordered pairs of legal moves: kept, and 'rescued' (broken, then repaired).

    Exhaustive while the number of pairs stays below ``max_pairs``; beyond
    that, first moves are sampled uniformly and *all* second moves of each
    sampled first move are checked (``exhaustive`` is then False). A BIC-cherry
    expansion of 12 genes already has ~2500 moves, i.e. ~6 million pairs.
    """
    from task_2_utils.fast_edit import Net, apply, best_match_arcs, enumerate_candidates

    net = Net.from_nx(N)
    reference = best_match_arcs(net, weak)
    firsts = []
    for m in enumerate_candidates(net, kinds):
        try:
            firsts.append(apply(net, m, reduce=False)[0])
        except ValueError:
            continue
    exhaustive = len(firsts) ** 2 <= max_pairs
    if not exhaustive:
        k = max(1, max_pairs // max(1, len(firsts)))
        firsts = random.Random(seed).sample(firsts, min(k, len(firsts)))

    checked = kept = rescued = 0
    for M1 in firsts:
        broke = best_match_arcs(M1, weak) != reference
        for m2 in enumerate_candidates(M1, kinds):
            try:
                M2, _ = apply(M1, m2, reduce=False)
            except ValueError:
                continue
            checked += 1
            if best_match_arcs(M2, weak) == reference:
                kept += 1
                rescued += broke
    return {"checked": checked, "kept": kept, "rescued": rescued,
            "exhaustive": exhaustive, "first_moves": len(firsts)}


def combination_scan(
    N: nx.DiGraph,
    depth: int = 2,
    max_sequences: int = 2000,
    weak: bool = False,
    kinds: tuple = DEFAULT_MOVES,
    rng: random.Random | None = None,
) -> dict:
    """Combinations of up to ``depth`` moves.

    Sequences are taken over *all* legal moves, not only the invariant ones,
    so the scan also finds the interesting case: a first move that destroys
    the graph followed by a second one that repairs it ("rescued" below).
    Exhaustive while the number of sequences stays under ``max_sequences``,
    uniformly sampled afterwards.
    """
    rng = rng or random.Random(0)
    bmg0, wbmg0 = _both(N)
    reference = arcs(wbmg0 if weak else bmg0)
    leaves = {v for v in N if N.out_degree(v) == 0}

    out = {}
    for d in range(1, depth + 1):
        checked = preserved = rescued = 0
        exhaustive = True

        def walk(state, remaining, broken):
            nonlocal checked, preserved, rescued
            if checked >= max_sequences:
                return
            moves = enumerate_moves(state, kinds)
            for move in moves:
                if checked >= max_sequences:
                    return
                try:
                    M = apply_move(state, move, leaves)
                except (ValueError, nx.NetworkXError):
                    continue
                bmg, wbmg = _both(M)
                ok = arcs(wbmg if weak else bmg) == reference
                if remaining == 1:
                    checked += 1
                    if ok:
                        preserved += 1
                        if broken:
                            rescued += 1
                else:
                    walk(M, remaining - 1, broken or not ok)

        walk(N, d, False)
        if checked >= max_sequences:
            exhaustive = False
        out[d] = {
            "checked": checked,
            "preserved": preserved,
            "rescued": rescued,
            "exhaustive": exhaustive,
        }
    return out


# ---------------------------------------------------------------------------
# the per-instance pipeline
# ---------------------------------------------------------------------------

#: The search configurations compared in the report, as
#: ``label -> (strategy, kwargs, max_leaves)``. ``kinds`` selects the move set,
#: ``strict`` whether intermediate networks have to keep explaining the graph,
#: and the last entry gates the configuration by instance size (beam search is
#: an order of magnitude more expensive than the greedy descent).
#: ``greedy_prefer_extended`` is the recommended configuration and is the one
#: :data:`task_2_utils.search.RECOMMENDED` / :func:`task_2_utils.search.simplify_bmg` use.
SEARCH_CONFIGS: dict = {
    "greedy_strict_named": (
        "greedy", {"kinds": DEFAULT_MOVES, "strict": True, "restarts": 3}, 99,
    ),
    "greedy_prefer_named": (
        "greedy", {"kinds": DEFAULT_MOVES, "strict": "prefer", "restarts": 3}, 99,
    ),
    "greedy_prefer_extended": (
        "greedy", {"kinds": EXTENDED_MOVES, "strict": "prefer", "restarts": 3}, 99,
    ),
    "beam_strict_named": (
        "beam",
        {"kinds": DEFAULT_MOVES, "strict": True, "width": 4, "max_steps": 25,
         "candidate_limit": 60},
        6,
    ),
}


#: The four configurations of 2e, as ``label -> (kinds, strict)``. The move set
#: and the invariant are the two knobs the work program leaves open; this is
#: their product. ``extended/prefer`` is :data:`task_2_utils.search.RECOMMENDED`.
SEARCH_CONFIGS_2E: dict = {
    "named/strict": (DEFAULT_MOVES, True),
    "named/prefer": (DEFAULT_MOVES, "prefer"),
    "extended/strict": (EXTENDED_MOVES, True),
    "extended/prefer": (EXTENDED_MOVES, "prefer"),
}


def compare_searches(
    N: nx.DiGraph,
    target: nx.DiGraph,
    configs: dict | None = None,
    restarts: int = 3,
    cost: str = "target",
) -> dict:
    """Run every configuration of ``configs`` on ``N`` and keep the results."""
    configs = SEARCH_CONFIGS_2E if configs is None else configs
    return {
        label: STRATEGIES["greedy"](
            N, target=target, cost=cost, kinds=kinds, strict=strict, restarts=restarts
        )
        for label, (kinds, strict) in configs.items()
    }


def complete_bmg(sizes: tuple[int, ...] = (1, 2)) -> nx.DiGraph:
    """The complete multicolored digraph with the given color class sizes.

    Its least resolved tree is the star, and its BIC-cherry expansion performs
    no extension (there is no missing arc to remove). For ``len(sizes) == 2``
    this is the *frozen* family of 2f: no single move preserves the BMG.
    """
    G = nx.DiGraph()
    colors = {}
    for c, count in enumerate(sizes):
        for i in range(count):
            v = f"c{c}_{i}"
            colors[v] = c
            G.add_node(v, color=c)
    G.add_edges_from((u, v) for u in colors for v in colors if colors[u] != colors[v])
    return G


def is_complete_bmg(G: nx.DiGraph) -> bool:
    """Is every bicolored ordered pair an arc?"""
    return G.number_of_edges() == sum(
        1 for u in G for v in G
        if u != v and G.nodes[u]["color"] != G.nodes[v]["color"]
    )


def strict_search_fails(
    G: nx.DiGraph,
    kinds: tuple = DEFAULT_MOVES,
    restricted: bool = True,
    restarts: int = 3,
) -> bool:
    """Does the *strict* greedy search miss ``T*`` on this graph?

    The predicate :func:`minimize_bmg` shrinks against: a graph is interesting
    exactly when the literal reading of the task (every intermediate network
    explains ``G``) cannot reach the target.
    """
    star = lrt_from_bmg(G)
    if star is None:
        return False
    return not STRATEGIES["greedy"](
        bic_cherry_expansion(G, restricted=restricted), target=star, cost="target",
        kinds=kinds, strict=True, restarts=restarts,
    ).reached_target


def analyse_instance(
    gene_tree: nx.DiGraph,
    expansions: Iterable[str] = ("restricted", "plain"),
    search_configs: dict | None = None,
    weak: bool = False,
    kinds: tuple = DEFAULT_MOVES,
    combination_depth: int = 2,
    combination_budget: int = 800,
    single_move_limit: int = 900,
    search_max_leaves: int = 9,
    combination_max_leaves: int = 5,
) -> dict:
    """Run 2a-2e for one gene tree and return a JSON-friendly record."""
    search_configs = SEARCH_CONFIGS if search_configs is None else search_configs

    # -- 2a ---------------------------------------------------------------
    bmg, wbmg = _both(gene_tree)
    record: dict = {
        "leaves": bmg.number_of_nodes(),
        "colors": len({c for _, c in bmg.nodes(data="color")}),
        "bmg_arcs": bmg.number_of_edges(),
        "bmg_equals_wbmg": arcs(bmg) == arcs(wbmg),
        "color_sink_free": check_color_sink_free(bmg),
        "sicor_in_hub": check_sicorinhub(bmg),
        "default_expansion": DEFAULT_EXPANSION,
        "expansions": {},
    }

    star = lrt_from_bmg(bmg)
    record["lrt_exists"] = star is not None
    if star is None:
        return record
    record["lrt_vertices"] = star.number_of_nodes()
    record["lrt_arcs"] = star.number_of_edges()
    record["lrt_explains_bmg"] = explains(star, bmg)

    target_graph = wbmg if weak else bmg

    for name in expansions:
        N = EXPANSIONS[name](target_graph)
        entry: dict = {
            "vertices": N.number_of_nodes(),
            "arcs": N.number_of_edges(),
            "reticulations": reticulation_number(N),
            "explains_bmg": explains(N, bmg),
            "explains_wbmg": explains(N, wbmg, weak=True),
            "n_moves": len(enumerate_moves(N, kinds)),
        }

        # -- 2d ------------------------------------------------------------
        if entry["n_moves"] <= single_move_limit:
            entry["single_moves"] = single_move_scan(N, kinds)
        if combination_depth >= 2 and record["leaves"] <= combination_max_leaves:
            entry["combinations"] = combination_scan(
                N,
                depth=combination_depth,
                max_sequences=combination_budget,
                weak=weak,
                kinds=kinds,
            )

        # -- 2e ------------------------------------------------------------
        entry["search"] = {}
        if record["leaves"] <= search_max_leaves:
            for label, config in search_configs.items():
                strategy, kwargs = config[0], config[1]
                gate = config[2] if len(config) > 2 else 99
                if record["leaves"] > gate:
                    continue
                result = STRATEGIES[strategy](N, target=star, weak=weak, cost="target", **kwargs)
                entry["search"][label] = result.summary()
                entry["search"][label]["path"] = [
                    [m.kind, [str(a) for a in m.args]] for m in result.path
                ]

        record["expansions"][name] = entry

    return record


# ---------------------------------------------------------------------------
# 2f -- minimal counterexamples
# ---------------------------------------------------------------------------

def _induced(G: nx.DiGraph, keep: set) -> nx.DiGraph:
    H = nx.DiGraph()
    for v in keep:
        H.add_node(v, color=G.nodes[v]["color"])
    for u, v in G.edges():
        if u in keep and v in keep:
            H.add_edge(u, v)
    return H


def minimize_bmg(bmg: nx.DiGraph, fails: Callable[[nx.DiGraph], bool]) -> nx.DiGraph:
    """Shrink ``bmg`` while ``fails`` stays true.

    Leaves are removed one at a time. The BMG of a tree restricted to a subset
    of its leaves is again a tree-BMG (the restriction of the explaining tree
    explains it), and the restriction is the induced subgraph -- *provided*
    the subgraph is still color-sink-free, which is checked. So the shrunken
    instance is a legitimate instance of the same problem.
    """
    current = bmg
    changed = True
    while changed and current.number_of_nodes() > 2:
        changed = False
        for v in sorted(current.nodes, key=str):
            keep = set(current.nodes) - {v}
            if len({current.nodes[u]["color"] for u in keep}) < 2:
                continue
            candidate = _induced(current, keep)
            if not check_color_sink_free(candidate):
                continue
            if lrt_from_bmg(candidate) is None:
                continue
            if fails(candidate):
                current = candidate
                changed = True
                break
    return current


# ---------------------------------------------------------------------------
# instance generation
# ---------------------------------------------------------------------------

def generate_instances(
    n: int,
    min_leaves: int = 4,
    max_leaves: int = 10,
    min_species: int = 2,
    max_species: int = 4,
    seed: int = 0,
) -> list:
    """``n`` AsymmeTree gene trees as ``nx.DiGraph`` (reproducible)."""
    import numpy as np

    rng = random.Random(seed)
    np.random.seed(seed)
    random.seed(seed)

    out = []
    while len(out) < n:
        species = rng.randint(min_species, max_species)
        leaves = rng.randint(max(species, min_leaves), max_leaves)
        d = create_gene_tree_n_leaves(leaves, species)
        tree = d.gene_tree
        bmg, _ = _both(tree)
        # degenerate draws (a single surviving color) carry no information
        if len({c for _, c in bmg.nodes(data="color")}) < 2:
            continue
        if not is_phylogenetic_tree(tree):
            continue
        out.append(tree)
    return out
