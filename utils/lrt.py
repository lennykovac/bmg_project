from dataclasses import dataclass, field
from typing import Hashable

import networkx as nx

from utils.graph_editing import contract_edge, try_edit
from utils.graph_utils import root_from_network
from utils.network_search import make_still_valid


@dataclass
class LrtReport:
    mode: str = "bmg"
    contracted_edges: list = field(default_factory=list)  # [(u, v), ...]
    rounds_run: int = 0


def _internal_edges(tree: nx.DiGraph):
    """
    Returns Edges (u, v) where v is NOT a leaf -- only these can be contracted
    """

    return [(u, v) for u, v in tree.edges if tree.out_degree(v) > 0]


def compute_lrt(tree: nx.DiGraph, mode: str = "bmg", max_rounds: int = 200) -> tuple:
    """
    Repeatedly contracts any redundant internal edge
    until a fixed point is reached. 

    Every attempt is guarded by `try_edit(..., still_valid=...)`
    a contraction that would change the (weak) BMG is never accepted.

    Returns (T_star, LrtReport). T_star is a NEW tree (the input `tree`
    is never mutated, because of how `try_edit` works).
    """

    still_valid = make_still_valid(mode)
    report = LrtReport(mode=mode)

    for round_i in range(max_rounds):
        report.rounds_run = round_i + 1
        progressed = False
        for u, v in _internal_edges(tree):
            result, applied = try_edit(tree, contract_edge, u, v, still_valid=still_valid)
            if applied:
                tree = result
                report.contracted_edges.append((u, v))
                progressed = True
                break  # topology changed, restart the edge scan
        if not progressed:
            break

    return tree, report


def is_least_resolved(tree: nx.DiGraph, mode: str = "bmg") -> bool:
    """
    Directly checks Def. 6: no remaining internal edge can be
    contracted without changing the (weak) BMG. Used in the tests to
    confirm that compute_lrt really reached a genuine fixed point (not
    just stopped due to max_rounds).
    """

    still_valid = make_still_valid(mode)
    for u, v in _internal_edges(tree):
        _, applied = try_edit(tree, contract_edge, u, v, still_valid=still_valid)
        if applied:
            return False
    return True
