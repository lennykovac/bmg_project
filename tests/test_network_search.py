"""utils/network_search.py"""
import networkx as nx

from tests.helpers import edges
from utils.bic_cherry import bic_cherry_extension
from utils.graph_editing import normalize
from utils.graph_utils import bmg_from_network, same_phylogeny
from utils.lrt import lrt_from_bmg
from utils.network_search import (
    SearchReport, candidate_moves, canonical, edit_search, hybrid_excess, inner_clusters,
    make_score_guided, neighbours, reduce_to_tree, score_agnostic,
)


def test_hybrid_excess_and_inner_clusters():
    N = nx.DiGraph([("r", "p"), ("r", "q"), ("p", "a"), ("q", "a"), ("p", "b"), ("q", "c")])
    assert hybrid_excess(N) == 1
    assert inner_clusters(N) == {frozenset("abc"), frozenset("ab"), frozenset("ac")}


def test_guided_score_zero_exactly_at_lrt(star_bmg):
    T = lrt_from_bmg(star_bmg)
    score = make_score_guided(T)
    assert score(T) == 0
    N = bic_cherry_extension(star_bmg)
    normalize(N, set(star_bmg))
    assert score(N) > 0
    assert score_agnostic(N) > score_agnostic(T)


def test_candidate_moves_kinds(star_bmg):
    N = bic_cherry_extension(star_bmg)
    normalize(N, set(star_bmg))
    kinds = {name for name, _, _ in candidate_moves(N)}
    assert {"merge", "contract", "pull_up", "delete_parent_edge"} <= kinds
    assert {name for name, _, _ in candidate_moves(N, compound=False)} <= {"delete_parent_edge", "pull_up", "pull_down"}


def test_canonical_ignores_inner_names(example_tree):
    renamed = nx.relabel_nodes(example_tree, {"r": "root", "x": "cherry"})
    assert canonical(example_tree) == canonical(renamed)


def test_neighbours_are_normalized_and_distinct(star_bmg):
    N = bic_cherry_extension(star_bmg)
    leaves = set(star_bmg)
    normalize(N, leaves)
    keys = []
    for _, M, key in neighbours(N, leaves):
        keys.append(key)
        assert all(M.out_degree(v) != 1 for v in M)
        assert {v for v in M if M.out_degree(v) == 0} == leaves
    assert len(keys) == len(set(keys)) > 0


def test_star_needs_compound_moves(star_bmg):
    T = lrt_from_bmg(star_bmg)
    N = bic_cherry_extension(star_bmg)
    _, rep = edit_search(N, T, compound=False)
    assert not rep.success
    M, rep = edit_search(N, T, compound=True)
    assert rep.success and same_phylogeny(M, T)


def test_report_fields(star_bmg):
    T = lrt_from_bmg(star_bmg)
    _, rep = edit_search(bic_cherry_extension(star_bmg), T)
    assert isinstance(rep, SearchReport) and rep.mode == "bmg" and rep.score_name == "guided"
    assert rep.final_is_tree and rep.final_score == 0
    assert rep.steps == len(rep.path) and rep.evaluated >= rep.steps
    assert rep.final_size[0] < rep.start_size[0]


def test_search_result_always_explains_g(small_instances):
    for d in small_instances:
        T = lrt_from_bmg(d.bmg)
        for guided in (True, False):
            M, rep = edit_search(bic_cherry_extension(d.bmg), T, guided=guided, beam_width=2)
            assert edges(bmg_from_network(M)) == edges(d.bmg)
        assert rep.success


def test_reduce_to_tree_keeps_bmg(small_instances):
    for d in small_instances:
        M, accepted = reduce_to_tree(bic_cherry_extension(d.bmg))
        assert accepted >= 0
        assert edges(bmg_from_network(M)) == edges(d.bmg)
