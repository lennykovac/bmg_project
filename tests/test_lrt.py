"""utils/lrt.py"""
import networkx as nx
import pytest

from tests.helpers import colored_graph, colored_tree, edges
from utils.graph_utils import bmg_from_network, same_phylogeny
from utils.lrt import (
    aho_build, contract_vertex, informative_triples, is_least_resolved, lrt_by_contraction,
    lrt_from_bmg, redundant_edges,
)


def cluster_set(T):
    from utils.graph_utils import clusters
    return set(clusters(T).values())


def test_informative_triples(example_tree):
    G = bmg_from_network(example_tree)
    assert informative_triples(G) == {("b1", "a1", "a2")}      # b1->a1 in E, b1->a2 not


def test_informative_triples_empty_for_star(star_bmg):
    assert informative_triples(star_bmg) == set()


def test_aho_build_consistent():
    T = aho_build(["a", "b", "c", "d"], {("a", "b", "c"), ("a", "b", "d")})
    assert {frozenset({"a", "b"}), frozenset("abcd")} <= cluster_set(T)
    assert T.in_degree("rho") == 0


def test_aho_build_inconsistent_and_trivial():
    assert aho_build(["a", "b", "c"], {("a", "b", "c"), ("a", "c", "b")}) is None
    assert aho_build(["a"], set()).number_of_nodes() == 1


def test_lrt_hand_example():
    T = colored_tree([("r", "x"), ("r", "y"), ("r", "a3"), ("x", "a1"), ("x", "b1"), ("y", "a2"), ("y", "b2")],
                     {"a1": "a", "a2": "a", "a3": "a", "b1": "b", "b2": "b"})
    G = bmg_from_network(T)
    L = lrt_from_bmg(G)
    assert same_phylogeny(L, T) and is_least_resolved(L, G)
    assert all(L.nodes[v]["color"] == G.nodes[v]["color"] for v in G)


def test_lrt_of_star(star_bmg):
    L = lrt_from_bmg(star_bmg)
    assert L.number_of_nodes() == 5 and L.out_degree("rho") == 4


def test_non_bmg_rejected():
    G = colored_graph({"a": 1, "a2": 1, "b": 2, "b2": 2}, [("a", "b"), ("b", "a"), ("b2", "a"), ("b2", "a2")])
    with pytest.raises(ValueError):
        lrt_from_bmg(G)


def test_redundant_edges_and_contraction():
    T = colored_tree([("r", "u"), ("r", "a2"), ("u", "w"), ("u", "a3"), ("w", "a1"), ("w", "b1")],
                     {"a1": "A", "a2": "A", "a3": "A", "b1": "B"})
    G = bmg_from_network(T)
    assert redundant_edges(T, G) == [("r", "u")]
    L = lrt_by_contraction(T, G)
    assert "u" not in L and set(L.successors("r")) == {"w", "a2", "a3"}
    assert edges(bmg_from_network(L)) == edges(G)
    assert is_least_resolved(L, G) and not is_least_resolved(T, G)


def test_contract_vertex():
    T = nx.DiGraph([("r", "u"), ("u", "a"), ("u", "b"), ("r", "c")])
    contract_vertex(T, "u")
    assert set(T.successors("r")) == {"a", "b", "c"}


def test_single_child_vertex_is_removed():
    """non-phylogenetic input: the edge into a single-child vertex is always redundant"""
    T = colored_tree([("top", "r"), ("r", "a"), ("r", "b")], {"a": 1, "b": 2})
    L = lrt_by_contraction(T, bmg_from_network(T))
    assert L.number_of_nodes() == 3
    assert same_phylogeny(L, colored_tree([("root", "a"), ("root", "b")], {}))


def test_three_constructions_agree(instances):
    from asymmetree.analysis import lrt_from_tree
    for d in instances:
        L1 = lrt_from_bmg(d.bmg)
        L2 = lrt_by_contraction(d.gene_tree, d.bmg)
        assert same_phylogeny(L1, L2)
        assert is_least_resolved(L1, d.bmg)
        A = lrt_from_tree(d.original_gene_tree)
        below = {}
        for v in A.postorder():
            below[v] = frozenset([v.label]) if not v.children else frozenset().union(*(below[c] for c in v.children))
        assert cluster_set(L1) == set(below.values())
