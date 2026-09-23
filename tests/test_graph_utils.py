"""utils/graph_utils.py"""
import networkx as nx
import pytest

from tests.helpers import colored_graph, colored_tree, edges
from utils.graph_utils import (
    add_hybrid_node, bmg_from_network, check_color_sink_free, check_sicorinhub, clusters,
    insert_node_on_edge, is_phylogenetic_tree, lca_dict_from_network, leaves_from_network,
    root_from_network, same_phylogeny, transform, wbmg_from_network,
)


@pytest.fixture
def weak_not_strict():
    """y1 is a weak but not a strict best match of x"""
    N = nx.DiGraph([("r", "u"), ("r", "v"), ("u", "w"), ("u", "y1"), ("w", "x"), ("w", "y2"), ("v", "x"), ("v", "y1")])
    for n in N:
        N.nodes[n]["color"] = {"x": "A", "y1": "B", "y2": "B"}.get(n)
    return N


# ---------------------------------------------------------------- basics
def test_root_and_leaves(example_tree):
    assert root_from_network(example_tree) == "r"
    assert set(leaves_from_network(example_tree)) == {"a1", "a2", "b1"}
    with pytest.raises(ValueError):
        root_from_network(nx.DiGraph([("a", "c"), ("b", "c")]))


def test_lca_dict(weak_not_strict):
    N = weak_not_strict
    reach = {n: nx.descendants(N, n) for n in N}
    lca = lca_dict_from_network(N, reach, leaves_from_network(N))
    assert lca[("x", "y1")] == {"u", "v"}
    assert lca[("x", "y2")] == {"w"}
    assert ("y1", "y2") not in lca          # same color -> no entry


# ---------------------------------------------------------------- (weak) BMG
def test_bmg_on_tree(example_tree):
    assert edges(bmg_from_network(example_tree)) == {("a1", "b1"), ("b1", "a1"), ("a2", "b1")}


def test_strict_vs_weak_on_network(weak_not_strict):
    assert edges(bmg_from_network(weak_not_strict)) == {("x", "y2"), ("y1", "x"), ("y2", "x")}
    assert edges(wbmg_from_network(weak_not_strict)) == {("x", "y1"), ("x", "y2"), ("y1", "x"), ("y2", "x")}


def test_strict_subset_of_weak_and_equal_on_trees(instances):
    for d in instances:
        assert edges(bmg_from_network(d.gene_tree)) == edges(wbmg_from_network(d.gene_tree))
        N = transform(d.gene_tree, 3)
        assert edges(bmg_from_network(N)) <= edges(wbmg_from_network(N))


def test_bmg_vertices_are_the_leaves(instances):
    for d in instances[:5]:
        N = transform(d.gene_tree, 2)
        assert set(bmg_from_network(N).nodes) == set(leaves_from_network(N))


# ---------------------------------------------------------------- properties
def test_sicor_in_hub():
    good = colored_graph({"a1": 1, "a2": 1, "c": 2}, [("a1", "c"), ("a2", "c"), ("c", "a1")])
    bad = colored_graph({"a1": 1, "a2": 1, "c": 2}, [("a1", "c"), ("c", "a1")])
    assert check_sicorinhub(good) and not check_sicorinhub(bad)


def test_color_sink_free():
    good = colored_graph({"a": 1, "b": 2}, [("a", "b"), ("b", "a")])
    bad = colored_graph({"a": 1, "b": 2}, [("a", "b")])
    assert check_color_sink_free(good) and not check_color_sink_free(bad)


def test_clusters(example_tree):
    cl = clusters(example_tree)
    assert cl["x"] == {"a1", "b1"} and cl["r"] == {"a1", "a2", "b1"} and cl["a1"] == {"a1"}


def test_is_phylogenetic_tree(example_tree):
    assert is_phylogenetic_tree(example_tree)
    hybrid = nx.DiGraph([("r", "u"), ("r", "w"), ("u", "a"), ("w", "a"), ("u", "b"), ("w", "c")])
    single = nx.DiGraph([("r", "u"), ("u", "a"), ("u", "b")])
    assert not is_phylogenetic_tree(hybrid) and not is_phylogenetic_tree(single)


def test_same_phylogeny_ignores_names_and_order(example_tree):
    other = colored_tree([("root", "a2"), ("root", "k"), ("k", "b1"), ("k", "a1")], {})
    assert same_phylogeny(example_tree, other)
    different = colored_tree([("r", "x"), ("r", "b1"), ("x", "a1"), ("x", "a2")], {})
    assert not same_phylogeny(example_tree, different)


# ---------------------------------------------------------------- network generation
def test_insert_node_on_edge(example_tree):
    G = example_tree.copy()
    insert_node_on_edge("m", ("r", "x"), G)
    assert G.has_edge("r", "m") and G.has_edge("m", "x") and not G.has_edge("r", "x")
    with pytest.raises(ValueError):
        insert_node_on_edge("n", ("a1", "b1"), G)
    with pytest.raises(ValueError):
        insert_node_on_edge("m", ("x", "a1"), G)


def test_add_hybrid_node(example_tree):
    G = example_tree.copy()
    assert add_hybrid_node(("r", "a2"), ("x", "a1"), "d", "h", G)
    assert set(G.predecessors("h")) == {"x", "d"}
    assert nx.is_directed_acyclic_graph(G)
    H = example_tree.copy()
    assert not add_hybrid_node(("x", "a1"), ("r", "x"), "d", "h", H)   # would close a cycle
    assert edges(H) == edges(example_tree)


def test_transform_keeps_leaves_and_is_dag(instances):
    for d in instances[:8]:
        N = transform(d.gene_tree, 4)
        assert nx.is_directed_acyclic_graph(N)
        assert set(leaves_from_network(N)) == set(leaves_from_network(d.gene_tree))
        assert d.gene_tree.number_of_edges() < N.number_of_edges()
