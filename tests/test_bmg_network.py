import networkx as nx
import random 
import pytest
from utils.bic_cherry import bic_cherry, restricted_bic_cherry_extension, wbmg_cherry_extension
from utils.graph_utils import (
    bmg_from_network,
    lca_dict_from_network,
    leaves_from_network,
    wbmg_from_network,
    transform,
    check_sicorinhub,
)

from utils.tree_utils import create_gene_tree_n_leaves
from asymmetree.analysis import bmg_from_tree


@pytest.fixture
def sample_graph_1():
    # here, bmg != wbmg
    G = nx.DiGraph()

    G.add_node(0, label="0", color="0")
    G.add_node(1, label="1", color="0")
    G.add_node(2, label="2", color="0")
    G.add_node(3, label="3", color="0")
    G.add_node(4, label="4", color="0")
    G.add_node(5, label="5", color="1")
    G.add_node(6, label="6", color="0")

    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4), (1, 4), (3, 5), (3, 6), (2, 5)])

    return G


@pytest.fixture
def sample_graph_2():
    # here, bmg != wbmg
    G = nx.DiGraph()

    G.add_node(0, label="0", color="0")
    G.add_node(1, label="1", color="1")
    G.add_node(2, label="2", color="0")
    G.add_node(3, label="3", color="1")
    G.add_node(4, label="4", color="0")
    G.add_node(5, label="5", color="1")
    G.add_node(6, label="6", color="0")

    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4), (1, 4), (3, 5), (3, 6), (2, 5)])

    return G


def test_bmg_properties():
    # generate 10 random gene networks
    species = 10
    leaves = 20
    species_tree_age = 1
    for i in range(10):
        trees = create_gene_tree_n_leaves(leaves, species, species_tree_age)

        gene_tree_di_graph = trees.gene_tree

        G_transformed = transform(gene_tree_di_graph, 10)
        # compute the bmg
        bmg = bmg_from_network(G_transformed)
        wbmg = wbmg_from_network(G_transformed)
        # check self-loop free
        for n in bmg.nodes:
            assert not bmg.has_edge(n, n)
        for n in wbmg.nodes:
            assert not wbmg.has_edge(n, n)
        # check sicor-in-hub property
        assert check_sicorinhub(bmg)
        assert check_sicorinhub(wbmg)
        # check proper coloring (edges connecting different colors)
        nodes_bmg = bmg.nodes(data=True)
        for u, v in bmg.edges():
            assert nodes_bmg[u]["color"] != nodes_bmg[v]["color"]
        nodes_wbmg = wbmg.nodes(data=True)
        for u, v in wbmg.edges():
            assert nodes_wbmg[u]["color"] != nodes_wbmg[v]["color"]


def test_bmg_structure(sample_graph_1):

    bmg = bmg_from_network(sample_graph_1)
    actual_nodes = list(bmg.nodes(data=True))
    expected_nodes = [(4, {"color": "0"}), (5, {"color": "1"}), (6, {"color": "0"})]
    assert actual_nodes == expected_nodes

    actual_edges = list(bmg.edges())
    expected_edges = [(4, 5), (5, 6), (6, 5)]
    assert actual_edges == expected_edges


def test_wbmg_structure(sample_graph_1):

    bmg = wbmg_from_network(sample_graph_1)
    actual_nodes = list(bmg.nodes(data=True))
    expected_nodes = [(4, {"color": "0"}), (5, {"color": "1"}), (6, {"color": "0"})]
    assert actual_nodes == expected_nodes

    actual_edges = list(bmg.edges())
    expected_edges = [(4, 5), (5, 4), (5, 6), (6, 5)]
    assert actual_edges == expected_edges


# for trees, bmg=wbmg and bmg_from_tree of Asymmetree should also return the same result
def test_bmg_wbmg_tree():
    species = 10
    leaves = 20
    species_tree_age = 1
    for i in range(10):
        # asymmetree uses Tree class, our methods use nx.DiGraph
        tree_class = create_gene_tree_n_leaves(leaves, species, species_tree_age)
        gene_tree = tree_class.gene_tree
        tree = tree_class.original_gene_tree
        bmg = bmg_from_network(gene_tree)
        wbmg = wbmg_from_network(gene_tree)

        assert nx.is_isomorphic(bmg_from_tree(tree), bmg)
        assert nx.is_isomorphic(bmg, wbmg)


# test if bmg from cherry network is actually a maximally connected bmg
def test_bmg_from_cherry():
    species = 2
    leaves = 10
    species_tree_age = 1
    for i in range(2):
        tree = create_gene_tree_n_leaves(leaves, species, species_tree_age).gene_tree

        network = transform(tree, 2)

        cherry, pairs = bic_cherry(network)

        resulting_bmg = bmg_from_network(cherry)

        all_pairs = [edge for (u, v) in pairs for edge in [(u, v), (v, u)]]

        assert set(all_pairs) == set(resulting_bmg.edges())


def test_bmg_from_known_cherry(sample_graph_1):

    network = sample_graph_1

    cherry, pairs = bic_cherry(network)

    resulting_bmg = bmg_from_network(cherry)

    all_pairs = [edge for (u, v) in pairs for edge in [(u, v), (v, u)]]

    assert set(all_pairs) == set(resulting_bmg.edges())


# TODO: test if lca_dict function works correctly
def test_lca_dict_from_network(sample_graph_1):

    reach = {
        n: nx.descendants(sample_graph_1, n) for n in sample_graph_1.nodes
    }  # pre-compute reachability in network as dict[{v:descendants of v}]

    leaves = leaves_from_network(sample_graph_1)

    lcas = lca_dict_from_network(sample_graph_1, reach, leaves)

    # returns both directions of lca tuples
    correct_lcas = {(5, 4): {1, 2}, (4, 5): {1, 2}, (5, 6): {3}, (6, 5): {3}}

    assert lcas == correct_lcas


@pytest.mark.exhaustive
def test_wbmg_from_network_with_trees():
    AMOUNT = 50

    for _ in range(AMOUNT):
        t =  create_gene_tree_n_leaves(leaves=random.randint(3,100), species=2)
        assert set(wbmg_from_network(t.gene_tree).edges()) == set(bmg_from_network(t.gene_tree).edges())


class TestMinimalExample:
    """
    Just an example i did on paper for debugging 
    """
    @pytest.fixture
    def example_network(self):
        #          r
        #        /   \
        #       a     b
        #      / \   / \
        #     c   \ /   \
        #    / \   y     |
        #   z   \        |
        #        x ------+
        N = nx.DiGraph()
        N.add_edges_from([
            ("r", "a"), ("r", "b"),
            ("a", "c"), ("a", "y"),
            ("b", "y"), ("b", "x"),
            ("c", "z"), ("c", "x"),
        ])
        nx.set_node_attributes(N, None, "color")
        N.nodes["x"]["color"] = "0"
        N.nodes["y"]["color"] = "1"
        N.nodes["z"]["color"] = "1"
        return N

    def test_lca_minimal_network(self, example_network):
        N = example_network
        reach = {n: nx.descendants(N, n) for n in N}
        lcas = lca_dict_from_network(N, reach, leaves_from_network(N))
        assert lcas[("x", "y")] == {"a", "b"}   # two incomparable LCAs
        assert lcas[("x", "z")] == {"c"}


    def test_strict_bmg_minimal_network(self, example_network):
        bmg = bmg_from_network(example_network)
        # y is not a best match of x: c in LCA(x,z) lies strictly below a in LCA(x,y)
        assert set(bmg.edges()) == {("x", "z"), ("y", "x"), ("z", "x")}


    def test_wbmg_minimal_network(self, example_network):
        wbmg = wbmg_from_network(example_network)
        # Q(x, 1) = {b, c}; LCA(x,y) = {a, b} meets Q in b, so x -> y is a weak match
        assert set(wbmg.edges()) == {("x", "y"), ("x", "z"), ("y", "x"), ("z", "x")}


class TestRunExperiment:
    """
    Task 1d: WBMG of N_og vs. WBMG of the network constructed from it
    """
    @pytest.fixture
    def minimal_wbmg(self):
        # smallest graph where variant (b) fails, it is even the BMG of the tree (x2, (x1, y))
        #   x1 -> y <- x2
        #   y  -> x1
        G = nx.DiGraph()
        G.add_nodes_from([("x1", {"color": "0"}), ("x2", {"color": "0"}), ("y", {"color": "1"})])
        G.add_edges_from([("x1", "y"), ("x2", "y"), ("y", "x1")])
        return G

    def test_restricted_extension_minimal_fails(self, minimal_wbmg):
        network = restricted_bic_cherry_extension(minimal_wbmg)
        # q:y|x1 blocks p:y|x2, but its twin p:x2|y stays in Q(y, 0), so y -> x2 survives
        assert set(wbmg_from_network(network).edges()) == set(minimal_wbmg.edges()) | {("y", "x2")}

    def test_wbmg_cherry_extension_minimal(self, minimal_wbmg):
        #          R
        #        /   \
        #       |   q:x2|y|x1
        #       |    /     \
        #        \  /       x2
        #       p:x1|y
        #        /  \
        #       x1   y
        network = wbmg_cherry_extension(minimal_wbmg)
        assert set(network.edges()) == {
            ("R", "p:x1|y"), ("R", "q:x2|y|x1"),
            ("q:x2|y|x1", "x2"), ("q:x2|y|x1", "p:x1|y"),
            ("p:x1|y", "x1"), ("p:x1|y", "y"),
        }
        assert set(wbmg_from_network(network).edges()) == set(minimal_wbmg.edges())

    def test_sink_free_but_no_wbmg(self):
        # x1 -> y1 -> x2 -> y2 -> x1 is a network BMG (sicor-in-hub), but a witness of x1 -> y1
        # would need blockers below it down the whole cycle until one reaches x1 and y2 again
        G = nx.DiGraph()
        G.add_nodes_from([("x1", {"color": "0"}), ("x2", {"color": "0"}), ("y1", {"color": "1"}), ("y2", {"color": "1"})])
        G.add_edges_from([("x1", "y1"), ("y1", "x2"), ("x2", "y2"), ("y2", "x1")])

        assert set(bmg_from_network(restricted_bic_cherry_extension(G)).edges()) == set(G.edges())
        with pytest.raises(ValueError):
            wbmg_cherry_extension(G)

    def test_run_trial(self):
        leaves = 10
        hybrids = 5
        for _ in range(10):
            #1. create a tree
            gene_tree = create_gene_tree_n_leaves(leaves, 2, max_attempts=10).gene_tree
            #2. transform into network with hybrid_nodes
            N_og = transform(gene_tree, hybrids)
            #3. get weak_best_match_graph from N_og and from the network explaining it
            wbmg_og = wbmg_from_network(N_og)
            wbmg_bic = wbmg_from_network(wbmg_cherry_extension(wbmg_og))

            assert set(wbmg_og.edges()) == set(wbmg_bic.edges())
