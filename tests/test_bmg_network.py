import networkx as nx
from utils.bic_cherry import bic_cherry
from utils.graph_utils import (
    bmg_from_network,
    wbmg_from_network,
    transform,
    check_sicorinhub,
)
import pytest
from utils.tree_utils import create_gene_tree
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
    species_tree_age = 1
    for i in range(10):
        trees = create_gene_tree(species, species_tree_age)

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
    species_tree_age = 1
    for i in range(10):
        # asymmetree uses Tree class, our methods use nx.DiGraph
        tree_class = create_gene_tree(species, species_tree_age)
        gene_tree = tree_class.gene_tree
        tree = tree_class.original_gene_tree
        bmg = bmg_from_network(gene_tree)
        wbmg = wbmg_from_network(gene_tree)

        assert nx.is_isomorphic(bmg_from_tree(tree), bmg)
        assert nx.is_isomorphic(bmg, wbmg)


# TODO: test if bmg from cherry network is fully connected bmg
def test_bmg_from_cherry():
    species = 2
    species_tree_age = 1
    for i in range(2):
        tree = create_gene_tree(species, species_tree_age).gene_tree

        network = transform(tree, 2)

        cherry, pairs = bic_cherry(network)

        resulting_bmg = bmg_from_network(cherry)

        all_pairs = [edge for (u, v) in pairs for edge in [(u, v), (v, u)]]

        assert set(all_pairs) == set(resulting_bmg.edges())


def test_bmg_from_known_cherry(sample_graph_2):

    network = sample_graph_2

    cherry, pairs = bic_cherry(network)

    resulting_bmg = bmg_from_network(cherry)

    all_pairs = [edge for (u, v) in pairs for edge in [(u, v), (v, u)]]

    assert set(all_pairs) == set(resulting_bmg.edges())
