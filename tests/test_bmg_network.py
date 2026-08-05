import networkx as nx
from utils.graph_utils import (
    bmg_from_network,
    wbmg_from_network,
    transform,
    check_sicorinhub,
)
import pytest
from utils.tree_utils import create_gene_tree


@pytest.fixture
def sample_graph_1():
    # here, bmg != wbmg
    G = nx.DiGraph()

    G.add_node(0, label="0", reconc="0")
    G.add_node(1, label="1", reconc="0")
    G.add_node(2, label="2", reconc="0")
    G.add_node(3, label="3", reconc="0")
    G.add_node(4, label="4", reconc="0")
    G.add_node(5, label="5", reconc="1")
    G.add_node(6, label="6", reconc="0")

    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4), (1, 4), (3, 5), (3, 6), (2, 5)])

    return G


def test_sicorinhub():
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
