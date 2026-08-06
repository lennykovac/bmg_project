from itertools import permutations
import networkx as nx
from utils.graph_utils import (
    bmg_from_network,
    transform,
    leaves_from_network,
    print_graph_diff,
)
from utils.tree_utils import create_gene_tree
from utils.bic_cherry import bic_cherry_extension, bic_cherry
import pytest


@pytest.fixture
def sample_bmg_1():
    G = nx.DiGraph()

    G.add_node(0, label="0", color="0")
    G.add_node(1, label="1", color="1")
    G.add_node(2, label="2", color="1")
    G.add_node(3, label="3", color="0")

    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 0), (3, 2)])

    return G


# TODO: convert to test with generated inputs
def test_bic_cherry(sample_bmg_1):

    cherry_network, pairs = bic_cherry(sample_bmg_1)

    # check leaves are same as in bmg
    assert set(sample_bmg_1.nodes()) == set(leaves_from_network(cherry_network))
    # check bmg from cherry_network is fully connected bmg
    full_bmg = (
        nx.DiGraph()
    )  # full bmg wrong! Can only have edges between differently colored nodes!!
    full_bmg.add_nodes_from(sample_bmg_1.nodes(data=True))
    full_bmg.add_edges_from(
        edge for (u, v) in pairs for edge in [(u, v), (v, u)]
    )  # add both directions for all mismatch colored nodes

    print(pairs)
    print_graph_diff(bmg_from_network(cherry_network), full_bmg)
    assert nx.is_isomorphic(bmg_from_network(cherry_network), full_bmg)


# TODO: convert to test with generated inputs
def test_bic_cherry_extension(sample_bmg_1):
    network = bic_cherry_extension(sample_bmg_1)

    new_bmg = bmg_from_network(network)

    assert nx.is_isomorphic(sample_bmg_1, new_bmg)
