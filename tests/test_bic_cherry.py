from itertools import permutations
import networkx as nx
from utils.graph_utils import (
    bmg_from_network,
    transform,
    leaves_from_network,
    print_graph_diff,
)
from utils.tree_utils import create_gene_tree
from utils.bic_cherry import (
    bic_cherry_extension,
    bic_cherry,
    restricted_bic_cherry_extension,
)
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


@pytest.fixture
def sample_bmg_2():
    G = nx.DiGraph()

    G.add_nodes_from(
        [
            (6, {"color": 2}),
            (7, {"color": 2}),
            (8, {"color": 3}),
            (10, {"color": 2}),
            (11, {"color": 3}),
        ]
    )
    G.add_edges_from(
        [
            (6, 8),
            (8, 6),
            (7, 8),
            (8, 7),
            (8, 10),
            (10, 8),
            (6, 11),
            (11, 6),
            (10, 11),
            (11, 10),
        ]
    )

    return G


def test_bic_cherry(sample_bmg_1):

    cherry_network, pairs = bic_cherry(sample_bmg_1)

    # check leaves are same as in bmg
    assert set(sample_bmg_1.nodes()) == set(leaves_from_network(cherry_network))
    # check bmg from cherry_network is fully connected bmg
    full_bmg = nx.DiGraph()
    full_bmg.add_nodes_from(sample_bmg_1.nodes(data=True))
    full_bmg.add_edges_from(
        edge for (u, v) in pairs for edge in [(u, v), (v, u)]
    )  # add both directions for all mismatch colored nodes

    assert nx.is_isomorphic(bmg_from_network(cherry_network), full_bmg)


def test_bic_cherry_extension(sample_bmg_1):
    network = bic_cherry_extension(sample_bmg_1)
    restricted_network = restricted_bic_cherry_extension(sample_bmg_1)

    new_bmg = bmg_from_network(network)
    restricted_new_bmg = bmg_from_network(restricted_network)

    assert nx.is_isomorphic(sample_bmg_1, new_bmg)
    assert nx.is_isomorphic(sample_bmg_1, restricted_new_bmg)


# guarantees, that the bic-cherry construction is correct (its bmg is fully connected bmg)
def test_bic_cherry_generated_examples():

    species = 10
    species_tree_age = 1
    for i in range(10):
        # asymmetree uses Tree class, our methods use nx.DiGraph
        tree = create_gene_tree(species, species_tree_age).gene_tree
        bmg = bmg_from_network(tree)
        cherry_network, pairs = bic_cherry(bmg)

        # check leaves are same as in bmg
        assert set(bmg.nodes()) == set(leaves_from_network(cherry_network))
        # check bmg from cherry_network is fully connected bmg
        full_bmg = nx.DiGraph()
        full_bmg.add_nodes_from(bmg.nodes(data=True))
        full_bmg.add_edges_from(
            edge for (u, v) in pairs for edge in [(u, v), (v, u)]
        )  # add both directions for all mismatch colored nodes

        assert nx.is_isomorphic(bmg_from_network(cherry_network), full_bmg)


def test_bic_cherry_extension_generated_examples():
    species = 2
    species_tree_age = 1
    for i in range(10):
        tree = create_gene_tree(species, species_tree_age).gene_tree
        graph = transform(tree, 2)
        bmg = bmg_from_network(graph)
        network = bic_cherry_extension(bmg)

        new_bmg = bmg_from_network(network)

        # check that bmg of computed network is same as staring bmg
        if not nx.is_isomorphic(bmg, new_bmg):
            print_graph_diff(bmg, new_bmg)
            assert False


# observation: only tests on trees! Fails if tested on networks...
def test_restricted_bic_cherry_extension_generated_examples():
    species = 2
    species_tree_age = 1
    for i in range(10):
        tree = create_gene_tree(species, species_tree_age).gene_tree
        trans_tree = transform(tree, 2)
        bmg = bmg_from_network(trans_tree)
        restricted_network = restricted_bic_cherry_extension(bmg)

        restricted_new_bmg = bmg_from_network(restricted_network)

        # check that bmg of computed network is same as staring bmg
        assert nx.is_isomorphic(bmg, restricted_new_bmg)


# this example used to fail before the fix
def test_bmg_extra(sample_bmg_2):

    network = bic_cherry_extension(sample_bmg_2)
    restricted_network = restricted_bic_cherry_extension(sample_bmg_2)

    new_bmg = bmg_from_network(network)
    restricted_new_bmg = bmg_from_network(restricted_network)

    print_graph_diff(sample_bmg_2, new_bmg)
    assert nx.is_isomorphic(sample_bmg_2, new_bmg)
    assert nx.is_isomorphic(sample_bmg_2, restricted_new_bmg)
