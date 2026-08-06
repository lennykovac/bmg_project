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
            (5, {"color": 2}),
            (6, {"color": 2}),
            (7, {"color": 2}),
            (8, {"color": 3}),
            (10, {"color": 2}),
            (11, {"color": 3}),
        ]
    )
    G.add_edges_from(
        [(5, 8), (6, 8), (7, 8), (8, 5), (8, 6), (8, 7), (10, 11), (11, 10)]
    )

    return G


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

    assert nx.is_isomorphic(bmg_from_network(cherry_network), full_bmg)


def test_bic_cherry_extension(sample_bmg_1):
    network = bic_cherry_extension(sample_bmg_1)
    restricted_network = restricted_bic_cherry_extension(sample_bmg_1)

    new_bmg = bmg_from_network(network)
    restricted_new_bmg = bmg_from_network(restricted_network)

    assert nx.is_isomorphic(sample_bmg_1, new_bmg)
    assert nx.is_isomorphic(sample_bmg_1, restricted_new_bmg)


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
        full_bmg = (
            nx.DiGraph()
        )  # full bmg wrong! Can only have edges between differently colored nodes!!
        full_bmg.add_nodes_from(bmg.nodes(data=True))
        full_bmg.add_edges_from(
            edge for (u, v) in pairs for edge in [(u, v), (v, u)]
        )  # add both directions for all mismatch colored nodes

        assert nx.is_isomorphic(bmg_from_network(cherry_network), full_bmg)


def test_bic_cherry_extension_generated_examples():
    species = 2
    species_tree_age = 1
    for i in range(10):
        # asymmetree uses Tree class, our methods use nx.DiGraph
        tree = create_gene_tree(species, species_tree_age).gene_tree
        bmg = bmg_from_network(tree)
        network = bic_cherry_extension(bmg)
        restricted_network = restricted_bic_cherry_extension(bmg)

        new_bmg = bmg_from_network(network)
        restricted_new_bmg = bmg_from_network(restricted_network)

        # check that bmg of computed network is same as staring bmg
        # hint1: new_bmg has less edges -> too many extensions applied in bic_cherry_ext?
        # hint2: restircted version works well?! -> no idea why...
        if not nx.is_isomorphic(bmg, new_bmg):
            print("bmg nodes: ", bmg.nodes(data=True))
            print("bmg edges: ", bmg.edges())
            print("\n\n network nodes: ", network.nodes(data=True))
            print("network edges: ", network.edges())
            print("\n\n new_bmg nodes: ", new_bmg.nodes(data=True))
            print("new_bmg edges: ", new_bmg.edges())
            print_graph_diff(bmg, new_bmg)
            assert False
        # assert nx.is_isomorphic(
        #     bmg, new_bmg
        # )  # doesnt work, but they theoretically proved correctness...
        assert nx.is_isomorphic(bmg, restricted_new_bmg)


# this fails, why?!?! -> very sensitive to choosing z! -> smallest possible z breaks this example, largest works. (different for other examples)
def test_bmg_extra(sample_bmg_2):

    network = bic_cherry_extension(sample_bmg_2)
    restricted_network = restricted_bic_cherry_extension(sample_bmg_2)

    new_bmg = bmg_from_network(network)
    restricted_new_bmg = bmg_from_network(restricted_network)

    print_graph_diff(sample_bmg_2, new_bmg)
    assert nx.is_isomorphic(sample_bmg_2, new_bmg)
    assert nx.is_isomorphic(sample_bmg_2, restricted_new_bmg)
