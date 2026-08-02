import networkx as nx
from obj_1a import bmg_wbmg_check
import pytest


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


@pytest.fixture
def sample_graph_2():
    # here, bmg = wbmg
    G = nx.DiGraph()

    G.add_node(0, label="0", reconc="0")
    G.add_node(1, label="1", reconc="0")
    G.add_node(2, label="2", reconc="1")

    G.add_edges_from([(0, 1), (0, 2)])

    return G


def test_bmg_structure(sample_graph_1, sample_graph_2):

    implies_wbmg_1, implies_bmg_1 = bmg_wbmg_check(sample_graph_1)
    implies_wbmg_2, implies_bmg_2 = bmg_wbmg_check(sample_graph_2)
    print(implies_bmg_1, implies_wbmg_1)
    print(implies_bmg_2, implies_wbmg_2)
    assert implies_wbmg_1 and implies_wbmg_2
    assert implies_bmg_2
    assert not implies_bmg_1
