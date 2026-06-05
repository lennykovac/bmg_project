import networkx as nx
from numpy import exp
from utils.graph_utils import show_graph
from utils.graph_utils import bmg_from_network
import pytest
# from utils.graph_utils import wbmg_from_network # TODO:


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


def test_bmg_structure(sample_graph_1):

    bmg = bmg_from_network(sample_graph_1)
    # wbmg = wbmg_from_network(G) # TODO:
    actual = list(bmg.nodes(data=True))
    expected = [(4, {"color": "0"}), (5, {"color": "1"}), (6, {"color": "0"})]
    assert actual == expected
