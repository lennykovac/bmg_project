from collections import defaultdict, deque
from typing import Dict, Hashable

from networkx.classes import nodes
from utils.graph_editing import make_guard, normalize
from utils.graph_utils import (
    bmg_from_network,
    leaves_from_network,
    root_from_network,
    show_graph,
    transform,
)
from utils.bic_cherry import bic_cherry_extension
from utils.tree_utils import create_gene_tree_n_leaves
import networkx as nx


network = nx.DiGraph()

network.add_node("R", label="R", color=None)
network.add_node("p:4|5", label="p:4|5", color=None)
network.add_node("p:5|4", label="p:5|4", color=None)
network.add_node("p:4|7", label="p:4|7", color=None)
network.add_node("p:7|4", label="p:7|4", color=None)
network.add_node(4, label="4", color=2)
network.add_node(5, label="5", color=1)
network.add_node(7, label="7", color=1)
network.add_node("q:4|5", label="q:4|5", color=None)

network.add_edges_from(
    [
        ("R", "p:4|5"),
        ("R", "p:5|4"),
        ("R", "p:4|7"),
        ("R", "p:7|4"),
        ("p:4|5", 4),
        ("p:4|5", 5),
        ("p:5|4", 4),
        ("p:5|4", 5),
        ("p:4|7", 4),
        ("p:4|7", 7),
        ("p:4|7", "q:4|5"),
        ("p:7|4", 4),
        ("p:7|4", 7),
        ("q:4|5", 4),
        ("q:4|5", 5),
    ]
)


tree = nx.DiGraph()

tree.add_node(1, label="1", color=None)
tree.add_node(2, label="2", color=None)
tree.add_node(7, label="7", color=1)
tree.add_node(4, label="4", color=2)
tree.add_node(5, label="5", color=1)

tree.add_edges_from(
    [
        (1, 2),
        (1, 7),
        (2, 4),
        (2, 5),
    ]
)

network_unravelled = nx.DiGraph()

network_unravelled.add_node("R_0", label="R", color=None)
network_unravelled.add_node("p:4|5_0", label="p:4|5", color=None)
network_unravelled.add_node("p:4|7_0", label="p:4|7", color=None)
network_unravelled.add_node("p:7|4_0", label="p:7|4", color=None)
network_unravelled.add_node("4_0", label="4", color=2)
network_unravelled.add_node("4_1", label="4_1", color=2)
network_unravelled.add_node("4_2", label="4_2", color=2)
network_unravelled.add_node("5_0", label="5", color=1)
network_unravelled.add_node("5_1", label="5_1", color=1)
network_unravelled.add_node("7_0", label="7", color=1)
network_unravelled.add_node("7_1", label="7_1", color=1)
network_unravelled.add_node("q:4|5_0", label="q:4|5", color=None)

network_unravelled.add_edges_from(
    [
        ("R_0", "p:4|5_0"),
        ("R_0", "p:4|7_0"),
        ("R_0", "p:7|4_0"),
        ("p:4|5_0", "4_0"),
        ("p:4|5_0", "5_0"),
        ("p:4|7_0", "7_0"),
        ("p:4|7_0", "q:4|5_0"),
        ("p:7|4_0", "4_1"),
        ("p:7|4_0", "7_1"),
        ("q:4|5_0", "4_2"),
        ("q:4|5_0", "5_1"),
    ]
)


def unravel(network: nx.DiGraph) -> nx.DiGraph:
    """
    Takes normalized DAG and produces its unravelling (needed for further reductions). Only input reduced, normalized networks, as unravellings
    can get very big! Unravelling = Tree that represents all possible paths through the input DAG.
    """
    if not nx.is_directed_acyclic_graph(network):
        raise ValueError("Input network not acyclic!")

    unravelling = nx.DiGraph()
    root = root_from_network(network)
    # candidates = pair of original node and one copy in unravelling
    candidates = [(root, "R_0")]
    # store number of node occurences in a dict - use for naming duplicate nodes in unravelling, e.g. "4_0" if first node named 4 etc.
    node_dict: defaultdict[Hashable, int] = defaultdict(int)
    node_dict[root] = 1
    # work through input graph and build unravelling
    while len(candidates) > 0:
        node, node_id = candidates.pop()
        successors = list(network.successors(node))
        new_successors = []
        for d in successors:
            id = f"{d}_{node_dict[d]}"
            # add new candidate - copy tuple to look at later
            candidates.append((d, id))
            unravelling.add_node(id, **network.nodes[d])
            new_successors.append(id)
            node_dict[d] += 1
        unravelling.add_edges_from([(node_id, v) for v in new_successors])

    return unravelling


if __name__ == "__main__":
    # print(tree.edges())
    # print(bmg.edges())
    # print(network.edges())
    guard = make_guard(network, "bmg")
    norm_net = network.copy()
    normalize(norm_net, set(leaves_from_network(network)))

    unr = unravel(norm_net)
    print(nx.is_isomorphic(unr, network_unravelled))
    print("graph 1:")
    print("nodes:", unr.nodes())
    print("edges:", unr.edges())
    print("graph 2:")
    print("nodes:", network_unravelled.nodes())
    print("edges:", network_unravelled.edges())


# bmg = nx.DiGraph()
#
# bmg.add_node(4, label="4", color=2)
# bmg.add_node(5, label="5", color=1)
# bmg.add_node(7, label="7", color=1)
#
# bmg.add_edges_from(
#     [
#         (4, 5),
#         (5, 4),
#         (7, 4),
#     ]
# )
