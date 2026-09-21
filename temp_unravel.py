from collections import defaultdict
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

unr_red_1 = nx.DiGraph()

unr_red_1.add_node("R", label="R", color=None)
unr_red_1.add_node("p:4|5", label="p:4|5", color=None)
unr_red_1.add_node("p:4|5_1", label="p:4|5_1", color=None)
unr_red_1.add_node("p:5|4", label="p:5|4", color=None)
unr_red_1.add_node("p:4|7", label="p:4|7", color=None)
unr_red_1.add_node("p:7|4", label="p:7|4", color=None)
unr_red_1.add_node(4, label="4", color=2)
unr_red_1.add_node(5, label="5", color=1)
unr_red_1.add_node(7, label="7", color=1)
unr_red_1.add_node("q:4|5", label="q:4|5", color=None)

unr_red_1.add_edges_from(
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

if __name__ == "__main__":
    # print(tree.edges())
    # print(bmg.edges())
    # print(network.edges())
    guard = make_guard(network, "bmg")
    norm_net = network.copy()
    normalize(norm_net, set(leaves_from_network(network)))
    print(norm_net.edges())
    print(guard(norm_net))

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
