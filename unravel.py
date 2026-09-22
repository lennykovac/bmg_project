from collections import defaultdict
from typing import Hashable

from utils.graph_editing import make_guard, normalize
from utils.graph_utils import root_from_network
import networkx as nx

G = nx.DiGraph()

G.add_node(0, label="0", color="0")
G.add_node(1, label="1", color="0")
G.add_node(2, label="2", color="0")
G.add_node(3, label="3", color="0")
G.add_node(4, label="4", color="0")
G.add_node(5, label="5", color="1")
G.add_node(6, label="6", color="0")
G.add_node(7, label="7", color="0")
G.add_node(8, label="8", color="0")
G.add_node(9, label="9", color="1")
G.add_node(10, label="10", color="0")

G.add_edges_from(
    [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 8),
        (2, 4),
        (2, 5),
        (3, 5),
        (3, 6),
        (4, 5),
        (4, 7),
        (4, 8),
        (5, 7),
        (5, 9),
        (6, 8),
        (6, 10),
        (7, 9),
        (7, 10),
    ]
)

norm_G = nx.DiGraph()

norm_G.add_node(0, label="0", color="0")
norm_G.add_node(3, label="3", color="0")
norm_G.add_node(4, label="4", color="0")
norm_G.add_node(6, label="6", color="0")
norm_G.add_node(7, label="7", color="0")
norm_G.add_node(8, label="8", color="0")
norm_G.add_node(9, label="9", color="1")
norm_G.add_node(10, label="10", color="0")

norm_G.add_edges_from(
    [
        (0, 4),
        (0, 3),
        (3, 7),
        (3, 6),
        (4, 7),
        (4, 8),
        (6, 8),
        (6, 10),
        (7, 9),
        (7, 10),
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


# TODO: reducing unravelling: start at root; breadth search for nodes with same name (last vertex in their path-name) that have exactly the same descendants;
# delete one of them AND THEIR SUBTREE as it is duplicate (the other one ensures, the leaf-contracted output network has the BMs that were lost on the deleted side)
# IDEA: delete any subtree where ALL nodes have id suffix > 0 -> work from leaves upwards... AND delete all internal nodes where no leaves are in subtree!

# FIRST TEST IF UNRAVELLING "BMG" == REDUCED UNRAVELLING "BMG"
# TODO: leaf-contraction: take reduced unravelling and join all leaves with the same name.

# TODO: verify this works: Unravelling works/produces what I expect; "BMG" of unravelling is always the same as in the OG network; reduction doesnt change BMG;
# whole pipeline doesnt change BMG.

# tests: test if unraveling "BMG" is always the same as network BMG;

g = make_guard(G, mode="bmg")
G_norm = G.copy()
normalize(
    G_norm, set([v for v in G.nodes() if G.out_degree(v) == 0])
)  # doesnt change anything
# print(g(G_norm))
# print(list(G_norm.edges()))
g_norm_unr = unravel(G_norm)
print(g_norm_unr.edges())
