"""
Utilities for working nx.DiGraphs.

One visulizezer functions and a few other handy utilities should be collected here.

"""

import random
from typing import Any, Tuple

from typing import Hashable
from itertools import permutations
import networkx as nx
from pyvis.network import Network


def show_graph(di_graph: nx.DiGraph):
    """
    Shows u a neat graph view of the DAG
    """
    nt = Network("1000px", "1000px", directed=True)
    for node in di_graph.nodes():
        di_graph.nodes[node]["label"] = str(node)

    nt.from_nx(di_graph)
    nt.show("nx.html", notebook=False)


def insert_node_on_edge(node_for_adding: Any, edge: Tuple[Any, Any], G: nx.DiGraph):
    """
    Inserts a node onto an edge and removes the old redundant edge
    Function is inplace!

    Parameters:
    node_for_adding: New node
    edge: on which the node should en placed
    G: Directed Graph
    """
    parent_node = edge[0]
    child_node = edge[1]
    # add edge from parent_node to new_node
    G.add_edge(parent_node, node_for_adding)
    # add edge from new_node to child_node
    G.add_edge(node_for_adding, child_node)
    # remove old edge
    G.remove_edge(parent_node, child_node)


# TODO: Check over dist. attribute in node if we have a cycle
def add_hybrid_node(
    donor_edge: Tuple[Any, Any],
    hybrid_edge: Tuple[Any, Any],
    donor: Any,
    hybrid: Any,
    G: nx.DiGraph,
):
    """
    Connects 2 nodes on 2 edges with each other and creates one hybrid node!

    Parameters:
    donor_edge: Edge on which the donor node is placed
    hybrid_edge: Edge on which the hybrid now is placed (it has two parents)
    donor: donor node which "donates" an edge
    hybrid: hybrid node which gets another parent (donor)
    """
    # we have to check if a path exists from the source of the donor edge to the source of the hybrid edge
    if nx.has_path(G, donor_edge[0], hybrid_edge[0]):
        print("Cant insert hybrid node")
        # raise Exception("Cant insert hybrid. It would make the Graph cyclic.")
    else:
        # first insert donor to donor_edge
        insert_node_on_edge(donor, donor_edge, G)
        # second insert hybrid to hybrid_edge
        insert_node_on_edge(hybrid, hybrid_edge, G)
        # third add edge between donor and hybrid
        G.add_edge(donor, hybrid)


# TODO: Naiv implementation for now we have to sort out a strategy
def transform(graph: nx.DiGraph, no_of_hybrid_nodes: int) -> nx.DiGraph:
    """
    GOAL: Edit a bicolored tree into a phylogenetic network by inserting random hybridization vertices
    We dont want this inplace i guess.

    Parameters:
    di_graph: The di_graph on which the hybrid nodes will be inserted
    no_of_hybrid_nodes: The amount of hybrid nodes we would like to have

    Returns:
    A nx.DiGraph object.
    """
    # keep old data intact for now
    transformer_graph = graph
    # get all edges
    edge_list = graph.edges

    # warum gibt es in python keine saubere funtion um zwei verschiedene elemente aus einer liste zu samplen!?
    for i in range(no_of_hybrid_nodes):
        e0, e1 = random.sample(edge_list, 2)

        # TODO: Think of a naming convention for hybrid nodes.
        add_hybrid_node(e0, e1, f"{0}_d", f"{0}_h", transformer_graph)

    return transformer_graph


def root_from_network(network: nx.DiGraph) -> Hashable:

    roots = [n for n in network.nodes() if network.in_degree(n) == 0]

    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, found {len(roots)}")

    root = roots[0]

    return root


def leaves_from_network(network: nx.DiGraph) -> list[Hashable]:

    return [n for n in network.nodes() if network.out_degree(n) == 0]


def bmg_from_network(
    network: nx.DiGraph,
) -> nx.DiGraph:
    """Construct a BMG from bic-network.

    Args:
        network: A network with leaves that has the `label` and `reconc` attribute set.

    Returns:
        The constructed BMG with attributes `label` and `color`
    """

    leaves = leaves_from_network(network)
    bmg = nx.DiGraph()
    colors = set()
    reach = {
        n: nx.descendants(network, n) for n in network.nodes
    }  # pre-compute reachability in network as dict[{v:descendants of v}]

    # collect all leaves and colors
    for v in leaves:
        colors.add(network.nodes[v]["reconc"])
        bmg.add_node(v, color=network.nodes[v]["reconc"])

    # collect all pairs of different color
    pairs = [
        (u, v)
        for u, v in permutations(leaves, 2)
        if network.nodes[u]["reconc"] != network.nodes[v]["reconc"]
    ]

    # create dict with pair -> LCA(pair) mapping
    lca_dict = dict()
    for x, y in pairs:
        pred_x = set()
        pred_y = set()
        for z in reach:
            if x in reach[z]:
                pred_x.add(z)
            if y in reach[z]:
                pred_y.add(z)

        common_ancestors = set.intersection(pred_x, pred_y)

        # remove all non minimal common ancestors
        high_ancestors = set()  # ancestors to be removed
        for u, v in permutations(common_ancestors, 2):
            if nx.has_path(network, u, v):
                high_ancestors.add(u)  # v < u, thus remove u from lca

        lca = common_ancestors - high_ancestors
        lca_dict.update({(x, y): lca})

    # check bm property for each pair
    delete_keys = set()
    for x, y in lca_dict.keys():
        # find all y' with same color as y
        alt_y = [
            u
            for u in leaves
            if network.nodes[u]["reconc"] == network.nodes[y]["reconc"]
        ]
        # iterate over lca(x, y') --> ALREADY IN lca_dict!!!!
        for ay in alt_y:
            lca_alt_y = lca_dict[(x, ay)]
            for u in lca_alt_y:
                for v in lca_dict[(x, y)]:
                    if u in reach[v]:  # i.e. u<v
                        delete_keys.add((x, y))
    # delete all marked keys from dict
    for x, y in delete_keys:
        lca_dict.pop((x, y))

    # add remaining bmg edges to bmg
    for x, y in lca_dict:
        bmg.add_edge(x, y)

    return bmg


if __name__ == "__main__":
    """
    EXAMPLES:
    """

    G = nx.read_gml("../tests/gene_tree_test_file.gml")

    show_graph(G)

    G_Transformed = transform(G, 5)

    show_graph(G_Transformed)
