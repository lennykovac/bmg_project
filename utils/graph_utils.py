"""
Utilities for working nx.DiGraphs.

One visulizezer functions and a few other handy utilities should be collected here.

"""

import random
from typing import Any, Tuple

import networkx as nx
from pyvis.network import Network


def show_graph(di_graph: nx.DiGraph):
    """
    Shows u a neat graph view of the DAG
    """
    nt = Network("1000px", "1000px", directed=True)

    for node, node_data in di_graph.nodes(data=True):
        node_data["label"] = str(node)
        title_parts = []
        if "reconc" in node_data:
            reconc_value = node_data["reconc"]
            title_parts.append(f"reconc: {reconc_value}")
        if "dist" in node_data:
            title_parts.append(f"dist: {node_data['dist']}")
        if title_parts:
            node_data["title"] = "\n".join(title_parts)

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


def add_hybrid_node(
    donor_edge: Tuple[Any, Any],
    hybrid_edge: Tuple[Any, Any],
    donor: Any,
    hybrid: Any,
    G: nx.DiGraph,
    tries: int,
    inserts: int
) -> bool:
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
        tries -= 1 
        return
    else:
        # first insert donor to donor_edge
        insert_node_on_edge(donor, donor_edge, G)
        # second insert hybrid to hybrid_edge
        insert_node_on_edge(hybrid, hybrid_edge, G)
        # third add edge between donor and hybrid
        G.add_edge(donor, hybrid)
        inserts += 1


def transform(graph: nx.DiGraph, num_of_hybrid_nodes: int, tries = 7) -> nx.DiGraph:
    """
    GOAL: Edit a bicolored tree into a phylogenetic network by inserting random hybridization vertices
    We dont want this inplace i guess.

    Parameters:
    di_graph: The di_graph on which the hybrid nodes will be inserted
    num_of_hybrid_nodes: The amount of hybrid nodes we would like to have
    tries: number of tries before we skip stop inserting nodes default 7 (cuz i like the number)

    Returns:
    A nx.DiGraph object.
    """
    # number of succesful inserts
    insertions = 0
    # keep old data intact for now
    transformer_graph = graph
    # get all edges
    edge_list = graph.edges

    # sample two random edges
    for i in range(num_of_hybrid_nodes):
        if tries == 0: 
            print(f"Number of succesfull insertions: {insertions}")
            return transformer_graph

        donor_e, hybrid_e = random.sample(edge_list, 2)
        add_hybrid_node(donor_e, hybrid_e, f"{i}_d", f"{i}_h", transformer_graph, tries, insertions)

    print(f"Number of succesfull insertions: {insertions}")
    return transformer_graph


if __name__ == "__main__":
    """
    EXAMPLES:
    """

    show_graph(G)

    G_Transformed = transform(G, 5)

    show_graph(G_Transformed)
