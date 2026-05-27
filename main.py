from typing import Any, Tuple

import networkx as nx
from asymmetree.utils.phylogenetic_trees import random_colored_tree
from pyvis.network import Network


def show_graph(di_graph: nx.DiGraph):
    nt = Network("1000px", "1000px", directed=True)
    for node in di_graph.nodes():
        di_graph.nodes[node]["label"] = str(node)

    nt.from_nx(di_graph)
    nt.show("nx.html", notebook=False)


def insert_node_between_two_nodes(
    node_for_adding: Any, parent_node: Any, child_node: Any, G: nx.DiGraph
):
    """
    Inserts a node between two existings nodes and removes old redundant edge

    Parameters:
    node_for_adding: New node
    parent_node: Parent node of node_for_adding
    child_node: Child node of node_for_adding
    G: Directed Graph
    """
    # add edge from parent_node to new_node
    G.add_edge(parent_node, node_for_adding)
    # add edge from new_node to child_node
    G.add_edge(node_for_adding, child_node)
    # remove old edge
    G.remove_edge(parent_node, child_node)


def insert_node_on_edge(node_for_adding: Any, edge: Tuple[Any, Any], G: nx.DiGraph):
    """
    Inserts a node onto an edge and removes the old redundant edge

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


# TODO clear definition of hybrid node
def add_hybrid_node(
    donor_edge: Tuple[Any, Any],
    hybrid_edge: Tuple[Any, Any],
    donor: Any,
    hybrid: Any,
    G: nx.DiGraph,
):
    """
    Connects 2 nodes on 2 edges with each other and creates one hybrid node?

    Parameters:
    donor_edge: Edge on which the donor node is placed
    hybrid_edge: Edge on which the hybrid now is placed (it has two parents)
    donor: donor node
    hybrid: hybrid node
    """
    # first insert donor to donor_edge
    G.add_node(donor)
    insert_node_between_two_nodes(donor, donor_edge[0], donor_edge[1], G)
    # second insert hybrid to hybrid_edge
    G.add_node(hybrid)
    insert_node_between_two_nodes(hybrid, hybrid_edge[0], hybrid_edge[1], G)
    # third add edge between donor and hybrid
    G.add_edge(donor, hybrid)


def transform(
    species: int, leaves: int, filename: str, generate_new_random=False
) -> nx.DiGraph:
    """
    GOAL: Edit a bicolored tree into a phylogenetic network by inserting hybridization vertices

    Parameters:
    species (int): Number of species (colors)
    leaves (int): Number of leaves (genes)
    filename (str): Filename on which to save or which to read from a phylogentic tree

    Returns:
    A nx.DiGraph object.
    """
    di_graph = nx.DiGraph()

    if generate_new_random:
        bic_tree = random_colored_tree(leaves, species)
        di_graph, root_id = bic_tree.to_nx()  # transform Tree() to networkx DiGraph
        # TODO save old root_id and make it label
        # safe old node ids as str
        for node in di_graph.nodes():
            di_graph.nodes[node]["asym_id"] = str(node)
        di_graph = nx.convert_node_labels_to_integers(di_graph)
        nx.write_gml(di_graph, "./test.gml")
        return di_graph

    else:
        di_graph = nx.read_gml(filename)
        # ATTENTION when reading a gml file all attributes of a node are parsed as str
        add_hybrid_node(("0", "2"), ("11", "12"), "donor", "hybrid", di_graph)
        show_graph(di_graph)
        return di_graph


if __name__ == "__main__":
    transform(2, 20, "test/test_phylo_tree.gml")
