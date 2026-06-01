"""
Utilities for simulating dated gene and species trees with asymmetree.

Trees are returned as networkx DiGraphs with simplified integer node labels
and JSON-friendly node attributes.
"""

from typing import NamedTuple

import asymmetree.treeevolve as te
import networkx as nx
import numpy as np
from tralda.datastructures import Tree


class GeneSpeciesTrees(NamedTuple):
    """Pair of simulated trees.

    as a names tuple to make unpacking easier.
    """

    gene_tree: nx.DiGraph
    species_tree: nx.DiGraph


def _to_clean_digraph(tree: Tree) -> nx.DiGraph:
    """Convert an asymmetree Tree into a clean networkx DiGraph.

    - relabels nodes to integers (the original id is kept in ``asym_id``)
    - normalises attribute types: ``None`` -> ``""`` and ``np.float64`` -> ``float``
    """
    di_graph, _root_id = tree.to_nx()  # transform Tree() to networkx DiGraph

    # save the old asymmetree ids onto each node before relabelling
    for node in di_graph.nodes():
        di_graph.nodes[node]["asym_id"] = str(node)

    # convert to simpler integer node labels
    di_graph = nx.convert_node_labels_to_integers(di_graph)

    # normalise attribute datatypes to usable Python types
    for _node_id, attrs in di_graph.nodes(data=True):
        for key, value in attrs.items():
            if value is None:
                attrs[key] = ""
            elif isinstance(value, np.float64):
                # looks hacky but converts no.float to python float data-type
                attrs[key] = value.item()

    return di_graph


def create_gene_tree(species: int, spt_age: float = 1.0) -> GeneSpeciesTrees:
    """Simulate a dated gene tree together with its species tree.

    Parameters:
        species: number of species in the species tree
        spt_age: age (depth) of the species tree

    Returns:
        GeneSpeciesTrees: (gene_tree, species_tree) as networkx DiGraphs,
        with distance attributes preserved.
    """
    # 1. simulate species tree
    species_tree = te.species_tree_n_age(species, age=spt_age)

    # 2. simulate the corresponding gene tree
    gene_tree = te.dated_gene_tree(
        species_tree, dupl_rate=1.0, loss_rate=0.5, hgt_rate=0.1
    )

    return GeneSpeciesTrees(
        gene_tree=_to_clean_digraph(gene_tree),
        species_tree=_to_clean_digraph(species_tree),
    )


if __name__ == "__main__":
    """
    EXAMPLES:
    """
    trees = create_gene_tree(species=10, spt_age=1.0)
    print("gene tree nodes:   ", trees.gene_tree.number_of_nodes())
    print("species tree nodes:", trees.species_tree.number_of_nodes())
    # save to file:
    filename_gene = "../tests/gene_tree_test_file.gml"
    filename_species = "../tests/species_tree_test_file.gml"

    nx.write_gml(trees.gene_tree, filename_gene)
    nx.write_gml(trees.species_tree, filename_species)
