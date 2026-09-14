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
    """
    Pair of simulated trees as a names tuple to make unpacking easier.
    """

    gene_tree: nx.DiGraph
    species_tree: nx.DiGraph
    original_gene_tree: Tree  # needed for testing with Asymmetree methods


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

    # rename all "reconc" to "color" for now
    for node in di_graph.nodes():
        if "reconc" in di_graph.nodes[node]:
            di_graph.nodes[node]["color"] = di_graph.nodes[node].pop("reconc")

    return di_graph


def create_gene_tree(species: int, spt_age: float = 1.0) -> GeneSpeciesTrees:
    """
    Simulate a dated gene tree together with its species tree.

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
        original_gene_tree=gene_tree,
    )


def create_gene_tree_n_leaves(
    leaves: int,
    species: int,
    spt_age: float = 1.0,
    loss_rate: float = 0.5,
    hgt_rate: float = 0.1,
    max_attempts: int = 7,
) -> GeneSpeciesTrees:
    """
    Simulate a dated gene tree with a given number of surviving leaves.

    asymmetree has no leaf count parameter, so the gene tree is re-simulated
    along one fixed species tree while the duplication rate is nudged up or
    down depending on whether the last attempt had too few or too many leaves.
    Loss branches are pruned, hence the leaf count refers to surviving genes.
    I took the default parametrs from the documentation. They can be adjusted.

    Parameters:
        leaves: wanted number of leaves (surviving genes) in the gene tree,
            has to be at least ``species`` since no species goes extinct
        species: number of species in the species tree
        spt_age: age (depth) of the species tree
        loss_rate: loss rate of the gene tree simulation
        hgt_rate: horizontal gene transfer rate of the gene tree simulation
        max_attempts: number of simulations before giving up on an exact hit

    Returns:
        GeneSpeciesTrees: (gene_tree, species_tree) as networkx DiGraphs. If no
        attempt hits the wanted leaf count, the closest one is returned.
    """
    if species < 2:
        raise ValueError("species has to be at least 2")
    if leaves < species:
        # every species keeps at least one gene (prohibit_extinction per species)
        raise ValueError(f"leaves ({leaves}) has to be at least species ({species})")

    # 0. simulate the species tree once and reuse it for every attempt
    species_tree = te.species_tree_n_age(species, age=spt_age)

    dupl_rate = 1.0
    best_tree, best_error = None, None

    for _attempt in range(max_attempts):
        # 1. simulate a gene tree and drop the branches leading to losses only, took it from asymmetry docs
        gene_tree = te.prune_losses(
            te.dated_gene_tree(
                species_tree,
                dupl_rate=dupl_rate,
                loss_rate=loss_rate,
                hgt_rate=hgt_rate,
            )
        )
        leaf_count = sum(1 for _leaf in gene_tree.leaves())

        # 3. keep the attempt that comes closest to the wanted leaf count
        error = abs(leaf_count - leaves)
        if best_error is None or error < best_error:
            best_tree, best_error = gene_tree, error
        if error == 0:
            break

        # 4. more duplications give more leaves, so steer the rate accordingly
        dupl_rate *= 1.3 if leaf_count < leaves else 1 / 1.3
        dupl_rate = min(max(dupl_rate, 0.01), 100.0)

    return GeneSpeciesTrees(
        gene_tree=_to_clean_digraph(best_tree),
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

    # gene tree with a wanted number of leaves
    sized_trees = create_gene_tree_n_leaves(leaves=25, species=10, spt_age=1.0)
    gene_tree = sized_trees.gene_tree
    print(
        "gene tree leaves:  ",
        sum(1 for node in gene_tree if gene_tree.out_degree(node) == 0),
    )
