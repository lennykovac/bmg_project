"""
Utilities for simulating dated gene and species trees with asymmetree.

Trees are returned as networkx DiGraphs with simplified integer node labels
and JSON-friendly node attributes.
"""

from typing import NamedTuple

import asymmetree.treeevolve as te
import networkx as nx
from tralda.datastructures import Tree



class GeneSpeciesTrees(NamedTuple):
    gene_tree: nx.DiGraph
    species_tree: nx.DiGraph
    original_gene_tree: Tree  # AsymmeTree object (for asymmetree.analysis cross-checks)


def _plain(value):
    """numpy scalars / tuples -> JSON-friendly python values."""
    if value is None:
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, tuple):
        return str(value)
    return value


def tree_to_digraph(tree: Tree) -> nx.DiGraph:
    """AsymmeTree ``Tree`` -> ``nx.DiGraph`` keyed by node label.

    Leaves get ``color = reconc`` (species), inner vertices ``color = None``."""
    D = nx.DiGraph()
    for v in tree.preorder():
        is_leaf = not v.children
        D.add_node(
            v.label,
            event=_plain(getattr(v, "event", None)),
            reconc=_plain(getattr(v, "reconc", None)),
            dist=_plain(getattr(v, "dist", None)),
            color=_plain(getattr(v, "reconc", None)) if is_leaf else None,
        )
        if v.parent is not None:
            D.add_edge(v.parent.label, v.label)
    return D


def create_gene_tree_n_leaves(
    leaves: int = 4, 
    species: int = 2,
    spt_age: float = 1.0,
    loss_rate: float = 0.5,
    hgt_rate: float = 0.1,
    max_attempts: int = 7,
) -> GeneSpeciesTrees:
    """Simulate a dated gene tree with (approximately) ``leaves`` surviving genes.

    The duplication rate is steered up/down between attempts; the attempt with
    the closest leaf count is returned. The BMG is computed with AsymmeTree and
    cross-checked against ``bmg_from_network`` (they must be equal).
    """
    if species < 2:
        raise ValueError("species has to be at least 2")
    if leaves < species:
        raise ValueError(f"leaves ({leaves}) has to be at least species ({species})")

    species_tree = te.species_tree_n_age(species, age=spt_age)

    dupl_rate = 1.0
    best_tree, best_error = None, None

    for _ in range(max_attempts):
        gene_tree = te.prune_losses(te.dated_gene_tree(species_tree, dupl_rate=dupl_rate, loss_rate=loss_rate, hgt_rate=hgt_rate))

        leaf_count = sum(1 for _ in gene_tree.leaves())
        error = abs(leaf_count - leaves)

        if best_error is None or error < best_error:
            best_tree, best_error = gene_tree, error
        if error == 0:
            break
        # 3. more duplications give more leaves, so steer the rate accordingly (by a third)
        dupl_rate *= 1.3 if leaf_count < leaves else 1 / 1.3
        dupl_rate = min(max(dupl_rate, 0.01), 25.0)

    gene_nx = tree_to_digraph(best_tree)


    return GeneSpeciesTrees(
        gene_tree=gene_nx,
        species_tree=tree_to_digraph(species_tree),
        original_gene_tree=best_tree,
    )

def import_good_trees() -> list[nx.DiGraph]:
    # import trees from a file and return as a list
    # smth like this?
    # good_trees_file = "../objects/good_trees.gml"
    # trees = nx.read_gml(good_trees_file)
    pass
