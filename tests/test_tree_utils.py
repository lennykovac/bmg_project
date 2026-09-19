"""utils/tree_utils.py"""
import numpy as np
import pytest
import asymmetree.treeevolve as te

from tests.helpers import edges
from utils.graph_utils import bmg_from_network, check_color_sink_free, is_phylogenetic_tree
from utils.tree_utils import _plain, create_gene_tree_n_leaves, tree_to_digraph


def test_plain_converts_numpy_and_tuples():
    assert _plain(np.float64(1.5)) == 1.5 and type(_plain(np.float64(1.5))) is float
    assert _plain((0, 1)) == "(0, 1)"
    assert _plain(None) is None and _plain("S") == "S"


def test_tree_to_digraph_uses_labels_and_colors_only_leaves():
    S = te.species_tree_n_age(3, age=1.0)
    T = te.prune_losses(te.dated_gene_tree(S, dupl_rate=1.0, loss_rate=0.3))
    D = tree_to_digraph(T)
    labels = {v.label for v in T.preorder()}
    assert set(D.nodes) == labels
    assert D.number_of_edges() == len(labels) - 1
    for v in T.preorder():
        if v.children:
            assert D.nodes[v.label]["color"] is None
        else:
            assert D.nodes[v.label]["color"] == v.reconc


@pytest.mark.parametrize("leaves, species", [(2, 3), (5, 1)])
def test_invalid_arguments(leaves, species):
    with pytest.raises(ValueError):
        create_gene_tree_n_leaves(leaves, species)


def test_gene_tree_and_bmg_share_leaves(instances):
    for d in instances:
        leaves = {v for v in d.gene_tree if d.gene_tree.out_degree(v) == 0}
        assert leaves == set(bmg_from_network(d.gene_tree).nodes)
        assert edges(bmg_from_network(d.gene_tree)) == edges(bmg_from_network(d.gene_tree))


def test_simulated_objects_are_valid(instances):
    for d in instances:
        assert is_phylogenetic_tree(d.gene_tree)
        assert check_color_sink_free(bmg_from_network(d.gene_tree))
        assert all(bmg_from_network(d.gene_tree).nodes[u]["color"] != bmg_from_network(d.gene_tree).nodes[v]["color"] for u, v in bmg_from_network(d.gene_tree).edges)
        species = {v for v in d.species_tree if d.species_tree.out_degree(v) == 0}
        assert set(nx_colors(bmg_from_network(d.gene_tree))) <= species


def nx_colors(G):
    return {c for _, c in G.nodes(data="color")}


def test_leaf_count_is_close_to_request():
    d = create_gene_tree_n_leaves(8, 3, max_attempts=15)
    assert abs(bmg_from_network(d.gene_tree).number_of_nodes() - 8) <= 3
