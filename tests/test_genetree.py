"""
Tests for create_gene_tree_n_leaves.

The simulation is stochastic, so the leaf count assertions use a tolerance
instead of demanding an exact hit. The structural assertions below are
deterministic and are where the real checking happens.
"""

import json

import networkx as nx
import numpy as np
import pytest

from utils.graph_utils import bmg_from_network, leaves_from_network
from utils.tree_utils import create_gene_tree_n_leaves

# the default (7) is tuned for speed, the tests want the rate steering to settle
ATTEMPTS = 40


def leaf_count(graph: nx.DiGraph) -> int:
    return len(leaves_from_network(graph))


@pytest.fixture(scope="module")
def trees():
    """One simulated pair reused by all structural tests (simulation is slow)."""
    return create_gene_tree_n_leaves(
        leaves=20, species=2, spt_age=1.0, max_attempts=ATTEMPTS
    )


# sanity checks ... 
class TestArgumentValidation:
    @pytest.mark.parametrize("species", [1, 0, -3])
    def test_fewer_than_two_species_raises(self, species):
        with pytest.raises(ValueError):
            create_gene_tree_n_leaves(leaves=10, species=species)

    @pytest.mark.parametrize("leaves", [4, 0, -1])
    def test_fewer_leaves_than_species_raises(self, leaves):
        # no species goes extinct, so a gene tree with less leaves than
        # species cannot exist
        with pytest.raises(ValueError):
            create_gene_tree_n_leaves(leaves=leaves, species=5)

    def test_species_is_checked_before_leaves(self):
        # both arguments are invalid -- the species message must win
        with pytest.raises(ValueError, match="species has to be at least 2"):
            create_gene_tree_n_leaves(leaves=0, species=1)


# "real" tests 
class TestLeafCount:
    @pytest.mark.parametrize("species", [2, 5])
    def test_minimum_case_leaves_equals_species_is_hit_exactly(self, species):
        # one gene per species is the smallest reachable tree, the rate
        # steering drives the duplication rate down until it gets there
        trees = create_gene_tree_n_leaves(
            leaves=species, species=species, max_attempts=ATTEMPTS
        )
        assert leaf_count(trees.gene_tree) == species

    @pytest.mark.parametrize("leaves,species", [(12, 5), (25, 10), (40, 8)])
    def test_lands_close_to_the_requested_leaf_count(self, leaves, species):
        trees = create_gene_tree_n_leaves(
            leaves=leaves, species=species, max_attempts=ATTEMPTS
        )
        # deliberately generous -- an exact hit is never guaranteed and this
        # only checks that the rate steering reaches the right neighbourhood.
        # A single unsteered draw is off by far more, see the max_attempts=1 test.
        assert abs(leaf_count(trees.gene_tree) - leaves) <= max(4, leaves // 4)

    def test_single_attempt_returns_a_usable_tree_instead_of_raising(self):
        # with one shot the leaf count is pure luck, but the function has to
        # return the only attempt rather than fail
        trees = create_gene_tree_n_leaves(leaves=30, species=5, max_attempts=1)
        assert leaf_count(trees.gene_tree) >= 5

    def test_unreachably_large_target_returns_best_effort(self):
        # 500 leaves over 3 species will not be hit within the attempts,
        # the closest tree must still come back intact
        trees = create_gene_tree_n_leaves(leaves=500, species=3, max_attempts=3)
        assert leaf_count(trees.gene_tree) >= 3
        assert nx.is_tree(trees.gene_tree.to_undirected())


class TestTreeStructure:
    def test_gene_tree_is_a_rooted_tree(self, trees):
        gene_tree = trees.gene_tree
        roots = [n for n in gene_tree if gene_tree.in_degree(n) == 0]
        assert len(roots) == 1
        assert all(gene_tree.in_degree(n) <= 1 for n in gene_tree)
        assert nx.is_tree(gene_tree.to_undirected())

    def test_no_internal_vertex_has_exactly_one_child(self, trees):
        # pruning suppresses the superfluous vertices left behind by losses
        gene_tree = trees.gene_tree
        assert all(gene_tree.out_degree(n) != 1 for n in gene_tree)

    def test_loss_branches_are_pruned_away(self, trees):
        gene_tree = trees.gene_tree
        events = {gene_tree.nodes[n].get("event") for n in gene_tree}
        assert "L" not in events

    def test_species_tree_has_the_requested_number_of_leaves(self):
        trees = create_gene_tree_n_leaves(leaves=15, species=7, max_attempts=3)
        assert leaf_count(trees.species_tree) == 7

    def test_original_gene_tree_matches_the_digraph(self, trees):
        # the asymmetree Tree is handed out for the analysis functions, it has
        # to be the very tree the DiGraph was built from
        assert sum(1 for _ in trees.original_gene_tree.leaves()) == leaf_count(
            trees.gene_tree
        )


class TestColors:
    def test_every_gene_leaf_is_reconciled_to_a_species_leaf(self, trees):
        gene_tree, species_tree = trees.gene_tree, trees.species_tree
        # the DiGraph is KEYED by the asymmetree label, there is no "label"
        # attribute -- the species leaves are their own ids
        species_labels = set(leaves_from_network(species_tree))
        leaf_colors = {
            gene_tree.nodes[n]["color"] for n in leaves_from_network(gene_tree)
        }
        # no species goes extinct, so the colors cover the species exactly
        assert leaf_colors == species_labels

    def test_leaf_colors_are_single_species_not_edges(self, trees):
        # inner vertices may sit on a species edge (reconc is a stringified
        # tuple there and color is None), leaves always belong to one species
        gene_tree = trees.gene_tree
        assert all(
            isinstance(gene_tree.nodes[n]["color"], int)
            for n in leaves_from_network(gene_tree)
        )


class TestAttributeCleaning:
    def test_attributes_are_json_friendly(self, trees):
        # None is a legal value (inner vertices have color None, the root has
        # no event); what has to be gone are numpy scalars and tuples
        for graph in (trees.gene_tree, trees.species_tree):
            assert json.dumps(dict(graph.nodes(data=True)))
            for _node, attrs in graph.nodes(data=True):
                for key, value in attrs.items():
                    assert not isinstance(value, (np.generic, tuple)), key

    def test_color_is_derived_from_reconc_which_is_kept(self, trees):
        # reconc is NOT renamed -- color is an extra attribute: the species
        # label on leaves, None on inner vertices (their reconc may be a
        # species EDGE, which is not a color)
        gene_tree = trees.gene_tree
        assert all("reconc" in attrs for _n, attrs in gene_tree.nodes(data=True))
        assert all(
            gene_tree.nodes[n]["color"] == gene_tree.nodes[n]["reconc"]
            for n in leaves_from_network(gene_tree)
        )
        assert all(
            gene_tree.nodes[n]["color"] is None
            for n in gene_tree
            if gene_tree.out_degree(n) > 0
        )

    def test_nodes_are_keyed_by_the_asymmetree_label(self, trees):
        # labels are unique integers but NOT renumbered to 0..n-1 (pruned
        # branches leave gaps); keying by the label is what makes the gene
        # tree leaves and the BMG vertices the same genes
        gene_tree = trees.gene_tree
        assert all(isinstance(n, int) for n in gene_tree)
        assert len(set(gene_tree.nodes)) == gene_tree.number_of_nodes()
        assert set(bmg_from_network(trees.gene_tree).nodes) == set(leaves_from_network(gene_tree))
