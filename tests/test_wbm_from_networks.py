import os
import random
import pytest
from collections import Counter

from utils.graph_utils import (
    transform,
    wbmg_from_network)
from utils.tree_utils import create_gene_tree_n_leaves
from bic_cherry_wmb import bic_cherry_for_wbm

@pytest.mark.exhaustive
def test_bic_cherry_for_wbm_generated_examples():
    """
    Round-trip on random gene trees: tree -> WBMG -> network -> WBMG must match.
    """
    runs = 0
    max_runs = 50 # this can be increased to 1000 or more, but the test will take a long time to run
    # (leaves, hybrids) -> number of mismatches
    mismatches = Counter()

    for leaves in range(2,30):
        print(f"Testing {max_runs} trees with {leaves} leaves and 2-30 random hybrid nodes!")
        species = 2
        species_tree_age = 1

        for _ in range(max_runs):
            tree = create_gene_tree_n_leaves(leaves, species, species_tree_age, max_attempts=25).gene_tree
            number_of_hybrid_nodes = random.randint(2,30)
            graph = transform(tree, number_of_hybrid_nodes)
            wbmg = wbmg_from_network(graph)
            network = bic_cherry_for_wbm(wbmg)

            new_wbmg = wbmg_from_network(network)

            actual_leaves = [n for n, d in graph.out_degree() if d == 0]
            runs += 1

            new_wbmg_edges = set(new_wbmg.edges())
            wbmg_edges = set(wbmg.edges())
            if new_wbmg_edges != wbmg_edges:
                mismatches[(len(actual_leaves), number_of_hybrid_nodes)] += 1
                print(
                    f"mismatch: {len(actual_leaves)} leaves, {number_of_hybrid_nodes} hybrids, "
                    f"missing {wbmg_edges - new_wbmg_edges}, "
                )

    # Summary: how often and under which conditions it went wrong
    total = sum(mismatches.values())
    print(f"{total} mismatches in {runs} runs ({100 * total / runs:.2f}%)")
    for (n_leaves, n_hybrids), count in sorted(mismatches.items()):
        print(f"  {n_leaves} leaves, {n_hybrids} hybrids: {count}")
