import os
import random
import pytest
from collections import Counter

from utils.graph_utils import (
    transform,
    wbmg_from_network)
from utils.tree_utils import create_gene_tree_n_leaves
from bic_cherry_wmb import bic_cherry_for_wbm

def export_mismatch(graph, n_leaves, n_hybrids, folder="mismatches"):
    """Write graph to <folder>/l_<leaves>h_<hybrids>.gml, numbered if the name is taken."""
    os.makedirs(folder, exist_ok=True)
    name = f"l_{n_leaves}h_{n_hybrids}"
    path = os.path.join(folder, f"{name}.gml")
    suffix = 1
    while os.path.exists(path):
        suffix += 1
        path = os.path.join(folder, f"{name}_{suffix}.gml")

    # GML can't store None, so leave those attributes out
    export = nx.DiGraph()
    for n, data in graph.nodes(data=True):
        export.add_node(n, **{k: v for k, v in data.items() if v is not None})
    export.add_edges_from(graph.edges())
    nx.write_gml(export, path)
    return path

@pytest.mark.exhaustive
def test_bic_cherry_for_wbm_generated_examples():
    """Round-trip on random gene trees: tree -> WBMG -> network -> WBMG must match."""
    runs = 0
    max_runs = 300
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
                path = export_mismatch(graph, len(actual_leaves), number_of_hybrid_nodes)
                print(
                    f"mismatch: {len(actual_leaves)} leaves, {number_of_hybrid_nodes} hybrids, "
                    f"missing {wbmg_edges - new_wbmg_edges}, "
                    f"extra {new_wbmg_edges - wbmg_edges} -> {path}"
                )

    # Summary: how often and under which conditions it went wrong
    total = sum(mismatches.values())
    print(f"{total} mismatches in {runs} runs ({100 * total / runs:.2f}%)")
    for (n_leaves, n_hybrids), count in sorted(mismatches.items()):
        print(f"  {n_leaves} leaves, {n_hybrids} hybrids: {count}")
