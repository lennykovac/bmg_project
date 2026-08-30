from itertools import permutations
import networkx as nx

from utils.graph_utils import (
    bmg_from_network,
    transform,
    wbmg_from_network,
    leaves_from_network,
    print_graph_diff,
    print_compare_bmg
)
from utils.tree_utils import create_gene_tree
from utils.bic_cherry_one_node_init import (
    bic_cherry_extension,
    bic_cherry,
    restricted_bic_cherry_extension,
)

"""
N -> BMG -> N' -> BMG'
with p: x|y = p: y|x the cherry initialization of BMG results in missing edges. Are the missing edges weak BMGs? 
Yes -> for WBMG we can use p: x|y = p: y|x for more correct reconstructed WBMG'. 
"""


"""
Question 1: Can  p:x|y = p:y|x introduce non-existent edges in the reconstructed BMG'? 
"""
def test_reconstructed_bmg_has_no_additional_edges():
    species = 2
    species_tree_age = 1
    for i in range(100):
        tree = create_gene_tree(species, species_tree_age).gene_tree
        graph = transform(tree, 2)
        bmg = bmg_from_network(graph)
        network = bic_cherry_extension(bmg)

        new_bmg = bmg_from_network(network)

        edges_only_new_bmg = set(new_bmg.edges) - set(bmg.edges)

        assert not edges_only_new_bmg, f"BMG' contains additional edge"



"""
Test 1 passed: edges(BMG´) subset of edges(BMG)
Question 2: are the edges missing in BMG' at least weak best matches
"""
def test_missing_edges_are_weak_best_matches():
    species = 2
    species_tree_age = 1
    for i in range(100):
        tree = create_gene_tree(species, species_tree_age).gene_tree
        graph = transform(tree, 2)
        bmg = bmg_from_network(graph)
        network = bic_cherry_extension(bmg)

        new_bmg = bmg_from_network(network)

        edges_only_bmg = set(bmg.edges) - set(new_bmg.edges)

        non_wbm = set()
        if edges_only_bmg:
            wbmg = wbmg_from_network(network)
            edges_wbmg = set(wbmg.edges)
            non_wbm = edges_only_bmg - edges_wbmg

        assert not non_wbm, (
            f"Missing BMG edges are not weak best matches"
        )



def wbm_minimal_example(max_leaves=10, runs_per_size=1000):
    species = 2
    species_tree_age = 1

    for n_leaves in range(3, max_leaves + 1):
        print(f"Suche mit {n_leaves} Blättern")

        for i in range(runs_per_size):
            tree = create_gene_tree(species, species_tree_age).gene_tree
            graph = transform(tree, 2)

            leaves = [n for n, d in graph.out_degree() if d == 0]

            if len(leaves) != n_leaves:
                continue

            wbmg = wbmg_from_network(graph)

            network = restricted_bic_cherry_extension(wbmg)
            new_wbmg = wbmg_from_network(network)

            if not nx.is_isomorphic(wbmg, new_wbmg):
                print(f"Minimales Beispiel gefunden: {n_leaves} Blätter")

                return graph, wbmg, network, new_wbmg

    print("Kein Gegenbeispiel gefunden.")
    return None, None, None, None


def wbmg_edge_count_relationship(total_runs):
    species = 2
    species_tree_age = 1

    isomorphic_count = 0
    more_edges = 0
    fewer_edges = 0
    same_count = 0
    mixed_edges = 0


    for _ in range(total_runs):
        tree = create_gene_tree(species, species_tree_age).gene_tree
        graph = transform(tree, 2)
        wbmg = wbmg_from_network(graph)

        network = restricted_bic_cherry_extension(wbmg)
        new_wbmg = wbmg_from_network(network)

        if nx.is_isomorphic(wbmg, new_wbmg):
            isomorphic_count += 1
        else:
            edges_wbmg = set(wbmg.edges)
            edges_new_wbmg = set(new_wbmg.edges)

            added = edges_new_wbmg - edges_wbmg
            missing = edges_wbmg - edges_new_wbmg

            if added and missing:
                mixed_edges += 1
            elif added:
                more_edges += 1
            elif missing:
                fewer_edges += 1
            else:
                same_count += 1

    print("\n--- comparison: WBMG vs. reconstructed WBMG' ---")
    print(f"Total runs: {total_runs}")
    print(f"  - Isomorphic): {isomorphic_count} ({isomorphic_count/total_runs:.1%})")
    print(f"  - Non-Isomorphic:")
    print(f"      * edges(WBMG') > edges(WBMG):  {more_edges} ({more_edges/total_runs:.1%})")
    print(f"      * edges(WBMG') < edges(WBMG): {fewer_edges} ({fewer_edges/total_runs:.1%})")
    print(f"      * added & missing edges:  {mixed_edges} ({mixed_edges/total_runs:.1%})")
    print(f"      * same edge count, diff structure: {same_count} ({same_count/total_runs:.1%})")
    print("--------------------------------------------------\n")




if __name__ == "__main__":
    print("Tests for p_xy = p_yx")
    print("Test 1: the new BMG has no additional edges")
    test_reconstructed_bmg_has_no_additional_edges()

    print("Test 2: all missing edges in the new BMG are weak best matches")
    test_missing_edges_are_weak_best_matches()




wbmg_edge_count_relationship(1000)

graph, wbmg, network, new_wbmg = wbm_minimal_example()

print_compare_bmg(graph, wbmg, network, new_wbmg)