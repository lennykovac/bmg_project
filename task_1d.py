import networkx as nx
from utils.graph_utils import (
    transform,
    wbmg_from_network,
    print_graph_diff,
)
from utils.tree_utils import create_gene_tree
from utils.bic_cherry import restricted_bic_cherry_extension


def generated_examples_test():
    species = 2
    species_tree_age = 1
    for i in range(10):
        tree = create_gene_tree(species, species_tree_age).gene_tree
        network = transform(tree, 10)
        wbmg = wbmg_from_network(network)
        network2 = restricted_bic_cherry_extension(wbmg)
        wbmg2 = wbmg_from_network(network2)

        if nx.is_isomorphic(wbmg, wbmg2):
            print("success")
        else:
            print("fail")
            print_graph_diff(wbmg, wbmg2)


# some fails, some successes...
generated_examples_test()
