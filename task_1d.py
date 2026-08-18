import networkx as nx
from utils.graph_utils import transform, wbmg_from_network
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

        # if not nx.is_isomorphic(bmg, new_bmg):
        #     print("bmg nodes: ", bmg.nodes(data=True))
        #     print("bmg edges: ", bmg.edges())
        #     print("\n\n network nodes: ", network.nodes(data=True))
        #     print("network edges: ", network.edges())
        #     print("\n\n new_bmg nodes: ", new_bmg.nodes(data=True))
        #     print("new_bmg edges: ", new_bmg.edges())
        #     print_graph_diff(bmg, new_bmg)
        #     assert False


# some fails, some successes...
generated_examples_test()
