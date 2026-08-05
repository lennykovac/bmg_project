import networkx as nx
from utils.graph_utils import (
    bmg_from_network,
    transform,
)
from utils.tree_utils import create_gene_tree
from utils.bic_cherry import bic_cherry_extension


def test_bic_cherry_extension():
    # generate 10 random gene networks
    species = 10
    species_tree_age = 1
    for i in range(10):
        trees = create_gene_tree(species, species_tree_age)

        gene_tree_di_graph = trees.gene_tree

        G_transformed = transform(gene_tree_di_graph, 10)
        # compute the bmg
        bmg = bmg_from_network(G_transformed)

        network = bic_cherry_extension(bmg)
        assert nx.is_isomorphic(bmg, bmg_from_network(network))
