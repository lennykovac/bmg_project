from utils.graph_utils import transform, bmg_from_network, wbmg_from_network
from utils.tree_utils import create_gene_tree_n_leaves
import networkx as nx


def bmg_wbmg_check(G: nx.DiGraph):
    """
    Checks whether best matches are also weak best matches and vice versa.

    Parameters:
    G: Directed Graph

    Returns:
    Boolean values bmg_implies_wbmg and wbmg_implies_bmg.
    """

    bmg = bmg_from_network(G)
    wbmg = wbmg_from_network(G)
    bmg_implies_wbmg = False
    wbmg_implies_bmg = False

    # check bmg -> wbmg
    edges = set(bmg.edges()).issubset(set(wbmg.edges()))
    nodes = set(bmg.nodes()).issubset(set(wbmg.nodes()))

    if edges and nodes:
        bmg_implies_wbmg = True
    else:
        bmg_implies_wbmg = False

    # check wbmg -> bmg
    edges = set(wbmg.edges()).issubset(set(bmg.edges()))
    nodes = set(wbmg.nodes()).issubset(set(bmg.nodes()))

    if edges and nodes:
        wbmg_implies_bmg = True
    else:
        wbmg_implies_bmg = False

    # sanity check, should be true if graphs are equal - was always correct
    # if bmg_implies_wbmg and wbmg_implies_bmg:
    #     assert nx.is_isomorphic(bmg, wbmg)

    return bmg_implies_wbmg, wbmg_implies_bmg


def main():
    species = 10
    leaves = 20
    species_tree_age = 1
    # we know bmg -> wbmg. But not wbmg->bmg. So if bmg_implies_wbmg is false, our theoretical prediction would be wrong
    wrong_combinations = 0
    implies_bmg = 0
    not_implies_bmg = 0
    for i in range(1000):
        trees = create_gene_tree_n_leaves(leaves, species, species_tree_age)

        gene_tree_di_graph = trees.gene_tree

        G_transformed = transform(gene_tree_di_graph, 10)
        bmg_implies_wbmg, wbmg_implies_bmg = bmg_wbmg_check(G_transformed)
        if not bmg_implies_wbmg:
            wrong_combinations += 1

        if wbmg_implies_bmg:
            implies_bmg += 1
        else:
            not_implies_bmg += 1

    print(wrong_combinations, implies_bmg, not_implies_bmg)


if __name__ == "__main__":
    main()
