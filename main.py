import matplotlib.pyplot as plt
import networkx as nx
from asymmetree.analysis.best_matches import bmg_from_tree
from asymmetree.utils.phylogenetic_trees import random_colored_tree

"""
Convert tralda.Tree into network x
"""


def main():
    n = 2
    colors = 2
    bic_tree = random_colored_tree(n, colors)
    print(bic_tree.to_newick())
    bmg = bmg_from_tree(bic_tree)
    nx.draw_networkx(bmg)
    plt.show()


"""
GOAL: Edit a bicolored tree into a phylogenetic network by inserting hybridization vertices
"""
if __name__ == "__main__":
    # main()
    n = 15
    colors = 5
    bic_tree = random_colored_tree(n, colors)

    graph, root_id = bic_tree.to_nx()

    nx.write_gml(graph, "yikes.gml")
    """
    for node, data in graph.nodes.data:
        print(node)
    """
