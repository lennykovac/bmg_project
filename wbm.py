import matplotlib.pyplot as plt
import networkx as nx
from asymmetree.analysis.best_matches import bmg_from_tree
from asymmetree.utils.phylogenetic_trees import random_colored_tree
from utils.graph_utils import show_graph


def main():
    n = 2
    colors = 2
    bic_tree = random_colored_tree(n, colors)
    print(bic_tree.to_newick())
    bmg = bmg_from_tree(bic_tree)
    nx.draw_networkx(bmg)
    plt.show()


def leaves_from_network():
    return 0


# TODO: adapt to networks
def bmg_from_network(
    network: nx.DiGraph,
) -> nx.DiGraph | tuple[nx.DiGraph, nx.Graph]:
    """Construct a BMG.

    Args:
        network: A network with leaves that has the `label` and `reconc` attribute set.

    Returns:
        The constructed BMG.
    """
    if not isinstance(network, nx.DiGraph):
        raise TypeError("not of type 'Tree'")

    leaves = network.leaf_dict()
    bmg = nx.DiGraph()
    colors = set()

    for v in leaves[network.root]:  # how to get all leaves of network?
        colors.add(v.reconc)
        bmg.add_node(v.label, color=v.reconc)

    for u in leaves[network.root]:
        remaining = colors - set(
            [u.reconc]
        )  # colors to which no best match has yet been found
        parent = u.parent  # parent no longer unique! How to traverse network backwards? variable tuples of vertices?

        while remaining and parent:  # exclude root and processed vertices
            colors_here = set()
            for v in leaves[parent]:  # leaves in subtree of parent??
                if v.reconc in remaining:  # best match found
                    colors_here.add(v.reconc)
                    bmg.add_edge(u.label, v.label)
            remaining -= colors_here
            parent = parent.parent

    return bmg

    # TODO: implement wbmg_from_network


"""
GOAL: experiment towards bmg & wbmg from networks
"""
if __name__ == "__main__":
    G = nx.DiGraph()

    G.add_node(0, label="0", color="0")
    G.add_node(1, label="1", color="0")
    G.add_node(2, label="2", color="0")
    G.add_node(3, label="3", color="0")
    G.add_node(4, label="4", color="1")
    G.add_node(5, label="5", color="1")
    G.add_node(6, label="6", color="0")
    G.add_node(7, label="7", color="0")

    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (1, 4), (3, 5), (3, 6), (2, 7)])

    # show_graph(G)
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]

    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, found {len(roots)}")

    root = roots[0]
    print(root)
    leaves = [n for n in G.nodes() if G.out_degree(n) == 0]
    print(leaves)
