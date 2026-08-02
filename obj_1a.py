from utils.graph_utils import show_graph, transform, bmg_from_network, wbmg_from_network
from utils.tree_utils import create_gene_tree
import networkx as nx

# False basic example
G = nx.DiGraph()

G.add_node(0, label="0", reconc="0")
G.add_node(1, label="1", reconc="0")
G.add_node(2, label="2", reconc="0")
G.add_node(3, label="3", reconc="0")
G.add_node(4, label="4", reconc="0")
G.add_node(5, label="5", reconc="1")
G.add_node(6, label="6", reconc="0")

G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4), (1, 4), (3, 5), (3, 6), (2, 5)])

# True basic example

Gt = nx.DiGraph()

Gt.add_node(0, label="0", reconc="0")
Gt.add_node(1, label="1", reconc="0")
Gt.add_node(2, label="2", reconc="1")

Gt.add_edges_from([(0, 1), (0, 2)])


def bmg_wbmg_check(G: nx.DiGraph):

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


# basic test
# also_wbmg, also_bmg = bmg_wbmg_check(Gt)
# print(also_bmg, also_wbmg)

species = 10
species_tree_age = 1
# we know bmg -> wbmg. But not wbmg->bmg. So if also_wbmg is false, our theoretical prediction would be wrong
expected_combination = 0
wrong_combinations = 0
for i in range(1000):
    trees = create_gene_tree(species, species_tree_age)

    gene_tree_di_graph = trees.gene_tree
    species_tree_di_graph = trees.gene_tree

    G_transformed = transform(gene_tree_di_graph, 10)
    also_bmg, also_wbmg = bmg_wbmg_check(G_transformed)
    if also_wbmg:
        expected_combination += 1
    else:
        wrong_combinations += 1

print(
    expected_combination, wrong_combinations
)  # checked with 1000 networks and 10 species each, all as expected
