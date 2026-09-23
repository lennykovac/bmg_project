import asymmetree.analysis.best_matches as bm
import asymmetree.utils.phylogenetic_trees as pt
import networkx as nx
from utils.tree_utils import create_gene_tree
from utils.lrt import(
    compute_lrt
)
from utils.graph_utils import (
    bmg_from_network,
    wbmg_from_network
)
from utils.graph_utils import (
    print_compare_bmg
)
from utils.bic_cherry import (
    bic_cherry_extension,
)

from utils.graph_editing import (
    pull_up,
    pull_down,
    pull_up_to_common_ancestor,
    find_twin_vertices,
    remove_redundant_vertex,
    remove_useless_vertex,
    try_edit,
)

from utils.network_search import (
    make_still_valid
)




'''
(a) Construct tree-BMGs (G, σ) from (AsymmeTree-generated) trees and
compute their least resolved trees. This is the target.
'''
species = 2
species_tree_age = 1

tree = create_gene_tree(species, species_tree_age).gene_tree
tree_bmg = bmg_from_network(tree)
tree_lrt, report  = compute_lrt(tree)

leaves = [v for v in tree.nodes if tree.out_degree(v) == 0]
for leaf in leaves:

    color = tree.nodes[leaf].get("color", "Keine Farbe")
    print(f"Blatt {leaf!r} hat die Farbe/Spezies: {color}")


'''
(b) Take the tree-BMGs from (a) and construct the BIC-cherry+expansion
explanations (N, σ).
'''

network = bic_cherry_extension(tree_bmg)


'''
(d) Check both for single moves and combinations of a small number of
moves whether the modified network (N′, σ) has the same (weak) best match graph as (N, σ).


def single_moves(network, mode="wbmg"):

    still_valid = make_still_valid(mode) #erzeugt eine funktion still_valid die prüft,ob N den selben wmbg erzeugt wie N'

    for v in network.nodes():
        net_after, applied = try_edit(
            network, remove_useless_vertex, v, still_valid=still_valid
    )



    valid_single_moves = []

'''
