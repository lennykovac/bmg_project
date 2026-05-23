from collections import defaultdict
import asymmetree.treeevolve as te
import asymmetree.visualization.tree_vis as vis
from asymmetree.analysis.best_matches import bmg_from_tree
from asymmetree.utils.phylogenetic_trees import to_newick
from create_trees import gather_trees
import numpy as np

from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt

import networkx as nx


s, t = gather_trees()



# G = bmg_from_tree(T)

t = nx.convert_node_labels_to_integers(t)

t.add_edge(4,8)

pos = graphviz_layout(t, prog="dot")
nx.draw(t, pos=pos, with_labels=True)
plt.show()

# T_leaves = T.leaves()

# gene_species = defaultdict(list)
# for x in T_leaves:
#     attr = list(x.attributes())
#     gene_species[attr[2][1]].append(attr[0][1])
#    event "S"
