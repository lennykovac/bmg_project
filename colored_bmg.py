import numpy as np
import networkx as nx
from collections import defaultdict

def random_color():
    color = np.random.rand(3)
    return color

def draw_colored_bmg(BMG_Graph, Gene_tree):
    T_leaves = Gene_tree.leaves()
    pos = nx.spring_layout(BMG_Graph)
    gene_species = defaultdict(list)
    for x in T_leaves:
        attr = list(x.attributes())
        gene_species[attr[2][1]].append(attr[0][1])


    for species, genes in gene_species.items():
        nx.draw_networkx_nodes(BMG_Graph,
                               pos=pos,
                               nodelist=genes,
                               node_color=[random_color()],
                               label=genes)

    nx.draw_networkx_edges(BMG_Graph, pos=pos)
    nx.draw_networkx_labels(BMG_Graph, pos=pos)
