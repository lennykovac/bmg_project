from os import path
import asymmetree.treeevolve as te
from asymmetree.treeevolve.species import species_tree_n
from asymmetree.utils import phylogenetic_trees
from networkx.drawing.nx_pydot import graphviz_layout

import networkx as nx 
import matplotlib.pyplot as plt

GENETREE_FILE="./trees/gene_tree.txt"
SPECIESTREE_FILE="./trees/species_tree.txt"

GML_SPECIES="trees/species_tree.gml"
GML_GENES="trees/genes_tree.gml"

def create_trees():
    s = te.species_tree_n(5)

    t = te.dated_gene_tree(s, dupl_rate=0.5, loss_rate=0.3, hgt_rate=0.1)
    t = te.rate_heterogeneity(t, s, base_rate=1, autocorr_variance=0.2, rate_increase=("gamma", 0.5, 2.2 ) )
    t = te.prune_losses(t)

    return s, t

def save_trees(species_tree, genes_tree):
    s_netwick = species_tree.to_newick()
    t_netwick = genes_tree.to_newick()

    s_nx, s_root = species_tree.to_nx()
    t_nx, t_root = genes_tree.to_nx()

    for node, data in s_nx.nodes.data():
        for k, v in data.items():
            if v == None:
                data[k] = "root"
            data["dist"] = float(data["dist"])

    for node, data in t_nx.nodes.data():
        for k, v in data.items():
            data["dist"] = float(data["dist"])

    nx.write_gml(s_nx, "trees/species_tree.gml")
    nx.write_gml(t_nx, "trees/genes_tree.gml")

    with open(GENETREE_FILE, "w") as f:
        f.write(t_netwick)
    with open(SPECIESTREE_FILE, "w") as f:
        f.write(s_netwick)

def gather_trees() -> tuple:
    if path.exists(GML_GENES) or path.exists(GML_SPECIES):
        t = nx.read_gml(GML_GENES)
        s = nx.read_gml(GML_SPECIES)
      
    else:
        s, t = create_trees()
        save_trees(s, t)
    return (s, t)

if __name__ == "__main__":
    # if not path.exists(GENETREE_FILE) or not path.exists(SPECIESTREE_FILE):
        s, t = create_trees()
        save_trees(s, t)
