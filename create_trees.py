from os import makedirs, path
import asymmetree.treeevolve as te
from tralda.datastructures import Tree

import networkx as nx 

TREE_DIR="./trees/"

NEWICK_GENES="./trees/gene_tree.txt"
NEWICK_SPECIES="./trees/species_tree.txt"

GML_SPECIES="trees/species_tree.gml"
GML_GENES="trees/genes_tree.gml"

TRALDA_SPECIES="trees/species_tree.pickle"
TRALDA_GENES="trees/genes_tree.pickle"

def create_trees():
    s = te.species_tree_n(3)

    t = te.dated_gene_tree(s, dupl_rate=0.5, loss_rate=0.3, hgt_rate=0.1)
    t = te.rate_heterogeneity(t, s, base_rate=1, autocorr_variance=0.2, rate_increase=("gamma", 0.5, 2.2 ) )
    t = te.prune_losses(t)

    return s, t

def save_trees(species_tree, genes_tree):
    s_netwick = species_tree.to_newick()
    t_netwick = genes_tree.to_newick()

    with open(NEWICK_GENES, "w") as f:
        f.write(t_netwick)
    with open(NEWICK_SPECIES, "w") as f:
        f.write(s_netwick)

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

    species_tree.serialize(TRALDA_SPECIES)
    genes_tree.serialize(TRALDA_GENES)


def gather_trees() -> tuple:
    if path.exists(TRALDA_SPECIES) or path.exists(TRALDA_GENES):
        t = Tree.load(TRALDA_GENES)
        s = Tree.load(TRALDA_SPECIES)
      
    else:
        s, t = create_trees()
        if not path.isdir(TREE_DIR):
            makedirs(TREE_DIR)
        save_trees(s, t)
    return (s, t)

def main() -> None:
    if not path.isdir(TREE_DIR):
        makedirs(TREE_DIR)
        print("creating tree files")
        s, t = create_trees()
        save_trees(s, t)
        return
    print("tree files already exist")

if __name__ == "__main__":
    main()
