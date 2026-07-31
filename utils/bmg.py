from os import makedirs, path
import asymmetree.treeevolve as te
from asymmetree.analysis.best_matches import bmg_from_tree
from tralda.datastructures import Tree

import networkx as nx 

TREE_DIR="./trees/"

NEWICK_GENES="./trees/gene_tree.txt"
NEWICK_SPECIES="./trees/species_tree.txt"

GML_SPECIES="trees/species_tree.gml"
GML_GENES="trees/genes_tree.gml"

TRALDA_SPECIES="trees/species_tree.pickle"
TRALDA_GENES="trees/genes_tree.pickle"

BMG_FILE="trees/bmg.gml"

AVAILABLE_COLORS = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:cyan', 'tab:purple', 'tab:gray']

def create_trees():
    s = te.species_tree_n(4)

    t = te.dated_gene_tree(s, dupl_rate=0.5, loss_rate=0.3, hgt_rate=0.1)
    t = te.rate_heterogeneity(t, s, base_rate=1, autocorr_variance=0.2, rate_increase=("gamma", 0.5, 2.2 ) )
    t = te.prune_losses(t)


    return s, t

def build_species_color_map(bmg, colors: list[str]) -> dict:
    species_color_map = dict()
    for node, data in bmg.nodes(data=True):
        node_species = data.get("color", "None")

        if node_species not in species_color_map.keys():
            species_color_map[node_species] = colors.pop(0)

    return species_color_map

def save_graphs(species_tree, genes_tree, g, color_map: dict):
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

    for node, data in t_nx.nodes(data=True):
        data["dist"] = float(data["dist"])
        # data['species'] = data.get('reconc', 'None')
        data['color'] = color_map.get(data.get('reconc', 'lightgray'), 'lightgray')

    nx.write_gml(s_nx, "trees/species_tree.gml")
    nx.write_gml(t_nx, "trees/genes_tree.gml")

    nx.write_gml(g, "trees/bmg.gml")

    species_tree.serialize(TRALDA_SPECIES)
    genes_tree.serialize(TRALDA_GENES)


def load_bmg() -> tuple:
    if path.exists(TRALDA_SPECIES) or path.exists(TRALDA_GENES):
        t = nx.read_gml(GML_GENES)
        s = Tree.load(TRALDA_SPECIES)
        g = nx.read_gml(BMG_FILE)
      
    else:
        s, t = create_trees()
        g = bmg_from_tree(t)
        if not path.isdir(TREE_DIR):
            makedirs(TREE_DIR)
        save_graphs(s, t, g. color_map)
    return (s, t, g)

def main():
    if not path.isdir(TREE_DIR):
        makedirs(TREE_DIR)
        print("creating tree files")
        s, t = create_trees()
        g = bmg_from_tree(t)
        color_map = build_species_color_map(g, AVAILABLE_COLORS)
        for node, data in g.nodes(data=True):
            data['species'] = data.get('color', 'None')
            data['color'] = color_map[data.get('color', 'None')]

        save_graphs(s, t, g, color_map)
        return
    else:
        return load_bmg()

if __name__ == "__main__":
    main()
