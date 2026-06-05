from utils.graph_utils import show_graph, transform
from utils.tree_utils import create_gene_tree

if __name__ == "__main__":
    # G = nx.read_gml("tests/gene_tree_test_file.gml")
    species = 10
    species_tree_age = 1
    trees = create_gene_tree(species, species_tree_age)

    gene_tree_di_graph = trees.gene_tree
    species_tree_di_graph = trees.gene_tree
    print("leah war hier")

    G_transformed = transform(gene_tree_di_graph, 10)
    show_graph(G_transformed)
