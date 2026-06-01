import networkx as nx

from utils.graph_utils import show_graph, transform

if __name__ == "__main__":
    G = nx.read_gml("tests/gene_tree_test_file.gml")

    G_transformed = transform(G, 10)
    show_graph(G_transformed)
