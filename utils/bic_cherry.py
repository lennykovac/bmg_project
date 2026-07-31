from os import path
from bmg import load_bmg

import networkx as nx

AVAILABLE_COLORS = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:cyan', 'tab:purple', 'tab:gray']
BIC_FILE='trees/bic_cherry.gml'

def build_species_color_map(bmg, colors: list[str]) -> dict:
    species_color_map = dict()
    for node, data in bmg.nodes(data=True):
        node_species = data.get("color", "None")

        if node_species not in species_color_map.keys():
            species_color_map[node_species] = colors.pop(0)

    return species_color_map

def init_graph(bmg, colors) -> nx.DiGraph:
    graph = nx.DiGraph(name="bic_cherry")
    graph.add_node("r", color="black", kind='root', layer=3, label="Root")

    for node, data in bmg.nodes(data=True):
        node_color = data.get("color", "None")
        node_species = data.get("species", "None")
        graph.add_node(node, species=node_species, color=node_color, kind='leaf', layer=0, label=node)
    return graph

def add_cherry_parents(graph: nx.DiGraph) -> nx.DiGraph:
    leaf_data = [
        (n, data['color'])
        for n, data in graph.nodes(data=True)
        if data.get('kind') == 'leaf'
    ]

    for i in range(len(leaf_data)):
        for j in range(i + 1, len(leaf_data)):
            label_i, color_i = leaf_data[i]
            label_j, color_j = leaf_data[j]
            if color_i != color_j:
                parent_name = f"p_{label_j}-{label_i}"
                parent_label = f"P_{label_j}-{label_i}"
                graph.add_node(parent_name, color="tab:gray", kind='cherry_parent', layer=2, label=parent_label)
                graph.add_edge("r", parent_name)
                graph.add_edge(parent_name, label_i)
                graph.add_edge(parent_name, label_j)
    return graph

def bic_extension(graph: nx.DiGraph, parent_node, child_node, extension_leaf_node) -> nx.DiGraph:
    if (extension_leaf_node, child_node) not in graph.edges:
        raise ValueError("extension edge not in bmg")

    if (parent_node, extension_leaf_node) not in graph.edges:
        extension_node = f"q_{child_node}-{extension_leaf_node}"
        extension_label = f"Q_{child_node}-{extension_leaf_node}"

        graph.add_node(extension_node, color="darkgray", kind="extension", layer=1, size=5000, label=extension_label)
        graph.add_edge(parent_node, extension_node)
        graph.remove_edge(parent_node, child_node)
        graph.add_edge(extension_node, child_node, edge_color="blue")
        graph.add_edge(extension_node, extension_leaf_node, edge_color="blue")

        return graph
    return graph

def save_cherry(graph: nx.DiGraph):
    print("saving bic_cherry")
    nx.write_gml(graph, BIC_FILE)
    
def load_cherry():
    print("loading bic_cherry")
    return nx.read_gml(BIC_FILE)

def main():
    s, t, g = load_bmg()

    if path.exists(BIC_FILE):
        return load_cherry()
    else:
        species_color_map = build_species_color_map(g, AVAILABLE_COLORS)
        bic_cherry = init_graph(g, species_color_map)
        bic_cherry = add_cherry_parents(bic_cherry)
        save_cherry(bic_cherry)
        # bic_cherry = bic_extension(bic_cherry, "p_3-7", 7, 5)
        # draw_graph(bic_cherry)

if __name__ == "__main__":
    main()
