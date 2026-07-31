from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

from bmg import load_bmg
from bic_cherry import load_cherry
from networkx.drawing.nx_pydot import graphviz_layout

SPECIES_TREE_PLOT_FILE = "plots/species_tree.png"
GENE_TREE_PLOT_FILE = "plots/gene_tree.png"
BMG_PLOT_FILE = "plots/bmg.png"
BIC_CHERRY_PLOT_FILE = "plots/bic_cherry.png"


def get_node_attributes(G):
    node_attr=defaultdict(list)
    for node, data in G.nodes(data=True):
        node_attr['colors'].append(data.get('color', 'lightgray'))
        node_attr['sizes'].append(data.get('size', 1000))
    # node_attr['labels']={n: n for n, data in G.nodes(data=True) }
    return node_attr

def get_edge_attributes(G):
    edge_attr=defaultdict(list)
    for node, data in G.nodes(data=True):
        edge_attr['edge_color'].append(data.get('color', 'black'))
    return edge_attr

def draw_graph(
    G,
    pos,
    output_file,
    figsize=(15, 10),
    draw_labels=True,
    ):
    fig, ax = plt.subplots(figsize=figsize)
    # fig, ax = plt.subplots()

    node_attr = get_node_attributes(G)
    edge_attr = get_edge_attributes(G)
    # print(node_attr)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_attr["colors"],
        node_size=node_attr["sizes"],
        edgecolors="black",
        linewidths=2,
        # margins=(0.1, 0.2)
    )

    nx.draw_networkx_edges(
        G,
        pos,
        # edge_color=edge_attr["colors"],
        # width=10,
    )

    if draw_labels:
        # print(node_attr['labels'])
        nx.draw_networkx_labels(
            G,
            pos,
            # labels=node_attr["labels"],
            font_size=10,
        )

    # ax.set_axis_off()

    # fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    fig.savefig(output_file)
    plt.close(fig)


def plot_gene_tree(t):
    pos = graphviz_layout(t, prog="dot")

    draw_graph(
        t,
        pos,
        GENE_TREE_PLOT_FILE,
    )


def plot_bmg(g):
    pos = nx.circular_layout(g)

    draw_graph(
        g,
        pos,
        BMG_PLOT_FILE,
    )

def plot_cherry(bc):
    pos = nx.multipartite_layout(
        bc,
        subset_key="layer",
        align="horizontal",
    )

    draw_graph(
        bc,
        pos,
        BIC_CHERRY_PLOT_FILE,
        figsize=(25,10)
    )


def main():
    s, t, g = load_bmg()
    bic_cherry = load_cherry()

    plot_gene_tree(t)
    plot_bmg(g)
    plot_cherry(bic_cherry)


if __name__ == "__main__":
    main()
