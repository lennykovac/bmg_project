"""Plain helper functions shared by the test modules."""
import networkx as nx


def edges(G):
    return set(G.edges)


def colored_tree(edge_list, colors):
    """DiGraph from an edge list; leaves get their color from ``colors``, inner vertices None."""
    T = nx.DiGraph(edge_list)
    for v in T:
        T.nodes[v]["color"] = colors.get(v)
    return T


def colored_graph(colors, arcs):
    G = nx.DiGraph()
    G.add_nodes_from((v, {"color": c}) for v, c in colors.items())
    G.add_edges_from(arcs)
    return G
