from itertools import combinations, product
import networkx as nx


def bic_cherry(bmg: nx.DiGraph):
    """
    Construct BIC-cherry network (before any extensions)

    Parameters:
    bmg: valid bmg graph, no self loops, sicor-in-hub property

    Returns:
    network: resulting bic-cherry network
    pairs: all node pairs of different color (needed for extensions)
    """
    # construct BIC-cherry network
    network = nx.DiGraph()
    network.add_nodes_from(  # careful: bmg.nodes(data="color") returns a tuple, not a dict!
        (n, {"color": color}) for n, color in bmg.nodes(data="color")
    )

    # find all pairs of different colored nodes
    color_groups = {}  # build dict with with list of all nodes by color
    for node, color in nx.get_node_attributes(bmg, "color").items():
        color_groups.setdefault(color, []).append(node)

    pairs = []  # collect all pairs of different color
    colors = list(color_groups.keys())

    for x, y in combinations(colors, 2):
        pairs.extend(product(color_groups[x], color_groups[y]))

    # build root and basic parent nodes and basic edges
    network.add_node("R", color=None)  # new root
    for x, y in pairs:
        network.add_node(f"p:{x}|{y}", color=None)
        network.add_edge("R", f"p:{x}|{y}")
        network.add_edge(f"p:{x}|{y}", x)
        network.add_edge(f"p:{x}|{y}", y)

        network.add_node(f"p:{y}|{x}", color=None)
        network.add_edge("R", f"p:{y}|{x}")
        network.add_edge(f"p:{y}|{x}", x)
        network.add_edge(f"p:{y}|{x}", y)
    return network, pairs


def bic_cherry_extension(bmg):
    """
    Construct explaining network from BMG using the BIC-cherry + Expansion Algo from the paper.

    Parameters:
    bmg: valid bmg graph, no self loops, sicor-in-hub property

    Returns:
    network: an explaining network for input BMG
    """
    network, pairs = bic_cherry(bmg)
    bmg_edges = set(bmg.edges())

    # pairs to do extensions for (direction sensitive)
    extend_pairs = [
        edge for (u, v) in pairs for edge in [(u, v), (v, u)] if edge not in bmg_edges
    ]

    for x, y in extend_pairs:
        z = [
            n
            for n, color in bmg.nodes(data="color")
            if n != y and color == bmg.nodes[y]["color"]
        ][0]

        network.add_node(f"q:{x}|{z}", color=None)
        network.add_edge(f"p:{x}|{y}", f"q:{x}|{z}")
        network.add_edge(f"q:{x}|{z}", x)
        network.add_edge(f"q:{x}|{z}", z)

    return network


def restricted_bic_cherry_extension(bmg):
    """
    Construct explaining network from BMG using the BIC-cherry + Expansion Algo from the paper where expansion partner nodes (z)
    must be chosen such that (x, z) is an edge in the BMG

    Parameters:
    bmg: valid bmg graph, no self loops, sicor-in-hub property

    Returns:
    network: an explaining network for input BMG
    """
    network, pairs = bic_cherry(bmg)
    bmg_edges = set(bmg.edges())

    # pairs to do extensions for (both directions!)
    extend_pairs = [
        edge for (u, v) in pairs for edge in [(u, v), (v, u)] if edge not in bmg_edges
    ]

    for x, y in extend_pairs:
        z = [
            n
            for n, color in bmg.nodes(data="color")
            if n != y and color == bmg.nodes[y]["color"] and (x, n) in set(bmg.edges())
        ][0]
        network.add_node(f"q:{x}|{z}", color=None)
        network.add_edge(f"p:{x}|{y}", f"q:{x}|{z}")
        network.add_edge(f"q:{x}|{z}", x)
        network.add_edge(f"q:{x}|{z}", z)

    return network
