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


def _witness_chain(wbmg, edges, x, y):
    """
    Search a chain x = c_0 -> c_1 = y -> c_2 -> ... -> c_k -> c_k+1 of wbmg edges from which
    wbmg_cherry_extension builds a vertex that witnesses (x, y) and only true edges.

    The chain alternates between the colors of x and y, has no repeated nodes, every c_i points to
    all c_j with j > i at odd distance and it ends in a reciprocal pair (c_k+1, c_k).
    For every edge of a WBMG such a chain exists (the minimal vertices blocking the reverse edges
    in any explaining network form one), so None means wbmg is not a WBMG.

    Parameters:
    wbmg: vertex colored digraph
    edges: set of the edges of wbmg
    x, y: edge (x, y) of wbmg

    Returns:
    chain as list of nodes, or None if there is none
    """
    color = wbmg.nodes(data="color")

    def extend(chain):
        if (chain[-1], chain[-2]) in edges:
            return chain
        m = len(chain)
        for c in wbmg.successors(chain[-1]):
            if c in chain or color[c] != color[chain[-2]]:
                continue
            if all((chain[i], c) in edges for i in range(m - 1, -1, -2)):
                found = extend(chain + [c])
                if found:
                    return found
        return None

    return extend([x, y])


def wbmg_cherry_extension(wbmg):
    """
    Construct explaining network for a WBMG (weak best matches).

    BIC-cherry + extension does not work for weak best matches: a single lowest common ancestor that
    is not strictly above another one is enough for a weak best match, and every p_xy or q_xz is a
    lowest common ancestor for both directions of its pair.

    Instead only reciprocal edges get a cherry p:x|y. A one directional edge (x, y) is explained by
    extending y: q has the children x and the vertex of the next chain edge (y, z) (see _witness_chain).
    q witnesses (x, w) for all leaves w below it, but no reverse edge, because the child already
    reaches every such w together with a leaf of x's color.

    Parameters:
    wbmg: weak best match graph (of some network)

    Returns:
    network: network whose weak best match graph is wbmg

    Raises:
    ValueError: if wbmg is not the weak best match graph of any network
    """
    network = nx.DiGraph()
    network.add_nodes_from((n, {"color": color}) for n, color in wbmg.nodes(data="color"))
    network.add_node("R", color=None)

    edges = set(wbmg.edges())
    witnessed = set()  # edges explained by an already built vertex
    for x, y in wbmg.edges():
        if (x, y) in witnessed:
            continue
        chain = _witness_chain(wbmg, edges, x, y)
        if chain is None:
            raise ValueError(f"no network has {(x, y)} as weak best match, input is not a WBMG")

        # innermost reciprocal cherry, reuse it if it already exists in the other orientation
        u, v = chain[-2], chain[-1]
        child = f"p:{v}|{u}" if f"p:{v}|{u}" in network else f"p:{u}|{v}"
        network.add_node(child, color=None)
        network.add_edge(child, u)
        network.add_edge(child, v)
        witnessed |= {(u, v), (v, u)}

        # extend outwards, q:c_i|...|c_k+1 witnesses (c_i, c_j) for odd j - i
        for i in range(len(chain) - 3, -1, -1):
            parent = "q:" + "|".join(map(str, chain[i:]))
            network.add_node(parent, color=None)
            network.add_edge(parent, chain[i])
            network.add_edge(parent, child)
            witnessed |= {(chain[i], c) for c in chain[i + 1 :: 2]}
            child = parent

        network.add_edge("R", child)

    return network
