from itertools import combinations, product
import networkx as nx

def bic_cherry(bmg: nx.DiGraph):
    network = nx.DiGraph()
    network.add_nodes_from(
        (n, {"color": color}) for n, color in bmg.nodes(data="color")
    )

    color_groups = {}
    for node, color in nx.get_node_attributes(bmg, "color").items():
        color_groups.setdefault(color, []).append(node)

    pairs = []
    colors = list(color_groups.keys())

    bmg_edges = set(bmg.edges())

    for x, y in combinations(colors, 2):
        # Nur hinzufügen, wenn mindestens eine Kante in bmg existiert
        pairs.extend([
            (u, v) for u, v in product(color_groups[x], color_groups[y])
            if (u, v) in bmg_edges or (v, u) in bmg_edges
        ])

        #pairs.extend(product(color_groups[x], color_groups[y]))

    network.add_node("R", color=None)


    created_cherry_nodes = set()

    for x, y in pairs:

        u, v = sorted([x, y], key=lambda item: str(item))
        cherry_node = f"p:{u}|{v}"

        if cherry_node not in created_cherry_nodes:
            network.add_node(cherry_node, color=None)
            network.add_edge("R", cherry_node)
            network.add_edge(cherry_node, u)
            network.add_edge(cherry_node, v)
            created_cherry_nodes.add(cherry_node)

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

        p_xy = f"p:{x}|{y}"
        p_yx = f"p:{y}|{x}"

        if p_xy not in network and p_yx in network:
            p_xy = p_yx

        network.add_node(f"q:{x}|{z}", color=None)
        network.add_edge(p_xy, f"q:{x}|{z}")
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

        p_xy = f"p:{x}|{y}"
        p_yx = f"p:{y}|{x}"

        if p_xy not in network and p_yx in network:
            p_xy = p_yx

        network.add_node(f"q:{x}|{z}", color=None)
        network.add_edge(p_xy, f"q:{x}|{z}")
        network.add_edge(f"q:{x}|{z}", x)
        network.add_edge(f"q:{x}|{z}", z)

    return network


def restricted_bic_cherry_more_extensions(bmg):
    network, pairs = bic_cherry(bmg)
    bmg_edges = set(bmg.edges())

    extend_pairs = [
        edge for (u, v) in pairs for edge in [(u, v), (v, u)] if edge not in bmg_edges
    ]

    for x, y in extend_pairs:



        z_candidates = [
            n
            for n, color in bmg.nodes(data="color")
            if n != y and color == bmg.nodes[y]["color"] and (x, n) in bmg_edges
        ]


        z = z_candidates[0]

        for z_search in z_candidates:
            if (z_search, x) in bmg_edges:
                z = z_search
                break

        u, v = sorted([x, y], key=lambda item: str(item))
        first_parent = f"p:{u}|{v}"
        parent_node = first_parent


        curr_x = x
        curr_z = z
        depth = 1


        while True:
            q_prefix = f"q{depth if depth > 1 else ''}"
            q_node = f"{q_prefix}:{curr_x}|{curr_z}"

            network.add_node(q_node, color=None)
            network.add_edge(parent_node, q_node)
            network.add_edge(q_node, curr_x)
            network.add_edge(q_node, curr_z)


            if (curr_z, curr_x) in bmg_edges:
                break


            '''
            w_candidates = [
                n for n, color in bmg.nodes(data="color")
                if n != curr_x and color == bmg.nodes[x]["color"] and (curr_z, n) in bmg_edges
            ]
            '''

            w_candidates = [
                n for n, color in bmg.nodes(data="color")
                if n != curr_x and color == bmg.nodes[curr_x]["color"]
                   and (curr_z, n) in bmg_edges and (y, n) in bmg_edges
            ]



            if len(w_candidates) == 0:
                break

            w = w_candidates[0]
            for w_search in w_candidates:
                if (w_search, curr_z) in bmg_edges:
                    w = w_search
                    break




            color_y = bmg.nodes[y]["color"]
            color_w = bmg.nodes[w]["color"]

            '''
            if color_y != color_w and (y,w) not in bmg_edges:
                r_node = f"r:{y}|{w}"
                network.add_node(r_node, color=None)
                network.add_edge("R", r_node)
                network.add_edge(r_node, first_parent)
                network.add_edge(r_node, w)
                network.add_edge(r_node, y)
            '''

            parent_node = q_node
            curr_x = curr_z
            curr_z = w
            depth += 1



    return network, pairs, extend_pairs