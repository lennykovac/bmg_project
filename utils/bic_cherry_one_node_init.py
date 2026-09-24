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

def bic_cherry_expansion(bmg):
    """
    Construct explaining network from BMG using the BIC-cherry + Expansion Algo from the paper.

    Parameters:
    bmg: valid bmg graph, no self loops, sicor-in-hub property

    Returns:
    network: an explaining network for input BMG
    """
    network, pairs = bic_cherry(bmg)
    bmg_edges = set(bmg.edges())

    # pairs to do expansions for (direction sensitive)
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


def restricted_bic_cherry_expansion(bmg):
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

    # pairs to do expansions for (both directions!)
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





def restricted_bic_cherry_more_expansions(bmg):
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




            w_candidates = [
                n for n, color in bmg.nodes(data="color")
                if n != curr_x and color == bmg.nodes[curr_x]["color"]
                   and (curr_z, n) in bmg_edges and (y, n) in bmg_edges
            ]



            if len(w_candidates) == 0:
                print("kein w candidat gefunden")
                break

            w = w_candidates[0]
            for w_search in w_candidates:
                if (w_search, curr_z) in bmg_edges:
                    w = w_search
                    break




            color_y = bmg.nodes[y]["color"]
            color_w = bmg.nodes[w]["color"]



            parent_node = q_node
            curr_x = curr_z
            curr_z = w
            depth += 1



    return network, pairs, extend_pairs







#[xy:xz]
def expansion(x, z, parent_node, network, depth):
    u, v = sorted([x, z], key=lambda item: str(item))
    q_prefix = f"q{depth if depth > 0 else ''}"
    q_node = f"{q_prefix}:{u}|{v}"

    network.add_node(q_node, color=None)
    network.add_edge(parent_node, q_node)
    network.add_edge(q_node, x)
    network.add_edge(q_node, z)

    return network




def wbmg_bic_cherry(wbmg):
    network, pairs = bic_cherry(wbmg)
    wbmg_edges = set(wbmg.edges())

    second_extend_pairs = set()
    third_extend_pairs = set()
    fourth_extend_pairs = set()
    fifth_extend_pairs = set()

    extend_pairs = [
        edge for (u, v) in pairs for edge in [(u, v), (v, u)] if edge not in wbmg_edges
    ]


    for x, y in extend_pairs:

        z_candidates = [
            n for n, color in wbmg.nodes(data="color")
            if n != y and color == wbmg.nodes[y]["color"] and (x, n) in wbmg_edges
        ]
        if not z_candidates:
            continue

        z = z_candidates[0]
        found = False

        for z_search in z_candidates:
            if (z_search, x) in wbmg_edges:
                z = z_search
                found = True
                break

        u, v = sorted([x, y], key=lambda item: str(item))
        parent_node = f"p:{u}|{v}"


        network = expansion(x, z, parent_node, network, depth=0)

        if not found:
            second_extend_pairs.add((x, y, z))


    for x, y, z in second_extend_pairs:

        py = y
        w_candidates = [
            n for n, color in wbmg.nodes(data="color")
            if n != x and color == wbmg.nodes[x]["color"] and (y, n) in wbmg_edges
        ]
        if not w_candidates:
            continue

        w = w_candidates[0]

        found = False

        for w_search in w_candidates:
            has_forward = (z, w_search) in wbmg_edges
            has_backward = (w_search, z) in wbmg_edges

            if has_forward and has_backward:
                w = w_search
                found = True
                break
            elif has_forward:
                w = w_search
                third_extend_pairs.add((w, z, py))
                found = True
                break
            elif has_backward:
                w = w_search
                third_extend_pairs.add((z, w, py))
                found = True
                break

        u, v = sorted([z, x], key=lambda item: str(item))
        parent_node = f"q:{u}|{v}"


        network = expansion(z, w, parent_node, network, depth=1)

        if not found:
            third_extend_pairs.add((w, z, py))
            third_extend_pairs.add((z, w, py))


    for x, y, py in fourth_extend_pairs:

        w_candidates = [
            n for n, color in wbmg.nodes(data="color")
            if n != y and color == wbmg.nodes[y]["color"] and (py, n) in wbmg_edges
        ]
        if not w_candidates:
            continue

        w = w_candidates[0]
        found = False

        for w_search in w_candidates:
            has_forward = (x, w_search) in wbmg_edges
            has_backward = (w_search, x) in wbmg_edges

            if has_forward and has_backward:
                w = w_search
                found = True
                break
            elif has_forward:
                w = w_search
                fourth_extend_pairs.add((w, x, py))
                found = True
                break
            elif has_backward:
                w = w_search
                fourth_extend_pairs.add((x, w, py))
                found = True
                break

        u, v = sorted([x, y], key=lambda item: str(item))
        parent_node = f"q1:{u}|{v}"

        network = expansion(x, w, parent_node, network, depth=2)

        if not found:
            fourth_extend_pairs.add((w, x, py))
            fourth_extend_pairs.add((x, w, py))


    for x, y, py in fourth_extend_pairs:

            w_candidates = [
                n for n, color in wbmg.nodes(data="color")
                if n != y and color == wbmg.nodes[y]["color"] and (py, n) in wbmg_edges
            ]
            if not w_candidates:
                continue

            w = w_candidates[0]
            found = False

            for w_search in w_candidates:
                has_forward = (x, w_search) in wbmg_edges
                has_backward = (w_search, x) in wbmg_edges

                if has_forward and has_backward:
                    w = w_search
                    found = True
                    break
                elif has_forward:
                    w = w_search
                    fifth_extend_pairs.add((w, x, py))
                    found = True
                    break
                elif has_backward:
                    w = w_search
                    fifth_extend_pairs.add((x, w, py))
                    found = True
                    break

            u, v = sorted([x, y], key=lambda item: str(item))
            parent_node = f"q2:{u}|{v}"

            network = expansion(x, w, parent_node, network, depth=3)



    return network, pairs, extend_pairs



def expansion2(x, z, parent_node, network, depth):
    u, v = sorted([x, z], key=lambda item: str(item))
    q_prefix = f"q{depth if depth > 0 else ''}"
    q_node = f"{q_prefix}:{u}|{v}"

    network.add_node(q_node, color=None)
    network.add_edge(parent_node, q_node)
    network.add_edge(q_node, x)
    network.add_edge(q_node, z)

    return network, q_node


def wbmg_bic_cherry2(wbmg, max_depth=5):
    network, pairs = bic_cherry(wbmg)
    wbmg_edges = set(wbmg.edges())


    current_extend_pairs = [
        edge for (u, v) in pairs for edge in [(u, v), (v, u)] if edge not in wbmg_edges
    ]


    for depth in range(max_depth):
        if not current_extend_pairs:
            break

        next_extend_pairs = set()

        for x, y in current_extend_pairs:

            z_candidates = [
                n for n, color in wbmg.nodes(data="color")
                if n != y and color == wbmg.nodes[y]["color"] and (x, n) in wbmg_edges
            ]



            z = z_candidates[0]
            found = False

            for z_search in z_candidates:
                if (z_search, x) in wbmg_edges:
                    z = z_search
                    found = True
                    break

            u, v = sorted([x, y], key=lambda item: str(item))
            parent_node = f"p:{u}|{v}" if depth == 0 else f"q{depth}:{u}|{v}"


            network, q = expansion2(x, z, parent_node, network, depth=depth)


            if not found:
                w_candidates = [
                    n for n, color in wbmg.nodes(data="color")
                    if n != x and color == wbmg.nodes[x]["color"] and (z, n) in wbmg_edges
                ]



                w = w_candidates[0]
                found2 = False

                for w_search in w_candidates:
                    if (w_search, z) in wbmg_edges:
                        w = w_search
                        found2 = True
                        break


                network, q1 = expansion2(z, w, q, network, depth=depth + 1)

                if (y, w) not in wbmg_edges:
                    t_node = f"lca:{y}|{w}"
                    x_node = f"lca:{y}|{x}"
                    network.add_node(x_node, color=None)
                    network.add_node(t_node, color=None)

                    network.add_edge(t_node, x_node)
                    network.add_edge(parent_node, t_node)
                    network.add_edge(t_node, q1)
                    network.add_edge(x_node, y)
                    network.add_edge(x_node, q)


                if not found2:
                    next_extend_pairs.add((w, z))


        current_extend_pairs = list(next_extend_pairs)

    return network, pairs, current_extend_pairs
















































