import networkx as nx
from utils.graph_utils import (
    bmg_from_network,
    transform,
    wbmg_from_network)
from utils.tree_utils import create_gene_tree_n_leaves
from itertools import combinations, product
import re

G = nx.DiGraph()

nodes_data = [
    (12, {"label": "0", "color": "2"}),
    (13, {"label": "1", "color": "2"}),
    (9,  {"label": "2", "color": "2"}),
    (14, {"label": "3", "color": "2"}),
    (10, {"label": "5", "color": "2"}),
    (15, {"label": "4", "color": "3"}),
    (6,  {"label": "6", "color": "3"}),
    (7,  {"label": "7", "color": "3"}),
]

for node, attr in nodes_data:
    G.add_node(node, **attr)

G.add_edges_from([(12, 15), (13, 15), (9, 15), (9, 6), (14, 15), (15, 14), (10, 15), (10, 6), (6, 10), (7, 9), (7, 10)])



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




#[xy: xz] parent = p_xy
def extension(x, z, parent_node, network, depth):
    global q_counter
    u, v = sorted([x, z], key=lambda item: str(item))
    q_prefix = f"q{depth if depth > 0 else ''}"
    q_node = f"{q_prefix}:{u}|{v}"


    network.add_node(q_node, color=None)
    network.add_edge(parent_node, q_node)
    network.add_edge(q_node, x)
    network.add_edge(q_node, z)

    return network, q_node


def bic_cherry_for_wbm(wbmg, max_depth=10):
    network, pairs = bic_cherry(wbmg)
    wbmg_edges = set(wbmg.edges())
    depth = 0

    first_extend_pairs = [
        edge for (u, v) in pairs for edge in [(u, v), (v, u)] if edge not in wbmg_edges
    ]

    second_extend_pairs = set()

    for x, y in first_extend_pairs:
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
        parent_node = f"p:{u}|{v}"

        network, q_node = extension(x, z, parent_node, network, depth=depth)

        if found == False:
            path_visited = (parent_node, q_node)
            second_extend_pairs.add((z, x, y, q_node, path_visited ))



    current_extend_pairs = second_extend_pairs

    for depth in range(1, max_depth + 1):
        if not current_extend_pairs:
            break

        next_extend_pairs = set()

        for z, x, y, q_node, path_visited in current_extend_pairs:


            w_candidates = [
                n for n, color in wbmg.nodes(data="color")
                if n != x and color == wbmg.nodes[x]["color"] and (y, n) in wbmg_edges
            ]
            print(path_visited)
            #print(w_candidates)

            #print(w_candidates)

            if len(w_candidates) == 0:
                print(path_visited)
                print("kein w")
                break

            w = w_candidates[0]

            found_forward = False
            found_backward = False


            for w_search in w_candidates:
                has_forward = (z, w_search) in wbmg_edges
                has_backward = (w_search, z) in wbmg_edges

                if has_forward and has_backward:
                    w = w_search
                    found_forward = True
                    found_backward = True
                    break



            if not (found_forward and found_backward):
                for w_search in w_candidates:
                    has_forward = (z, w_search) in wbmg_edges
                    has_backward = (w_search, z) in wbmg_edges

                    u, v = sorted([z, w_search], key=lambda item: str(item))

                    already_visited = any(
                        path_node.endswith(f":{u}|{v}")
                        for path_node in path_visited
                    )

                    if already_visited:
                     #   print(w_candidates)
                     #   continue

                    if has_forward:
                        w = w_search
                        found_forward = True
                        found_backward = False
                        break

                    elif has_backward:
                        w = w_search
                        found_backward = True
                        found_forward = False
                        break

                    else:
                        w = w_search
                        found_forward = False
                        found_backward = False
                        break


            network, q_next = extension(z, w, q_node, network, depth=depth)

            new_path_visited = path_visited + (q_next,)

            if found_forward == False:
                next_extend_pairs.add((z, w, y, q_next, new_path_visited))


            if found_backward == False:
                next_extend_pairs.add((w, z, x, q_next, new_path_visited))



        current_extend_pairs = next_extend_pairs

    return network




network = bic_cherry_for_wbm(G, max_depth=30)
print(G.edges())
new_wbmg = wbmg_from_network(network)

only_new_wbmg = set(new_wbmg.edges() ) - set(G.edges())
print(network.edges())
print(only_new_wbmg)





def test_bic_cherry_for_wbm_generated_examples():
    species = 2
    leaves = 10
    species_tree_age = 1
    for i in range(1000):
        tree = create_gene_tree_n_leaves(leaves, species, species_tree_age).gene_tree
        graph = transform(tree, 2)
        wbmg = wbmg_from_network(graph)
        network = bic_cherry_for_wbm(wbmg)

        new_wbmg = wbmg_from_network(network)

        actual_leaves = [n for n, d in graph.out_degree() if d == 0]
        n_leaves = len(actual_leaves)
        print(f"Suche mit {n_leaves} Blättern")

        if set(new_wbmg.edges()) != set(wbmg.edges()):
            print(wbmg.edges())
            print(new_wbmg.edges())
            print(new_wbmg_cleaned_edges)
            print(network.edges())
            break




#test_bic_cherry_for_wbm_generated_examples()


