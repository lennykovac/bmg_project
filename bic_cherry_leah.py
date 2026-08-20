import random
import itertools
import networkx as nx

# groups all nodes by their color. returns a dictinary: key: color, value: node
def sort_nodes_color(graph):
    node_dic = {}

    for node in graph.nodes():
        if "color" not in graph.nodes[node]:
            raise ValueError(
                f"Node {node} has no color attribute: "
                f"{graph.nodes[node]}"
            )

        color = graph.nodes[node]["color"]
        node_dic.setdefault(color, []).append(node)

    return node_dic


# returns all pairs (x, y) where x and y have different colors.
def different_color_pairs(graph: nx.DiGraph):
    pairs = []
    node_dic = sort_nodes_color(graph)
    colors = list(node_dic.keys())

    for col1, col2 in itertools.combinations(colors, 2):
        node_col1 = node_dic[col1]
        node_col2= node_dic[col2]


        for x,y in itertools.product(node_col1, node_col2):
            pairs.append((x,y))

    return pairs

# initialization of bic cherry network. for every pair of differently colored nodes a parent Px:y is created
def bic_cherry_network_initialization(graph: nx.DiGraph):
    color_pairs = different_color_pairs(graph)
    N = nx.DiGraph()
    N.add_node("root")

    for cherry1, cherry2 in color_pairs:

        N.add_node(f"P{cherry1}:{cherry2}")
        N.add_edge("root", f"P{cherry1}:{cherry2}")
        N.add_edge(f"P{cherry1}:{cherry2}", cherry1)
        N.add_edge(f"P{cherry1}:{cherry2}", cherry2)

    return N



# extends the network by [xy : xz].
def extension(x, y, z,  N: nx.DiGraph):
    p_xy = f"P{x}:{y}"
    p_yx = f"P{y}:{x}"

    if p_xy not in N:
        if p_yx in N:
            p_xy = p_yx

    q_xz = f"Q{x}:{z}"

    N.add_node(q_xz)
    N.add_edge(p_xy, q_xz)
    N.add_edge(q_xz, x)
    N.add_edge(q_xz, z)

    return N

# returns a random node y' != y with the same color as y
def choose_same_color_node(y, graph: nx.DiGraph):
    color_y = graph.nodes[y]["color"]
    node_dic = sort_nodes_color(graph)
    candidates = [node for node in node_dic[color_y] if node != y]

    return random.choice(candidates)


# bic cherry algo for colored digraph with sicor-in-hub property
def bic_cherry(graph: nx.DiGraph):
    N = bic_cherry_network_initialization(graph)
    color_pairs = different_color_pairs(graph)
    for x,y in color_pairs:
        if graph.has_edge(x, y) == False:
            z = choose_same_color_node(y, graph)
            N = extension(x, y, z, N)

        if graph.has_edge(y, x) == False:
            z = choose_same_color_node(x, graph)
            N = extension(y, x, z, N)

    return N


# checks whether every sicor is an in-hub
def is_sicor_in_hub(graph: nx.DiGraph):
    node_dic = sort_nodes_color(graph)

    for color, nodes in node_dic.items():

        if len(nodes) == 1:
            sicor = nodes[0]

            for u in graph.nodes():
                if u != sicor and not graph.has_edge(u, sicor):
                    return False

    return True

