from typing import Hashable

import networkx as nx
from utils.graph_utils import show_graph
from itertools import permutations


def root_from_network(network: nx.DiGraph) -> Hashable:

    roots = [n for n in network.nodes() if network.in_degree(n) == 0]

    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, found {len(roots)}")

    root = roots[0]

    return root


def leaves_from_network(network: nx.DiGraph) -> list[Hashable]:

    return [n for n in network.nodes() if network.out_degree(n) == 0]


def bmg_from_network(
    network: nx.DiGraph,
) -> nx.DiGraph:
    """Construct a BMG from bic-network.

    Args:
        network: A network with leaves that has the `label` and `reconc` attribute set.

    Returns:
        The constructed BMG with attributes `label` and `color`
    """

    leaves = leaves_from_network(network)
    bmg = nx.DiGraph()
    colors = set()
    reach = {
        n: nx.descendants(network, n) for n in network.nodes
    }  # pre-compute reachability in network as dict[{v:descendants of v}]

    # collect all leaves and colors
    for v in leaves:
        colors.add(network.nodes[v]["reconc"])
        bmg.add_node(v, color=network.nodes[v]["reconc"])

    # collect all pairs of different color
    pairs = [
        (u, v)
        for u, v in permutations(leaves, 2)
        if network.nodes[u]["reconc"] != network.nodes[v]["reconc"]
    ]

    # create dict with pair -> LCA(pair) mapping
    lca_dict = dict()
    for x, y in pairs:
        pred_x = set()
        pred_y = set()
        for z in reach:
            if x in reach[z]:
                pred_x.add(z)
            if y in reach[z]:
                pred_y.add(z)

        common_ancestors = set.intersection(pred_x, pred_y)

        # remove all non minimal common ancestors
        high_ancestors = set()  # ancestors to be removed
        for u, v in permutations(common_ancestors, 2):
            if nx.has_path(network, u, v):
                high_ancestors.add(u)  # v < u, thus remove u from lca

        lca = common_ancestors - high_ancestors
        lca_dict.update({(x, y): lca})

    # check bm property for each pair
    delete_keys = set()
    for x, y in lca_dict.keys():
        # find all y' with same color as y
        alt_y = [
            u
            for u in leaves
            if network.nodes[u]["reconc"] == network.nodes[y]["reconc"]
        ]
        # iterate over lca(x, y') --> ALREADY IN lca_dict!!!!
        for ay in alt_y:
            lca_alt_y = lca_dict[(x, ay)]
            for u in lca_alt_y:
                for v in lca_dict[(x, y)]:
                    if u in reach[v]:  # i.e. u<v
                        delete_keys.add((x, y))
    # delete all marked keys from dict
    for x, y in delete_keys:
        lca_dict.pop((x, y))

    # add remaining bmg edges to bmg
    for x, y in lca_dict:
        bmg.add_edge(x, y)

    return bmg

    # TODO: implement wbmg_from_network


"""
GOAL: experiment towards bmg & wbmg from networks
"""
if __name__ == "__main__":
    # Leahs example where bmg != wbmg
    G = nx.DiGraph()

    G.add_node(0, label="0", reconc="0")
    G.add_node(1, label="1", reconc="0")
    G.add_node(2, label="2", reconc="0")
    G.add_node(3, label="3", reconc="0")
    G.add_node(4, label="4", reconc="0")
    G.add_node(5, label="5", reconc="1")
    G.add_node(6, label="6", reconc="0")

    G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4), (1, 4), (3, 5), (3, 6), (2, 5)])

    bmg = bmg_from_network(G)
    print(bmg.nodes(data=True))
    show_graph(bmg)
