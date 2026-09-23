"""
Utilities for working nx.DiGraphs.

"""

import random
from typing import Any, Tuple

from typing import Hashable
from itertools import permutations
import networkx as nx
from pyvis.network import Network
from collections import Counter

###############################################################################
# GENERIC HELPER
###############################################################################
def insert_node_on_edge(node_for_adding: Any, edge: Tuple[Any, Any], G: nx.DiGraph):
    """
    Inserts a node onto an edge and removes the old redundant edge
    Function is inplace!

    Parameters:
    node_for_adding: New node
    edge: on which the node should be placed
    G: Directed Graph

    Raises:
    ValueError: if the edge is not in G or the node is already in G. Both are
    checked up front, so a failing call leaves G untouched.
    """
    parent_node, child_node = edge

    # validate before we touch G, otherwise a bad edge leaves a half inserted
    # node behind (add_node and add_edge would already have run)
    if not G.has_edge(parent_node, child_node):
        raise ValueError(f"{edge} is not an edge of G")
    if node_for_adding in G:
        raise ValueError(f"{node_for_adding} is already a vertex of G")

    # add new node TODO: think about default color
    G.add_node(node_for_adding, color=None)
    # add edge from parent_node to new_node
    G.add_edge(parent_node, node_for_adding)
    # add edge from new_node to child_node
    G.add_edge(node_for_adding, child_node)
    # remove old edge
    G.remove_edge(parent_node, child_node)


def root_from_network(network: nx.DiGraph) -> Any:
    """
    Returns the root from
    """

    roots = [n for n in network.nodes() if network.in_degree(n) == 0]

    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, found {len(roots)}")

    root = roots[0]

    return root


def leaves_from_network(network: nx.DiGraph) -> list[Hashable]:

    return [n for n in network.nodes() if network.out_degree(n) == 0]

# used in testing if bmg/wbmg_from_network works correctly
def check_sicorinhub(G: nx.DiGraph):
    """
    Checks if given DiGraph has the sicor-in-hub property.

    Parameters:
    G: DiGraph with no self-loops! BMGs and WBMGs should not have self loops.

    Returns:
    Boolean value True, iff G has sicor-in-hub property.
    """
    color_counts = Counter(nx.get_node_attributes(G, "color").values())
    unique_nodes = [n for n, d in G.nodes(data=True) if color_counts[d["color"]] == 1]
    for n in unique_nodes:
        # because no Multigraph and self-loop-free
        if G.in_degree(n) != G.number_of_nodes() - 1:
            return False
    return True


###############################################################################
# VISUALISATION AND PRINTING
###############################################################################
def show_graph(di_graph: nx.DiGraph):
    """
    Shows u a neat graph view of the DAG
    """
    nt = Network("1000px", "1000px", directed=True)

    for node, node_data in di_graph.nodes(data=True):
        node_data["label"] = str(node)
        title_parts = []
        if "reconc" in node_data:
            reconc_value = node_data["reconc"]
            title_parts.append(f"reconc: {reconc_value}")
        if "dist" in node_data:
            title_parts.append(f"dist: {node_data['dist']}")
        if title_parts:
            node_data["title"] = "\n".join(title_parts)

    nt.from_nx(di_graph)
    nt.show("nx.html", notebook=False)


def print_graph_diff(g1, g2):
    """
    For debugging: prints edges and node differences between two graphs
    """
    print("=== Node differences ===")
    print("Only in g1:", set(g1.nodes) - set(g2.nodes))
    print("Only in g2:", set(g2.nodes) - set(g1.nodes))

    print("\n=== Edge differences ===")
    print("Only in g1:", set(g1.edges) - set(g2.edges))
    print("Only in g2:", set(g2.edges) - set(g1.edges))


###############################################################################
# Hybridization
###############################################################################
def add_hybrid_node(
    donor_edge: Tuple[Any, Any],
    hybrid_edge: Tuple[Any, Any],
    donor: Any,
    hybrid: Any,
    G: nx.DiGraph,
) -> bool:
    """
    Connects 2 nodes on 2 edges with each other and creates one hybrid node!
    Function is inplace, but only touches G if it returns True.

    Parameters:
    donor_edge: Edge on which the donor node is placed
    hybrid_edge: Edge on which the hybrid now is placed (it has two parents)
    donor: donor node which "donates" an edge
    hybrid: hybrid node which gets another parent (donor)
    G: Directed Graph

    Returns:
    True if the hybrid node was inserted, False if the pair of edges was
    refused because it would have closed a cycle. The caller counts.
    """
    # after the insertion the only parent of donor is donor_edge[0] and the
    # only child of hybrid is hybrid_edge[1], so the new donor -> hybrid edge
    # closes a cycle exactly if hybrid can already reach donor. Subdividing an
    # edge keeps reachability
    if nx.has_path(G, hybrid_edge[1], donor_edge[0]):
        return False
    # validate everything up front, the two inserts below run one after the
    # other, so a failing second insert would leave the donor behind
    for edge in (donor_edge, hybrid_edge):
        if not G.has_edge(*edge):
            raise ValueError(f"{edge} is not an edge of G")
    if donor_edge == hybrid_edge:
        raise ValueError("donor_edge and hybrid_edge must be different edges")
    for node in (donor, hybrid):
        if node in G:
            raise ValueError(f"{node} is already a vertex of G")
    if donor == hybrid:
        raise ValueError("donor and hybrid need different names")

    if nx.has_path(G, hybrid_edge[1], donor_edge[0]):
        return False

    # first insert donor to donor_edge
    insert_node_on_edge(donor, donor_edge, G)
    # second insert hybrid to hybrid_edge
    insert_node_on_edge(hybrid, hybrid_edge, G)
    # third add edge between donor and hybrid
    G.add_edge(donor, hybrid)
    return True


def transform(graph: nx.DiGraph, num_of_hybrid_nodes: int, attempts = 7) -> nx.DiGraph:
    """
    Edit a bicolored tree into a phylogenetic network by inserting random hybridization vertices
    This is not inplace, so the original network will be conserved.

    Parameters:
    di_graph: The di_graph on which the hybrid nodes will be inserted
    num_of_hybrid_nodes: The amount of hybrid nodes we would like to have, cant be guaranteed.
    attempts: number of refused edge pairs we tolerate before we stop
        inserting nodes, default 7 (cuz i like the number)

    Returns:
    A nx.DiGraph object.
    """
    # number of succesful inserts
    insertions = 0
    # keep old data intact
    transformer_graph = graph.copy()
    # a refused pair of edges costs budget, a successful one does not, so we
    # really get num_of_hybrid_nodes as long as the graph allows it
    budget = attempts

    # names of the inserted vertices, counts on past the ones an earlier
    # transform() call may already have put into this graph
    label = 0

    while insertions < num_of_hybrid_nodes and budget > 0:
        # fresh pull out of the Urne :D , the edges change with every insertion
        edge_list = list(transformer_graph.edges)
        if len(edge_list) < 2:
            break

        donor, hybrid = f"{label}_d", f"{label}_h"
        while donor in transformer_graph or hybrid in transformer_graph:
            label += 1
            donor, hybrid = f"{label}_d", f"{label}_h"

        # sample two random edges
        donor_e, hybrid_e = random.sample(edge_list, 2)
        if add_hybrid_node(donor_e, hybrid_e, donor, hybrid, transformer_graph):
            insertions += 1
            label += 1
        else:
            budget -= 1

    print(f"Number of succesfull insertions: {insertions}")
    return transformer_graph

###############################################################################
# BEST MATCHES and WEAK BEST MATCHES
###############################################################################

def lca_dict_from_network(
    network: nx.DiGraph,
    reach: dict[Hashable, set[Hashable]],
    leaves: list[Hashable],
) -> dict[tuple[Hashable, Hashable], set[Hashable]]:
    """Compute dict with all LCAs in network

    Args:
        network: network with leaves that have the `label` and `reconc` attributes set.
        reach: reachability sets for all nodes in network (for example see bmg_from_network)
        leaves: leafset of network

    """

    # collect all pairs of different color
    pairs = [
        (u, v)
        for u, v in permutations(leaves, 2)
        if network.nodes[u]["color"] != network.nodes[v]["color"]
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
        eliminate_ancestors = set()  # ancestors to be removed
        for u, v in permutations(common_ancestors, 2):
            if v in reach[u]:
                eliminate_ancestors.add(u)  # v < u, thus remove u from lca

        lca = common_ancestors - eliminate_ancestors
        lca_dict.update({(x, y): lca})

    return lca_dict


def bmg_from_network(
    network: nx.DiGraph,
) -> nx.DiGraph:
    """Construct a BMG from bic-network.

    Args:
        network: A network with leaves that has the `label` and `color` attribute set.

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
        colors.add(network.nodes[v]["color"])
        bmg.add_node(v, color=network.nodes[v]["color"])

    lca_dict = lca_dict_from_network(network, reach, leaves)

    # check bm property for each pair
    delete_keys = set()
    for x, y in lca_dict.keys():
        # find all y' with same color as y
        alt_y = [
            u for u in leaves if network.nodes[u]["color"] == network.nodes[y]["color"]
        ]
        # iterate over lca(x, y')
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


def wbmg_from_network(
    network: nx.DiGraph,
) -> nx.DiGraph:
    """Construct a WBMG from bic-network.

    Args:
        network: A network with leaves that has the `label` and `reconc` attribute set.

    Returns:
        The constructed WBMG with attributes `label` and `color`
    """

    leaves = leaves_from_network(network)
    wbmg = nx.DiGraph()
    colors = set()
    reach = {
        n: nx.descendants(network, n) for n in network.nodes
    }  # pre-compute reachability in network as dict[{v:descendants of v}]

    # collect all leaves and colors
    for v in leaves:
        colors.add(network.nodes[v]["color"])
        wbmg.add_node(v, color=network.nodes[v]["color"])

    lca_dict = lca_dict_from_network(network, reach, leaves)

    # check bm property for each pair
    delete_keys = set()
    for x, y in lca_dict.keys():
        # compute Q
        alt_y = [
            n for n in leaves if network.nodes[n]["color"] == network.nodes[y]["color"]
        ]
        q = set()
        for element in alt_y:
            q |= lca_dict[(x, element)]
        eliminate_q = set()
        for u, v in permutations(q, 2):
            if v in reach[u]:
                eliminate_q.add(u)
        q = q - eliminate_q
        # check intersection of lca is non-empty
        if len(set.intersection(lca_dict[(x, y)], q)) == 0:
            delete_keys.add((x, y))
    # delete all marked keys from dict
    for x, y in delete_keys:
        lca_dict.pop((x, y))

    # add remaining wbmg edges to wbmg
    for x, y in lca_dict:
        wbmg.add_edge(x, y)

    return wbmg

# ---------------------------------------------------------------------------
# Graph properties
# ---------------------------------------------------------------------------

def check_color_sink_free(G: nx.DiGraph) -> bool:
    """Every vertex has an out-neighbor of every other color."""
    colors = set(nx.get_node_attributes(G, "color").values())
    for x in G.nodes:
        own = G.nodes[x]["color"]
        seen = {G.nodes[y]["color"] for y in G.successors(x)}
        if seen != colors - {own}:
            return False
    return True

# ---------------------------------------------------------------------------
# Comparing phylogenies
# ---------------------------------------------------------------------------

def clusters(network: nx.DiGraph) -> dict:
    """ leaves of the subtree"""
    """vertex -> frozenset of leaf descendants (cluster C(v))."""
    leaves = {v for v in network.nodes if network.out_degree(v) == 0}
    out = {}
    for v in reversed(list(nx.topological_sort(network))):
        if v in leaves:
            out[v] = frozenset([v])
        else:
            out[v] = frozenset().union(*(out[c] for c in network.successors(v)))
    return out


def is_phylogenetic_tree(network: nx.DiGraph) -> bool:
    """Rooted tree (single root, in-degree <= 1) and no inner vertex with one child."""
    if network.number_of_nodes() == 0 or not nx.is_directed_acyclic_graph(network):
        return False
    roots = [v for v in network if network.in_degree(v) == 0]
    if len(roots) != 1 or any(network.in_degree(v) > 1 for v in network):
        return False
    return all(network.out_degree(v) != 1 for v in network)


def same_phylogeny(n1: nx.DiGraph, n2: nx.DiGraph) -> bool:
    """Leaf-labelled isomorphism of two phylogenetic TREES.

    A phylogenetic tree is uniquely determined by its hierarchy of clusters
    (Semple & Steel 2003, Prop. 2.1), so comparing cluster sets suffices."""
    if not (is_phylogenetic_tree(n1) and is_phylogenetic_tree(n2)):
        return False
    return set(clusters(n1).values()) == set(clusters(n2).values())