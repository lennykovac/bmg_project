import networkx as nx
import os
import random
from collections import Counter
from utils.graph_utils import (
    transform,
    wbmg_from_network)
from utils.tree_utils import create_gene_tree_n_leaves
from itertools import combinations, product
import warnings


def bic_cherry(bmg: nx.DiGraph):
    """Build the initial network: root R with one cherry p:u|v per connected pair."""
    # Copy all leaves with their colors into the new network
    network = nx.DiGraph()
    network.add_nodes_from(
        (n, {"color": color}) for n, color in bmg.nodes(data="color")
    )

    # Group leaves by color
    color_groups = {}
    for node, color in nx.get_node_attributes(bmg, "color").items():
        color_groups.setdefault(color, []).append(node)

    pairs = []
    colors = list(color_groups.keys())

    bmg_edges = set(bmg.edges())

    # Collect all pairs of different colors that share an edge (in either direction)
    for x, y in combinations(colors, 2):
        # Nur hinzufügen, wenn mindestens eine Kante in bmg existiert
        pairs.extend([
            (u, v) for u, v in product(color_groups[x], color_groups[y])
            if (u, v) in bmg_edges or (v, u) in bmg_edges
        ])

    network.add_node("R", color=None)

    created_cherry_nodes = set()

    # Hang one cherry node p:u|v under R for every pair (u, v sorted for a stable name)
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




def pair_key(a, b):
    """Order two leaves by str, so the same pair always gives the same key/name."""
    return tuple(sorted([a, b], key=lambda item: str(item)))


#[xy: xz] parent = p_xy
def extension(x, z, parent_node, network, depth):
    """Add q:x|z under parent_node so x prefers z, suppressing x's old partner."""
    # Node name: q:u|v at depth 0, q<depth>:u|v deeper
    u, v = pair_key(x, z)
    q_prefix = f"q{depth if depth > 0 else ''}"
    q_node = f"{q_prefix}:{u}|{v}"
    # Names must be unique: a shared node would be reached from two branches,
    # and undoing one branch would delete it for the other one too
    suffix = 1
    while q_node in network:
        suffix += 1
        q_node = f"{q_prefix}:{u}|{v}_{suffix}"

    network.add_node(q_node, color=None)
    network.add_edge(parent_node, q_node)
    network.add_edge(q_node, x)
    network.add_edge(q_node, z)

    return network, q_node


def bic_cherry_for_wbm(wbmg):
    """
    Build a network whose WBMG equals wbmg by extending the cherry network.

    Every false edge is fixed by a depth first search with backtracking: pick
    a partner, hang an extension node, fix the false edges this node creates
    below it, and on a dead end restore the network and try the next partner.
    """
    network, pairs = bic_cherry(wbmg)
    wbmg_edges = set(wbmg.edges())
    colors = dict(wbmg.nodes(data="color"))

    # Cherry directions (x, y) that are NOT edges in the WBMG: the cherry
    # wrongly implies x -> y, so it must be suppressed
    false_edges = [
        edge for (u, v) in pairs for edge in [(u, v), (v, u)] if edge not in wbmg_edges
    ]

    # Extension nodes in the order they were created. A dead end restores the
    # network by removing everything created after a saved length
    stack = []

    def undo(mark):
        while len(stack) > mark:
            network.remove_node(stack.pop())  # also removes its 3 edges

    # (a, b, visited) that already ended in a dead end. The search below a
    # false edge only depends on these three, so it would fail again
    dead_ends = set()

    def candidates(a, b, visited):
        """
        Partners c for the false edge a -> b, best ones first.
        """
        # Every node on this path also reaches c. For a leaf g up there of
        # another color than c, that node becomes LCA(g, c), so g -> c
        # appears and has to be real (a is fine, it gets its own node with c)
        path_leaves = {leaf for pair in visited for leaf in pair}
        found = []
        for c in wbmg.nodes:
            if c == b or colors[c] != colors[b]:
                continue
            if any(
                (g, c) not in wbmg_edges
                for g in path_leaves
                if g not in (a, c) and colors[g] != colors[c]
            ):
                continue
            # Pair already used on this path: we would run in a cycle
            if pair_key(a, c) in visited:
                continue
            found.append(c)
        # Both directions real first, then only a -> c, then only c -> a,
        # then neither (sort is stable, so node order is kept within a group)
        found.sort(key=lambda c: ((a, c) not in wbmg_edges, (c, a) not in wbmg_edges))
        return found

    def resolve(a, b, node, depth, visited):
        """Fix the false edge a -> b implied by node. False means dead end.

        visited: leaf pairs of the nodes on this path, from the cherry down
        """
        if (a, b, visited) in dead_ends:
            return False

        for c in candidates(a, b, visited):
            mark = len(stack)
            # q:a|c under node makes a prefer c, so a -> b disappears
            _, q_node = extension(a, c, node, network, depth=depth)
            stack.append(q_node)

            # q:a|c itself implies a -> c and c -> a, fix the false ones below it
            todo = [edge for edge in [(a, c), (c, a)] if edge not in wbmg_edges]

            path = visited | {pair_key(a, c)}
            if all(resolve(s, t, q_node, depth + 1, path) for s, t in todo):
                return True

            # Dead end below this partner: restore the network, try the next one
            undo(mark)

        dead_ends.add((a, b, visited))
        return False

    for x, y in false_edges:
        cherry = pair_key(x, y)
        if not resolve(x, y, f"p:{cherry[0]}|{cherry[1]}", 0, frozenset({cherry})):
            warnings.warn(f"dead end: no extension removes the false edge {x} -> {y}")

    return network




def export_mismatch(graph, n_leaves, n_hybrids, folder="mismatches"):
    """Write graph to <folder>/l_<leaves>h_<hybrids>.gml, numbered if the name is taken."""
    os.makedirs(folder, exist_ok=True)
    name = f"l_{n_leaves}h_{n_hybrids}"
    path = os.path.join(folder, f"{name}.gml")
    suffix = 1
    while os.path.exists(path):
        suffix += 1
        path = os.path.join(folder, f"{name}_{suffix}.gml")

    # GML can't store None, so leave those attributes out
    export = nx.DiGraph()
    for n, data in graph.nodes(data=True):
        export.add_node(n, **{k: v for k, v in data.items() if v is not None})
    export.add_edges_from(graph.edges())
    nx.write_gml(export, path)
    return path


def test_bic_cherry_for_wbm_generated_examples():
    """Round-trip on random gene trees: tree -> WBMG -> network -> WBMG must match."""
    runs = 0
    max_runs = 300
    # (leaves, hybrids) -> number of mismatches
    mismatches = Counter()

    for leaves in range(2,30):
        print(f"Testing {max_runs} trees with {leaves} leaves and 2-30 random hybrid nodes!")
        species = 2
        species_tree_age = 1

        for _ in range(max_runs):
            tree = create_gene_tree_n_leaves(leaves, species, species_tree_age, max_attempts=25).gene_tree
            number_of_hybrid_nodes = random.randint(2,30)
            graph = transform(tree, number_of_hybrid_nodes)
            wbmg = wbmg_from_network(graph)
            network = bic_cherry_for_wbm(wbmg)

            new_wbmg = wbmg_from_network(network)

            actual_leaves = [n for n, d in graph.out_degree() if d == 0]
            runs += 1

            new_wbmg_edges = set(new_wbmg.edges())
            wbmg_edges = set(wbmg.edges())
            if new_wbmg_edges != wbmg_edges:
                mismatches[(len(actual_leaves), number_of_hybrid_nodes)] += 1
                path = export_mismatch(graph, len(actual_leaves), number_of_hybrid_nodes)
                print(
                    f"mismatch: {len(actual_leaves)} leaves, {number_of_hybrid_nodes} hybrids, "
                    f"missing {wbmg_edges - new_wbmg_edges}, "
                    f"extra {new_wbmg_edges - wbmg_edges} -> {path}"
                )

    # Summary: how often and under which conditions it went wrong
    total = sum(mismatches.values())
    print(f"{total} mismatches in {runs} runs ({100 * total / runs:.2f}%)")
    for (n_leaves, n_hybrids), count in sorted(mismatches.items()):
        print(f"  {n_leaves} leaves, {n_hybrids} hybrids: {count}")


if __name__ == '__main__':

    test_bic_cherry_for_wbm_generated_examples()
