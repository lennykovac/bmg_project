"""
Task 2.2(a): least resolved tree (LRT) T* of a tree-BMG (G, sigma).

Two independent constructions:

1. ``lrt_from_bmg(G)``.
   Corrigendum, Def. 11: informative triples
       R(G, sigma) = { ab|b' : sigma(a) != sigma(b) = sigma(b'),
                               ab ∈ E(G), ab' ∉ E(G) }.
   Theorem 15: (G, sigma) is a BMG  <=>  G(Aho(R(G, sigma)), sigma) = (G, sigma),
   and then Aho(R(G, sigma)) is the UNIQUE least resolved tree.
   Aho(R) is computed with BUILD (Aho et al. 1981; Sec. 3.4 of the paper).

2. ``lrt_by_contraction(T, G)`` -- *from any explaining tree* (e.g. the
   AsymmeTree gene tree). Theorem 13 / Cor. 2: T* is obtained by contracting
   all redundant edges in arbitrary order. Redundant edges are characterised
   by Lemma 21 (corrigendum):
       inner edge uv (v ≺ u) is redundant  <=>  there is NO arc ab ∈ E(G) with
       lca_T(a, b) = v and sigma(b) ∈ sigma(L(T(u)) \\ L(T(v))).
"""

from itertools import count

import networkx as nx

from utils.graph_utils import bmg_from_network, clusters


# ---------------------------------------------------------------------------
# informative triples + BUILD
# ---------------------------------------------------------------------------

def informative_triples(G: nx.DiGraph) -> set:
    """R(G, sigma) as a set of tuples (a, b, c) meaning the triple ab|c."""
    color = nx.get_node_attributes(G, "color")
    by_color: dict = {}
    for v, c in color.items():
        by_color.setdefault(c, set()).add(v)
    R = set()
    for a in G.nodes:
        out = set(G.successors(a))
        for s, vertexes_s_color in by_color.items():
            if s == color[a]:
                continue
            # vertexes that are not of the same color as a but are ist successors
            hits = out & vertexes_s_color
            # vertexes that are not of the same color as a minus ist successors
            misses = vertexes_s_color - out
            for b in hits:
                for b2 in misses:
                    R.add((a, b, b2))
    return R


def aho_build(leaves, triples) -> nx.DiGraph | None:
    """BUILD: returns Aho(R) as nx.DiGraph (leaves = given leaves) or None if R
    is inconsistent. Inner vertices are named 'rho' (root), 'v1', 'v2', ..."""
    leaves = list(leaves)
    T = nx.DiGraph()
    T.add_nodes_from(leaves)
    if len(leaves) == 1:
        return T
    ids = count(1)

    def recurse(node, L, R):
        # Aho graph [R, L]: edge a-b for every ab|c with a, b, c ∈ L
        aho = nx.Graph()
        aho.add_nodes_from(L)
        aho.add_edges_from((a, b) for a, b, _ in R)
        comps = [frozenset(c) for c in nx.connected_components(aho)]
        if len(comps) == 1:
            return False  # connected Aho graph with |L| > 1 -> inconsistent
        for C in comps:
            if len(C) == 1:
                T.add_edge(node, next(iter(C)))
                continue
            child = f"v{next(ids)}"
            T.add_edge(node, child)
            R_C = [t for t in R if t[0] in C and t[1] in C and t[2] in C]
            if not recurse(child, C, R_C):
                return False
        return True

    Lset = frozenset(leaves)
    return T if recurse("rho", Lset, list(triples)) else None


def lrt_from_bmg(G: nx.DiGraph, verify: bool = True) -> nx.DiGraph:
    """Unique least resolved tree of a (tree-)BMG via Theorem 15.

    Raises ValueError if (G, sigma) is not a BMG (R inconsistent, or
    G(Aho(R)) != G)."""
    T = aho_build(G.nodes, informative_triples(G))
    if T is None:
        raise ValueError("informative triples are inconsistent: G is not a BMG")
    for v in T.nodes:
        T.nodes[v]["color"] = G.nodes[v]["color"] if v in G else None
    if verify:
        H = bmg_from_network(T)
        if set(H.edges) != set(G.edges):
            raise ValueError("G(Aho(R)) != G: G is not a BMG (Thm. 15)")
    return T


# ---------------------------------------------------------------------------
# redundant edges (Lemma 21) and contraction
# ---------------------------------------------------------------------------

def _tree_lca(T: nx.DiGraph, parent: dict, depth: dict, a, b):
    while depth[a] > depth[b]:
        a = parent[a]
    while depth[b] > depth[a]:
        b = parent[b]
    while a != b:
        a, b = parent[a], parent[b]
    if a==b:
        return a
    raise ValueError


def redundant_edges(T: nx.DiGraph, G: nx.DiGraph) -> list:
    """All redundant edges (u, v) of a tree T explaining G (Lemma 21)."""
    root = next(v for v in T if T.in_degree(v) == 0)
    parent = {v: next(iter(T.predecessors(v))) for v in T if v != root}
    depth = nx.single_source_shortest_path_length(T, root)
    cl = clusters(T)
    color = {x: G.nodes[x]["color"] for x in G.nodes}

    # colors of arcs ab grouped by lca_T(a, b)
    arc_colors_at: dict = {}
    for a, b in G.edges:
        arc_colors_at.setdefault(_tree_lca(T, parent, depth, a, b), set()).add(color[b])

    red = []
    for u, v in T.edges:
        if T.out_degree(v) == 0:
            continue  # outer edge: never redundant
        rest_colors = {color[x] for x in cl[u] - cl[v]}
        if not (arc_colors_at.get(v, set()) & rest_colors):
            red.append((u, v))
    return red


def contract_vertex(T: nx.DiGraph, v) -> None:
    """Contract the edge (parent(v), v) in place (v must have in-degree 1)."""
    (u,) = T.predecessors(v)
    children = list(T.successors(v))
    T.remove_node(v)
    T.add_edges_from((u, c) for c in children)


def lrt_by_contraction(T: nx.DiGraph, G: nx.DiGraph) -> nx.DiGraph:
    """Contract all redundant edges of T (Thm. 13: order is irrelevant)."""
    T = T.copy()
    for _, v in redundant_edges(T, G):
        contract_vertex(T, v)  # identified by lower endpoint -> order-independent

    # suppress a possible single-child root (non-phylogenetic input)
    root = next(v for v in T if T.in_degree(v) == 0)
    while T.out_degree(root) == 1:
        (c,) = T.successors(root)
        T.remove_node(root)
        root = c
    return T


def is_least_resolved(T: nx.DiGraph, G: nx.DiGraph) -> bool:
    """T explains G and has no redundant edge (Def. 6 + Lemma 21)."""
    H = bmg_from_network(T)
    return set(H.edges) == set(G.edges) and not redundant_edges(T, G)
