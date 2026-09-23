"""Least resolved trees -- the bridge between our ``nx.DiGraph`` world and
``asymmetree.analysis.best_matches`` / ``tralda``.

Everything in this project is a ``networkx.DiGraph`` whose vertices carry a
``color`` attribute (``None`` on inner vertices). AsymmeTree instead works on
``tralda.datastructures.Tree`` objects whose *leaves* carry ``label`` and
``reconc``. 

``lrt_from_tree(T: Tree) -> Tree``
    contracts all redundant edges of a gene tree -- the least resolved tree of
    the BMG of that tree, computed *from the tree*.
``lrt_from_colored_graph(G, mincut=False, weighted_mincut=False) -> Tree | None``
    BUILD on the informative triples of ``G`` -- the least resolved tree of a
    graph, ``None`` if ``G`` is not a BMG (unless a mincut heuristic is asked
    for).
``lrt_from_2bmg(G) -> Tree | None``
    the linear-time special case for two colors.

The two routes must agree: for a gene tree ``T`` with BMG ``G(T)`` we have
``LRT(T) == LRT(G(T))``, which is exactly what :func:`lrt_cross_check` asserts
and what the test suite uses as an oracle for our own ``bmg_from_network``.

"""

from __future__ import annotations

from typing import Hashable

import networkx as nx
from asymmetree.analysis import best_matches as _bm
from tralda.datastructures import Tree, TreeNode

from utils.graph_utils import root_from_network

__all__ = [
    "tralda_to_digraph",
    "digraph_to_tralda",
    "lrt_from_bmg",
    "lrt_from_two_colored_bmg",
    "lrt_of_tree",
    "asymmetree_bmg",
    "lrt_cross_check",
    "is_bmg",
]


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------

def tralda_to_digraph(tree: Tree, inner_prefix: str = "i") -> nx.DiGraph:
    """``tralda`` tree -> ``nx.DiGraph`` with ``color`` on the leaves.

    Leaves keep their ``label``; inner vertices are renamed to
    ``f"{inner_prefix}{k}"`` in preorder because their labels are unusable
    (empty or absent, see the module docstring).
    """
    D = nx.DiGraph()
    ids: dict[int, Hashable] = {}
    counter = 0

    for v in tree.preorder():
        if v.children:
            ids[id(v)] = f"{inner_prefix}{counter}"
            counter += 1
            D.add_node(ids[id(v)], color=None)
        else:
            label = getattr(v, "label", None)
            if label is None or label == "":
                raise ValueError("leaf without a usable label in the tralda tree")
            ids[id(v)] = label
            D.add_node(label, color=getattr(v, "reconc", None))

    for v in tree.preorder():
        for c in v.children:
            D.add_edge(ids[id(v)], ids[id(c)])

    return D


def digraph_to_tralda(tree: nx.DiGraph) -> Tree:
    """``nx.DiGraph`` (a rooted tree!) -> ``tralda`` tree.

    Leaves receive ``label`` (the vertex itself) and ``reconc`` (its color),
    which is what ``bmg_from_tree`` and ``lrt_from_tree`` read.
    """
    root = root_from_network(tree)
    if any(tree.in_degree(v) > 1 for v in tree):
        raise ValueError("digraph_to_tralda expects a tree, not a network")

    nodes: dict[Hashable, TreeNode] = {}
    for v in nx.topological_sort(tree):
        node = TreeNode()
        node.label = v
        if tree.out_degree(v) == 0:
            node.reconc = tree.nodes[v].get("color")
        nodes[v] = node
        for parent in tree.predecessors(v):
            nodes[parent].add_child(node)

    return Tree(nodes[root])


# ---------------------------------------------------------------------------
# LRT entry points
# ---------------------------------------------------------------------------

def lrt_from_bmg(
    graph: nx.DiGraph,
    mincut: bool = False,
    weighted_mincut: bool = False,
) -> nx.DiGraph | None:
    """Least resolved tree of a colored digraph, as an ``nx.DiGraph``.

    Thin wrapper around ``asymmetree.analysis.best_matches.lrt_from_colored_graph``.
    Returns ``None`` exactly when that function does, i.e. when ``graph`` is not
    a BMG and no mincut heuristic was requested.
    """
    tree = _bm.lrt_from_colored_graph(graph, mincut=mincut, weighted_mincut=weighted_mincut)
    if tree is None:
        return None
    return tralda_to_digraph(tree)


def lrt_from_two_colored_bmg(graph: nx.DiGraph) -> nx.DiGraph | None:
    """Least resolved tree of a 2-colored BMG (linear-time algorithm)."""
    tree = _bm.lrt_from_2bmg(graph)
    if tree is None:
        return None
    return tralda_to_digraph(tree)


def lrt_of_tree(tree: nx.DiGraph) -> nx.DiGraph:
    """Least resolved tree *of a gene tree*, by contracting redundant edges."""
    return tralda_to_digraph(_bm.lrt_from_tree(digraph_to_tralda(tree)))


def asymmetree_bmg(tree: nx.DiGraph) -> nx.DiGraph:
    """BMG of a gene tree computed by AsymmeTree (independent oracle)."""
    return _bm.bmg_from_tree(digraph_to_tralda(tree))


def is_bmg(graph: nx.DiGraph) -> bool:
    """True iff ``graph`` is a (tree-)BMG, via AsymmeTree's characterisation."""
    return _bm.is_bmg(graph) is not None


def lrt_cross_check(gene_tree: nx.DiGraph, bmg: nx.DiGraph) -> bool:
    """``LRT(T)`` and ``LRT(G(T))`` have to be the same tree.

    Used as a consistency oracle in the tests: if our ``bmg_from_network`` and
    AsymmeTree's ``bmg_from_tree`` disagree, or if either LRT route is wrong,
    this returns ``False``.
    """
    from utils.graph_utils import same_phylogeny

    from_tree = lrt_of_tree(gene_tree)
    from_graph = lrt_from_bmg(bmg)
    if from_graph is None:
        return False
    return same_phylogeny(from_tree, from_graph)


def lrt_target(gene_tree: nx.DiGraph) -> tuple[nx.DiGraph, nx.DiGraph]:
    """Task 2a in one call: ``(tree-BMG, least resolved tree T*)``.

    The BMG is taken from our own ``bmg_from_network`` (the object the rest of
    the pipeline works with) and ``T*`` is computed from that graph, so the
    target really is the LRT *of the graph we are trying to explain*. The
    leaves of ``T*`` are the vertices of the BMG.
    """
    from utils.graph_utils import bmg_from_network

    bmg = bmg_from_network(gene_tree)
    star = lrt_from_bmg(bmg)
    if star is None:
        raise ValueError("the BMG of a tree must have a least resolved tree")
    return bmg, star
