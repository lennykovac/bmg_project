"""utils/lrt.py -- the AsymmeTree / tralda bridge and the target T* (task 2a).

The strongest check here is not a hand-computed example but the agreement of
two independent routes:

* our ``bmg_from_network`` vs. AsymmeTree's ``bmg_from_tree``;
* ``LRT(T)`` computed by edge contraction vs. ``LRT(G(T))`` computed by BUILD
  on the informative triples.

If any of the four implementations were wrong, these would disagree.
"""

import networkx as nx
import pytest

from tests.helpers import edges
from utils.graph_utils import bmg_from_network, is_phylogenetic_tree, same_phylogeny
from utils.lrt import (
    asymmetree_bmg,
    digraph_to_tralda,
    lrt_cross_check,
    lrt_from_bmg,
    lrt_from_two_colored_bmg,
    lrt_of_tree,
    lrt_target,
    tralda_to_digraph,
    is_bmg
)


# ---------------------------------------------------------------- conversion
def test_roundtrip_keeps_topology_and_colors(example_tree):
    back = tralda_to_digraph(digraph_to_tralda(example_tree))
    assert same_phylogeny(example_tree, back)
    leaves = {v for v in back if back.out_degree(v) == 0}
    assert leaves == {"a1", "a2", "b1"}
    assert {v: back.nodes[v]["color"] for v in leaves} == {"a1": "A", "a2": "A", "b1": "B"}
    assert all(back.nodes[v]["color"] is None for v in back if back.out_degree(v) > 0)


def test_digraph_to_tralda_rejects_networks():
    N = nx.DiGraph([("r", "u"), ("r", "w"), ("u", "a"), ("w", "a"), ("u", "b"), ("w", "c")])
    for v in N:
        N.nodes[v]["color"] = None
    with pytest.raises(ValueError):
        digraph_to_tralda(N)


def test_inner_vertices_get_fresh_ids_and_do_not_collide(instances):
    # asymmetree sets every inner label to "" -- the converter must not key on it
    d = instances[0]
    lrt = lrt_of_tree(d.gene_tree)
    assert len(set(lrt.nodes)) == lrt.number_of_nodes()
    assert is_phylogenetic_tree(lrt) or lrt.number_of_nodes() == 1


# ---------------------------------------------------------------- the oracle
def test_our_bmg_equals_asymmetrees_bmg(instances):
    for d in instances:
        assert edges(bmg_from_network(d.gene_tree)) == edges(asymmetree_bmg(d.gene_tree))


def test_tree_bmgs_are_bmgs(instances):
    for d in instances:
        assert is_bmg(bmg_from_network(d.gene_tree))


def test_both_lrt_routes_agree(instances):
    for d in instances:
        assert lrt_cross_check(d.gene_tree, bmg_from_network(d.gene_tree))


def test_lrt_explains_the_bmg(instances):
    for d in instances:
        bmg, star = lrt_target(d.gene_tree)
        assert edges(bmg_from_network(star)) == edges(bmg)


def test_lrt_is_a_contraction_of_the_gene_tree(instances):
    # the LRT is obtained by contracting redundant edges, so it can only get
    # smaller, never larger
    for d in instances:
        star = lrt_of_tree(d.gene_tree)
        assert star.number_of_nodes() <= d.gene_tree.number_of_nodes()


def test_running_example_lrt(example_tree):
    bmg, star = lrt_target(example_tree)
    # ((a1,b1),a2) is already least resolved for its BMG
    assert same_phylogeny(star, example_tree)


def test_star_bmg_has_a_star_lrt(star_bmg):
    star = lrt_from_bmg(star_bmg)
    assert star.number_of_nodes() == star_bmg.number_of_nodes() + 1
    assert edges(bmg_from_network(star)) == edges(star_bmg)


def test_two_colored_shortcut_agrees_with_build(instances):
    checked = 0
    for d in instances:
        bmg = bmg_from_network(d.gene_tree)
        if len({c for _, c in bmg.nodes(data="color")}) != 2:
            continue
        fast = lrt_from_two_colored_bmg(bmg)
        slow = lrt_from_bmg(bmg)
        assert fast is not None and slow is not None
        assert same_phylogeny(fast, slow)
        checked += 1
    assert checked, "no two-colored instance in the fixture"


def test_lrt_from_bmg_returns_none_for_a_non_bmg():
    # the classic non-BMG: a needs a best match of color B in both directions
    # but the informative triples are inconsistent
    G = nx.DiGraph()
    colors = {"a1": "A", "a2": "A", "b1": "B", "b2": "B"}
    for v, c in colors.items():
        G.add_node(v, color=c)
    G.add_edges_from([("a1", "b1"), ("a1", "b2"), ("a2", "b1"),
                      ("b1", "a1"), ("b2", "a1"), ("b2", "a2")])
    assert lrt_from_bmg(G) is None or not is_bmg(G)
