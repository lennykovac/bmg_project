import random

import networkx as nx
import numpy as np
import pytest

from utils.bic_cherry import bic_cherry_expansion
from utils.graph_utils import bmg_from_network
from task_2_utils.labels import class_names, leaf_labels, thinness_classes
from utils.lrt import lrt_from_bmg, lrt_of_tree
from utils.plotting import layout, plot_graph
from utils.tree_utils import create_gene_tree_n_leaves


def _instance(seed, n=8, species=3):
    random.seed(seed)
    np.random.seed(seed)
    T = create_gene_tree_n_leaves(n, species, 1).gene_tree
    return T, bmg_from_network(T)


def _toy_bmg():
    # colours 7 and 9 -> a, b; ids chosen so that id order != insertion order
    G = nx.DiGraph()
    for v, c in [("x10", 7), ("x2", 7), ("y1", 9), ("y3", 9), ("y2", 9)]:
        G.add_node(v, color=c)
    # x2, x10 have identical neighbourhoods; y1, y3 too; y2 differs
    G.add_edges_from([("x2", "y1"), ("x10", "y1"), ("x2", "y3"), ("x10", "y3"),
                      ("y1", "x2"), ("y1", "x10"), ("y3", "x2"), ("y3", "x10"),
                      ("y2", "x2"), ("y2", "x10")])
    return G


def test_names_follow_colour_letter_and_id_order():
    names = leaf_labels(_toy_bmg())
    assert names == {"x2": "a1", "x10": "a2", "y1": "b1", "y2": "b2", "y3": "b3"}


def test_classes_and_greek_names():
    G = _toy_bmg()
    classes = {frozenset(c) for c in thinness_classes(G)}
    assert classes == {frozenset({"x2", "x10"}), frozenset({"y1", "y3"}), frozenset({"y2"})}
    assert class_names(G) == {"x2": "α", "x10": "α", "y1": "β", "y3": "β"}  # y2: singleton, no class


def test_several_classes_of_one_colour_are_numbered():
    G = _toy_bmg()
    G.add_node("y4", color=9)
    G.add_edges_from([("y4", "x2"), ("y4", "x10")])  # y4 ~ y2 now
    names = class_names(G)
    assert names["y1"] == names["y3"] == "β1"
    assert names["y2"] == names["y4"] == "β2"


@pytest.mark.parametrize("seed", range(6))
def test_same_names_and_classes_everywhere(seed):
    T, G = _instance(seed)
    graphs = (T, G, lrt_of_tree(T), lrt_from_bmg(G), bic_cherry_expansion(G, restricted=True))
    names = [leaf_labels(H) for H in graphs]
    assert all(n == names[0] for n in names)
    assert all(T.nodes[v]["label"] == names[0][v] for v in names[0])  # stored at creation
    cls = [class_names(H) for H in graphs[:4]]
    assert all(c == cls[0] for c in cls)


@pytest.mark.parametrize("seed", range(6))
def test_class_members_are_adjacent_in_bmg_and_lrt(seed):
    T, G = _instance(seed)
    L = lrt_of_tree(T)
    names = class_names(G)
    for H in (G, L):
        pos = layout(H)
        # leaves of one colour sorted along their row/column
        order = sorted((v for v in H if v in G), key=lambda v: (H.nodes[v]["color"], pos[v][0], pos[v][1]))
        for cls in set(names.values()):
            idx = [i for i, v in enumerate(order) if names.get(v) == cls]
            assert idx == list(range(idx[0], idx[0] + len(idx)))


def test_plot_marks_classes(tmp_path):
    T, G = _instance(1, n=9)
    html = open(plot_graph(G, output_file=str(tmp_path / "g.html")), encoding="utf-8").read()
    assert "thinness classes:" in html
    assert "const HULLS" in html  # classes circled in the BMG


# ---------------------------------------------------------------------------
# collapsed drawing of the classes
# ---------------------------------------------------------------------------

def _shapes(html):
    import json
    import re

    nodes = json.loads(re.search(r"nodes = new vis\.DataSet\((\[.*?\])\);", html, re.S).group(1))
    return {n["id"]: n["shape"] for n in nodes}


@pytest.mark.parametrize("seed", range(6))
def test_quotient_bmg_is_well_defined(seed):
    from utils.plotting import collapse_thin_classes

    _, G = _instance(seed)
    names = class_names(G)
    H, members = collapse_thin_classes(G, names, "bmg")
    rep = {v: node for node, vs in members.items() for v in vs}
    image = lambda v: rep.get(v, v)  # noqa: E731
    assert set(H.edges) == {(image(u), image(v)) for u, v in G.edges}
    # every member pair realises every quotient arc -> nothing was invented
    for A, B in H.edges:
        for a in members.get(A, [A]):
            for b in members.get(B, [B]):
                assert G.has_edge(a, b)


def _hulls(html):
    import json
    import re

    m = re.search(r"const HULLS = (\[.*?\]);", html, re.S)
    return json.loads(m.group(1)) if m else []


@pytest.mark.parametrize("seed", range(6))
def test_gene_tree_plain_bmg_enclosed_lrt_collapsed(seed, tmp_path):
    T, G = _instance(seed, n=9)
    names = class_names(G)
    named = set(names.values())
    for H, mode in ((T, "plain"), (G, "enclosed"), (lrt_of_tree(T), "collapsed")):
        html = open(plot_graph(H, output_file=str(tmp_path / "p.html")), encoding="utf-8").read()
        shapes = _shapes(html)
        class_nodes = {i for i in shapes if i.startswith("class:")}
        assert all(shapes[i] == "dot" for i in shapes if i not in class_nodes)  # individuals round
        hulls = _hulls(html)
        if mode == "collapsed":
            assert {i[len("class:"):] for i in class_nodes} == named
            assert all(shapes[i] != "dot" for i in class_nodes) and not hulls
        elif mode == "enclosed":
            # every member is still its own round vertex, and circled with its class
            assert not class_nodes
            assert {h["name"] for h in hulls} == named
            for h in hulls:
                assert {str(v) for v, c in names.items() if c == h["name"]} == set(h["ids"])
        else:
            assert not class_nodes and not hulls
