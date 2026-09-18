"""utils/graph_editing.py"""
import networkx as nx
import pytest

from tests.helpers import colored_tree, edges
from utils.bic_cherry import bic_cherry_extension
from utils.graph_editing import (
    bmg_is_same, contract_edge, contract_into_parents, delete_parent_edge, find_twin_vertices,
    group_children, make_guard, merge_siblings, normalize, pull_down, pull_up,
    pull_up_to_common_ancestor, remove_dead_vertex, remove_one_to_one_vertex, remove_shortcut_edge,
    remove_single_child_vertex, remove_twin_vertex, transfer_edge, try_edit,
)
from utils.graph_utils import transform


@pytest.fixture
def T():
    return colored_tree([("r", "u"), ("r", "c"), ("u", "a"), ("u", "b")], {"a": 1, "b": 2, "c": 1})


# ---------------------------------------------------------------- structural
def test_pull_up(T):
    pull_up(T, "u", "a")                       # default target = only parent
    assert T.has_edge("r", "a") and not T.has_edge("u", "a")
    with pytest.raises(ValueError):
        pull_up(T, "u", "b", "c")              # c is not an ancestor


def test_pull_down():
    N = nx.DiGraph([("u", "t"), ("u", "v"), ("t", "a"), ("t", "b")])
    pull_down(N, "u", "v", "t")
    assert N.has_edge("t", "v") and not N.has_edge("u", "v")
    with pytest.raises(ValueError):
        pull_down(N.copy(), "u", "t", "a")     # leaf target


def test_reattach_rejects_cycles():
    N = nx.DiGraph([("r", "u"), ("u", "w"), ("w", "a"), ("r", "b"), ("u", "c")])
    with pytest.raises(ValueError):
        pull_down(N, "r", "u", "w")            # w is below u -> cycle


def test_delete_parent_edge():
    N = nx.DiGraph([("r", "p"), ("r", "q"), ("p", "v"), ("q", "v"), ("p", "a"), ("q", "b")])
    delete_parent_edge(N, "p", "v")
    assert list(N.predecessors("v")) == ["q"]
    with pytest.raises(ValueError):
        delete_parent_edge(N, "q", "v")        # only parent left


def test_pull_up_to_common_ancestor():
    N = nx.DiGraph([("c", "p1"), ("c", "p2"), ("p1", "x"), ("p1", "v"), ("p2", "v"), ("p2", "y")])
    pull_up_to_common_ancestor(N, "v", "c")
    assert list(N.predecessors("v")) == ["c"]


def test_contract_edge(T):
    contract_edge(T, "r", "u")
    assert set(T.successors("r")) == {"a", "b", "c"}
    with pytest.raises(ValueError):
        contract_edge(T, "r", "a")             # outer edge


def test_transfer_edge():
    N = nx.DiGraph([("p", "u"), ("p", "t"), ("u", "a"), ("u", "v"), ("t", "b"), ("t", "c")])
    transfer_edge(N, "u", "v", "p", "t")
    assert N.has_edge("t", "v") and not N.has_edge("u", "v") and not N.has_edge("p", "v")


def test_merge_siblings():
    N = nx.DiGraph([("p", "u"), ("p", "w"), ("u", "a"), ("u", "x"), ("w", "x"), ("w", "c")])
    merge_siblings(N, "u", "w")
    assert "w" not in N and set(N.successors("u")) == {"a", "x", "c"}
    with pytest.raises(ValueError):
        merge_siblings(N, "u", "a")


def test_group_children():
    N = nx.DiGraph([("u", c) for c in ("c1", "c2", "c3")])
    group_children(N, "u", "c1", "c2")
    assert set(N.successors("g0")) == {"c1", "c2"} and set(N.successors("u")) == {"g0", "c3"}
    with pytest.raises(ValueError):
        group_children(nx.DiGraph([("u", "a"), ("u", "b")]), "u", "a", "b")


def test_contract_into_parents():
    N = nx.DiGraph([("p1", "v"), ("p2", "v"), ("v", "a"), ("v", "b")])
    contract_into_parents(N, "v")
    assert edges(N) == {("p1", "a"), ("p1", "b"), ("p2", "a"), ("p2", "b")}
    with pytest.raises(ValueError):
        contract_into_parents(N, "p1")         # root


# ---------------------------------------------------------------- invariant
def test_twins():
    N = nx.DiGraph([("r", "p"), ("r", "q"), ("p", "a"), ("p", "b"), ("q", "a"), ("q", "b")])
    assert find_twin_vertices(N) == [["p", "q"]]
    remove_twin_vertex(N, "q")
    assert "q" not in N
    with pytest.raises(ValueError):
        remove_twin_vertex(N, "p")


def test_single_child_and_one_to_one():
    N = nx.DiGraph([("p1", "v"), ("p2", "v"), ("v", "c"), ("c", "a"), ("c", "b")])
    with pytest.raises(ValueError):
        remove_one_to_one_vertex(N.copy(), "v")        # two parents
    remove_single_child_vertex(N, "v")
    assert set(N.predecessors("c")) == {"p1", "p2"}


def test_dead_vertex_and_shortcut():
    N = nx.DiGraph([("r", "d"), ("r", "u"), ("u", "a"), ("u", "b"), ("r", "a")])
    remove_dead_vertex(N, "d", leaves={"a", "b"})
    assert "d" not in N
    with pytest.raises(ValueError):
        remove_dead_vertex(N, "a", leaves={"a", "b"})
    remove_shortcut_edge(N, "r", "a")
    assert not N.has_edge("r", "a")
    with pytest.raises(ValueError):
        remove_shortcut_edge(N, "u", "b")


def test_normalize_on_bic_network(example_tree):
    from utils.graph_utils import bmg_from_network
    N = bic_cherry_extension(bmg_from_network(example_tree))
    stats = normalize(N, {"a1", "a2", "b1"})
    assert stats["twins"] >= 1 and stats["shortcuts"] >= 1
    assert not find_twin_vertices(N)
    assert all(N.out_degree(v) != 1 for v in N)


def test_invariant_edits_preserve_bmg_and_wbmg(instances):
    for d in instances[:15]:
        for N in (bic_cherry_extension(d.bmg), transform(d.gene_tree, 4)):
            leaves = {v for v in N if N.out_degree(v) == 0}
            gb, gw = make_guard(N, "bmg"), make_guard(N, "wbmg")
            M = N.copy()
            normalize(M, leaves)
            assert gb(M) and gw(M)
            assert {v for v in M if M.out_degree(v) == 0} == leaves


# ---------------------------------------------------------------- guard
def test_make_guard_modes(T):
    assert make_guard(T, "bmg")(T.copy()) and make_guard(T, "wbmg")(T.copy())
    with pytest.raises(ValueError):
        make_guard(T, "nope")


def test_try_edit_never_mutates(T):
    res, applied = try_edit(T, pull_up, "u", "a", target="r")
    assert applied and res is not T and T.has_edge("u", "a")
    res, applied = try_edit(T, pull_up, "u", "a", target="c")     # ValueError
    assert not applied and res is T


def test_guard_rejects_bmg_change():
    T = colored_tree([("r", "x"), ("r", "y"), ("x", "a1"), ("x", "b1"), ("y", "a2"), ("y", "b2")],
                     {"a1": "a", "a2": "a", "b1": "b", "b2": "b"})
    guard = make_guard(T)
    res, applied = try_edit(T, contract_into_parents, "x", still_valid=lambda _, c: guard(c))
    assert not applied and res is T
    C = T.copy()
    contract_into_parents(C, "x")
    assert not bmg_is_same(T, C)
