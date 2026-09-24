"""Tests for ``graph_editing.py`` (task 2c: edit moves on explaining networks).

Run from the project root:

    python -m pytest -q tests/task_2/test_graph_editing.py

Two kinds of tests:

* **exact** tests on small hand-built networks, where the expected result of a
  move is written down explicitly;
* **property** tests on BIC-cherry expansions of AsymmeTree tree-BMGs, checking
  invariants that must hold for *every* move (the result is a phylogenetic
  network, twin merging never changes the BMG/WBMG, ...).

The module is looked up in ``task_2_utils`` first and in ``utils`` second, so
the file works before and after the move of the task-2 modules.
"""
from __future__ import annotations

import random

import networkx as nx
import pytest

try:
    import task_2_utils.graph_editing as ge
except ImportError:  # pragma: no cover - old layout
    import utils.graph_editing as ge

from utils.bic_cherry import bic_cherry_expansion
from utils.graph_utils import bmg_from_network, clusters, is_phylogenetic_tree

Move = ge.Move


# ---------------------------------------------------------------------------
# helpers and fixtures
# ---------------------------------------------------------------------------

def net(arcs, colors):
    """Leaf-colored network from an arc list; ``colors`` maps leaf -> colour."""
    N = nx.DiGraph(arcs)
    for v in N:
        N.nodes[v]["color"] = colors.get(v)
    return N


def arcs(N):
    return set(N.edges())


def bmg_arcs(N, weak=False):
    return set(bmg_from_network(N, weak=weak).edges())


@pytest.fixture
def cherries():
    """Twin-reduced BIC-cherry network of the complete BMG on a | b1, b2.

         R
       /   \\
      p1    p2
     / \\   / \\
    a   b1 a   b2
    """
    return net([("R", "p1"), ("R", "p2"), ("p1", "a"), ("p1", "b1"),
                ("p2", "a"), ("p2", "b2")],
               {"a": 0, "b1": 1, "b2": 1})


@pytest.fixture
def tree():
    """((a1, b1)u, (a2, b2)w)R -- a binary tree, no reticulation."""
    return net([("R", "u"), ("R", "w"), ("u", "a1"), ("u", "b1"),
                ("w", "a2"), ("w", "b2")],
               {"a1": 0, "a2": 0, "b1": 1, "b2": 1})


@pytest.fixture
def twins():
    """R -> t1, t2, c; t1 and t2 both have children x, y (twins)."""
    return net([("R", "t1"), ("R", "t2"), ("R", "z"),
                ("t1", "x"), ("t1", "y"), ("t2", "x"), ("t2", "y")],
               {"x": 0, "y": 1, "z": 1})


def _instances(n=6, seed=5):
    from task_2_utils.experiments import generate_instances
    out = []
    for t in generate_instances(n, min_leaves=3, max_leaves=6, max_species=3, seed=seed):
        G = bmg_from_network(t)
        out.append(bic_cherry_expansion(G, restricted=True))
    return out


EXPANSIONS = _instances()


# ---------------------------------------------------------------------------
# invariants: is_network, reticulation_number, tree_likeness
# ---------------------------------------------------------------------------

class TestInvariants:
    def test_tree_is_network(self, tree):
        assert ge.is_network(tree)
        assert ge.is_network(tree, leaves={"a1", "a2", "b1", "b2"})

    def test_wrong_leaf_set(self, tree):
        assert not ge.is_network(tree, leaves={"a1", "a2", "b1"})

    def test_cycle_is_not_network(self):
        N = net([("R", "u"), ("u", "v"), ("v", "u"), ("u", "x"), ("R", "y")],
                {"x": 0, "y": 1})
        assert not ge.is_network(N)

    def test_two_roots_is_not_network(self):
        N = net([("R", "x"), ("R", "y"), ("S", "x"), ("S", "y")], {"x": 0, "y": 1})
        assert not ge.is_network(N)

    def test_single_child_vertex_is_not_network(self):
        N = net([("R", "u"), ("R", "y"), ("u", "x")], {"x": 0, "y": 1})
        assert not ge.is_network(N)

    def test_empty_graph_is_not_network(self):
        assert not ge.is_network(nx.DiGraph())

    def test_reticulation_number(self, tree, cherries):
        assert ge.reticulation_number(tree) == 0
        assert ge.reticulation_number(cherries) == 1      # a has two parents

    def test_tree_likeness(self, cherries):
        assert ge.tree_likeness(cherries) == (1, 6, 6)


# ---------------------------------------------------------------------------
# canonical_key
# ---------------------------------------------------------------------------

class TestCanonicalKey:
    def test_invariant_under_renaming_inner_vertices(self, tree):
        renamed = nx.relabel_nodes(tree, {"R": "root", "u": "X", "w": "Y"})
        assert ge.canonical_key(tree) == ge.canonical_key(renamed)

    def test_leaf_labelled(self, tree):
        # swapping two leaves of different cherries is a different phylogeny
        swapped = nx.relabel_nodes(tree, {"b1": "b2", "b2": "b1"})
        assert ge.canonical_key(tree) != ge.canonical_key(swapped)

    def test_distinguishes_shared_from_duplicated_vertex(self, twins):
        # twins t1, t2 carry the same signature; the multiset must see two of them
        merged = ge.merge_twins(twins, "t1", "t2")
        assert ge.canonical_key(twins) != ge.canonical_key(merged)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_removes_childless_inner_vertex(self, tree):
        N = tree.copy()
        N.add_node("dead", color=None)
        N.add_edge("u", "dead")
        M = ge.normalize(N, leaves={"a1", "a2", "b1", "b2"})
        assert "dead" not in M
        assert arcs(M) == arcs(tree)

    def test_suppresses_single_child_chain(self):
        N = net([("R", "s1"), ("s1", "s2"), ("s2", "u"), ("u", "x"), ("u", "y"),
                 ("R", "z")], {"x": 0, "y": 1, "z": 1})
        M = ge.normalize(N)
        assert arcs(M) == {("R", "u"), ("u", "x"), ("u", "y"), ("R", "z")}

    def test_suppresses_root_with_single_child(self):
        N = net([("R", "u"), ("u", "x"), ("u", "y")], {"x": 0, "y": 1})
        M = ge.normalize(N)
        assert arcs(M) == {("u", "x"), ("u", "y")}
        assert ge.is_network(M)

    def test_does_not_modify_input(self, tree):
        N = tree.copy()
        N.add_edge("u", "dead")
        before = arcs(N)
        ge.normalize(N, leaves={"a1", "a2", "b1", "b2"})
        assert arcs(N) == before

    def test_raises_if_a_leaf_is_lost(self, tree):
        N = tree.copy()
        N.remove_node("a1")
        with pytest.raises(ValueError):
            ge.normalize(N, leaves={"a1", "a2", "b1", "b2"})

    def test_single_child_suppression_keeps_all_parents(self, cherries):
        # s has two parents and one child; both parents must get the child
        N = net([("R", "p"), ("R", "q"), ("p", "s"), ("q", "s"), ("s", "x"),
                 ("p", "y"), ("q", "z")], {"x": 0, "y": 1, "z": 1})
        M = ge.normalize(N)
        assert {("p", "x"), ("q", "x")} <= arcs(M)
        assert "s" not in M


# ---------------------------------------------------------------------------
# the four moves
# ---------------------------------------------------------------------------

class TestPullUp:
    def test_exact_result(self, tree):
        M = ge.pull_up(tree, "u", "a1", "R")
        assert arcs(M) == arcs(tree) - {("u", "a1")} | {("R", "a1")}

    def test_degenerates_to_deletion_when_arc_exists(self, cherries):
        # pull a up from p1 to R, then from p2 to R: second time (R, a) exists
        M = ge.pull_up(cherries, "p1", "a", "R")
        M2 = ge.pull_up(M, "p2", "a", "R")
        assert not M2.has_edge("p2", "a") and M2.has_edge("R", "a")
        assert M2.number_of_edges() == M.number_of_edges() - 1

    def test_does_not_modify_input(self, tree):
        before = arcs(tree)
        ge.pull_up(tree, "u", "a1", "R")
        assert arcs(tree) == before

    @pytest.mark.parametrize("args", [
        ("u", "a2", "R"),     # (u, a2) is not an arc
        ("u", "a1", "w"),     # w is not a parent of u
    ])
    def test_illegal(self, tree, args):
        with pytest.raises(ValueError):
            ge.pull_up(tree, *args)

    def test_p_equals_v_is_illegal(self):
        # p is a parent of u and a child of u at the same time is impossible in a
        # DAG, so construct the degenerate call directly
        N = net([("R", "u"), ("u", "x"), ("u", "y"), ("R", "z")],
                {"x": 0, "y": 1, "z": 1})
        with pytest.raises(ValueError):
            ge.pull_up(N, "u", "R", "R")


class TestPullDown:
    def test_exact_result(self):
        N = net([("R", "u"), ("R", "w"), ("R", "x"), ("w", "y"), ("w", "z"),
                 ("u", "a"), ("u", "b")], {"x": 0, "y": 1, "z": 0, "a": 1, "b": 0})
        M = ge.pull_down(N, "R", "x", "w")
        assert arcs(M) == arcs(N) - {("R", "x")} | {("w", "x")}

    def test_leaf_target_is_illegal(self, tree):
        with pytest.raises(ValueError):
            ge.pull_down(tree, "u", "a1", "b1")

    def test_cycle_is_illegal(self):
        # v = u is an ancestor of w = c: pulling u down below c closes a cycle
        N = net([("R", "u"), ("R", "c"), ("u", "c"), ("u", "x"), ("c", "y"), ("c", "z")],
                {"x": 0, "y": 1, "z": 0})
        with pytest.raises(ValueError):
            ge.pull_down(N, "R", "u", "c")

    @pytest.mark.parametrize("args", [
        ("R", "a1", "u"),     # (R, a1) is not an arc
        ("R", "u", "x"),      # x is not a child of R
        ("R", "u", "u"),      # w == v
    ])
    def test_illegal(self, tree, args):
        with pytest.raises(ValueError):
            ge.pull_down(tree, *args)


class TestMergeTwins:
    def test_exact_result(self, twins):
        M = ge.merge_twins(twins, "t1", "t2")
        assert "t2" not in M and "t1" in M
        assert arcs(M) == arcs(twins) - {("R", "t2"), ("t2", "x"), ("t2", "y")}

    def test_not_twins_is_illegal(self, tree):
        with pytest.raises(ValueError):
            ge.merge_twins(tree, "u", "w")          # same parent, other children

    def test_leaves_are_not_merged(self, twins):
        with pytest.raises(ValueError):
            ge.merge_twins(twins, "x", "y")

    def test_same_vertex_is_illegal(self, twins):
        with pytest.raises(ValueError):
            ge.merge_twins(twins, "t1", "t1")

    def test_preserves_bmg_and_wbmg(self, twins):
        M = ge.merge_twins(twins, "t1", "t2")
        assert bmg_arcs(M) == bmg_arcs(twins)
        assert bmg_arcs(M, weak=True) == bmg_arcs(twins, weak=True)


class TestRemoveArc:
    def test_exact_result(self, cherries):
        M = ge.remove_arc(cherries, "p1", "a")
        assert arcs(M) == arcs(cherries) - {("p1", "a")}

    def test_last_parent_is_illegal(self, cherries):
        with pytest.raises(ValueError):
            ge.remove_arc(cherries, "p1", "b1")

    def test_not_an_arc_is_illegal(self, cherries):
        with pytest.raises(ValueError):
            ge.remove_arc(cherries, "p1", "b2")


# ---------------------------------------------------------------------------
# apply_move = move + normalisation
# ---------------------------------------------------------------------------

class TestApplyMove:
    def test_normalises(self, cherries):
        # removing (p1, a) leaves p1 with the single child b1 -> suppressed
        M = ge.apply_move(cherries, Move("remove_arc", ("p1", "a")))
        assert "p1" not in M and M.has_edge("R", "b1")
        assert ge.is_network(M, leaves={"a", "b1", "b2"})

    def test_unknown_kind(self, cherries):
        with pytest.raises(KeyError):
            ge.apply_move(cherries, Move("teleport", ("p1", "a")))

    def test_complete_bmg_to_star_in_two_moves(self, cherries):
        """The 'rescued pair' of 2d/2f: the first move breaks G, the second repairs it."""
        G0 = bmg_arcs(cherries)
        M1 = ge.apply_move(cherries, Move("pull_up", ("p1", "a", "R")))
        assert bmg_arcs(M1) != G0
        M2 = ge.apply_move(M1, Move("pull_up", ("p2", "a", "R")))
        assert bmg_arcs(M2) == G0
        assert is_phylogenetic_tree(M2)
        assert arcs(M2) == {("R", "a"), ("R", "b1"), ("R", "b2")}


# ---------------------------------------------------------------------------
# twins
# ---------------------------------------------------------------------------

class TestTwins:
    def test_twin_pairs(self, twins, tree):
        assert {frozenset(p) for p in ge.twin_pairs(twins)} == {frozenset({"t1", "t2"})}
        assert ge.twin_pairs(tree) == []

    def test_reduce_twins_on_bic_cherry_network(self):
        # every bicolored pair gets two copies p:x|y, p:y|x -> one each after reduction
        from task_2_utils.experiments import complete_bmg
        N = bic_cherry_expansion(complete_bmg((1, 2)), restricted=True)
        R = ge.reduce_twins(N)
        assert N.number_of_nodes() == 8 and R.number_of_nodes() == 6
        assert ge.twin_pairs(R) == []

    @pytest.mark.parametrize("N", EXPANSIONS)
    def test_reduce_twins_preserves_graphs_and_is_idempotent(self, N):
        R = ge.reduce_twins(N)
        assert bmg_arcs(R) == bmg_arcs(N)
        assert bmg_arcs(R, weak=True) == bmg_arcs(N, weak=True)
        assert ge.twin_pairs(R) == []
        assert arcs(ge.reduce_twins(R)) == arcs(R)


# ---------------------------------------------------------------------------
# cluster_multiset and target_distance
# ---------------------------------------------------------------------------

class TestCosts:
    def test_cluster_multiset(self, twins):
        # was broken (NameError: 'ml' instead of 'cl') before the fix
        m = ge.cluster_multiset(twins)
        assert m == {frozenset({"x", "y", "z"}): 1, frozenset({"x", "y"}): 2}

    def test_target_distance_zero_at_target(self, tree):
        assert ge.target_distance(tree, tree)[0] == 0

    def test_target_distance_counts_reticulations(self, cherries):
        star = net([("R", "a"), ("R", "b1"), ("R", "b2")], {"a": 0, "b1": 1, "b2": 1})
        cost = ge.target_distance(cherries, star)
        assert cost[1] == ge.reticulation_number(cherries) == 1
        assert cost[0] > 0

    def test_target_distance_decreases_along_the_rescued_pair(self, cherries):
        star = net([("R", "a"), ("R", "b1"), ("R", "b2")], {"a": 0, "b1": 1, "b2": 1})
        M1 = ge.apply_move(cherries, Move("pull_up", ("p1", "a", "R")))
        M2 = ge.apply_move(M1, Move("pull_up", ("p2", "a", "R")))
        d0, d1, d2 = (ge.target_distance(x, star)[0] for x in (cherries, M1, M2))
        assert d0 > d1 > d2 == 0


# ---------------------------------------------------------------------------
# enumerate_moves / legal_successors (property tests)
# ---------------------------------------------------------------------------

class TestEnumeration:
    def test_no_moves_on_a_single_cherry(self):
        N = net([("p", "x"), ("p", "y")], {"x": 0, "y": 1})
        assert ge.enumerate_moves(N, ge.EXTENDED_MOVES) == []

    def test_kinds_filter(self, cherries):
        kinds = {m.kind for m in ge.enumerate_moves(cherries, ("remove_arc",))}
        assert kinds <= {"remove_arc"}

    def test_deterministic_order(self, cherries):
        assert ge.enumerate_moves(cherries, ge.EXTENDED_MOVES) == \
            ge.enumerate_moves(cherries.copy(), ge.EXTENDED_MOVES)

    def test_complete_bmg_is_frozen(self, cherries):
        """2f: every legal atomic move changes the BMG."""
        G0 = bmg_arcs(cherries)
        succ = list(ge.legal_successors(cherries, ge.EXTENDED_MOVES))
        assert len(succ) == 8
        assert all(bmg_arcs(M) != G0 for _m, M in succ)

    @pytest.mark.parametrize("N", EXPANSIONS)
    def test_every_enumerated_move_is_legal_and_yields_a_network(self, N):
        N = ge.reduce_twins(N)
        leaves = {v for v in N if N.out_degree(v) == 0}
        moves = ge.enumerate_moves(N, ge.EXTENDED_MOVES)
        for m in random.Random(0).sample(moves, min(60, len(moves))):
            M = ge.apply_move(N, m, leaves)      # must not raise
            assert ge.is_network(M, leaves=leaves), m

    @pytest.mark.parametrize("N", EXPANSIONS)
    def test_moves_keep_the_leaf_set_and_clusters_of_leaves(self, N):
        leaves = {v for v in N if N.out_degree(v) == 0}
        for _m, M in list(ge.legal_successors(N, ge.DEFAULT_MOVES, leaves))[:40]:
            cl = clusters(M)
            assert all(cl[x] == frozenset([x]) for x in leaves)
            assert set(M) >= leaves


# ---------------------------------------------------------------------------
# random editing: single_moves / combination (task 2d)
# ---------------------------------------------------------------------------

class TestRandomEditing:
    def test_single_move_records_and_is_reproducible(self, tree):
        rec1, rec2 = [], []
        M1 = ge.single_moves(tree, rng=3, record=rec1)
        M2 = ge.single_moves(tree, rng=3, record=rec2)
        assert rec1 == rec2 and len(rec1) == 1
        assert arcs(M1) == arcs(M2)
        assert arcs(tree) == {("R", "u"), ("R", "w"), ("u", "a1"), ("u", "b1"),
                              ("w", "a2"), ("w", "b2")}           # input untouched

    def test_no_legal_move(self):
        N = net([("p", "x"), ("p", "y")], {"x": 0, "y": 1})
        with pytest.raises(ge.NoLegalMove):
            ge.single_moves(N, rng=0)

    def test_frozen_instance_raises_with_preserve_wbmg(self, cherries):
        with pytest.raises(ge.NoLegalMove, match="preserves the WBMG"):
            ge.single_moves(cherries, preserve_wbmg=True, kinds=ge.EXTENDED_MOVES, rng=0)

    @pytest.mark.parametrize("N", EXPANSIONS[:3])
    def test_preserve_wbmg_single(self, N):
        w0 = bmg_arcs(N, weak=True)
        try:
            M = ge.single_moves(N, preserve_wbmg=True, rng=1)
        except ge.NoLegalMove:
            pytest.skip("frozen instance")
        assert bmg_arcs(M, weak=True) == w0

    @pytest.mark.parametrize("N", EXPANSIONS[:3])
    def test_combination_checks_every_intermediate(self, N):
        """With preserve_wbmg=True *every* step keeps the WBMG, not only the end."""
        w0 = bmg_arcs(N, weak=True)
        leaves = {v for v in N if N.out_degree(v) == 0}
        rec = []
        try:
            M = ge.combination(N, k=3, preserve_wbmg=True, rng=2, record=rec)
        except ge.NoLegalMove:
            pytest.skip("no WBMG-preserving path of length 3")
        assert len(rec) == 3
        cur = N
        for m in rec:                                  # replay step by step
            cur = ge.apply_move(cur, m, leaves)
            assert bmg_arcs(cur, weak=True) == w0
        assert arcs(cur) == arcs(M)

    def test_combination_k_must_be_positive(self, tree):
        with pytest.raises(ValueError):
            ge.combination(tree, k=0)
