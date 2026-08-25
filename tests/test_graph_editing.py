import networkx as nx
import pytest

from utils.graph_editing import (
    find_twin_vertices,
    pull_down,
    pull_up,
    pull_up_to_common_ancestor,
    remove_redundant_vertex,
    remove_useless_vertex,
    try_edit,
)


def node(kind):
    return {"kind": kind}


@pytest.fixture
def diamond_network():
    """
    R -> p1 -> x
    R -> p1 -> y
    R -> p2 -> x
    R -> p2 -> z

    x has two parents (p1, p2), both children of the shared root R.
    """
    g = nx.DiGraph()
    g.add_node("R", **node("root"))
    g.add_node("p1", **node("cherry_parent"))
    g.add_node("p2", **node("cherry_parent"))
    for leaf in ("x", "y", "z"):
        g.add_node(leaf, **node("leaf"))
    g.add_edges_from([
        ("R", "p1"), ("R", "p2"),
        ("p1", "x"), ("p1", "y"),
        ("p2", "x"), ("p2", "z"),
    ])
    return g


@pytest.fixture
def chain_network():
    """R -> a -> b -> leaf, plus R -> leaf2 (a sibling edge to pull down)."""
    g = nx.DiGraph()
    g.add_node("R", **node("root"))
    g.add_node("a", **node("cherry_parent"))
    g.add_node("b", **node("cherry_parent"))
    g.add_node("leaf", **node("leaf"))
    g.add_node("leaf2", **node("leaf"))
    g.add_edges_from([("R", "a"), ("a", "b"), ("b", "leaf"), ("R", "leaf2")])
    return g

class TestPullUp:
    def test_moves_edge_source_to_explicit_ancestor(self, diamond_network):
        pull_up(diamond_network, "p1", "x", target="R")

        assert not diamond_network.has_edge("p1", "x")
        assert diamond_network.has_edge("R", "x")
        # unrelated edges untouched
        assert diamond_network.has_edge("p1", "y")
        assert diamond_network.has_edge("p2", "x")

    def test_default_target_is_the_sole_parent(self, chain_network):
        # b has exactly one parent (a) -> pull_up(b, leaf) with no target
        # slides the attachment from b up to a.
        pull_up(chain_network, "b", "leaf", target=None)

        assert not chain_network.has_edge("b", "leaf")
        assert chain_network.has_edge("a", "leaf")

    def test_ambiguous_default_target_raises(self, diamond_network):
        # x has two parents (p1, p2); pulling an edge that targets x's
        # *own* parent chain isn't the scenario here, but pulling up FROM
        # a vertex with 2 parents without saying which one is ambiguous.
        with pytest.raises(ValueError):
            pull_up(diamond_network, "R", "p1", target=None)

    def test_target_must_be_an_ancestor(self, diamond_network):
        # p2 is not an ancestor of p1 -- this is a sideways move, not "up".
        with pytest.raises(ValueError):
            pull_up(diamond_network, "p1", "x", target="p2")

    def test_missing_edge_raises(self, diamond_network):
        with pytest.raises(ValueError):
            pull_up(diamond_network, "p2", "y", target="R")


class TestPullDown:
    def test_moves_edge_source_to_explicit_descendant(self, chain_network):
        pull_down(chain_network, "R", "leaf2", target="a")

        assert not chain_network.has_edge("R", "leaf2")
        assert chain_network.has_edge("a", "leaf2")

    def test_target_must_be_a_descendant(self, chain_network):
        with pytest.raises(ValueError):
            pull_down(chain_network, "b", "leaf", target="R")

    def test_unrelated_target_raises(self, diamond_network):
        with pytest.raises(ValueError):
            pull_down(diamond_network, "R", "y", target="p2")

    def test_refuses_a_reattachment_that_would_create_a_cycle(self):
        """target is a genuine descendant of `a` (a->v->m->target), so the
        "descendant of u" precondition alone would let this through -- but
        target is ALSO a descendant of v itself, so reattaching (a, v) as
        (target, v) would close a cycle: v -> m -> target -> v. This is
        exactly the case pull_up can never hit (an ancestor of u can never
        be a descendant of v) but pull_down can, hence the extra
        acyclicity check in `_reattach`.
        """
        g = nx.DiGraph()
        for n in ("R", "a", "v", "m", "target"):
            g.add_node(n, **node("cherry_parent"))
        g.add_edges_from([("R", "a"), ("a", "v"), ("v", "m"), ("m", "target")])

        with pytest.raises(ValueError):
            pull_down(g, "a", "v", target="target")


def test_pull_up_to_common_ancestor_collapses_all_parents(diamond_network):
    pull_up_to_common_ancestor(diamond_network, "x", "R")

    assert set(diamond_network.predecessors("x")) == {"R"}
    # p1/p2 keep their other children -- only x's edges moved.
    assert diamond_network.has_edge("p1", "y")
    assert diamond_network.has_edge("p2", "z")


def test_pull_up_to_common_ancestor_is_a_noop_for_a_single_parent(chain_network):
    pull_up_to_common_ancestor(chain_network, "leaf2", "R")
    assert set(chain_network.predecessors("leaf2")) == {"R"}


@pytest.fixture
def twin_network():
    """p1 and p2 both end up with parent {R} and children {x, y} -- exact
    structural twins after some earlier edit collapsed their leaves onto
    the same pair."""
    g = nx.DiGraph()
    g.add_node("R", **node("root"))
    g.add_node("p1", **node("cherry_parent"))
    g.add_node("p2", **node("cherry_parent"))
    g.add_node("x", **node("leaf"))
    g.add_node("y", **node("leaf"))
    g.add_edges_from([
        ("R", "p1"), ("R", "p2"),
        ("p1", "x"), ("p1", "y"),
        ("p2", "x"), ("p2", "y"),
    ])
    return g


class TestTwinVertices:
    def test_find_twin_vertices_detects_the_group(self, twin_network):
        groups = find_twin_vertices(twin_network)
        assert groups == [["p1", "p2"]]

    def test_leaves_are_never_reported_as_twins(self, diamond_network):
        # y and z are both leaves with a single parent each, but distinct
        # parents (p1 vs p2) -- and leaves are excluded regardless.
        groups = find_twin_vertices(diamond_network)
        assert groups == []

    def test_remove_redundant_vertex_deletes_the_duplicate(self, twin_network):
        remove_redundant_vertex(twin_network, "p2")

        assert "p2" not in twin_network
        # p1 still provides every path p2 used to.
        assert twin_network.has_edge("R", "p1")
        assert twin_network.has_edge("p1", "x")
        assert twin_network.has_edge("p1", "y")

    def test_raises_without_a_twin(self, diamond_network):
        with pytest.raises(ValueError):
            remove_redundant_vertex(diamond_network, "p1")

    def test_refuses_to_remove_a_leaf(self, twin_network):
        with pytest.raises(ValueError):
            remove_redundant_vertex(twin_network, "x")


class TestRemoveUselessVertex:
    def test_suppresses_a_one_parent_one_child_vertex(self, chain_network):
        remove_useless_vertex(chain_network, "a")

        assert "a" not in chain_network
        assert chain_network.has_edge("R", "b")
        assert chain_network.has_edge("b", "leaf")

    def test_raises_for_a_vertex_with_more_than_one_child(self, diamond_network):
        with pytest.raises(ValueError):
            remove_useless_vertex(diamond_network, "p1")

    def test_refuses_to_suppress_a_leaf(self, chain_network):
        with pytest.raises(ValueError):
            remove_useless_vertex(chain_network, "leaf")


class TestTryEdit:
    def test_applies_and_returns_a_new_network_without_mutating_the_original(self, diamond_network):
        before_edges = set(diamond_network.edges())

        result, applied = try_edit(diamond_network, pull_up, "p1", "x", target="R")

        assert applied is True
        assert result is not diamond_network
        assert set(diamond_network.edges()) == before_edges  # original untouched
        assert result.has_edge("R", "x")

    def test_rejects_when_still_valid_returns_false(self, diamond_network):
        before_edges = set(diamond_network.edges())

        result, applied = try_edit(
            diamond_network, pull_up, "p1", "x", target="R",
            still_valid=lambda before, after: False,
        )

        assert applied is False
        assert result is diamond_network
        assert set(diamond_network.edges()) == before_edges

    def test_rejects_when_the_edit_itself_is_invalid(self, diamond_network):
        result, applied = try_edit(diamond_network, pull_up, "p1", "x", target="p2")

        assert applied is False
        assert result is diamond_network

    def test_still_valid_receives_before_and_after(self, diamond_network):
        seen = {}

        def still_valid(before, after):
            seen["before_has_edge"] = before.has_edge("p1", "x")
            seen["after_has_edge"] = after.has_edge("p1", "x")
            return True

        try_edit(diamond_network, pull_up, "p1", "x", target="R", still_valid=still_valid)

        assert seen == {"before_has_edge": True, "after_has_edge": False}
