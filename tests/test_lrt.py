import networkx as nx
import pytest

from utils.graph_editing import contract_edge, try_edit
from utils.graph_utils import bmg_from_network, transform, wbmg_from_network
from utils.lrt import compute_lrt, is_least_resolved
from utils.tree_utils import create_gene_tree


def node(color=None):
    return {"color": color}


@pytest.fixture
def chain_network():
    """R -> a -> b -> leaf, plus R -> leaf2."""
    g = nx.DiGraph()
    g.add_node("R", **node())
    g.add_node("a", **node())
    g.add_node("b", **node())
    g.add_node("leaf", **node("X"))
    g.add_node("leaf2", **node("Y"))
    g.add_edges_from([("R", "a"), ("a", "b"), ("b", "leaf"), ("R", "leaf2")])
    return g


class TestContractEdge:
    def test_contracts_internal_edge_reparenting_children(self, chain_network):
        contract_edge(chain_network, "R", "a")

        assert "a" not in chain_network
        assert chain_network.has_edge("R", "b")
        assert chain_network.has_edge("b", "leaf")
        assert chain_network.has_edge("R", "leaf2")  # untouched

    def test_missing_edge_raises(self, chain_network):
        with pytest.raises(ValueError):
            contract_edge(chain_network, "R", "b")  # (R, b) is not an edge

    def test_refuses_external_edge_into_a_leaf(self, chain_network):
        with pytest.raises(ValueError):
            contract_edge(chain_network, "b", "leaf")  # leaf is a leaf

    def test_refuses_when_v_has_more_than_one_parent(self):
        g = nx.DiGraph()
        g.add_node("R", **node())
        g.add_node("p1", **node())
        g.add_node("p2", **node())
        g.add_node("v", **node())
        g.add_node("x", **node("X"))
        g.add_edges_from([("R", "p1"), ("R", "p2"), ("p1", "v"), ("p2", "v"), ("v", "x")])
        with pytest.raises(ValueError):
            contract_edge(g, "p1", "v")

    def test_try_edit_integration_does_not_mutate_original(self, chain_network):
        before_edges = set(chain_network.edges())
        result, applied = try_edit(chain_network, contract_edge, "R", "a")
        assert applied is True
        assert set(chain_network.edges()) == before_edges  # original untouched
        assert "a" not in result


class TestComputeLrtHandExample:
    """The example from tree_bmg_lrt.html: x1, x2, x3 (color X) carry no
    information relevant to the BMG among themselves (same distance to
    y1 and y2), so B and C are redundant and the LRT flattens all three
    directly under A."""

    @pytest.fixture
    def hand_tree(self):
        T = nx.DiGraph()
        for n in ["r", "A", "B", "C"]:
            T.add_node(n, color=None)
        for leaf in ["x1", "x2", "x3"]:
            T.add_node(leaf, color="X")
        for leaf in ["y1", "y2"]:
            T.add_node(leaf, color="Y")
        T.add_edges_from([
            ("r", "A"), ("r", "y1"),
            ("A", "B"), ("A", "y2"),
            ("B", "x1"), ("B", "C"),
            ("C", "x2"), ("C", "x3"),
        ])
        return T

    @pytest.mark.parametrize("mode", ["bmg", "wbmg"])
    def test_lrt_matches_hand_computed_expectation(self, hand_tree, mode):
        reference_bmg = (bmg_from_network if mode == "bmg" else wbmg_from_network)(hand_tree)

        lrt, report = compute_lrt(hand_tree, mode=mode)

        # B and C must have been contracted -- the exact labels of the
        # intermediate steps may vary (e.g. after contracting A-B, B's
        # former child C is already a direct child of A, so the 2nd
        # contraction shows up as (A, C) instead of (B, C) -- same final
        # result, only the intermediate step's label changes). What
        # matters is the NUMBER of contractions and the final structure
        # (checked below).
        assert len(report.contracted_edges) == 2
        assert set(lrt.nodes) == {"r", "A", "y1", "y2", "x1", "x2", "x3"}
        assert set(lrt.successors("A")) == {"x1", "x2", "x3", "y2"}
        assert set(lrt.successors("r")) == {"A", "y1"}

        compute = bmg_from_network if mode == "bmg" else wbmg_from_network
        assert set(compute(lrt).edges()) == set(reference_bmg.edges())
        assert is_least_resolved(lrt, mode=mode)

    def test_contracting_one_more_edge_would_break_the_bmg(self, hand_tree):
        # sanity check that r-A is NOT redundant (unlike A-B, B-C)
        lrt, _ = compute_lrt(hand_tree, mode="bmg")
        still_valid_result, applied = try_edit(
            lrt, contract_edge, "r", "A",
            still_valid=lambda before, after: set(bmg_from_network(before).edges())
            == set(bmg_from_network(after).edges()),
        )
        assert applied is False


class TestComputeLrtRealTrees:
    @pytest.mark.parametrize("mode", ["bmg", "wbmg"])
    def test_lrt_preserves_bmg_and_is_a_genuine_fixed_point(self, mode):
        compute = bmg_from_network if mode == "bmg" else wbmg_from_network
        for seed_species in [3, 5, 8]:
            tree = create_gene_tree(species=seed_species, spt_age=1.0).gene_tree
            reference = compute(tree)

            lrt, report = compute_lrt(tree, mode=mode)

            assert set(compute(lrt).edges()) == set(reference.edges())
            assert is_least_resolved(lrt, mode=mode)
            # still a genuine tree (in-degree <= 1 at every vertex)
            assert all(lrt.in_degree(v) <= 1 for v in lrt.nodes)

    def test_lrt_is_never_larger_than_the_input_tree(self):
        tree = create_gene_tree(species=6, spt_age=1.0).gene_tree
        lrt, report = compute_lrt(tree, mode="bmg")
        assert lrt.number_of_nodes() <= tree.number_of_nodes()
        assert len(report.contracted_edges) == (
            tree.number_of_nodes() - lrt.number_of_nodes()
        )

    def test_lrt_of_hybridized_tree_still_preserves_bmg(self):
        # transform() inserts hybridization -- here we only check that
        # compute_lrt (which only contracts single-parent internal
        # edges) does nothing wrong when applied to a network that is no
        # longer a pure tree (it should simply find no contractible edge
        # at the multi-parent vertices and stop).
        tree = create_gene_tree(species=5, spt_age=1.0).gene_tree
        hybrid = transform(tree.copy(), 2)
        reference = bmg_from_network(hybrid)

        reduced, report = compute_lrt(hybrid, mode="bmg")

        assert set(bmg_from_network(reduced).edges()) == set(reference.edges())
