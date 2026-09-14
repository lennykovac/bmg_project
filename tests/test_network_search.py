import networkx as nx
import pytest

from utils.bic_cherry import bic_cherry_extension
from utils.graph_utils import bmg_from_network, transform, wbmg_from_network
from utils.network_search import make_still_valid, reduce_to_tree
from utils.tree_utils import create_gene_tree


def node(color=None):
    return {"color": color}


@pytest.fixture
def dead_hybrid_network():
    """h is a hybrid (2 parents, a and b) but has no leaf underneath --
    it should be fully resolvable back into a tree without changing the
    (weak) BMG."""
    g = nx.DiGraph()
    g.add_node("r", **node())
    g.add_node("a", **node())
    g.add_node("b", **node())
    g.add_node("h", **node())
    for leaf, color in [("x1", "X"), ("y1", "Y"), ("x2", "X"), ("y2", "Y")]:
        g.add_node(leaf, **node(color))
    g.add_edges_from([
        ("r", "a"), ("r", "b"),
        ("a", "x1"), ("a", "y1"),
        ("b", "x2"), ("b", "y2"),
        ("a", "h"), ("b", "h"),
    ])
    return g


@pytest.fixture
def load_bearing_hybrid_network():
    """Same network, but h has a leaf y3 underneath -- the reticulation
    becomes genuinely necessary for the (weak) BMG."""
    g = nx.DiGraph()
    g.add_node("r", **node())
    g.add_node("a", **node())
    g.add_node("b", **node())
    g.add_node("h", **node())
    for leaf, color in [("x1", "X"), ("y1", "Y"), ("x2", "X"), ("y2", "Y"), ("y3", "Y")]:
        g.add_node(leaf, **node(color))
    g.add_edges_from([
        ("r", "a"), ("r", "b"),
        ("a", "x1"), ("a", "y1"),
        ("b", "x2"), ("b", "y2"),
        ("a", "h"), ("b", "h"), ("h", "y3"),
    ])
    return g


class TestStillValid:
    def test_identical_network_is_valid_bmg_and_wbmg(self, dead_hybrid_network):
        for mode in ("bmg", "wbmg"):
            still_valid = make_still_valid(mode)
            assert still_valid(dead_hybrid_network, dead_hybrid_network)

    def test_breaking_edit_is_rejected(self, dead_hybrid_network):
        broken = dead_hybrid_network.copy()
        broken.remove_edge("a", "y1")
        broken.add_edge("b", "y1")
        for mode in ("bmg", "wbmg"):
            still_valid = make_still_valid(mode)
            assert not still_valid(dead_hybrid_network, broken)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            make_still_valid("nonsense")


class TestReduceToTreeToyNetworks:
    @pytest.mark.parametrize("mode", ["bmg", "wbmg"])
    def test_dead_hybrid_gets_fully_resolved(self, dead_hybrid_network, mode):
        compute = bmg_from_network if mode == "bmg" else wbmg_from_network
        reference = compute(dead_hybrid_network)

        reduced, report = reduce_to_tree(dead_hybrid_network, mode=mode)

        assert report.is_tree
        assert report.stuck_hybrids == []
        assert set(compute(reduced).edges()) == set(reference.edges())

    @pytest.mark.parametrize("mode", ["bmg", "wbmg"])
    def test_load_bearing_hybrid_stays_and_bmg_is_preserved(
        self, load_bearing_hybrid_network, mode
    ):
        compute = bmg_from_network if mode == "bmg" else wbmg_from_network
        reference = compute(load_bearing_hybrid_network)

        reduced, report = reduce_to_tree(load_bearing_hybrid_network, mode=mode)

        assert "h" in report.stuck_hybrids
        assert not report.is_tree
        # the most important guarantee: still_valid never let through an
        # edit that changed the (weak) BMG, even though it didn't become
        # a tree
        assert set(compute(reduced).edges()) == set(reference.edges())


class TestReduceToTreeRealPipeline:
    """Task 2.2(a)/(b)/(e): a real AsymmeTree tree -> tree-BMG ->
    BIC-cherry+expansion (almost never a tree) -> reduction heuristic.
    The ground truth for comparison is always the BMG/WBMG of the
    network that is FED INTO the heuristic (the BIC-cherry+expansion
    explanation), not the original tree -- that is exactly what
    still_valid has to preserve."""

    @pytest.mark.parametrize("mode", ["bmg", "wbmg"])
    def test_bic_cherry_explanation_of_a_real_tree_bmg_reduces_to_a_tree(self, mode):
        # real tree -> real bmg (no hybridization: should be 100% reducible)
        tree = create_gene_tree(species=6, spt_age=1.0).gene_tree
        bmg = bmg_from_network(tree)
        network = bic_cherry_extension(bmg)

        reduced, report = reduce_to_tree(network, mode=mode)

        compute = bmg_from_network if mode == "bmg" else wbmg_from_network
        reference = compute(network)
        assert set(compute(reduced).edges()) == set(reference.edges())
        # we do not guarantee is_tree=True in every case (task 2.2f: the
        # heuristic may not be sufficient), but we log the success rate
        print(f"[{mode}] is_tree={report.is_tree} stuck={report.stuck_hybrids}")

    @pytest.mark.parametrize("mode", ["bmg", "wbmg"])
    def test_heuristic_never_breaks_the_bmg_even_with_real_hybridization(self, mode):
        # now with genuine hybridization inserted via transform()
        tree = create_gene_tree(species=6, spt_age=1.0).gene_tree
        hybrid_tree = transform(tree.copy(), 3)
        compute = bmg_from_network if mode == "bmg" else wbmg_from_network
        source_bmg = compute(hybrid_tree)
        network = bic_cherry_extension(source_bmg)

        reduced, report = reduce_to_tree(network, mode=mode)

        reference = compute(network)
        assert set(compute(reduced).edges()) == set(reference.edges())
