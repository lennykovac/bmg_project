"""
Tests for hybrid nodes...
- After an insert, the donor and hybrid have the right parents and children, and the graph gains 2 nodes and 3 edges.
- The graph has no cycles, and the root and leaves stay the same.
- Pairs that would create a cycle and leave the graph unchanged.
- Invalid edges and a donor name that already exists raise ValueError.
"""

import random

import networkx as nx
import pytest

from utils.graph_utils import (
    add_hybrid_node,
    insert_node_on_edge,
    leaves_from_network,
    root_from_network,
    transform,
)

from utils.tree_utils import create_gene_tree_n_leaves

def hybrid_nodes(G: nx.DiGraph):
    return [n for n in G.nodes if G.in_degree(n) >= 2]


@pytest.fixture
def path_graph():
    """a -> b -> c -> d"""
    G = nx.DiGraph()
    G.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])
    return G


@pytest.fixture
def cherry_tree():
    """
    r -> x -> l1
    r -> x -> l2
    r -> y -> l3
    r -> y -> l4
    """
    G = nx.DiGraph()
    G.add_edges_from(
        [("r", "x"), ("r", "y"), ("x", "l1"), ("x", "l2"), ("y", "l3"), ("y", "l4")]
    )
    for leaf, color in [("l1", 0), ("l2", 1), ("l3", 0), ("l4", 1)]:
        G.nodes[leaf]["color"] = color
    return G


@pytest.fixture
def big_tree():
    """Balanced binary tree with 32 leaves, edges point away from the root 0."""
    return nx.balanced_tree(2, 5, create_using=nx.DiGraph)

"""
- The old edge is replaced by two new ones, so the graph gains one node and one edge, and the other edges stay the same.
- The new node gets color=None.
- Which nodes can reach which doesn't change.
- A missing edge, an edge given in reverse, or a node name that already exists raises ValueError and leaves the graph unchanged.
"""

class TestInsertNodeOnEdge:
    def test_subdivides_edge(self, path_graph):
        insert_node_on_edge("new", ("b", "c"), path_graph)

        assert not path_graph.has_edge("b", "c")
        assert path_graph.has_edge("b", "new")
        assert path_graph.has_edge("new", "c")
        assert list(path_graph.predecessors("new")) == ["b"]
        assert list(path_graph.successors("new")) == ["c"]

    def test_node_and_edge_counts(self, path_graph):
        n, m = path_graph.number_of_nodes(), path_graph.number_of_edges()
        insert_node_on_edge("new", ("a", "b"), path_graph)

        assert path_graph.number_of_nodes() == n + 1
        # one edge removed, two added
        assert path_graph.number_of_edges() == m + 1

    def test_new_node_has_color_attribute(self, path_graph):
        insert_node_on_edge("new", ("a", "b"), path_graph)
        assert "color" in path_graph.nodes["new"]
        assert path_graph.nodes["new"]["color"] is None

    def test_other_edges_untouched(self, path_graph):
        insert_node_on_edge("new", ("b", "c"), path_graph)
        assert path_graph.has_edge("a", "b")
        assert path_graph.has_edge("c", "d")

    def test_reachability_preserved(self, cherry_tree):
        before = {n: nx.descendants(cherry_tree, n) for n in cherry_tree}
        insert_node_on_edge("new", ("r", "x"), cherry_tree)

        for n, desc in before.items():
            assert nx.descendants(cherry_tree, n) - {"new"} == desc
        assert nx.is_directed_acyclic_graph(cherry_tree)

    def test_returns_none_and_is_inplace(self, path_graph):
        assert insert_node_on_edge("new", ("a", "b"), path_graph) is None
        assert "new" in path_graph


class TestAddHybridNode:
    def test_structure_after_insertion(self, cherry_tree):
        ok = add_hybrid_node(("x", "l1"), ("y", "l3"), "d", "h", cherry_tree)

        assert ok is True
        # donor subdivides x -> l1 and gets an extra child
        assert set(cherry_tree.predecessors("d")) == {"x"}
        assert set(cherry_tree.successors("d")) == {"l1", "h"}
        # hybrid subdivides y -> l3 and has two parents
        assert set(cherry_tree.predecessors("h")) == {"y", "d"}
        assert set(cherry_tree.successors("h")) == {"l3"}
        assert not cherry_tree.has_edge("x", "l1")
        assert not cherry_tree.has_edge("y", "l3")

    def test_counts(self, cherry_tree):
        n, m = cherry_tree.number_of_nodes(), cherry_tree.number_of_edges()
        add_hybrid_node(("x", "l1"), ("y", "l3"), "d", "h", cherry_tree)

        assert cherry_tree.number_of_nodes() == n + 2
        # each subdivision adds one edge, plus the donor -> hybrid edge
        assert cherry_tree.number_of_edges() == m + 3
        assert hybrid_nodes(cherry_tree) == ["h"]

    def test_stays_acyclic_with_same_root_and_leaves(self, cherry_tree):
        root = root_from_network(cherry_tree)
        leaves = set(leaves_from_network(cherry_tree))
        add_hybrid_node(("r", "x"), ("y", "l4"), "d", "h", cherry_tree)

        assert nx.is_directed_acyclic_graph(cherry_tree)
        assert root_from_network(cherry_tree) == root
        assert set(leaves_from_network(cherry_tree)) == leaves

    def test_donor_above_hybrid_on_path_is_allowed(self, path_graph):
        # donor on a -> b, hybrid on c -> d: a shortcut, no cycle
        assert add_hybrid_node(("a", "b"), ("c", "d"), "d0", "h0", path_graph)
        assert nx.is_directed_acyclic_graph(path_graph)
        assert path_graph.has_edge("d0", "h0")

    def test_adjacent_edges_donor_above_is_allowed(self, path_graph):
        # hybrid_edge[0] == donor_edge[1], still no cycle
        assert add_hybrid_node(("a", "b"), ("b", "c"), "d0", "h0", path_graph)
        assert nx.is_directed_acyclic_graph(path_graph)

    @pytest.mark.parametrize(
        "donor_edge, hybrid_edge, donor, hybrid",
        [
            (("x", "l1"), ("y", "l3"), "d", "l2"),   # hybrid name taken
            (("x", "l1"), ("x", "l1"), "d", "h"),    # same edge twice
            (("x", "l1"), ("y", "l3"), "d", "d"),    # same name twice
            (("x", "l1"), ("x", "l3"), "d", "h"),    # hybrid edge missing
        ],
    )
    def test_invalid_input_leaves_graph_untouched(
        self, cherry_tree, donor_edge, hybrid_edge, donor, hybrid
    ):
        nodes, edges = set(cherry_tree.nodes), set(cherry_tree.edges)
        with pytest.raises(ValueError):
            add_hybrid_node(donor_edge, hybrid_edge, donor, hybrid, cherry_tree)
        assert set(cherry_tree.nodes) == nodes 
        assert set(cherry_tree.edges) == edges


class TestTransform:

    @pytest.mark.exhaustive
    @pytest.mark.parametrize("seed", range(10))
    def test_result_is_valid_network(self, seed):
        random.seed(seed)
        # np.random.seed(seed)  # if AsymmeTree / your wrapper uses numpy

        for _ in range(50):
            t = create_gene_tree_n_leaves(leaves=random.randint(3, 250), species=2)
            g = t.gene_tree
            root = root_from_network(g)
            leaves = set(leaves_from_network(g))
            colors = {x: g.nodes[x].get("color") for x in leaves}
            n0, m0 = g.number_of_nodes(), g.number_of_edges()
            edges0 = set(g.edges)

            requested = random.randint(3, 20)
            result = transform(g, requested)
            k = len(hybrid_nodes(result))

            # input is not mutated
            assert set(g.edges) == edges0
            # still a valid network over the same leaves
            assert nx.is_directed_acyclic_graph(result)
            assert root_from_network(result) == root
            assert set(leaves_from_network(result)) == leaves
            assert all(result.nodes[x].get("color") == colors[x] for x in leaves)
            # exactly k successful insertions, never more than requested
            assert 0 < k <= requested
            assert result.number_of_nodes() == n0 + 2 * k
            assert result.number_of_edges() == m0 + 3 * k
            assert all(result.in_degree(h) == 2 for h in hybrid_nodes(result))

        
    def test_gets_requested_count_with_enough_attempts(self, big_tree):
        random.seed(42)
        result = transform(big_tree, 8, attempts=1000)
        assert len(hybrid_nodes(result)) == 8

    def test_repeated_transform_does_not_clash_labels(self, big_tree):
        random.seed(3)
        once = transform(big_tree, 3, attempts=1000)
        twice = transform(once, 3, attempts=1000)

        assert len(hybrid_nodes(twice)) == 6
        assert twice.number_of_nodes() == big_tree.number_of_nodes() + 12
        assert nx.is_directed_acyclic_graph(twice)

    def test_path_graph_terminates(self, path_graph):
        # many pairs on a path are refused, transform must still stop
        random.seed(0)
        result = transform(path_graph, 50, attempts=5)
        assert nx.is_directed_acyclic_graph(result)
        assert root_from_network(result) == "a"
        assert leaves_from_network(result) == ["d"]
