"""Cross-checks of the fast engine against the networkx reference.

Run:  python -m pytest -q tests/
"""
import random

import networkx as nx
import pytest

from utils.bic_cherry import bic_cherry_expansion
from task_2_utils.fast_bmg import fast_bmg
from task_2_utils.fast_edit import (Move, Net, apply, best_match_arcs, cluster_set,
                             enumerate_candidates, target_cost, MOVE_KINDS)
from utils.graph_editing import (EXTENDED_MOVES, apply_move, enumerate_moves,
                                 reduce_twins, target_distance)
from utils.graph_editing import Move as RefMove
from utils.graph_utils import bmg_from_network, transform
from utils.lrt import lrt_from_bmg, lrt_of_tree
from utils.graph_utils import same_phylogeny
from utils.tree_utils import create_gene_tree_n_leaves


def _random_networks(n, seed=0):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    out = []
    while len(out) < n:
        t = create_gene_tree_n_leaves(random.randint(3, 8), random.randint(2, 3)).gene_tree
        leaves = [v for v in t if t.out_degree(v) == 0]
        if len({t.nodes[v]["color"] for v in leaves}) < 2:
            continue
        out.append(t)
        out.append(transform(t, random.randint(1, 3)))
    return out


NETS = _random_networks(15)


@pytest.mark.parametrize("N", NETS)
def test_bmg_bitsets_match_reference(N):
    ref_b = set(bmg_from_network(N).edges())
    ref_w = set(bmg_from_network(N, weak=True).edges())
    net = Net.from_nx(N)
    assert best_match_arcs(net) == ref_b
    assert best_match_arcs(net, weak=True) == ref_w
    fb, fw = fast_bmg(N)
    assert set(fb.edges()) == ref_b and set(fw.edges()) == ref_w


def _explanations():
    out = []
    for N in NETS[::2]:  # trees
        G = bmg_from_network(N)
        out.append((bic_cherry_expansion(G, restricted=True), lrt_from_bmg(G)))
    return out


@pytest.mark.parametrize("N,T", _explanations())
def test_atomic_moves_match_reference(N, T):
    leaves = {v for v in N if N.out_degree(v) == 0}
    net = Net.from_nx(N)
    want = cluster_set(Net.from_nx(T))
    for m in enumerate_moves(N, EXTENDED_MOVES)[:120]:
        try:
            ref = apply_move(N, m, leaves)
        except (ValueError, nx.NetworkXError):
            with pytest.raises(ValueError):
                apply(net, Move(m.kind, m.args), reduce=False)
            continue
        got, _ = apply(net, Move(m.kind, m.args), reduce=False)
        assert set(got.to_nx().edges()) == set(ref.edges()), m
        assert best_match_arcs(got) == set(bmg_from_network(ref).edges())
        assert target_cost(got, want) == target_distance(ref, T)


@pytest.mark.parametrize("N,T", _explanations())
def test_twin_reduction_matches_reference(N, T):
    from task_2_utils.fast_edit import reduce_twins_net
    got = Net.from_nx(N)
    reduce_twins_net(got)
    ref = reduce_twins(N)
    # vertex names may differ (which twin survives), compare up to relabelling
    assert got.n_vertices() == ref.number_of_nodes()
    assert best_match_arcs(got) == set(bmg_from_network(ref).edges())
    assert best_match_arcs(got) == set(bmg_from_network(N).edges())


@pytest.mark.parametrize("N,T", _explanations())
def test_macros_are_sequences_of_named_moves(N, T):
    """Replaying the reported atomic path with the reference gives the same net."""
    leaves = {v for v in N if N.out_degree(v) == 0}
    start = reduce_twins(N)
    net = Net.from_nx(start)
    for m in [c for c in enumerate_candidates(net) if c.kind in ("merge", "contract")][:40]:
        try:
            got, path = apply(net, m, reduce=False)
        except ValueError:
            continue
        ref = start
        for a in path:
            ref = apply_move(ref, RefMove(a.kind, a.args), leaves)
        assert set(got.to_nx().edges()) == set(ref.edges()), m


@pytest.mark.parametrize("N", NETS[::2])
def test_lrt_routes_agree(N):
    G = bmg_from_network(N)
    assert same_phylogeny(lrt_of_tree(N), lrt_from_bmg(G))
