import json
import random
import re

import numpy as np
import pytest

from task_2_utils.animation import animate, load_trace, replay, save_trace, trace_path, trace_search
from utils.bic_cherry import bic_cherry_expansion
from utils.graph_editing import DEFAULT_MOVES, EXTENDED_MOVES, Move
from utils.graph_utils import bmg_from_network
from utils.lrt import lrt_of_tree
from task_2_utils.search import STRATEGIES
from utils.tree_utils import create_gene_tree_n_leaves


def _search(seed, kinds=EXTENDED_MOVES, strict="prefer"):
    random.seed(seed)
    np.random.seed(seed)
    T = create_gene_tree_n_leaves(5, 3, 1).gene_tree
    G = bmg_from_network(T)
    L = lrt_of_tree(T)
    N = bic_cherry_expansion(G, restricted=True)
    r = STRATEGIES["greedy"](N, target=L, cost="target", kinds=kinds, strict=strict, restarts=3)
    return N, L, r


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("kinds,strict", [(EXTENDED_MOVES, "prefer"), (DEFAULT_MOVES, True)])
def test_replay_reproduces_the_search(seed, kinds, strict):
    N, L, r = _search(seed, kinds, strict)
    trace = trace_search(N, r, target=L)          # raises if the replay drifts
    assert trace.reached_target == r.reached_target
    assert len([f for f in trace.frames if f.kind == "move"]) == len(r.path)
    if strict is True:                            # strict search never leaves G
        assert all(f.explains for f in trace.frames)


def test_wrong_path_is_rejected():
    N, L, r = _search(0)
    with pytest.raises(ValueError):
        replay(N, [Move("remove_arc", ("no", "such arc"))])


@pytest.mark.parametrize("seed", range(3))
def test_save_load_roundtrip(seed, tmp_path):
    N, L, r = _search(seed)
    trace = trace_search(N, r, target=L)
    path = save_trace(trace, tmp_path / "t.json")
    data = json.loads(path.read_text())
    assert data["format"] == "bmg-edit-trace/1" and len(data["moves"]) == len(r.path)
    again = load_trace(path)
    assert again.moves == trace.moves
    assert set(again.final.edges) == set(r.network.edges)
    assert [f.explains for f in again.frames] == [f.explains for f in trace.frames]


def test_html_is_self_contained(tmp_path):
    N, L, r = _search(1)
    out = animate(trace_search(N, r, target=L), tmp_path / "a.html")
    html = open(out, encoding="utf-8").read()
    assert not re.search(r"<script[^>]+src=", html)       # no CDN, works offline
    data = json.loads(re.search(r"const DATA = (\{.*?\});\n", html, re.S).group(1))
    assert len(data["frames"]) == len(trace_path(N, r.path, target=L).frames)
    assert data["frames"][-1]["at_target"] == r.reached_target
