"""The search: correctness of reported paths and the known 2f facts."""
import pytest

from utils.bic_cherry import bic_cherry_expansion
from task_2_utils.edit_search import ATOMIC, WITH_MACROS, find_path
from task_2_utils.experiments import complete_bmg, generate_instances
from utils.graph_utils import bmg_from_network, same_phylogeny
from utils.lrt import lrt_from_bmg

TREES = generate_instances(10, min_leaves=4, max_leaves=8, max_species=3, seed=3)


@pytest.mark.parametrize("tree", TREES)
def test_macro_strict_reaches_lrt_and_path_replays(tree):
    G = bmg_from_network(tree)
    T = lrt_from_bmg(G)
    N = bic_cherry_expansion(G, restricted=True)
    r = find_path(N, T, kinds=WITH_MACROS, mode="strict")
    assert r.reached
    assert same_phylogeny(r.final, T)
    # replaying the atomic path with the networkx reference ends in T*,
    # and T* explains G
    assert r.verified


def test_complete_bmg_is_frozen_for_atomic_strict():
    """2f: no atomic path at all (proof by exhaustion), macro path of length 1."""
    G = complete_bmg((1, 2))
    T = lrt_from_bmg(G)
    N = bic_cherry_expansion(G, restricted=True)
    atomic = find_path(N, T, kinds=ATOMIC, mode="strict", prune=False, fallback=False)
    assert not atomic.reached and atomic.exhausted
    macro = find_path(N, T, kinds=WITH_MACROS, mode="strict")
    assert macro.reached and macro.macro_steps == 1 and macro.verified
