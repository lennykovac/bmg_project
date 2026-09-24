"""
Task 2 -- Simplification of explaining networks.

One tree, start to finish, printed as it goes. Every measurement is a call
into `utils/`; what is spelled out here is only which measurement belongs to
which sub-task. The same measurements over many trees: `batch_task_2.py`.
"""

import random

import numpy as np

from utils.bic_cherry import bic_cherry_expansion
from task_2_utils.expansions import network_depth
from task_2_task_2_utils.experiments import (
    arcs,
    compare_searches,
    complete_bmg,
    explains,
    minimize_bmg,
    pair_scan,
    single_move_scan,
    strict_search_fails,
)
from utils.graph_utils import (
    bmg_from_network,
    leaves_from_network,
    same_phylogeny,
    wbmg_from_network,
)
from utils.lrt import is_bmg, lrt_from_bmg, lrt_of_tree
from task_2_utils.report import table
from utils.tree_utils import create_gene_tree_n_leaves, import_good_trees
from utils.plotting import plot_graph
from task_2_utils.graph_editing import *
from task_2_utils.edit_search import ATOMIC, WITH_MACROS, find_path
from task_2_utils.failures import proven_failure, minimal_counterexample, signature
from utils.bic_cherry import bic_cherry_expansion as _bce

# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def tree_structure_str(network, max_depth=10) -> str:
    """Simple ASCII visualization of tree structure."""
    from utils.graph_utils import root_from_network
    
    lines = []
    root = root_from_network(network)
    
    def visit(node, indent=0, is_last=True):
        if indent > max_depth:
            return
        prefix = "└── " if is_last else "├── "
        lines.append("  " * indent + prefix + str(node))
        children = list(network.successors(node))
        for i, child in enumerate(children):
            visit(child, indent + 1, i == len(children) - 1)
    
    visit(root)
    return "\n".join(lines) if lines else "(empty tree)"


def leaves_and_colors_str(network) -> str:
    """Format leaves with their color assignments."""
    leaves = leaves_from_network(network)
    if not leaves:
        return "(no leaves)"
    
    color_groups = {}
    for leaf in leaves:
        color = network.nodes[leaf].get("color", "None")
        if color not in color_groups:
            color_groups[color] = []
        color_groups[color].append(leaf)
    
    lines = []
    for color in sorted(color_groups.keys(), key=str):
        nodes = color_groups[color]
        lines.append(f"  Color {color!r:>12}: {nodes}")
    return "\n".join(lines)


'''
(a) Construct tree-BMGs (G, σ) from (AsymmeTree-generated) trees and
compute their least resolved trees. This is the target.
'''

n_leaves = 12
n_species = 2
species_tree_age = 7

# AsymmeTree is stochastic; seed it so re-running the notebook while writing
# up the numbers does not quietly change them. SEED = 0 draws an instance that
# actually needs expansions -- raise n_leaves / n_species for a richer one.
SEED = 2
random.seed(SEED)
np.random.seed(SEED)

tree = create_gene_tree_n_leaves(n_leaves, n_species, species_tree_age).gene_tree
# trees: list[nx.DiGraph] = import_good_trees()
tree_bmg = bmg_from_network(tree)
tree_wbmg = wbmg_from_network(tree)
tree_lrt = lrt_of_tree(tree)
bmg_lrt = lrt_from_bmg(tree_bmg)
leaves = leaves_from_network(tree)

print(f"""
(a) GENE TREE STRUCTURE
{tree_structure_str(tree)}

    Leaves and color assignments:
{leaves_and_colors_str(tree)}

    Properties:
    - Gene tree: {tree.number_of_nodes()} vertices, {tree.number_of_edges()} edges, depth {network_depth(tree)}
    - Leaves: {len(leaves)}
    - BMG (strict best matches): {tree_bmg.number_of_nodes()} vertices, {tree_bmg.number_of_edges()} arcs
    - WBMG (weak best matches): {tree_wbmg.number_of_nodes()} vertices, {tree_wbmg.number_of_edges()} arcs
    - Is a valid BMG: {is_bmg(tree_bmg)}
    - BMG == WBMG: {arcs(tree_bmg) == arcs(tree_wbmg)}

    T* (Least Resolved Tree) - from gene tree:
{tree_structure_str(tree_lrt)}

    T* properties: {tree_lrt.number_of_nodes()} vertices, depth {network_depth(tree_lrt)}
    
    T* (Least Resolved Tree) - from BMG:
{tree_structure_str(bmg_lrt)}

    T* properties: {bmg_lrt.number_of_nodes()} vertices, depth {network_depth(bmg_lrt)}
    
    Verification:
    - LRT(T) == LRT(G(T)): {same_phylogeny(tree_lrt, bmg_lrt)}
    - LRT* explains BMG: {explains(tree_lrt, tree_bmg)}""")

print("""Plotting gene tree...""")
plot_graph(tree, title="Gene Tree", output_file="01_gene_tree.html")
print("""Plotting BMG...""")
plot_graph(tree_bmg, title="Best Match Graph (BMG)", output_file="02_bmg.html")
print("""Plotting T* (from gene tree)...""")
plot_graph(tree_lrt, title="Least Resolved Tree (LRT) from Gene Tree", output_file="03_lrt_tree.html")
print("""Plotting T* (from BMG)...""")
plot_graph(bmg_lrt, title="Least Resolved Tree (LRT) from BMG", output_file="04_lrt_bmg.html")


'''
(b) Take the tree-BMGs from (a) and construct the BIC-cherry+expansion
explanations (N, σ).
'''

N_bic_cherry = bic_cherry_expansion(tree_bmg, restricted=True)

print(f"""
(b) BIC-CHERRY EXPANSION NETWORK
    Structure:
    - Vertices: {N_bic_cherry.number_of_nodes()}
    - Arcs: {N_bic_cherry.number_of_edges()}
    - Depth (longest root-to-leaf path): {network_depth(N_bic_cherry)}
    
    Explanation power:
    - Explains strict BMG: {explains(N_bic_cherry, tree_bmg)}
    - Explains weak BMG: {explains(N_bic_cherry, tree_wbmg, weak=True)}
    
    Network leaves and colors:
{leaves_and_colors_str(N_bic_cherry)}
    
    Note: N explains G by construction; explaining WBMG is task 1b""")

print("""
    Plotting BIC-cherry expansion network...""")

plot_graph(N_bic_cherry, title="BIC-Cherry Expansion Network", output_file="05_bic_cherry_expansion.html")


'''
(c) Implement graph editing by "pulling up" or "pulling down" edges,
and removing vertices with the same parents and children, as de-
scribed in class.
'''

print("""
(c) IMPORTING GRAPH EDITING MOVES...""")

from task_2_utils.graph_editing import (
    DEFAULT_MOVES,
    EXTENDED_MOVES,
    NoLegalMove,
    combination,
    enumerate_moves,
    merge_twins,
    pull_down,
    pull_up,
    reduce_twins,
    single_moves,
    twin_pairs,
)

print(f"""
    ✓ Successfully imported:
      - Move sets: DEFAULT_MOVES, EXTENDED_MOVES
      - Exceptions: NoLegalMove
      - Basic moves: pull_up, pull_down, merge_twins, reduce_twins, twin_pairs
      - Combinators: single_moves, combination, enumerate_moves
    
    Available legal moves on N:
    - Basic moves (pull_up / pull_down / merge_twins): {len(enumerate_moves(N_bic_cherry, DEFAULT_MOVES))}
    - Extended (including arc deletion): {len(enumerate_moves(N_bic_cherry, EXTENDED_MOVES))}
    - Twin pairs awaiting merge: {len(twin_pairs(N_bic_cherry))}""")


'''
(d) Check both for single moves and combinations of a small number of
moves whether the modified network (N′, σ) has the same (weak) best match graph as (N, σ).

'''

# single_moves() applies ONE random move; combination() chains several.
# preserve_wbmg=True draws only among the moves that keep the WBMG, and
# combination() re-checks after EVERY step, so every intermediate network
# explains the same weak graph. Both raise NoLegalMove when nothing qualifies.

path_single: list = []
path_comb: list = []
path_guarded: list = []

# the reference is the WBMG of N itself -- NOT the tree's WBMG, which N does
# not explain in the first place (see (b))
wbmg_N = wbmg_from_network(N_bic_cherry)

N1_single = single_moves(N_bic_cherry, rng=0, record=path_single)
N1_comb = combination(N_bic_cherry, k=3, rng=0, record=path_comb)
try:
    N1_guarded = combination(
        N_bic_cherry, k=3, preserve_wbmg=True, rng=0, record=path_guarded)
    # True by construction -- if a move exists at all. On an expansion most
    # moves are safe anyway, so the guard often picks the same path as the
    # free draw; it earns its keep on the frozen instances of (f).
    guarded_line = (f"    combo   {' -> '.join(str(m) for m in path_guarded)}\n"
                    f"            WBMG kept: {explains(N1_guarded, wbmg_N, weak=True)}"
                    f"   (same path as the free draw: {path_guarded == path_comb})")
except NoLegalMove as exc:
    # not an edge case: for a complete bicolored BMG no move preserves the
    # graph at all, so the guarded editor cannot take a single step
    guarded_line = f"    no WBMG-preserving move exists -- {exc}"

print(f"""
(d) unguarded random edits
    single  {path_single[0]}
            WBMG kept: {explains(N1_single, wbmg_N, weak=True)}
    combo   {' -> '.join(str(m) for m in path_comb)}
            WBMG kept: {explains(N1_comb, wbmg_N, weak=True)}

    guarded (preserve_wbmg=True)
{guarded_line}""")

# One random draw says nothing about how *often* a move is safe. Every legal
# move, classified -- this is what task 2d actually asks.
scan = single_move_scan(N_bic_cherry, DEFAULT_MOVES)
totals = scan["ALL"]

# The interesting case is the one a per-step guard forbids: a first move that
# destroys the graph and a second that repairs it. Those pairs exist, which is
# why "every intermediate is valid" is strictly stronger than "the endpoints
# agree" -- see (f).
pairs = pair_scan(N_bic_cherry, DEFAULT_MOVES)

print(f"""
    every single move, classified
{table(("kind", "moves", "BMG kept", "WBMG kept", "both"),
       [(kind, row["total"], row["bmg"], row["wbmg"], row["both"])
        for kind, row in sorted(scan.items(), key=lambda kv: kv[0] == "ALL")])}
    -> {100 * totals['bmg'] / totals['total']:.0f}% keep the BMG, {100 * totals['wbmg'] / totals['total']:.0f}% keep the WBMG

    pairs of moves ({'exhaustive' if pairs['exhaustive'] else f"sampled: {pairs['first_moves']} first moves x all second moves"})
    {pairs['checked']} pairs, {pairs['kept']} keep the BMG, {pairs['rescued']} of those were
    'rescued' (first move broke it, second repaired it)""")


'''
(e) Devise heuristics to search for edit paths that allow the stepwise
conversion of the BIC-cherry+expansion explanations of (N, σ) in
the least resolved tree-explanation (lrt) T ∗ .
'''

from task_2_utils.search import RECOMMENDED

# The cost the descent minimises is a *guided* one: how far the cluster system
# of the current network is from that of T*, plus the reticulation number. A
# plain "is this cluster right" counter is flat almost everywhere and the
# search stalls after a handful of steps.
#
# strict=True      every intermediate network must explain G
# strict="prefer"  take an explaining step whenever one improves, and detour
#                  through a non-explaining network only when nothing else does


# results = compare_searches(N_bic_cherry, tree_lrt)
# best = results["extended/prefer"]

from task_2_utils.edit_search import find_path
best = find_path(N_bic_cherry, tree_lrt)   # macro moves, strict invariant

from task_2_utils.animation import trace_search, save_trace, animate

# the path of find_path starts at N itself (twin merges are part of it)
trace = trace_search(N_bic_cherry, best, target=tree_lrt,
                     reduce_first=False, reduce_each=False)
save_trace(trace, "results/trace_2e.json")
animate(trace, "07_bic_to_lrt.html")

if best.reached_target:
    # depth is not an invariant of the moves: it is 3 after the expansion and
    # whatever T* needs at the end
    verdict = (
        f"    reached T* in {len(best.path)} moves; "
        f"same_phylogeny(result, T*) = {same_phylogeny(best.network, tree_lrt)}\n"
        f"    first moves: {' -> '.join(str(m) for m in best.path[:4])} ...\n"
        f"    depth {network_depth(N_bic_cherry)} -> {network_depth(best.network)} "
        f"(T* has {network_depth(tree_lrt)})")
else:
    verdict = (f"    did NOT reach T*: stopped at cost {best.cost}, "
               f"{best.network.number_of_nodes()} vertices")

# print(f"""
# (e) greedy descent from N towards T*
# {table(("configuration", "steps", "-> T*", "tree", "explains", "cost"), [(label, len(r.path), r.reached_target, r.is_tree, r.explains, r.cost) for label, r in results.items()])}
#
#     recommended: extended moves, strict={RECOMMENDED['strict']!r}, {RECOMMENDED['restarts']} restarts
# {verdict}""")

if best.reached_target:
    print("""
    Plotting final simplified network...""")
    plot_graph(best.network, title="Simplified Network (reached T*)", output_file="06_final_simplified.html")


'''
(f) If this is not always successful, find minimal examples that fail, try to
find out why they fail, and if possible use this information to improve
the heuristic for editing schedules.
'''

# A failure of a heuristic can be a budget artefact. proven_failure() searches
# the *whole* reachable state space (no pruning); if it runs empty without
# meeting T*, no strict path with that move set exists -- a proof.
verdict = proven_failure(tree_bmg, kinds=ATOMIC)
if verdict:
    minimal = minimal_counterexample(tree_bmg, kinds=ATOMIC)
    origin = (f"this instance: no strict atomic path exists (search exhausted); "
              f"shrunk from {tree_bmg.number_of_nodes()} to {minimal.number_of_nodes()} genes")
else:
    minimal = complete_bmg((1, 2))
    origin = ("this instance: strict atomic path " +
              ("exists" if verdict is False else "undecided within budget") +
              " -- showing the minimal counterexample found in batch_task_2.py")

star = lrt_from_bmg(minimal)
N_min = bic_cherry_expansion(minimal, restricted=True)
N_min_red = reduce_twins(N_min)
frozen = single_move_scan(N_min_red, EXTENDED_MOVES)["ALL"]
atomic_min = find_path(N_min, star, kinds=ATOMIC, mode="strict", prune=False, fallback=False)
relaxed_min = find_path(N_min, star, kinds=ATOMIC, mode="relaxed")
macro_min = find_path(N_min, star, kinds=WITH_MACROS, mode="strict")

print(f'''
(f) failure analysis
    {origin}
    minimal counterexample: {minimal.number_of_nodes()} genes, \
{len({c for _, c in minimal.nodes(data='color')})} colours, {minimal.number_of_edges()} arcs, \
isomorphism class {signature(minimal)}
    T* is the star: {star.number_of_nodes() == minimal.number_of_nodes() + 1}
    N after twin merging: {N_min_red.number_of_nodes()} vertices
    atomic moves that keep the BMG: {frozen['bmg']} of {frozen['total']}
    atomic / strict : reached T* = {atomic_min.reached}, state space exhausted = {atomic_min.exhausted}
    atomic / relaxed: reached T* = {relaxed_min.reached} in {len(relaxed_min.path)} moves, \
{relaxed_min.broken_atomic} intermediate(s) not explaining G
    macro  / strict : reached T* = {macro_min.reached} in {macro_min.macro_steps} macro = \
{len(macro_min.path)} atomic moves: {' -> '.join(map(str, macro_min.path))}

    Why it fails
    ------------
    G is the complete bicolored graph on a1, a2 | b: every gene is a best match
    of every gene of the other colour, so the expansion adds nothing and N
    (after twin merging) is  rho -> p(a1,b), p(a2,b). The target is the star.
    b sits in two cherries. Any atomic move that takes b (or a_i) out of one
    cherry lifts lca(b, a_i) to rho while lca(b, a_j) stays at p(a_j, b):
    a_j is now strictly closer, the arc b -> a_i disappears. Every one of the
    moves changes G, so the strict search has no first step although the
    target is an ordinary tree. The obstruction is the *move set*, not the
    expansion: the two cherries of b have to be joined in one step.
    Every proven atomic failure in the batch run shrinks to exactly this graph.

    What fixes it
    -------------
    1. Check the invariant per *macro*: merge(p,u,w) = pull_down(p,w,u) followed
       by pull_up(w,c,u) for all children c. Its first atomic step breaks G,
       the last one repairs it -- exactly the 'rescued pairs' counted in (d).
    2. Guide merges by the target: merging fragments of *different* vertices of
       T* creates clusters T* does not have (a vertex straddling two subtrees)
       which the search could not take apart again; dropping those merges and
       merging all same-target siblings at once (merge_group) removed the
       remaining failures on the instances tested (see batch_task_2.py).''')
