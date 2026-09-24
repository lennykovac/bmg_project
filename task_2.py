from utils.graph_utils import *
from utils.tree_utils import create_gene_tree_n_leaves, import_good_trees
from utils.bic_cherry import bic_cherry_expansion
from task_2_utils.expansions import network_depth
from task_2_utils.report import table
from utils.lrt import (
    lrt_from_bmg,
    lrt_of_tree,
    is_bmg,
)
from utils.graph_utils import same_phylogeny
from task_2_utils.edit_search import CONFIGS, ATOMIC, WITH_MACROS, find_path, solve_many
from task_2_utils.failures import proven_failure, minimal_counterexample, signature
from task_2_utils.fast_edit import Net, cluster_set

from task_2_utils.experiments import (
    arcs,
    compare_searches,
    complete_bmg,
    explains,
    minimize_bmg,
    pair_scan,
    single_move_scan,
    strict_search_fails,
)



'''
(a) Construct tree-BMGs (G, σ) from (AsymmeTree-generated) trees and
compute their least resolved trees. This is the target.
'''

leaves = 4
species = 2
species_tree_age = 1

tree = create_gene_tree_n_leaves(leaves, species, species_tree_age).gene_tree
# trees: list[nx.DiGraph] = import_good_trees()
tree_bmg = bmg_from_network(tree)
tree_lrt  = lrt_of_tree(tree)
# the two routes to T* have to agree: LRT(T) == LRT(G(T))
assert same_phylogeny(tree_lrt, lrt_from_bmg(tree_bmg)), "LRT(T) != LRT(G(T))"

leaves = leaves_from_network(tree)
for leaf in leaves:
    color = tree.nodes[leaf].get("color", "Keine Farbe")
    # print(f"Blatt {leaf!r} hat die Farbe/Spezies: {color}")

'''
(b) Take the tree-BMGs from (a) and construct the BIC-cherry+expansion
explanations (N, σ).
'''

N_bic_cherry = bic_cherry_expansion(tree_bmg, restricted=True)
assert explains(N_bic_cherry, tree_bmg), "BIC-cherry expansion does not explain G"

print(f"""
(a) gene tree: {len(leaves)} genes, {len({tree.nodes[v]['color'] for v in leaves})} species
    BMG: {tree_bmg.number_of_edges()} arcs, T*: {tree_lrt.number_of_nodes()} vertices
(b) BIC-cherry+expansion N: {N_bic_cherry.number_of_nodes()} vertices, \
{N_bic_cherry.number_of_edges()} arcs, explains G: True""")

'''
(c) Implement graph editing by “pulling up” or “pulling down” edges,
and removing vertices with the same parents and children, as de-
scribed in class.
'''

from task_2_utils.graph_editing import (
    pull_down,
    pull_up,
    merge_twins,  
)
from task_2_utils.graph_editing import *


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
wbmg_N = bmg_from_network(N_bic_cherry, weak=True)
 
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
scan = single_move_scan(N_bic_cherry, EXTENDED_MOVES)
totals = scan["ALL"]
 
# The interesting case is the one a per-step guard forbids: a first move that
# destroys the graph and a second that repairs it. Those pairs exist, which is
# why "every intermediate is valid" is strictly stronger than "the endpoints
# agree" -- see (f).
pairs = pair_scan(N_bic_cherry, EXTENDED_MOVES)
 
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
the least resolved tree-explanation (lrt) T*.
'''

# Two knobs, four configurations (task_2_utils.edit_search.CONFIGS):
#   atomic  the moves of (c) only             macro  + merge / contract / merge_group
#   strict  every search step explains G      relaxed  non-explaining steps penalised
# A macro is a fixed sequence of pull downs / pull ups; the path printed and
# replayed is always the atomic one. "broken atomic" counts the atomic
# intermediates that do not explain G (inside macros for strict).
#
# Guidance: every vertex v is a *fragment* of phi(v) = lca_{T*}(C(v)). Twins
# are merged first, then fragments are contracted into a parent of the same
# target vertex, then siblings that are fragments of the same target vertex
# are merged (all at once if possible). Merging fragments of different target
# vertices is not proposed. The search descends on the first improving,
# explaining candidate and backtracks lazily.
#
# The four searches are independent -> solve_many runs them in parallel
# processes (os.cpu_count() workers).

jobs = [(label, N_bic_cherry, tree_lrt, dict(cfg)) for label, cfg in CONFIGS.items()]
results = dict(solve_many(jobs))
best = results["macro/strict"]

print(f'''
(e) edit paths N -> T*
{table(("configuration", "-> T*", "atomic moves", "search steps", "broken steps",
        "broken atomic", "replay ok", "time"),
       [(label, r.reached, len(r.path), r.macro_steps, r.broken_steps,
         r.broken_atomic, r.verified, f"{r.seconds:.2f}s")
        for label, r in results.items()])}''')

if best.reached:
    print(f'''
    recommended: macro moves, strict invariant
    reached T* in {best.macro_steps} search steps = {len(best.path)} atomic moves
    same_phylogeny(result, T*) = {same_phylogeny(best.final, tree_lrt)},
    replayed with graph_editing.apply_move: {best.verified}
    first moves: {' -> '.join(map(str, best.path[:4]))} ...
    depth {network_depth(N_bic_cherry)} -> {network_depth(best.final)} (T* has {network_depth(tree_lrt)})''')
else:
    print(f"    macro/strict did NOT reach T*: best cost {best.final_cost} "
          f"after {best.expanded} candidates")


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
